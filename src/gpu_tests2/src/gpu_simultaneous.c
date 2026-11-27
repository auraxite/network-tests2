#include "gpu_simultaneous.h"

#include <stdlib.h>
#include <string.h>

static int pair_tag(int src, int dst, int nproc) { return src * nproc + dst; }

/* Идёт ли передача src -> dst в раунде с корнем root (в all_to_all root не важен). */
static int active(Mode mode, int root, int src, int dst) {
	if (src == dst) return 0;
	if (mode == MODE_ONE_TO_ALL) return src == root;
	if (mode == MODE_ALL_TO_ONE) return dst == root;
	return 1;
}

typedef struct {
	int          rank, nproc;
	const Args  *args;
	char        *d_send, *d_recv;
	char       **recv_dev;       /* буфер приёма в памяти GPU на каждого соседа */
	double     **samples;        /* samples[src][i] */
	size_t      *n_samples;
	long        *v_checked, *v_bad, *v_early;
	cudaStream_t v_stream;
	long         iteration;      /* общий счётчик итераций: одинаков на всех рангах */
	MPI_Request *recv, *send;
	int         *got;
} State;

static void one_iteration(State *s, int root, int measure) {
	const Args *args  = s->args;
	const int   nproc = s->nproc, rank = s->rank;
	const int   count = (int)args->nbytes;
	const long  it    = s->iteration++;
	const int   check = args->verify && measure;

	if (check) {
		/* Новый шаблон в свой буфер отправки, нули в буферы приёма. До выставления приёмов
		   и до барьера, поэтому в замер не попадает. */
		int sends = 0;
		for (int dst = 0; dst < nproc; ++dst) sends = sends || active(args->mode, root, rank, dst);
		if (sends) verify_fill(s->d_send, args->nbytes, verify_tag(it, rank, 0));
		cuda_ok(cudaMemset(s->d_recv, 0, args->nbytes), "verify: cudaMemset(d_recv)");
		for (int src = 0; src < nproc; ++src)
			if (active(args->mode, root, src, rank))
				cuda_ok(cudaMemset(s->recv_dev[src], 0, args->nbytes), "verify: cudaMemset(recv)");
	}
	for (int i = 0; i < nproc; ++i) {
		s->recv[i] = MPI_REQUEST_NULL;
		s->send[i] = MPI_REQUEST_NULL;
		s->got[i]  = -1;
	}

	int expected = 0;
	for (int src = 0; src < nproc; ++src) {
		if (!active(args->mode, root, src, rank)) continue;
		++expected;
		mpi_ok(MPI_Irecv(s->recv_dev[src], count, MPI_BYTE, src, pair_tag(src, rank, nproc),
		                 MPI_COMM_WORLD, &s->recv[src]), "MPI_Irecv");
	}

	/* Общий старт итерации. Без барьера ранги расходятся по фазе, и в замер попадает
	   ожидание старта отправителя. Приёмы уже выставлены, поэтому барьер не задерживает передачу. */
	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(iteration)");
	const double t0 = measure ? MPI_Wtime() : 0.0;

	for (int dst = 0; dst < nproc; ++dst) {
		if (!active(args->mode, root, rank, dst)) continue;
		mpi_ok(MPI_Isend(s->d_send, count, MPI_BYTE, dst, pair_tag(rank, dst, nproc),
		                 MPI_COMM_WORLD, &s->send[dst]), "MPI_Isend device");
	}

	for (int done = 0; done < expected; ++done) {
		int        idx = MPI_UNDEFINED;
		MPI_Status st;
		mpi_ok(MPI_Waitany(nproc, s->recv, &idx, check ? &st : MPI_STATUS_IGNORE), "MPI_Waitany");
		if (idx == MPI_UNDEFINED) break;
		const int src = idx;
		if (check) mpi_ok(MPI_Get_count(&st, MPI_BYTE, &s->got[src]), "MPI_Get_count");
		/* Данные уже в recv_dev[src]; синхронизация устройства здесь намеренно не делается:
		   она ждала бы и чужие передачи (NOTES.md gpu_tests). */
		if (measure)
			s->samples[src][s->n_samples[src]++] = (MPI_Wtime() - t0) * 1e6;
		if (check &&
		    !verify_edges_ok(s->recv_dev[src], args->nbytes, verify_tag(it, src, 0), s->v_stream))
			++s->v_early[src];   /* к моменту отметки времени края сообщения ещё не те */
	}

	/* Одна синхронизация устройства на итерацию, уже после замеров. */
	cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after exchange)");
	if (check)
		for (int src = 0; src < nproc; ++src) {
			if (!active(args->mode, root, src, rank)) continue;
			++s->v_checked[src];
			if (s->got[src] != count ||
			    !verify_ok(s->recv_dev[src], args->nbytes, verify_tag(it, src, 0)))
				++s->v_bad[src];
		}
	mpi_ok(MPI_Waitall(nproc, s->send, MPI_STATUSES_IGNORE), "MPI_Waitall sends");
}

/* Замеры на этом ранге как на получателе. row (nproc * ACK_FIELDS): элемент src — статистики
   пары (src, rank), свой элемент помечен valid = 1. */
static void run_simultaneous(int rank, int nproc, const Args *args, char **rank_labels,
                             double *row) {
	DBG_LOG(rank, args, "%s RUN_BEGIN bytes=%zu env=%s", mode_to_string(args->mode), args->nbytes,
	        env_to_string(args->env));

	State s;
	memset(&s, 0, sizeof(s));
	s.rank = rank;
	s.nproc = nproc;
	s.args = args;
	cuda_ok(cudaMalloc((void **)&s.d_send, args->nbytes), "cudaMalloc(d_send)");
	cuda_ok(cudaMalloc((void **)&s.d_recv, args->nbytes), "cudaMalloc(d_recv)");
	cuda_ok(cudaMemset(s.d_send, 0xAA, args->nbytes), "cudaMemset");
	s.recv_dev  = xcalloc((size_t)nproc, sizeof(char *));
	s.samples   = xcalloc((size_t)nproc, sizeof(double *));
	s.n_samples = xcalloc((size_t)nproc, sizeof(size_t));
	s.v_checked = xcalloc((size_t)nproc, sizeof(long));
	s.v_bad     = xcalloc((size_t)nproc, sizeof(long));
	s.v_early   = xcalloc((size_t)nproc, sizeof(long));
	s.recv      = xcalloc((size_t)nproc, sizeof(MPI_Request));
	s.send      = xcalloc((size_t)nproc, sizeof(MPI_Request));
	s.got       = xcalloc((size_t)nproc, sizeof(int));
	for (int peer = 0; peer < nproc; ++peer) {
		if (peer == rank) continue;
		cuda_ok(cudaMalloc((void **)&s.recv_dev[peer], args->nbytes), "cudaMalloc(recv_dev)");
		s.samples[peer] = xmalloc((size_t)args->iters * sizeof(double));
	}
	/* --verify: свой неблокирующий поток для чтения краёв сообщения в момент отметки времени */
	if (args->verify)
		cuda_ok(cudaStreamCreateWithFlags(&s.v_stream, cudaStreamNonBlocking), "verify stream");

	/* Методика (шаги 1–3), как в one_to_one:
	   1) --prespin-sec: нагрузка CPU на всех рангах перед первым раундом;
	   2) --order random: раунды (корни) в случайном порядке, зерно --seed одинаково на всех
	      рангах, поэтому порядок совпадает без обмена; в all_to_all раунд один;
	   3) --probe: тесты ядра и L3 на каждом ранге до и после раунда (.sprobe как отправитель,
	      .rprobe как получатель). */
	if (args->prespin_sec > 0) {
		const double t_spin = MPI_Wtime();
		while (MPI_Wtime() - t_spin < args->prespin_sec) {}
		mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(prespin)");
	}
	/* Раунд на каждый корень. Каждая пара активна ровно в одном раунде. */
	const int rounds = args->mode == MODE_ALL_TO_ALL ? 1 : nproc;
	int *order = xmalloc((size_t)rounds * sizeof(int));
	for (int r = 0; r < rounds; ++r) order[r] = r;
	if (args->random_order) {
		Rng g;
		rng_seed(&g, args->seed);
		shuffle_ints(order, (size_t)rounds, &g);
	}
	for (int ri = 0; ri < rounds; ++ri) {
		const int root = order[ri];
		ProbeResult before = {0.0, 0.0};
		if (args->probe) before = cpu_probe();
		for (int i = 0; i < args->warmup; ++i) one_iteration(&s, root, 0);
		mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(iters)");
		for (int i = 0; i < args->iters; ++i) one_iteration(&s, root, 1);
		if (args->probe) {
			const ProbeResult after = cpu_probe();
			const double v[4] = {before.core_ns, before.l3_ns, after.core_ns, after.l3_ns};
			for (int peer = 0; peer < nproc; ++peer) {
				if (active(args->mode, root, rank, peer))
					append_raw_series(args, rank_labels[rank], rank_labels[peer], v, 4, ".sprobe", 4);
				if (active(args->mode, root, peer, rank))
					append_raw_series(args, rank_labels[peer], rank_labels[rank], v, 4, ".rprobe", 4);
			}
		}
	}
	if (rank == 0 && args->debug) {
		fprintf(stderr, "[DBG r0] ROUND_ORDER");
		for (int r = 0; r < rounds; ++r) fprintf(stderr, " %d", order[r]);
		fputc('\n', stderr);
	}

	for (int src = 0; src < nproc; ++src) {
		double *ack = row + (size_t)src * ACK_FIELDS;
		for (int i = 0; i < ACK_FIELDS; ++i) ack[i] = 0.0;
		if (src == rank) {
			ack[6] = 1.0;
			continue;
		}
		append_raw_samples(args, rank_labels[src], rank_labels[rank], s.samples[src],
		                   s.n_samples[src], NULL);
		fill_ack(s.samples[src], s.n_samples[src], ack);
		ack[7] = (double)s.v_checked[src];
		ack[8] = (double)s.v_bad[src];
		ack[9] = (double)s.v_early[src];
	}

	if (s.v_stream) cudaStreamDestroy(s.v_stream);
	for (int peer = 0; peer < nproc; ++peer) {
		if (s.recv_dev[peer]) cudaFree(s.recv_dev[peer]);
		free(s.samples[peer]);
	}
	cudaFree(s.d_send);
	cudaFree(s.d_recv);
	free(s.recv_dev); free(s.samples); free(s.n_samples);
	free(s.v_checked); free(s.v_bad); free(s.v_early);
	free(s.recv); free(s.send); free(s.got);
	free(order);
	DBG_LOG(rank, args, "%s RUN_DONE", mode_to_string(args->mode));
}

void schedule_simultaneous(int rank, int nproc, const Args *args, char **rank_labels) {
	if (rank == 0)
		out_printf("Order: %s seed=%u prespin_sec=%g probe=%d verify=%d\n",
		           args->random_order ? "random" : "seq", args->seed, args->prespin_sec,
		           args->probe ? 1 : 0, args->verify ? 1 : 0);
	const double t0 = MPI_Wtime();
	const size_t row_len = (size_t)nproc * ACK_FIELDS;
	double *row = xmalloc(row_len * sizeof(double));
	run_simultaneous(rank, nproc, args, rank_labels, row);

	/* Сбор после всех замеров: строка ранга dst — его статистики по источникам. */
	double *all = rank == 0 ? xmalloc((size_t)nproc * row_len * sizeof(double)) : NULL;
	mpi_ok(MPI_Gather(row, (int)row_len, MPI_DOUBLE, all, (int)row_len, MPI_DOUBLE, 0,
	                  MPI_COMM_WORLD), "MPI_Gather results");
	free(row);
	if (rank != 0) return;

	for (int src = 0; src < nproc; ++src)
		for (int dst = 0; dst < nproc; ++dst)
			print_pair_line("pair", rank_labels[src], rank_labels[dst],
			                all + ((size_t)dst * nproc + src) * ACK_FIELDS, args->stat_out);
	if (args->verify)
		for (int src = 0; src < nproc; ++src)
			for (int dst = 0; dst < nproc; ++dst) {
				const double *ack = all + ((size_t)dst * nproc + src) * ACK_FIELDS;
				if (src != dst && ack[7] > 0)
					print_verify_line(rank_labels[src], rank_labels[dst], ack);
			}

	out_printf("TotalTimeSec: %.*f\n", TOTAL_TIME_DIGITS, MPI_Wtime() - t0);
	free(all);
}
