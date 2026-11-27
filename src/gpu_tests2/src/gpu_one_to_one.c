#include "gpu_one_to_one.h"

#include <stdlib.h>
#include <string.h>

enum {
	TAG_TASK         = 1,
	TAG_ACK          = 2,
	TAG_DATA         = 0,
	RDV_TAG_FORWARD  = 42,
	RDV_TAG_BACKWARD = 43
};

/* Буферы ранга: выделяются один раз на весь режим, а не на каждую пару. */
typedef struct {
	char            *d_send, *d_recv;
	char            *h_buf;        /* --sys-via-host: закреплённый буфер в RAM */
	int              has_shared;
	SharedHostWindow shared;       /* --sys-via-host: общий буфер узла */
	char            *shared_buf;
	int              got_count;    /* --verify: длина принятого (MPI_Get_count) */
} Buffers;

static int same_node(const Task *t, const int *on_my_node) {
	return on_my_node[t->src_rank] && on_my_node[t->dst_rank];
}

/* Пара идёт через RAM силами программы: только с --sys-via-host и только пары GPU разных
   процессоров одного узла (прямой путь между ними на G5500 в тысячи раз медленнее, NOTES.md gpu_tests).
   Среда host сюда не относится: в ней через RAM копирует сам UCX. */
static int pair_via_host(const Task *t, const Args *args, int same, const int *gpu_numa) {
	if (!args->sys_via_host || !same) return 0;
	const int a = gpu_numa[t->src_rank], b = gpu_numa[t->dst_rank];
	return a >= 0 && b >= 0 && a != b;
}

/* Пустое сообщение туда и обратно: держит отправителя и получателя в одной фазе. */
static void rendezvous(int is_sender, int peer) {
	char dummy = 0;
	if (is_sender)
		mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, peer, RDV_TAG_FORWARD, &dummy, 0, MPI_BYTE, peer,
		                    RDV_TAG_BACKWARD, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
		       "Sendrecv rendezvous(sender)");
	else
		mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, peer, RDV_TAG_BACKWARD, &dummy, 0, MPI_BYTE, peer,
		                    RDV_TAG_FORWARD, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
		       "Sendrecv rendezvous(receiver)");
}

/* Блокирующие MPI_Send/MPI_Recv или (--p2p nonblocking) MPI_Isend/MPI_Irecv + MPI_Wait.
   Open MPI передаёт UCX для неблокирующих вызовов флаг UCP_OP_ATTR_FLAG_MULTI_SEND, для
   MPI_Send — UCP_OP_ATTR_FLAG_FAST_CMPL, и UCX выбирает протокол по разным таблицам. */
static void p2p_send(const void *buf, int count, int dst, int nonblocking, const char *what) {
	if (!nonblocking) {
		mpi_ok(MPI_Send(buf, count, MPI_BYTE, dst, TAG_DATA, MPI_COMM_WORLD), what);
		return;
	}
	MPI_Request r;
	mpi_ok(MPI_Isend(buf, count, MPI_BYTE, dst, TAG_DATA, MPI_COMM_WORLD, &r), what);
	mpi_ok(MPI_Wait(&r, MPI_STATUS_IGNORE), what);
}

/* got_count != NULL (--verify): длина принятого сообщения по статусу. */
static void p2p_recv(void *buf, int count, int src, int nonblocking, const char *what,
                     int *got_count) {
	MPI_Status  st;
	MPI_Status *stp = got_count ? &st : MPI_STATUS_IGNORE;
	if (!nonblocking) {
		mpi_ok(MPI_Recv(buf, count, MPI_BYTE, src, TAG_DATA, MPI_COMM_WORLD, stp), what);
	} else {
		MPI_Request r;
		mpi_ok(MPI_Irecv(buf, count, MPI_BYTE, src, TAG_DATA, MPI_COMM_WORLD, &r), what);
		mpi_ok(MPI_Wait(&r, stp), what);
	}
	if (got_count) mpi_ok(MPI_Get_count(&st, MPI_BYTE, got_count), "MPI_Get_count");
}

/* Одна передача src -> dst. Вызывается и отправителем, и получателем. */
static void transfer_once(const Task *t, int is_sender, int via_host, int same, size_t nbytes,
                          Buffers *b, int nonblocking, int verify) {
	const int count = (int)nbytes;
	int *got = verify ? &b->got_count : NULL;
	b->got_count = count;   /* путь через общую память длину не проверяет: копируется ровно count */

	/* --sys-via-host внутри узла: копия GPU -> общий буфер у отправителя, сигнал готовности,
	   копия общий буфер -> GPU у получателя. */
	if (via_host && same && b->shared_buf) {
		if (is_sender) {
			cuda_ok(cudaMemcpy(b->shared_buf, b->d_send, nbytes, cudaMemcpyDeviceToHost),
			        "D2H(shared)");
			shared_window_sync(&b->shared);
			rendezvous(1, t->dst_rank);
		} else {
			rendezvous(0, t->src_rank);
			shared_window_sync(&b->shared);
			cuda_ok(cudaMemcpy(b->d_recv, b->shared_buf, nbytes, cudaMemcpyHostToDevice),
			        "H2D(shared)");
		}
		return;
	}
	/* --sys-via-host без общего окна (не выделилось): через MPI с буфером в RAM */
	if (via_host) {
		if (is_sender) {
			cuda_ok(cudaMemcpy(b->h_buf, b->d_send, nbytes, cudaMemcpyDeviceToHost), "D2H");
			p2p_send(b->h_buf, count, t->dst_rank, nonblocking, "send(host)");
		} else {
			p2p_recv(b->h_buf, count, t->src_rank, nonblocking, "recv(host)", got);
			cuda_ok(cudaMemcpy(b->d_recv, b->h_buf, nbytes, cudaMemcpyHostToDevice), "H2D");
		}
		return;
	}

	/* Все среды: буферы в памяти GPU, путь выбирает UCX по своим переменным. */
	if (is_sender) {
		p2p_send(b->d_send, count, t->dst_rank, nonblocking, "send(device)");
	} else {
		p2p_recv(b->d_recv, count, t->src_rank, nonblocking, "recv(device)", got);
		/* UCX может сообщить о завершении MPI раньше, чем копия на устройстве действительно
		   закончится (cuda_ipc, cuda_copy): ждём её внутри замера. */
		cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize(recv device)");
	}
}

static void make_buffers(Buffers *b, const Args *args, MPI_Comm node_comm, int node_rank, int rank) {
	memset(b, 0, sizeof(*b));
	cuda_ok(cudaMalloc((void **)&b->d_send, args->nbytes), "cudaMalloc(d_send)");
	cuda_ok(cudaMalloc((void **)&b->d_recv, args->nbytes), "cudaMalloc(d_recv)");
	cuda_ok(cudaMemset(b->d_send, 0xAA, args->nbytes), "cudaMemset(d_send)");
	if (args->sys_via_host) {
		cuda_ok(cudaMallocHost((void **)&b->h_buf, args->nbytes), "cudaMallocHost");
		/* один общий буфер на узел: память выделяет ранг 0 узла */
		shared_window_create(&b->shared, node_comm, node_rank == 0 ? args->nbytes : 0, rank, args);
		b->has_shared = 1;
		b->shared_buf = shared_window_segment(&b->shared, 0, args->nbytes);
	}
}

static void free_buffers(Buffers *b) {
	if (b->has_shared) shared_window_free(&b->shared);   /* коллективно по node_comm */
	if (b->h_buf) cudaFreeHost(b->h_buf);
	cudaFree(b->d_send);
	cudaFree(b->d_recv);
	memset(b, 0, sizeof(*b));
}

/* Замер пары. ack заполнен у получателя, у отправителя valid = 0.
   Получатель пишет в .raw на каждую итерацию время её начала по системным часам
   (снимается вне замеряемого интервала) и задержку. */
static void measure_pair(int rank, const Task *t, const Args *args, int via_host, int same,
                         Buffers *b, char **rank_labels, double *ack) {
	for (int i = 0; i < ACK_FIELDS; ++i) ack[i] = 0.0;
	const int is_sender   = rank == t->src_rank;
	const int is_receiver = rank == t->dst_rank;
	if (!is_sender && !is_receiver) return;

	DBG_LOG(rank, args, "RUN src=%d dst=%d bytes=%zu env=%s via_host=%d", t->src_rank, t->dst_rank,
	        args->nbytes, env_to_string(args->env), via_host);

	ProbeResult before = {0.0, 0.0};
	if (args->probe) before = cpu_probe();

	for (int i = 0; i < args->warmup; ++i)
		transfer_once(t, is_sender, via_host, same, args->nbytes, b, args->p2p_nonblocking, 0);

	const size_t n_it = (size_t)args->iters;
	long    v_checked = 0, v_bad = 0;
	double *samples_us = is_receiver ? xmalloc(n_it * sizeof(double)) : NULL;
	double *times      = is_receiver ? xmalloc(n_it * sizeof(double)) : NULL;
	double *cpus       = args->cpu_log ? xmalloc(n_it * sizeof(double)) : NULL;   /* обе стороны */
	size_t  n_samples  = 0;
	const int peer = is_sender ? t->dst_rank : t->src_rank;

	for (int i = 0; i < args->iters; ++i) {
		unsigned tag = 0;
		if (args->verify) {
			/* Новый шаблон у отправителя, нули у получателя. Время заполнения в замер не
			   попадает: после него идёт синхронизация пары. */
			tag = verify_tag(i, t->src_rank, t->dst_rank);
			if (is_sender) {
				verify_fill(b->d_send, args->nbytes, tag);
			} else {
				cuda_ok(cudaMemset(b->d_recv, 0, args->nbytes), "verify: cudaMemset(d_recv)");
				if (b->h_buf) memset(b->h_buf, 0, args->nbytes);
			}
		}
		/* Синхронизация пары перед каждой измеряемой передачей, в том числе первой (после
		   прогрева): передача начинается у обоих одновременно. */
		rendezvous(is_sender, peer);
		if (is_receiver) times[i] = epoch_now();
		if (cpus) cpus[i] = current_cpu();
		const double t0 = MPI_Wtime();
		transfer_once(t, is_sender, via_host, same, args->nbytes, b, args->p2p_nonblocking,
		              args->verify);
		const double t1 = MPI_Wtime();
		if (is_receiver) {
			samples_us[n_samples++] = (t1 - t0) * 1e6;
			if (args->verify) {
				++v_checked;
				const int ok = b->got_count == (int)args->nbytes &&
				               verify_ok(b->d_recv, args->nbytes, tag);
				if (!ok) {
					++v_bad;
					DBG_LOG(rank, args, "VERIFY_BAD src=%d dst=%d iter=%d got=%d", t->src_rank,
					        t->dst_rank, i, b->got_count);
				}
			}
		}
	}

	const char *src_l = rank_labels[t->src_rank], *dst_l = rank_labels[t->dst_rank];
	if (cpus)
		append_raw_series(args, src_l, dst_l, cpus, n_it, is_sender ? ".scpu" : ".rcpu", 0);
	if (args->probe) {
		const ProbeResult after = cpu_probe();
		const double v[4] = {before.core_ns, before.l3_ns, after.core_ns, after.l3_ns};
		append_raw_series(args, src_l, dst_l, v, 4, is_sender ? ".sprobe" : ".rprobe", 4);
	}
	if (is_receiver) {
		append_raw_samples(args, src_l, dst_l, samples_us, n_samples, times);
		fill_ack(samples_us, n_samples, ack);
		ack[7] = (double)v_checked;
		ack[8] = (double)v_bad;
	}
	free(samples_us);
	free(times);
	free(cpus);
}

/* Ранги, кроме 0: ждут задание, участвуют в замере, возвращают ack. */
static void worker_loop(int rank, const Args *args, Buffers *b, const int *on_my_node,
                        char **rank_labels, const int *gpu_numa) {
	double ack[ACK_FIELDS];
	for (;;) {
		Task t;
		mpi_ok(MPI_Recv(&t, (int)sizeof(Task), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD,
		                MPI_STATUS_IGNORE), "MPI_Recv Task");
		DBG_LOG(rank, args, "WORKER_TASK src=%d dst=%d stop=%d", t.src_rank, t.dst_rank, t.stop);
		if (t.stop) break;
		const int same = same_node(&t, on_my_node);
		measure_pair(rank, &t, args, pair_via_host(&t, args, same, gpu_numa), same, b, rank_labels,
		             ack);
		mpi_ok(MPI_Send(ack, ACK_FIELDS, MPI_DOUBLE, 0, TAG_ACK, MPI_COMM_WORLD), "MPI_Send ack");
	}
}

void schedule_one_to_one(int rank, int nproc, const Args *args, MPI_Comm node_comm, int node_rank,
                         const int *on_my_node, char **rank_labels, const int *gpu_numa) {
	Buffers b;
	make_buffers(&b, args, node_comm, node_rank, rank);

	/* Разгон CPU до первой пары: без него первая пара мерится на низкой частоте ядер. */
	if (args->prespin_sec > 0) {
		const double t_spin = MPI_Wtime();
		while (MPI_Wtime() - t_spin < args->prespin_sec) {}
		mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(prespin)");
	}

	if (rank != 0) {
		worker_loop(rank, args, &b, on_my_node, rank_labels, gpu_numa);
		free_buffers(&b);
		return;
	}

	/* Ранг 0 перебирает пары и сам участвует в парах со своим номером.
	   Порядок пар: по источникам или случайный с --seed. Пара k — (order[k] / nproc, order[k] % nproc). */
	const size_t n_pairs = (size_t)nproc * (size_t)(nproc - 1);
	int   *order = xmalloc((n_pairs ? n_pairs : 1) * sizeof(int));
	size_t k = 0;
	for (int src = 0; src < nproc; ++src)
		for (int dst = 0; dst < nproc; ++dst)
			if (src != dst) order[k++] = src * nproc + dst;
	if (args->random_order) {
		Rng r;
		rng_seed(&r, args->seed);
		shuffle_ints(order, n_pairs, &r);
	}

	out_printf("Order: %s seed=%u prespin_sec=%g cpu_log=%d p2p=%s verify=%d\n",
	           args->random_order ? "random" : "seq", args->seed, args->prespin_sec,
	           args->cpu_log ? 1 : 0, args->p2p_nonblocking ? "nonblocking" : "blocking",
	           args->verify ? 1 : 0);

	double *metric = xcalloc((size_t)nproc * (size_t)nproc * ACK_FIELDS, sizeof(double));
	/* Строки pair_time копятся и печатаются после всех пар, как в gpu_tests */
	char  **times = xcalloc(n_pairs ? n_pairs : 1, sizeof(char *));
	const double test_t0 = MPI_Wtime();
	for (k = 0; k < n_pairs; ++k) {
		const int src = order[k] / nproc, dst = order[k] % nproc;
		const Task t = {src, dst, 0};
		DBG_LOG(rank, args, "PAIR src=%d dst=%d", src, dst);
		const double start = epoch_now();
		const int peers[2] = {src, dst};
		for (int p = 0; p < 2; ++p)
			if (peers[p] != 0)
				mpi_ok(MPI_Send(&t, (int)sizeof(Task), MPI_BYTE, peers[p], TAG_TASK, MPI_COMM_WORLD),
				       "MPI_Send Task");
		double *m = metric + ((size_t)src * nproc + dst) * ACK_FIELDS;
		double  ack[ACK_FIELDS];
		if (src == 0 || dst == 0) {
			const int same = same_node(&t, on_my_node);
			measure_pair(rank, &t, args, pair_via_host(&t, args, same, gpu_numa), same, &b,
			             rank_labels, ack);
			if (ack[6] == 1.0) memcpy(m, ack, sizeof(ack));
		}
		for (int p = 0; p < 2; ++p) {
			if (peers[p] == 0) continue;
			mpi_ok(MPI_Recv(ack, ACK_FIELDS, MPI_DOUBLE, peers[p], TAG_ACK, MPI_COMM_WORLD,
			                MPI_STATUS_IGNORE), "MPI_Recv ack");
			if (ack[6] == 1.0) memcpy(m, ack, sizeof(ack));
		}
		/* Время по системным часам ранга 0: от рассылки задания до последнего ack. */
		times[k] = xmalloc(3 * LABEL_LEN + 96);
		snprintf(times[k], 3 * LABEL_LEN + 96, "pair_time %s -> %s index=%zu start=%.6f end=%.6f\n",
		         rank_labels[src], rank_labels[dst], k, start, epoch_now());
	}
	for (k = 0; k < n_pairs; ++k) {
		out_printf("%s", times[k]);
		free(times[k]);
	}
	free(times);

	/* Строки pair в обычном порядке, чтобы разбор не зависел от порядка замеров. */
	for (int src = 0; src < nproc; ++src)
		for (int dst = 0; dst < nproc; ++dst)
			print_pair_line("pair", rank_labels[src], rank_labels[dst],
			                metric + ((size_t)src * nproc + dst) * ACK_FIELDS, args->stat_out);
	if (args->verify)
		for (int src = 0; src < nproc; ++src)
			for (int dst = 0; dst < nproc; ++dst)
				if (src != dst)
					print_verify_line(rank_labels[src], rank_labels[dst],
					                  metric + ((size_t)src * nproc + dst) * ACK_FIELDS);

	out_printf("TotalTimeSec: %.*f\n", TOTAL_TIME_DIGITS, MPI_Wtime() - test_t0);

	const Task stop = {0, 0, 1};
	for (int r = 1; r < nproc; ++r)
		mpi_ok(MPI_Send(&stop, (int)sizeof(Task), MPI_BYTE, r, TAG_TASK, MPI_COMM_WORLD),
		       "MPI_Send stop");
	free(metric);
	free(order);
	free_buffers(&b);
}
