#include "gpu_all_to_all.hpp"

#include <algorithm>     // std::max
#include <cuda_runtime.h> // CUDA: устройство, события, cudaMalloc, ...
#include <functional>    // std::function — параметр mirror в schedule_all_to_all
#include <iomanip>       // std::setprecision, std::fixed
#include <sstream>       // std::ostringstream — строки TotalTimeSec и т.п.
#include <vector>        // буферы ack, samples, MPI_Request

namespace gpu_benchmark {

namespace {
}

static int alltoall_pair_tag(int src_rank, int dst_rank, int nproc) {
	return src_rank * nproc + dst_rank;
}

std::vector<double> run_all_to_all(int rank, int nproc, const Args &args,
								   bool check_host, int local_gpu,
								   const std::vector<std::string> &rank_labels) {
	std::vector<double> ack(ACK_FIELDS, 0.0);

	char *d_send = nullptr;
	char *d_recv = nullptr;
	const int count = static_cast<int>(args.nbytes);
	DBG_LOG(rank, args,
			"all_to_all RUN_BEGIN local_gpu=" << local_gpu << " bytes=" << args.nbytes
			<< " nproc=" << nproc << " env_path=" << (check_host ? "host" : "auto"));

	cuda_ok(cudaSetDevice(local_gpu), "cudaSetDevice(all_to_all)");
	cuda_ok(cudaMalloc(&d_send, args.nbytes), "cudaMalloc(send)");
	cuda_ok(cudaMemset(d_send, 0xAA, args.nbytes), "cudaMemset(send)");
	cuda_ok(cudaMalloc(&d_recv, args.nbytes), "cudaMalloc(recv)");
	DBG_LOG(rank, args, "all_to_all ALLOC_DONE");

	std::vector<char *> send_bufs(static_cast<size_t>(nproc), nullptr);
	std::vector<char *> recv_bufs(static_cast<size_t>(nproc), nullptr);
	for (int peer_rank = 0; peer_rank < nproc; ++peer_rank) { // заполняем буферы для каждого peer_rank
		if (peer_rank == rank)
			continue;
		if (check_host) {
			cuda_ok(cudaMallocHost(&send_bufs[static_cast<size_t>(peer_rank)], args.nbytes),
					"cudaMallocHost(send)");
			cuda_ok(cudaMallocHost(&recv_bufs[static_cast<size_t>(peer_rank)], args.nbytes),
					"cudaMallocHost(recv)");
		} else {
			send_bufs[static_cast<size_t>(peer_rank)] = d_send;
			recv_bufs[static_cast<size_t>(peer_rank)] = d_recv;
		}
	}

	auto do_one = [&](int iter_idx, bool measure,
					  std::vector<std::vector<double>> *samples_mpi_us,
					  std::vector<std::vector<double>> *samples_cpu_us,
					  std::vector<std::vector<double>> *samples_gpu_us) {
		DBG_LOG(rank, args,
				"all_to_all ITER_BEGIN idx=" << iter_idx
				<< " measure=" << (measure ? 1 : 0));

		/* Receiver-side e2e (вариант B). Семантика: t0 берётся у ПОЛУЧАТЕЛЯ
		   один раз на итерацию — сразу после выставления всех MPI_Irecv (это
		   «старт раунда» с точки зрения этого ранга). t1 берётся у того же
		   получателя для каждого пришедшего src — после MPI_Waitany и H2D.
		   Сэмпл = t1 - t0 хранится в samples[src] и описывает «сколько мне
		   пришлось ждать и получить данные от конкретного src в условиях
		   текущего all-to-all раунда».

		   Все метки времени снимаются на ОДНОМ процессе и одних часах, поэтому
		   отрицательных значений быть не может; межузловой синхронизации часов
		   не требуется. В сэмпл попадает: ожидание сети + Recv + (если host) H2D.
		   D2H/Send отправителя физически выполняются параллельно ВНУТРИ интервала
		   [t0, t1] (отправитель тоже только что начал свой раунд), поэтому
		   реальная дорога «GPU отправителя → GPU получателя» в основном входит
		   в замер; недоучёт ограничен только зазором «вход в send-фазу» у
		   отправителя — единицы микросекунд. */

		std::vector<MPI_Request> recv_data_req(static_cast<size_t>(nproc),
											   MPI_REQUEST_NULL);

		for (int src_rank = 0; src_rank < nproc; ++src_rank) {
			if (src_rank == rank)
				continue;
			char *recv_buf = recv_bufs[static_cast<size_t>(src_rank)];
			DBG_LOG(rank, args,
					"all_to_all IRECV_POST src=" << src_rank
					<< " tag=" << alltoall_pair_tag(src_rank, rank, nproc)
					<< " bytes=" << count << " idx=" << iter_idx);
			mpi_ok(MPI_Irecv(recv_buf, count, MPI_BYTE, src_rank,
							 alltoall_pair_tag(src_rank, rank, nproc),
							 MPI_COMM_WORLD,
							 &recv_data_req[static_cast<size_t>(src_rank)]),
				   "MPI_Irecv(all_to_all data)");
		}

		/* Единая стартовая метка раунда у получателя — после Irecv, до send-фазы.
		   cudaEventRecord(start) делается ДО взятия t0, чтобы его собственная
		   стоимость (несколько мкс на вызов CUDA driver) не попадала в
		   samples_mpi_us. Аналогично ev_stop[src] и cudaEventSynchronize ниже —
		   ПОСЛЕ взятия t1. */
		double t0_mpi_s = 0.0;
		double t0_clk_us = 0.0;
		cudaEvent_t ev_start = nullptr;
		std::vector<cudaEvent_t> ev_stop(static_cast<size_t>(nproc), nullptr);
		if (measure) {
			cuda_ok(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
			cuda_ok(cudaEventRecord(ev_start), "cudaEventRecord(start)");
			t0_mpi_s = MPI_Wtime();
			t0_clk_us = clock_gettime_wrapper();
		}

		DBG_LOG(rank, args, "all_to_all SEND_PHASE_BEGIN");
		for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
			if (dst_rank == rank)
				continue;
			char *send_buf = send_bufs[static_cast<size_t>(dst_rank)];

			if (check_host) {
				cuda_ok(cudaMemcpy(send_buf, d_send, args.nbytes,
								   cudaMemcpyDeviceToHost),
						"D2H(all_to_all)");
				mpi_ok(MPI_Send(send_buf, count, MPI_BYTE, dst_rank,
								alltoall_pair_tag(rank, dst_rank, nproc),
								MPI_COMM_WORLD),
					   "MPI_Send(all_to_all host)");
			} else {
				mpi_ok(MPI_Send(send_buf, count, MPI_BYTE, dst_rank,
								alltoall_pair_tag(rank, dst_rank, nproc),
								MPI_COMM_WORLD),
					   "MPI_Send(all_to_all dev)");
			}
		}
		DBG_LOG(rank, args, "all_to_all SEND_PHASE_DONE");

		DBG_LOG(rank, args, "all_to_all RECV_PHASE_BEGIN");
		const int n_peers = nproc - 1;
		for (int k = 0; k < n_peers; ++k) {
			int idx = MPI_UNDEFINED;
			mpi_ok(MPI_Waitany(nproc, recv_data_req.data(), &idx,
							   MPI_STATUS_IGNORE),
				   "MPI_Waitany(all_to_all data)");
			if (idx == MPI_UNDEFINED)
				break;
			const int src_rank = idx;
			DBG_LOG(rank, args,
					"all_to_all WAIT_RECV_DONE src=" << src_rank
					<< " idx=" << iter_idx);
			if (check_host) {
				cuda_ok(cudaMemcpy(d_recv,
								   recv_bufs[static_cast<size_t>(src_rank)],
								   args.nbytes, cudaMemcpyHostToDevice),
						"H2D(all_to_all)");
				DBG_LOG(rank, args,
						"all_to_all H2D_DONE src=" << src_rank
						<< " idx=" << iter_idx);
			}
			if (measure) {
				/* t1 у получателя: ПОСЛЕ H2D. Сэмпл = t1 - t0 на одних часах. */
				const double t1_mpi_s = MPI_Wtime();
				const double t1_clk_us = clock_gettime_wrapper();
				const double mpi_sample_us = (t1_mpi_s - t0_mpi_s) * 1e6;
				const double cpu_sample_us = t1_clk_us - t0_clk_us;
				(*samples_mpi_us)[static_cast<size_t>(src_rank)].push_back(
					mpi_sample_us);
				(*samples_cpu_us)[static_cast<size_t>(src_rank)].push_back(
					cpu_sample_us);
				/* CUDA-event ПОСЛЕ взятия t1 — стоимость EventRecord/Sync/Elapsed
				   не попадает в samples_mpi/cpu. Свой ev_stop на каждый src,
				   чтобы CUDA-сэмпл считался от общего ev_start до момента
				   завершения H2D для этого src, а не перетирался следующей
				   итерацией Waitany. В env=auto default stream обычно пуст
				   (UCX делает H2D в собственных потоках) — значит samples_gpu_us
				   будет около нуля, это нормально и говорит «здесь GPU-таймер
				   не информативен», а не о баге. */
				cudaEvent_t &ev_stop_src = ev_stop[static_cast<size_t>(src_rank)];
				cuda_ok(cudaEventCreate(&ev_stop_src),
						"cudaEventCreate(stop[src])");
				cuda_ok(cudaEventRecord(ev_stop_src),
						"cudaEventRecord(stop[src])");
				cuda_ok(cudaEventSynchronize(ev_stop_src),
						"cudaEventSynchronize(stop[src])");
				float elapsed_ms = 0.0f;
				cuda_ok(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop_src),
						"cudaEventElapsedTime");
				(*samples_gpu_us)[static_cast<size_t>(src_rank)].push_back(
					static_cast<double>(elapsed_ms) * 1e3);
			}
		}
		DBG_LOG(rank, args, "all_to_all RECV_PHASE_DONE");

		if (measure) {
			if (ev_start)
				cuda_ok(cudaEventDestroy(ev_start), "cudaEventDestroy(start)");
			for (int src_rank = 0; src_rank < nproc; ++src_rank) {
				if (ev_stop[static_cast<size_t>(src_rank)])
					cuda_ok(cudaEventDestroy(
								ev_stop[static_cast<size_t>(src_rank)]),
							"cudaEventDestroy(stop[src])");
			}
		}
		DBG_LOG(rank, args,
				"all_to_all ITER_DONE idx=" << iter_idx
				<< " measure=" << (measure ? 1 : 0));
	};

	/* Сэмплы лежат по «источнику» src: для каждого src != rank копится N сэмплов
	   (по числу итераций) — это receiver-side e2e задержка ребра (src → этот
	   ранг), измеренная на часах ПОЛУЧАТЕЛЯ как t1 - t0, где t0 — общая для
	   итерации стартовая метка раунда у этого ранга, а t1 — момент завершения
	   H2D от данного src. */
	std::vector<std::vector<double>> samples_mpi_us(static_cast<size_t>(nproc));
	std::vector<std::vector<double>> samples_cpu_us(static_cast<size_t>(nproc));
	std::vector<std::vector<double>> samples_gpu_us(static_cast<size_t>(nproc));
	for (int src_rank = 0; src_rank < nproc; ++src_rank) {
		if (src_rank == rank)
			continue;
		samples_mpi_us[static_cast<size_t>(src_rank)].reserve(
			static_cast<size_t>(std::max(1, args.iters)));
		samples_cpu_us[static_cast<size_t>(src_rank)].reserve(
			static_cast<size_t>(std::max(1, args.iters)));
		samples_gpu_us[static_cast<size_t>(src_rank)].reserve(
			static_cast<size_t>(std::max(1, args.iters)));
	}

	DBG_LOG(rank, args, "all_to_all WARMUP_BEGIN");
	for (int i = 0; i < args.warmup; ++i)
		do_one(i, false, nullptr, nullptr, nullptr);
	DBG_LOG(rank, args, "all_to_all WARMUP_DONE");

	/* Между warmup и измерениями выровняем все ранги по входу в ITERS. Без
	   этого медленные ранги могли внести в первую измеряемую итерацию
	   «фоновый лаг» — он бы попал в samples_mpi_us. На каждой итерации внутри
	   этого цикла ранги синхронизируются сами через rendezvous Send/Recv в
	   send/recv-фазах, поэтому Barrier здесь нужен только один раз. */
	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(all_to_all iters)");

	DBG_LOG(rank, args, "all_to_all ITERS_BEGIN");
	for (int i = 0; i < args.iters; ++i)
		do_one(i, true, &samples_mpi_us, &samples_cpu_us, &samples_gpu_us);
	DBG_LOG(rank, args, "all_to_all ITERS_DONE");

	/* Этот ранг — получатель пары (src → rank). Сохраняем его сэмплы как
	   raw-файл для каждой такой пары. */
	for (int src_rank = 0; src_rank < nproc; ++src_rank) {
		if (src_rank == rank)
			continue;
		Task t{};
		t.src_rank = src_rank;
		t.src_gpu = local_gpu;
		t.dst_rank = rank;
		t.dst_gpu = local_gpu;
		switch (args.timer) {
		case Timer::All:
		case Timer::Mpi:
			append_raw_samples(args, rank, t, rank_labels,
							   samples_mpi_us[static_cast<size_t>(src_rank)]);
			break;
		case Timer::Cpu:
			append_raw_samples(args, rank, t, rank_labels,
							   samples_cpu_us[static_cast<size_t>(src_rank)]);
			break;
		case Timer::Cuda:
			append_raw_samples(args, rank, t, rank_labels,
							   samples_gpu_us[static_cast<size_t>(src_rank)]);
			break;
		}
	}

	std::vector<double> results;
	if (rank == 0) {
		results.assign(static_cast<size_t>(nproc) * static_cast<size_t>(nproc) *
						   static_cast<size_t>(ACK_FIELDS),
					   0.0);
	}

	const auto result_ptr = [&](int src_rank, int dst_rank) -> double * {
		return results.data() +
			   (static_cast<size_t>(src_rank) * static_cast<size_t>(nproc) +
				static_cast<size_t>(dst_rank)) *
				   static_cast<size_t>(ACK_FIELDS);
	};

	/* Для receiver-side e2e ячейку (src, dst) формирует получатель (rank == dst):
	   у него лежат сэмплы samples_*[src] для пары src → dst. Ранг 0 собирает
	   итоговые ack от всех получателей в общую матрицу results. */
	for (int src_rank = 0; src_rank < nproc; ++src_rank) {
		for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
			if (src_rank == dst_rank) {
				if (rank == 0)
					result_ptr(src_rank, dst_rank)[6] = 1.0;
				continue;
			}

			if (rank == dst_rank) {
				fill_ack(samples_mpi_us[static_cast<size_t>(src_rank)],
						 samples_cpu_us[static_cast<size_t>(src_rank)],
						 samples_gpu_us[static_cast<size_t>(src_rank)], args,
						 ack.data());
				if (rank == 0) {
					std::copy(ack.begin(), ack.end(), result_ptr(src_rank, dst_rank));
				} else {
					mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0,
									nproc * nproc + alltoall_pair_tag(src_rank, dst_rank, nproc),
									MPI_COMM_WORLD),
						   "MPI_Send ack(all_to_all)");
					DBG_LOG(rank, args,
							"all_to_all ACK_SENT src=" << src_rank << " dst=" << dst_rank);
				}
			} else if (rank == 0) {
				mpi_ok(MPI_Recv(result_ptr(src_rank, dst_rank), ACK_FIELDS, MPI_DOUBLE,
								dst_rank,
								nproc * nproc + alltoall_pair_tag(src_rank, dst_rank, nproc),
								MPI_COMM_WORLD, MPI_STATUS_IGNORE),
					   "MPI_Recv ack(all_to_all)");
				DBG_LOG(rank, args,
						"all_to_all ACK_RECV src=" << src_rank << " dst=" << dst_rank);
			}
		}
	}

	for (int peer_rank = 0; peer_rank < nproc; ++peer_rank) {
		if (peer_rank == rank)
			continue;
		if (check_host) {
			if (send_bufs[static_cast<size_t>(peer_rank)])
				cudaFreeHost(send_bufs[static_cast<size_t>(peer_rank)]);
			if (recv_bufs[static_cast<size_t>(peer_rank)])
				cudaFreeHost(recv_bufs[static_cast<size_t>(peer_rank)]);
		}
	}
	if (d_send)
		cudaFree(d_send);
	if (d_recv)
		cudaFree(d_recv);
	DBG_LOG(rank, args, "all_to_all RUN_DONE");
	return results;
}

void schedule_all_to_all(
	int rank, int nproc, const Args &args, bool via_host,
	const std::vector<int> &rank_to_gpu,
	const std::vector<std::string> &rank_labels,
	const std::function<void(const std::string &)> &mirror) {
	DBG_LOG(rank, args, "all_to_all SCHEDULE_BEGIN");
	/* Раньше тут хардкоженно передавался local_gpu=0 для всех ранков; при
	   --gres=gpu:N (когда у одного процесса видны все GPU узла) это означало,
	   что 4 процесса узла стучатся в физический GPU 0, а GPU 1..3 простаивают.
	   Теперь берём личный device из глобального справочника rank_to_gpu, который
	   gpu_benchmark.cpp заполнил по локальному рангу процесса на узле. */
	const int my_gpu = rank_to_gpu[static_cast<size_t>(rank)];
	if (rank != 0) {
		run_all_to_all(rank, nproc, args, via_host, my_gpu, rank_labels);
		DBG_LOG(rank, args, "all_to_all SCHEDULE_DONE");
		return;
	}

	const double test_t0 = MPI_Wtime();
	std::vector<double> results =
		run_all_to_all(rank, nproc, args, via_host, my_gpu, rank_labels);

	auto metric_ptr = [&](int src_rank, int dst_rank) -> const double * {
		return results.data() +
			   (static_cast<size_t>(src_rank) * static_cast<size_t>(nproc) +
				static_cast<size_t>(dst_rank)) *
				   static_cast<size_t>(ACK_FIELDS);
	};

	for (int src_rank = 0; src_rank < nproc; ++src_rank) {
		for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
			std::vector<double> metric(metric_ptr(src_rank, dst_rank),
									   metric_ptr(src_rank, dst_rank) + ACK_FIELDS);
			DBG_LOG(rank, args,
					"all_to_all PAIR_MERGE src=" << src_rank << " dst=" << dst_rank);
			if (src_rank == dst_rank) {
				if (args.timer == Timer::All) {
					mirror(print_pair_line("pair_mpi",
											rank_labels[static_cast<size_t>(src_rank)],
											rank_labels[static_cast<size_t>(dst_rank)],
											0.0, 0.0, 0.0, 0.0, 0.0, 0.0, args.stat_out));
					mirror(print_pair_line("pair_cpu",
											rank_labels[static_cast<size_t>(src_rank)],
											rank_labels[static_cast<size_t>(dst_rank)],
											0.0, 0.0, 0.0, 0.0, 0.0, 0.0, args.stat_out));
					mirror(print_pair_line("pair_cuda",
											rank_labels[static_cast<size_t>(src_rank)],
											rank_labels[static_cast<size_t>(dst_rank)],
											0.0, 0.0, 0.0, 0.0, 0.0, 0.0, args.stat_out));
				} else {
					mirror(print_pair_line("pair",
											rank_labels[static_cast<size_t>(src_rank)],
											rank_labels[static_cast<size_t>(dst_rank)],
											0.0, 0.0, 0.0, 0.0, 0.0, 0.0, args.stat_out));
				}
				continue;
			}

			if (args.timer == Timer::All) {
				mirror(print_pair_line("pair_mpi",
										rank_labels[static_cast<size_t>(src_rank)],
										rank_labels[static_cast<size_t>(dst_rank)], metric[13],
										metric[14], metric[15], metric[16],
										metric[17], metric[18], args.stat_out));
				mirror(print_pair_line("pair_cpu",
										rank_labels[static_cast<size_t>(src_rank)],
										rank_labels[static_cast<size_t>(dst_rank)], metric[7],
										metric[8], metric[9], metric[10],
										metric[11], metric[12], args.stat_out));
				mirror(print_pair_line("pair_cuda",
										rank_labels[static_cast<size_t>(src_rank)],
										rank_labels[static_cast<size_t>(dst_rank)], metric[19],
										metric[20], metric[21], metric[22],
										metric[23], metric[24], args.stat_out));
			} else {
				mirror(print_pair_line("pair",
										rank_labels[static_cast<size_t>(src_rank)],
										rank_labels[static_cast<size_t>(dst_rank)], metric[0],
										metric[1], metric[2], metric[3],
										metric[4], metric[5], args.stat_out));
			}
		}
	}

	const double total_elapsed_s = MPI_Wtime() - test_t0;
	{
		/* Этот ostringstream НЕ под if(args.debug): TotalTimeSec — обычная
		   часть отчёта в --out, а не отладка. */
		std::ostringstream oss;
		oss << "TotalTimeSec: " << std::fixed
			<< std::setprecision(TOTAL_TIME_DIGITS) << total_elapsed_s << "\n";
		mirror(oss.str());
	}
	DBG_LOG(rank, args, "all_to_all SCHEDULE_DONE");
}

} // namespace gpu_benchmark
