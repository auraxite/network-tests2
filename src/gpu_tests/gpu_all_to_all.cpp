#include "gpu_all_to_all.hpp"

#include <algorithm>     // std::max
#include <cuda_runtime.h> // CUDA: устройство, события, cudaMalloc, ...
#include <functional>    // std::function — параметр mirror в schedule_all_to_all
#include <iomanip>       // std::setprecision, std::fixed
#include <sstream>       // std::ostringstream — строки TotalTimeSec и т.п.
#include <vector>        // буферы ack, samples, MPI_Request

namespace gpu_benchmark {

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

	cuda_ok(cudaSetDevice(local_gpu), "cudaSetDevice(all_to_all)");
	cuda_ok(cudaMalloc(&d_send, args.nbytes), "cudaMalloc(send)");
	cuda_ok(cudaMemset(d_send, 0xAA, args.nbytes), "cudaMemset(send)");
	cuda_ok(cudaMalloc(&d_recv, args.nbytes), "cudaMalloc(recv)");

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

	auto do_one = [&](bool measure,
					  std::vector<std::vector<double>> *samples_mpi_us,
					  std::vector<std::vector<double>> *samples_cpu_us,
					  std::vector<std::vector<double>> *samples_gpu_us) {
		std::vector<MPI_Request> recv_req(static_cast<size_t>(nproc), MPI_REQUEST_NULL);
		for (int src_rank = 0; src_rank < nproc; ++src_rank) {
			if (src_rank == rank)
				continue;
			char *recv_buf = recv_bufs[static_cast<size_t>(src_rank)];
			mpi_ok(MPI_Irecv(recv_buf, count, MPI_BYTE, src_rank,
							 alltoall_pair_tag(src_rank, rank, nproc),
							 MPI_COMM_WORLD, &recv_req[static_cast<size_t>(src_rank)]),
				   "MPI_Irecv(all_to_all)");
		}

		cudaEvent_t ev_start = nullptr;
		cudaEvent_t ev_stop = nullptr;
		cuda_ok(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
		cuda_ok(cudaEventCreate(&ev_stop), "cudaEventCreate(stop)");

		for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
			if (dst_rank == rank)
				continue;

			char *send_buf = send_bufs[static_cast<size_t>(dst_rank)];
			const double t0_mpi = MPI_Wtime();
			const double t0_clk = clock_gettime_wrapper();
			cuda_ok(cudaEventRecord(ev_start), "cudaEventRecord(start)");

			if (check_host) {
				cuda_ok(cudaMemcpy(send_buf, d_send, args.nbytes, cudaMemcpyDeviceToHost),
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

			cuda_ok(cudaEventRecord(ev_stop), "cudaEventRecord(stop)");
			cuda_ok(cudaEventSynchronize(ev_stop), "cudaEventSynchronize(stop)");

			if (measure) {
				float elapsed_ms = 0.0f;
				cuda_ok(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop),
						"cudaEventElapsedTime");
				const double t1_mpi = MPI_Wtime();
				const double t1_clk = clock_gettime_wrapper();
				(*samples_mpi_us)[static_cast<size_t>(dst_rank)].push_back(
					(t1_mpi - t0_mpi) * 1e6);
				(*samples_cpu_us)[static_cast<size_t>(dst_rank)].push_back(t1_clk - t0_clk);
				(*samples_gpu_us)[static_cast<size_t>(dst_rank)].push_back(
					static_cast<double>(elapsed_ms) * 1e3);
			}
		}

		cudaEventDestroy(ev_start);
		cudaEventDestroy(ev_stop);

		for (int src_rank = 0; src_rank < nproc; ++src_rank) {
			if (src_rank == rank)
				continue;
			mpi_ok(MPI_Wait(&recv_req[static_cast<size_t>(src_rank)], MPI_STATUS_IGNORE),
				   "MPI_Wait(all_to_all recv)");
			if (check_host) {
				cuda_ok(cudaMemcpy(d_recv, recv_bufs[static_cast<size_t>(src_rank)], args.nbytes,
								   cudaMemcpyHostToDevice),
						"H2D(all_to_all)");
			}
		}
	};

	std::vector<std::vector<double>> samples_mpi_us(static_cast<size_t>(nproc));
	std::vector<std::vector<double>> samples_cpu_us(static_cast<size_t>(nproc));
	std::vector<std::vector<double>> samples_gpu_us(static_cast<size_t>(nproc));
	for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
		if (dst_rank == rank)
			continue;
		samples_mpi_us[static_cast<size_t>(dst_rank)].reserve(
			static_cast<size_t>(std::max(1, args.iters)));
		samples_cpu_us[static_cast<size_t>(dst_rank)].reserve(
			static_cast<size_t>(std::max(1, args.iters)));
		samples_gpu_us[static_cast<size_t>(dst_rank)].reserve(
			static_cast<size_t>(std::max(1, args.iters)));
	}

	for (int i = 0; i < args.warmup; ++i)
		do_one(false, nullptr, nullptr, nullptr);

	for (int i = 0; i < args.iters; ++i)
		do_one(true, &samples_mpi_us, &samples_cpu_us, &samples_gpu_us);

	for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
		if (dst_rank == rank)
			continue;
		Task t{};
		t.src_rank = rank;
		t.src_gpu = local_gpu;
		t.dst_rank = dst_rank;
		t.dst_gpu = local_gpu;
		switch (args.timer) {
		case Timer::All:
		case Timer::Mpi:
			append_raw_samples(args, rank, t, rank_labels,
							   samples_mpi_us[static_cast<size_t>(dst_rank)]);
			break;
		case Timer::Cpu:
			append_raw_samples(args, rank, t, rank_labels,
							   samples_cpu_us[static_cast<size_t>(dst_rank)]);
			break;
		case Timer::Cuda:
			append_raw_samples(args, rank, t, rank_labels,
							   samples_gpu_us[static_cast<size_t>(dst_rank)]);
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

	for (int src_rank = 0; src_rank < nproc; ++src_rank) {
		for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
			if (src_rank == dst_rank) {
				if (rank == 0)
					result_ptr(src_rank, dst_rank)[6] = 1.0;
				continue;
			}

			if (rank == src_rank) {
				fill_ack(samples_mpi_us[static_cast<size_t>(dst_rank)],
											 samples_cpu_us[static_cast<size_t>(dst_rank)],
											 samples_gpu_us[static_cast<size_t>(dst_rank)], args,
											 ack.data());
				if (rank == 0) {
					std::copy(ack.begin(), ack.end(), result_ptr(src_rank, dst_rank));
				} else {
					mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0,
									nproc * nproc + alltoall_pair_tag(src_rank, dst_rank, nproc),
									MPI_COMM_WORLD),
						   "MPI_Send ack(all_to_all)");
				}
			} else if (rank == 0) {
				mpi_ok(MPI_Recv(result_ptr(src_rank, dst_rank), ACK_FIELDS, MPI_DOUBLE,
								src_rank,
								nproc * nproc + alltoall_pair_tag(src_rank, dst_rank, nproc),
								MPI_COMM_WORLD, MPI_STATUS_IGNORE),
					   "MPI_Recv ack(all_to_all)");
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
	return results;
}

void schedule_all_to_all(
	int rank, int nproc, const Args &args, bool via_host,
	const std::vector<std::string> &rank_labels,
	const std::function<void(const std::string &)> &mirror, NetcdfBundle *nc,
	int matrix_idx) {
	if (rank != 0) {
		run_all_to_all(rank, nproc, args, via_host, 0, rank_labels);
		return;
	}

	if (nc != nullptr)
		netcdf_reset_matrix(*nc);
	const double test_t0 = MPI_Wtime();
	std::vector<double> results =
		run_all_to_all(rank, nproc, args, via_host, 0, rank_labels);

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
			if (nc != nullptr)
				netcdf_store_pair(*nc, src_rank, dst_rank, metric);
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
		std::ostringstream oss;
		oss << "TotalTimeSec: " << std::fixed
			<< std::setprecision(TOTAL_TIME_DIGITS) << total_elapsed_s << "\n";
		mirror(oss.str());
	}
	if (nc != nullptr)
		netcdf_write_matrix_slice(*nc, matrix_idx);
}

} // namespace gpu_benchmark
