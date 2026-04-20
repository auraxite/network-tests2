#include "gpu_one_to_one_node.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <sstream>
#include <vector>

namespace gpu_benchmark {

namespace {

static constexpr int RDV_TAG_FORWARD = 142;
static constexpr int RDV_TAG_BACKWARD = 143;

int global_gpu_offset(const std::vector<int> &gpu_counts, int rank) {
	int offset = 0;
	for (int r = 0; r < rank; ++r) {
		if (gpu_counts[static_cast<size_t>(r)] > 0)
			offset += gpu_counts[static_cast<size_t>(r)];
	}
	return offset;
}

const std::string &gpu_label(const std::vector<int> &gpu_counts,
							 const std::vector<std::string> &global_gpu_labels, int rank,
							 int gpu) {
	return global_gpu_labels[static_cast<size_t>(global_gpu_offset(gpu_counts, rank) + gpu)];
}

void enable_local_peer_access(int local_gpu_count) {
	for (int src = 0; src < local_gpu_count; ++src) {
		for (int dst = 0; dst < local_gpu_count; ++dst) {
			if (src == dst)
				continue;
			int can_access = 0;
			cuda_ok(cudaSetDevice(src), "cudaSetDevice(enable peer)");
			cuda_ok(cudaDeviceCanAccessPeer(&can_access, src, dst),
					"cudaDeviceCanAccessPeer");
			if (!can_access)
				continue;
			cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0);
			if (err == cudaErrorPeerAccessAlreadyEnabled) {
				cudaGetLastError();
				continue;
			}
			cuda_ok(err, "cudaDeviceEnablePeerAccess");
		}
	}
}

std::vector<double> run_one_to_one_node(
	int rank, const Task &t, const Args &args, bool check_host,
	const std::vector<int> &gpu_counts,
	const std::vector<std::string> &global_gpu_labels,
	const std::vector<char *> &d_send_bufs, const std::vector<char *> &d_recv_bufs,
	char *h_buf) {
	std::vector<double> ack(ACK_FIELDS, 0.0);
	const bool is_sender = (rank == t.src_rank);
	const bool is_receiver = (rank == t.dst_rank);
	const bool is_local_pair = (t.src_rank == t.dst_rank);
	const bool is_same_gpu = is_local_pair && (t.src_gpu == t.dst_gpu);
	/* Таймер закреплён за получателем, как и в обычном one_to_one.
	   Для межузловой пары старт итерации выравниваем pair-local rendezvous
	   на КАЖДОЙ итерации, чтобы receiver-side окно лучше покрывало sender D2H. */
	const bool collect_raw_samples_here = is_receiver;

	if (!is_sender && !is_receiver)
		return ack;
	if (is_same_gpu) {
		ack[6] = 1.0;
		return ack;
	}

	const int count = static_cast<int>(args.nbytes);
	char *d_send = nullptr;
	char *d_recv = nullptr;
	if (is_sender) {
		d_send = d_send_bufs[static_cast<size_t>(t.src_gpu)];
		cuda_ok(cudaSetDevice(t.src_gpu), "cudaSetDevice(src)");
		cuda_ok(cudaMemset(d_send, 0xAA, args.nbytes), "cudaMemset(src)");
	}
	if (is_receiver) {
		d_recv = d_recv_bufs[static_cast<size_t>(t.dst_gpu)];
	}

	DBG_LOG(rank, args,
			"one_to_one_node RUN_BEGIN src=" << t.src_rank << "." << t.src_gpu
			<< " dst=" << t.dst_rank << "." << t.dst_gpu << " bytes=" << args.nbytes
			<< " local_pair=" << (is_local_pair ? 1 : 0)
			<< " env_path=" << (check_host ? "host" : "auto"));

	int current_iter = -1;
	const char *current_phase = "none";
	auto do_one = [&]() {
		if (is_local_pair) {
			if (check_host) {
				DBG_LOG(rank, args,
						"one_to_one_node LOCAL_D2H_BEGIN idx=" << current_iter
						<< " phase=" << current_phase);
				cuda_ok(cudaSetDevice(t.src_gpu), "cudaSetDevice(local src)");
				cuda_ok(cudaMemcpy(h_buf, d_send, count, cudaMemcpyDeviceToHost),
						"D2H(local host)");
				DBG_LOG(rank, args,
						"one_to_one_node LOCAL_D2H_DONE idx=" << current_iter
						<< " phase=" << current_phase);
				DBG_LOG(rank, args,
						"one_to_one_node LOCAL_H2D_BEGIN idx=" << current_iter
						<< " phase=" << current_phase);
				cuda_ok(cudaSetDevice(t.dst_gpu), "cudaSetDevice(local dst)");
				cuda_ok(cudaMemcpy(d_recv, h_buf, count, cudaMemcpyHostToDevice),
						"H2D(local host)");
				DBG_LOG(rank, args,
						"one_to_one_node LOCAL_H2D_DONE idx=" << current_iter
						<< " phase=" << current_phase);
			} else {
				DBG_LOG(rank, args,
						"one_to_one_node LOCAL_P2P_BEGIN idx=" << current_iter
						<< " phase=" << current_phase);
				cuda_ok(cudaMemcpyPeer(d_recv, t.dst_gpu, d_send, t.src_gpu, count),
						"cudaMemcpyPeer(local auto)");
				DBG_LOG(rank, args,
						"one_to_one_node LOCAL_P2P_DONE idx=" << current_iter
						<< " phase=" << current_phase);
			}
			return;
		}

		if (is_sender) {
			cuda_ok(cudaSetDevice(t.src_gpu), "cudaSetDevice(send)");
			if (check_host) {
				DBG_LOG(rank, args,
						"one_to_one_node D2H_BEGIN idx=" << current_iter
						<< " phase=" << current_phase);
				cuda_ok(cudaMemcpy(h_buf, d_send, count, cudaMemcpyDeviceToHost),
						"D2H(remote host)");
				DBG_LOG(rank, args,
						"one_to_one_node D2H_DONE idx=" << current_iter
						<< " phase=" << current_phase);
				mpi_ok(MPI_Send(h_buf, count, MPI_BYTE, t.dst_rank, 0, MPI_COMM_WORLD),
					   "MPI_Send(node host)");
			} else {
				mpi_ok(MPI_Send(d_send, count, MPI_BYTE, t.dst_rank, 0, MPI_COMM_WORLD),
					   "MPI_Send(node auto)");
			}
		}
		if (is_receiver) {
			MPI_Status st{};
			cuda_ok(cudaSetDevice(t.dst_gpu), "cudaSetDevice(recv)");
			if (check_host) {
				mpi_ok(MPI_Recv(h_buf, count, MPI_BYTE, t.src_rank, 0, MPI_COMM_WORLD, &st),
					   "MPI_Recv(node host)");
				DBG_LOG(rank, args,
						"one_to_one_node H2D_BEGIN idx=" << current_iter
						<< " phase=" << current_phase);
				cuda_ok(cudaMemcpy(d_recv, h_buf, count, cudaMemcpyHostToDevice),
						"H2D(remote host)");
				DBG_LOG(rank, args,
						"one_to_one_node H2D_DONE idx=" << current_iter
						<< " phase=" << current_phase);
			} else {
				mpi_ok(MPI_Recv(d_recv, count, MPI_BYTE, t.src_rank, 0, MPI_COMM_WORLD, &st),
					   "MPI_Recv(node auto)");
			}
		}
	};

	std::vector<double> samples_mpi_us;
	std::vector<double> samples_cpu_us;
	std::vector<double> samples_gpu_us;
	cudaEvent_t ev_start = nullptr;
	cudaEvent_t ev_stop = nullptr;
	if (collect_raw_samples_here) {
		cuda_ok(cudaSetDevice(t.dst_gpu), "cudaSetDevice(timer init)");
		cuda_ok(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
		cuda_ok(cudaEventCreate(&ev_stop), "cudaEventCreate(stop)");
	}
	samples_mpi_us.reserve(static_cast<size_t>(std::max(1, args.iters)));
	samples_cpu_us.reserve(static_cast<size_t>(std::max(1, args.iters)));
	samples_gpu_us.reserve(static_cast<size_t>(std::max(1, args.iters)));

	DBG_LOG(rank, args, "one_to_one_node WARMUP_BEGIN");
	current_phase = "warmup";
	for (int i = 0; i < args.warmup; ++i) {
		current_iter = i;
		do_one();
	}
	DBG_LOG(rank, args, "one_to_one_node WARMUP_DONE");

	DBG_LOG(rank, args, "one_to_one_node ITERS_BEGIN");
	current_phase = "iters";
	for (int i = 0; i < args.iters; ++i) {
		current_iter = i;

		if (!is_local_pair) {
			char dummy = 0;
			if (is_sender) {
				mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, t.dst_rank, RDV_TAG_FORWARD,
									&dummy, 0, MPI_BYTE, t.dst_rank, RDV_TAG_BACKWARD,
									MPI_COMM_WORLD, MPI_STATUS_IGNORE),
					   "MPI_Sendrecv iter rendezvous (node sender)");
			} else if (is_receiver) {
				mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, t.src_rank, RDV_TAG_BACKWARD,
									&dummy, 0, MPI_BYTE, t.src_rank, RDV_TAG_FORWARD,
									MPI_COMM_WORLD, MPI_STATUS_IGNORE),
					   "MPI_Sendrecv iter rendezvous (node receiver)");
			}
		}

		if (collect_raw_samples_here) {
			cuda_ok(cudaSetDevice(t.dst_gpu), "cudaSetDevice(timer start)");
			cuda_ok(cudaEventRecord(ev_start), "cudaEventRecord(start)");
		}
		const double t0_mpi = collect_raw_samples_here ? MPI_Wtime() : 0.0;
		const double t0_clk = collect_raw_samples_here ? clock_gettime_wrapper() : 0.0;
		do_one();
		const double t1_mpi = collect_raw_samples_here ? MPI_Wtime() : 0.0;
		const double t1_clk = collect_raw_samples_here ? clock_gettime_wrapper() : 0.0;

		if (collect_raw_samples_here) {
			samples_mpi_us.push_back((t1_mpi - t0_mpi) * 1e6);
			samples_cpu_us.push_back(t1_clk - t0_clk);
			cuda_ok(cudaSetDevice(t.dst_gpu), "cudaSetDevice(timer stop)");
			cuda_ok(cudaEventRecord(ev_stop), "cudaEventRecord(stop)");
			cuda_ok(cudaEventSynchronize(ev_stop), "cudaEventSynchronize(stop)");
			float elapsed_ms = 0.0f;
			cuda_ok(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop),
					"cudaEventElapsedTime");
			samples_gpu_us.push_back(static_cast<double>(elapsed_ms) * 1e3);
		}
	}
	DBG_LOG(rank, args, "one_to_one_node ITERS_DONE");

	if (collect_raw_samples_here) {
		const std::string &src_label =
			gpu_label(gpu_counts, global_gpu_labels, t.src_rank, t.src_gpu);
		const std::string &dst_label =
			gpu_label(gpu_counts, global_gpu_labels, t.dst_rank, t.dst_gpu);
		switch (args.timer) {
		case Timer::All:
		case Timer::Mpi:
			append_raw_samples_named(args, src_label, dst_label, samples_mpi_us);
			break;
		case Timer::Cpu:
			append_raw_samples_named(args, src_label, dst_label, samples_cpu_us);
			break;
		case Timer::Cuda:
			append_raw_samples_named(args, src_label, dst_label, samples_gpu_us);
			break;
		}
	}

	if (!samples_gpu_us.empty() && !samples_mpi_us.empty() && !samples_cpu_us.empty())
		fill_ack(samples_mpi_us, samples_cpu_us, samples_gpu_us, args, ack.data());
	if (ev_start)
		cuda_ok(cudaEventDestroy(ev_start), "cudaEventDestroy(start)");
	if (ev_stop)
		cuda_ok(cudaEventDestroy(ev_stop), "cudaEventDestroy(stop)");
	return ack;
}

} // namespace

void schedule_one_to_one_node(int rank, int nproc, int local_gpu_count,
							  const std::vector<int> &gpu_counts, const Args &args,
							  bool check_host,
							  const std::vector<std::string> &global_gpu_labels,
							  const std::function<void(const std::string &)> &mirror) {
	std::vector<char *> d_send_bufs(static_cast<size_t>(local_gpu_count), nullptr);
	std::vector<char *> d_recv_bufs(static_cast<size_t>(local_gpu_count), nullptr);
	for (int g = 0; g < local_gpu_count; ++g) {
		cuda_ok(cudaSetDevice(g), "cudaSetDevice(alloc)");
		cuda_ok(cudaMalloc(&d_send_bufs[static_cast<size_t>(g)], args.nbytes),
				"cudaMalloc(d_send)");
		cuda_ok(cudaMalloc(&d_recv_bufs[static_cast<size_t>(g)], args.nbytes),
				"cudaMalloc(d_recv)");
	}
	if (!check_host)
		enable_local_peer_access(local_gpu_count);

	char *h_buf = nullptr;
	if (check_host)
		cuda_ok(cudaMallocHost(&h_buf, args.nbytes), "cudaMallocHost(h_buf)");

	if (rank != 0) {
		while (true) {
			Task tw{};
			mpi_ok(MPI_Recv(&tw, sizeof(Task), MPI_BYTE, 0, 1, MPI_COMM_WORLD,
							MPI_STATUS_IGNORE),
				   "MPI_Recv Task (node worker)");
			if (tw.stop)
				break;
			auto ack = run_one_to_one_node(rank, tw, args, check_host, gpu_counts,
										   global_gpu_labels, d_send_bufs, d_recv_bufs, h_buf);
			mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD),
				   "MPI_Send ack (node worker)");
		}
		if (h_buf)
			cuda_ok(cudaFreeHost(h_buf), "cudaFreeHost(h_buf)");
		for (int g = 0; g < local_gpu_count; ++g) {
			cuda_ok(cudaSetDevice(g), "cudaSetDevice(free)");
			cuda_ok(cudaFree(d_send_bufs[static_cast<size_t>(g)]), "cudaFree(d_send)");
			cuda_ok(cudaFree(d_recv_bufs[static_cast<size_t>(g)]), "cudaFree(d_recv)");
		}
		return;
	}

	const auto label_for = [&](int node_rank, int local_gpu) -> const std::string & {
		return gpu_label(gpu_counts, global_gpu_labels, node_rank, local_gpu);
	};

	const double test_t0 = MPI_Wtime();
	Task t{};
	for (int src_rank = 0; src_rank < nproc; ++src_rank) {
		const int src_gpu_count = gpu_counts[static_cast<size_t>(src_rank)];
		for (int src_gpu = 0; src_gpu < src_gpu_count; ++src_gpu) {
			for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
				const int dst_gpu_count = gpu_counts[static_cast<size_t>(dst_rank)];
				for (int dst_gpu = 0; dst_gpu < dst_gpu_count; ++dst_gpu) {
					t.src_rank = src_rank;
					t.src_gpu = src_gpu;
					t.dst_rank = dst_rank;
					t.dst_gpu = dst_gpu;
					t.stop = 0;

					if (src_rank == dst_rank && src_gpu == dst_gpu) {
						if (args.timer == Timer::All) {
							mirror(print_pair_line("pair_mpi", label_for(src_rank, src_gpu),
												 label_for(dst_rank, dst_gpu), 0.0, 0.0, 0.0,
												 0.0, 0.0, 0.0, args.stat_out));
							mirror(print_pair_line("pair_cpu", label_for(src_rank, src_gpu),
												 label_for(dst_rank, dst_gpu), 0.0, 0.0, 0.0,
												 0.0, 0.0, 0.0, args.stat_out));
							mirror(print_pair_line("pair_cuda", label_for(src_rank, src_gpu),
												 label_for(dst_rank, dst_gpu), 0.0, 0.0, 0.0,
												 0.0, 0.0, 0.0, args.stat_out));
						} else {
							mirror(print_pair_line("pair", label_for(src_rank, src_gpu),
												 label_for(dst_rank, dst_gpu), 0.0, 0.0, 0.0,
												 0.0, 0.0, 0.0, args.stat_out));
						}
						continue;
					}

					if (src_rank != 0)
						mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, src_rank, 1,
										MPI_COMM_WORLD),
							   "MPI_Send Task (node src)");
					if (dst_rank != 0 && dst_rank != src_rank)
						mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, dst_rank, 1,
										MPI_COMM_WORLD),
							   "MPI_Send Task (node dst)");

					std::vector<double> metric(ACK_FIELDS, 0.0);
					if (src_rank == 0 || dst_rank == 0) {
						auto ack0 = run_one_to_one_node(rank, t, args, check_host, gpu_counts,
														global_gpu_labels, d_send_bufs,
														d_recv_bufs, h_buf);
						if (ack0[6] == 1.0)
							metric = ack0;
					}
					if (src_rank != 0) {
						std::vector<double> ack(ACK_FIELDS, 0.0);
						mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, src_rank, 2,
										MPI_COMM_WORLD, MPI_STATUS_IGNORE),
							   "MPI_Recv ack (node src)");
						if (ack[6] == 1.0)
							metric = ack;
					}
					if (dst_rank != 0 && dst_rank != src_rank) {
						std::vector<double> ack(ACK_FIELDS, 0.0);
						mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, dst_rank, 2,
										MPI_COMM_WORLD, MPI_STATUS_IGNORE),
							   "MPI_Recv ack (node dst)");
						if (ack[6] == 1.0)
							metric = ack;
					}

					if (args.timer == Timer::All) {
						mirror(print_pair_line("pair_mpi", label_for(src_rank, src_gpu),
											 label_for(dst_rank, dst_gpu), metric[13],
											 metric[14], metric[15], metric[16],
											 metric[17], metric[18], args.stat_out));
						mirror(print_pair_line("pair_cpu", label_for(src_rank, src_gpu),
											 label_for(dst_rank, dst_gpu), metric[7],
											 metric[8], metric[9], metric[10], metric[11],
											 metric[12], args.stat_out));
						mirror(print_pair_line("pair_cuda", label_for(src_rank, src_gpu),
											 label_for(dst_rank, dst_gpu), metric[19],
											 metric[20], metric[21], metric[22],
											 metric[23], metric[24], args.stat_out));
					} else {
						mirror(print_pair_line("pair", label_for(src_rank, src_gpu),
											 label_for(dst_rank, dst_gpu), metric[0],
											 metric[1], metric[2], metric[3], metric[4],
											 metric[5], args.stat_out));
					}
				}
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

	t.stop = 1;
	for (int r = 1; r < nproc; ++r)
		mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, r, 1, MPI_COMM_WORLD),
			   "MPI_Send stop Task (node mode)");

	if (h_buf)
		cuda_ok(cudaFreeHost(h_buf), "cudaFreeHost(h_buf)");
	for (int g = 0; g < local_gpu_count; ++g) {
		cuda_ok(cudaSetDevice(g), "cudaSetDevice(free root)");
		cuda_ok(cudaFree(d_send_bufs[static_cast<size_t>(g)]), "cudaFree(d_send)");
		cuda_ok(cudaFree(d_recv_bufs[static_cast<size_t>(g)]), "cudaFree(d_recv)");
	}
}

} // namespace gpu_benchmark
