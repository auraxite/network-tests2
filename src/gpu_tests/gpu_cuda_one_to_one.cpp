#include "gpu_common.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace gpu_benchmark {
namespace {

constexpr int CUDA_READY_TAG = 4101;
constexpr int CUDA_ITER_TAG  = 4102;

void require_single_node(int rank, int nproc, const std::vector<int> &node_ranks) {
	if (static_cast<int>(node_ranks.size()) == nproc) return;
	if (rank == 0)
		std::cerr << "cuda_one_to_one requires all ranks on one node: CUDA IPC "
		             "cannot access GPU memory on another node.\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
}

void close_peer_handles(std::vector<char *> &peer_recv) {
	for (char *p : peer_recv)
		if (p) cuda_ok(cudaIpcCloseMemHandle(p), "cudaIpcCloseMemHandle");
}

std::vector<double> run_pair(
    int rank, const Task &task, const Args &args, char *d_send,
    const std::vector<char *> &peer_recv,
    const std::vector<std::string> &rank_labels) {
	std::vector<double> ack(ACK_FIELDS, 0.0);
	const bool is_sender = rank == task.src_rank;
	const bool is_receiver = rank == task.dst_rank;
	if (!is_sender && !is_receiver) return ack;

	auto transfer = [&](bool measure, std::vector<double> *samples) {
		if (is_receiver) {
			const double t0 = measure ? MPI_Wtime() : 0.0;
			char ready = 0;
			mpi_ok(MPI_Recv(&ready, 0, MPI_BYTE, task.src_rank, CUDA_READY_TAG,
			                MPI_COMM_WORLD, MPI_STATUS_IGNORE),
			       "MPI_Recv CUDA IPC ready");
			// The sender synchronizes its copy stream before this signal. This
			// synchronize only orders receiver-side work before recording t1.
			cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize(receiver)");
			if (measure) samples->push_back((MPI_Wtime() - t0) * 1e6);
		}

		if (is_sender) {
			char *remote_recv = peer_recv[static_cast<size_t>(task.dst_rank)];
			if (!remote_recv) {
				std::cerr << "rank " << rank << ": CUDA IPC handle for rank "
				          << task.dst_rank << " is unavailable\n";
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			cuda_ok(cudaMemcpyAsync(remote_recv, d_send, args.nbytes,
			                        cudaMemcpyDeviceToDevice, nullptr),
			        "cudaMemcpyAsync(CUDA IPC P2P)");
			cuda_ok(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize(sender)");
			char ready = 0;
			mpi_ok(MPI_Send(&ready, 0, MPI_BYTE, task.dst_rank, CUDA_READY_TAG,
			                MPI_COMM_WORLD),
			       "MPI_Send CUDA IPC ready");
		}

		char sync = 0;
		const int peer = is_sender ? task.dst_rank : task.src_rank;
		mpi_ok(MPI_Sendrecv(&sync, 0, MPI_BYTE, peer, CUDA_ITER_TAG,
		                    &sync, 0, MPI_BYTE, peer, CUDA_ITER_TAG,
		                    MPI_COMM_WORLD, MPI_STATUS_IGNORE),
		       "MPI_Sendrecv CUDA IPC iteration");
	};

	std::vector<double> samples;
	samples.reserve(static_cast<size_t>(std::max(1, args.iters)));
	for (int i = 0; i < args.warmup; ++i) transfer(false, nullptr);
	for (int i = 0; i < args.iters; ++i) transfer(true, &samples);

	if (is_receiver) {
		append_raw_samples(args, rank, task, rank_labels, samples);
		fill_ack(samples, ack.data());
	}
	return ack;
}

} // namespace

void schedule_cuda_one_to_one(
    int rank, int nproc, const Args &args, MPI_Comm node_comm,
    const std::vector<int> &node_ranks,
    const std::vector<std::string> &rank_labels,
    const std::function<void(const std::string &)> &mirror) {
	require_single_node(rank, nproc, node_ranks);

	char *d_send = nullptr;
	char *d_recv = nullptr;
	cuda_ok(cudaMalloc(&d_send, args.nbytes), "cudaMalloc(CUDA IPC d_send)");
	cuda_ok(cudaMalloc(&d_recv, args.nbytes), "cudaMalloc(CUDA IPC d_recv)");
	cuda_ok(cudaMemset(d_send, 0xAA, args.nbytes), "cudaMemset(CUDA IPC d_send)");

	cudaIpcMemHandle_t own_handle{};
	cuda_ok(cudaIpcGetMemHandle(&own_handle, d_recv), "cudaIpcGetMemHandle");
	std::vector<cudaIpcMemHandle_t> handles(static_cast<size_t>(nproc));
	mpi_ok(MPI_Allgather(&own_handle, static_cast<int>(sizeof(own_handle)), MPI_BYTE,
	                     handles.data(), static_cast<int>(sizeof(own_handle)), MPI_BYTE,
	                     node_comm),
	       "MPI_Allgather CUDA IPC handles");
	std::vector<cudaIpcMemHandle_t> global_handles(static_cast<size_t>(nproc));
	for (int local = 0; local < nproc; ++local)
		global_handles[static_cast<size_t>(node_ranks[static_cast<size_t>(local)])] =
		    handles[static_cast<size_t>(local)];

	std::vector<char *> peer_recv(static_cast<size_t>(nproc), nullptr);
	for (int peer = 0; peer < nproc; ++peer) {
		if (peer == rank) continue;
		cuda_ok(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&peer_recv[static_cast<size_t>(peer)]),
		                             global_handles[static_cast<size_t>(peer)],
		                             cudaIpcMemLazyEnablePeerAccess),
		        "cudaIpcOpenMemHandle");
	}

	if (rank != 0) {
		while (true) {
			Task task{};
			mpi_ok(MPI_Recv(&task, sizeof(Task), MPI_BYTE, 0, 1, MPI_COMM_WORLD,
			                MPI_STATUS_IGNORE),
			       "MPI_Recv CUDA IPC task");
			if (task.stop) break;
			auto ack = run_pair(rank, task, args, d_send, peer_recv, rank_labels);
			mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD),
			       "MPI_Send CUDA IPC ack");
		}
		close_peer_handles(peer_recv);
		cudaFree(d_send);
		cudaFree(d_recv);
		return;
	}

	const double test_t0 = MPI_Wtime();
	Task task{};
	for (int src = 0; src < nproc; ++src) {
		for (int dst = 0; dst < nproc; ++dst) {
			task = {src, 0, dst, 0, 0};
			if (src == dst) {
				mirror(print_pair_line("pair", rank_labels[static_cast<size_t>(src)],
				       rank_labels[static_cast<size_t>(dst)], 0, 0, 0, 0, 0, 0,
				       args.stat_out));
				continue;
			}

			if (src != 0)
				mpi_ok(MPI_Send(&task, sizeof(Task), MPI_BYTE, src, 1, MPI_COMM_WORLD),
				       "MPI_Send CUDA IPC src task");
			if (dst != 0)
				mpi_ok(MPI_Send(&task, sizeof(Task), MPI_BYTE, dst, 1, MPI_COMM_WORLD),
				       "MPI_Send CUDA IPC dst task");

			std::vector<double> metric(ACK_FIELDS, 0.0);
			if (src == 0 || dst == 0) {
				auto ack = run_pair(rank, task, args, d_send, peer_recv, rank_labels);
				if (ack[6] == 1.0) metric = ack;
			}
			if (src != 0) {
				std::vector<double> ack(ACK_FIELDS, 0.0);
				mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, src, 2,
				                MPI_COMM_WORLD, MPI_STATUS_IGNORE),
				       "MPI_Recv CUDA IPC src ack");
				if (ack[6] == 1.0) metric = ack;
			}
			if (dst != 0) {
				std::vector<double> ack(ACK_FIELDS, 0.0);
				mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, dst, 2,
				                MPI_COMM_WORLD, MPI_STATUS_IGNORE),
				       "MPI_Recv CUDA IPC dst ack");
				if (ack[6] == 1.0) metric = ack;
			}
			mirror(print_pair_line("pair", rank_labels[static_cast<size_t>(src)],
			       rank_labels[static_cast<size_t>(dst)], metric[0], metric[1],
			       metric[2], metric[3], metric[4], metric[5], args.stat_out));
		}
	}

	std::ostringstream oss;
	oss << "TotalTimeSec: " << std::fixed << std::setprecision(TOTAL_TIME_DIGITS)
	    << (MPI_Wtime() - test_t0) << "\n";
	mirror(oss.str());

	task.stop = 1;
	for (int peer = 1; peer < nproc; ++peer)
		mpi_ok(MPI_Send(&task, sizeof(Task), MPI_BYTE, peer, 1, MPI_COMM_WORLD),
		       "MPI_Send CUDA IPC stop");
	close_peer_handles(peer_recv);
	cudaFree(d_send);
	cudaFree(d_recv);
}

} // namespace gpu_benchmark
