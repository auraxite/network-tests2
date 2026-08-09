#include "gpu_common.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace gpu_benchmark {
namespace {

int pair_tag(int src, int dst, int nproc) {
	return src * nproc + dst;
}

void require_single_node(int rank, int nproc, const std::vector<int> &node_ranks) {
	if (static_cast<int>(node_ranks.size()) == nproc) return;
	if (rank == 0)
		std::cerr << "cuda_all_to_all requires all ranks on one node: CUDA IPC "
		             "cannot access GPU memory on another node.\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
}

void close_peer_handles(std::vector<char *> &peer_slots) {
	for (char *p : peer_slots)
		if (p) cuda_ok(cudaIpcCloseMemHandle(p), "cudaIpcCloseMemHandle");
}

} // namespace

void schedule_cuda_all_to_all(
    int rank, int nproc, const Args &args, MPI_Comm node_comm,
    const std::vector<int> &node_ranks,
    const std::vector<std::string> &rank_labels,
    const std::function<void(const std::string &)> &mirror) {
	require_single_node(rank, nproc, node_ranks);

	char *d_send = nullptr;
	cuda_ok(cudaMalloc(&d_send, args.nbytes), "cudaMalloc(CUDA IPC all d_send)");
	cuda_ok(cudaMemset(d_send, 0xAA, args.nbytes), "cudaMemset(CUDA IPC all d_send)");

	// Slot src belongs to this rank and is written only by source src.
	std::vector<char *> recv_slots(static_cast<size_t>(nproc), nullptr);
	std::vector<cudaIpcMemHandle_t> own_handles(static_cast<size_t>(nproc));
	for (int src = 0; src < nproc; ++src) {
		if (src == rank) continue;
		cuda_ok(cudaMalloc(&recv_slots[static_cast<size_t>(src)], args.nbytes),
		        "cudaMalloc(CUDA IPC receive slot)");
		cuda_ok(cudaIpcGetMemHandle(&own_handles[static_cast<size_t>(src)],
		                            recv_slots[static_cast<size_t>(src)]),
		        "cudaIpcGetMemHandle(receive slot)");
	}

	std::vector<cudaIpcMemHandle_t> local_handle_matrix(
	    static_cast<size_t>(nproc) * static_cast<size_t>(nproc));
	mpi_ok(MPI_Allgather(own_handles.data(),
	                     static_cast<int>(nproc * sizeof(cudaIpcMemHandle_t)), MPI_BYTE,
	                     local_handle_matrix.data(),
	                     static_cast<int>(nproc * sizeof(cudaIpcMemHandle_t)), MPI_BYTE,
	                     node_comm),
	       "MPI_Allgather CUDA IPC receive-slot handles");

	// MPI local ranks need not numerically equal world ranks.
	std::vector<cudaIpcMemHandle_t> handles(
	    static_cast<size_t>(nproc) * static_cast<size_t>(nproc));
	for (int local = 0; local < nproc; ++local) {
		const int global = node_ranks[static_cast<size_t>(local)];
		for (int src = 0; src < nproc; ++src)
			handles[static_cast<size_t>(global) * static_cast<size_t>(nproc) +
			        static_cast<size_t>(src)] =
			    local_handle_matrix[static_cast<size_t>(local) * static_cast<size_t>(nproc) +
			                        static_cast<size_t>(src)];
	}

	// On source rank, peer_slots[dst] is the slot owned by destination dst
	// specifically for messages from this source.
	std::vector<char *> peer_slots(static_cast<size_t>(nproc), nullptr);
	for (int dst = 0; dst < nproc; ++dst) {
		if (dst == rank) continue;
		const auto &handle = handles[static_cast<size_t>(dst) * static_cast<size_t>(nproc) +
		                             static_cast<size_t>(rank)];
		cuda_ok(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&peer_slots[static_cast<size_t>(dst)]),
		                             handle, cudaIpcMemLazyEnablePeerAccess),
		        "cudaIpcOpenMemHandle(receive slot)");
	}

	std::vector<std::vector<double>> samples(static_cast<size_t>(nproc));
	for (int src = 0; src < nproc; ++src)
		if (src != rank)
			samples[static_cast<size_t>(src)].reserve(static_cast<size_t>(std::max(1, args.iters)));
	const double test_t0 = MPI_Wtime();

	auto do_one = [&](bool measure) {
		std::vector<MPI_Request> recv_req(static_cast<size_t>(nproc), MPI_REQUEST_NULL);
		std::vector<MPI_Request> send_req(static_cast<size_t>(nproc), MPI_REQUEST_NULL);
		for (int src = 0; src < nproc; ++src) {
			if (src == rank) continue;
			mpi_ok(MPI_Irecv(nullptr, 0, MPI_BYTE, src, pair_tag(src, rank, nproc),
			                 MPI_COMM_WORLD, &recv_req[static_cast<size_t>(src)]),
			       "MPI_Irecv CUDA IPC ready");
		}

		// Same receiver-side timing model as the report: t0 is once before
		// the exchange phase and t1 is recorded per completed receive.
		const double t0 = measure ? MPI_Wtime() : 0.0;
		for (int dst = 0; dst < nproc; ++dst) {
			if (dst == rank) continue;
			cuda_ok(cudaMemcpyAsync(peer_slots[static_cast<size_t>(dst)], d_send,
			                        args.nbytes, cudaMemcpyDeviceToDevice, nullptr),
			        "cudaMemcpyAsync(CUDA IPC all-to-all)");
		}
		// A ready signal is sent only after every P2P write issued by this
		// source is complete, so a destination may safely consume its slot.
		cuda_ok(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize(CUDA IPC all)");
		for (int dst = 0; dst < nproc; ++dst) {
			if (dst == rank) continue;
			mpi_ok(MPI_Isend(nullptr, 0, MPI_BYTE, dst, pair_tag(rank, dst, nproc),
			                 MPI_COMM_WORLD, &send_req[static_cast<size_t>(dst)]),
			       "MPI_Isend CUDA IPC ready");
		}

		for (int done = 0; done < nproc - 1; ++done) {
			int src = MPI_UNDEFINED;
			mpi_ok(MPI_Waitany(nproc, recv_req.data(), &src, MPI_STATUS_IGNORE),
			       "MPI_Waitany CUDA IPC ready");
			if (src == MPI_UNDEFINED) break;
			if (measure)
				samples[static_cast<size_t>(src)].push_back((MPI_Wtime() - t0) * 1e6);
		}
		mpi_ok(MPI_Waitall(nproc, send_req.data(), MPI_STATUSES_IGNORE),
		       "MPI_Waitall CUDA IPC ready");
	};

	for (int i = 0; i < args.warmup; ++i) do_one(false);
	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(CUDA IPC iterations)");
	for (int i = 0; i < args.iters; ++i) do_one(true);

	for (int src = 0; src < nproc; ++src) {
		if (src == rank) continue;
		Task task{src, 0, rank, 0, 0};
		append_raw_samples(args, rank, task, rank_labels, samples[static_cast<size_t>(src)]);
	}

	std::vector<double> results;
	if (rank == 0)
		results.assign(static_cast<size_t>(nproc) * static_cast<size_t>(nproc) *
		               static_cast<size_t>(ACK_FIELDS), 0.0);
	auto result_ptr = [&](int src, int dst) -> double * {
		return results.data() +
		       (static_cast<size_t>(src) * static_cast<size_t>(nproc) +
		        static_cast<size_t>(dst)) * static_cast<size_t>(ACK_FIELDS);
	};

	std::vector<double> ack(ACK_FIELDS, 0.0);
	for (int src = 0; src < nproc; ++src) {
		for (int dst = 0; dst < nproc; ++dst) {
			if (src == dst) {
				if (rank == 0) result_ptr(src, dst)[6] = 1.0;
				continue;
			}
			if (rank == dst) {
				fill_ack(samples[static_cast<size_t>(src)], ack.data());
				if (rank == 0)
					std::copy(ack.begin(), ack.end(), result_ptr(src, dst));
				else
					mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0,
					                nproc * nproc + pair_tag(src, dst, nproc),
					                MPI_COMM_WORLD),
					       "MPI_Send CUDA IPC all ack");
			} else if (rank == 0) {
				mpi_ok(MPI_Recv(result_ptr(src, dst), ACK_FIELDS, MPI_DOUBLE, dst,
				                nproc * nproc + pair_tag(src, dst, nproc),
				                MPI_COMM_WORLD, MPI_STATUS_IGNORE),
				       "MPI_Recv CUDA IPC all ack");
			}
		}
	}

	if (rank == 0) {
		for (int src = 0; src < nproc; ++src)
			for (int dst = 0; dst < nproc; ++dst) {
				const double *m = result_ptr(src, dst);
				mirror(print_pair_line("pair", rank_labels[static_cast<size_t>(src)],
				       rank_labels[static_cast<size_t>(dst)], m[0], m[1], m[2], m[3],
				       m[4], m[5], args.stat_out));
			}
		std::ostringstream oss;
		oss << "TotalTimeSec: " << std::fixed << std::setprecision(TOTAL_TIME_DIGITS)
		    << (MPI_Wtime() - test_t0) << "\n";
		mirror(oss.str());
	}

	close_peer_handles(peer_slots);
	for (char *slot : recv_slots)
		if (slot) cudaFree(slot);
	cudaFree(d_send);
}

} // namespace gpu_benchmark
