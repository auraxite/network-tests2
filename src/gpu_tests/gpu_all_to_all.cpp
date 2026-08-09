#include "gpu_all_to_all.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <functional>
#include <iomanip>
#include <sstream>
#include <vector>

namespace gpu_benchmark {

static int pair_tag(int src, int dst, int nproc) {
	return src * nproc + dst;
}
static int local_ready_tag(int src, int dst, int nproc) {
	return nproc * nproc + pair_tag(src, dst, nproc);
}

void schedule_all_to_all(
    int rank, int nproc, const Args &args, bool via_host,
    MPI_Comm node_comm, int node_rank,
    const std::vector<int> &on_my_node,
    const std::vector<int> &node_ranks,
    const std::vector<std::string> &rank_labels,
    const std::function<void(const std::string &)> &mirror) {

	(void)node_rank;
	DBG_LOG(rank, args, "all_to_all SCHEDULE_BEGIN");

	if (rank != 0) {
		run_all_to_all(rank, nproc, args, via_host, 0,
		               node_comm, on_my_node, node_ranks, rank_labels);
		return;
	}

	const double t0 = MPI_Wtime();
	std::vector<double> results =
		run_all_to_all(rank, nproc, args, via_host, 0,
		               node_comm, on_my_node, node_ranks, rank_labels);

	auto metric_ptr = [&](int src, int dst) -> const double * {
		return results.data() +
		       (static_cast<size_t>(src) * static_cast<size_t>(nproc) +
		        static_cast<size_t>(dst)) * static_cast<size_t>(ACK_FIELDS);
	};

	for (int src = 0; src < nproc; ++src) {
		for (int dst = 0; dst < nproc; ++dst) {
			const double *m = metric_ptr(src, dst);
			mirror(print_pair_line("pair",
			       rank_labels[static_cast<size_t>(src)],
			       rank_labels[static_cast<size_t>(dst)],
			       m[0], m[1], m[2], m[3], m[4], m[5], args.stat_out));
		}
	}

	std::ostringstream oss;
	oss << "TotalTimeSec: " << std::fixed << std::setprecision(TOTAL_TIME_DIGITS)
	    << (MPI_Wtime() - t0) << "\n";
	mirror(oss.str());
}

std::vector<double> run_all_to_all(int rank, int nproc, const Args &args,
                                    bool check_host, int local_gpu,
                                    MPI_Comm node_comm,
                                    const std::vector<int> &on_my_node,
                                    const std::vector<int> &node_ranks,
                                    const std::vector<std::string> &rank_labels) {
	std::vector<double> ack(ACK_FIELDS, 0.0);
	const int count = static_cast<int>(args.nbytes);

	DBG_LOG(rank, args, "all_to_all RUN_BEGIN bytes=" << args.nbytes
	        << " path=" << (check_host ? "host" : "auto"));

	(void)local_gpu;
	cuda_ok(cudaGetDevice(&local_gpu), "cudaGetDevice(all_to_all)");

	char *d_send = nullptr;
	char *d_recv = nullptr;
	cuda_ok(cudaMalloc(&d_send, args.nbytes), "cudaMalloc(d_send)");
	cuda_ok(cudaMalloc(&d_recv, args.nbytes), "cudaMalloc(d_recv)");
	cuda_ok(cudaMemset(d_send, 0xAA, args.nbytes), "cudaMemset");

	// Per-peer host buffers (inter-node host path) and shared slots (intra-node)
	MPI_Win shared_h_win = MPI_WIN_NULL;
	std::vector<char *> shared_h_slots(static_cast<size_t>(nproc), nullptr);
	std::vector<char *> shared_h_registered;
	std::vector<char *> send_host(static_cast<size_t>(nproc), nullptr);
	std::vector<char *> recv_host(static_cast<size_t>(nproc), nullptr);
	std::vector<char *> recv_dev(static_cast<size_t>(nproc), nullptr);

	if (check_host) {
		void *shared_base = nullptr;
		mpi_ok(MPI_Win_allocate_shared(static_cast<MPI_Aint>(args.nbytes),
		                               sizeof(char), MPI_INFO_NULL, node_comm,
		                               &shared_base, &shared_h_win),
		       "MPI_Win_allocate_shared");
		const int node_size = static_cast<int>(node_ranks.size());
		for (int li = 0; li < node_size; ++li) {
			MPI_Aint sb = 0; int du = 0; void *pb = nullptr;
			mpi_ok(MPI_Win_shared_query(shared_h_win, li, &sb, &du, &pb),
			       "MPI_Win_shared_query");
			if (!pb || sb < static_cast<MPI_Aint>(args.nbytes)) continue;
			const int pr = node_ranks[static_cast<size_t>(li)];
			char *pp = static_cast<char *>(pb);
			shared_h_slots[static_cast<size_t>(pr)] = pp;
			cudaError_t reg = cudaHostRegister(pp, args.nbytes, cudaHostRegisterPortable);
			if (reg == cudaSuccess || reg == cudaErrorHostMemoryAlreadyRegistered) {
				if (reg != cudaSuccess) cudaGetLastError();
				shared_h_registered.push_back(pp);
			} else {
				DBG_LOG(rank, args, "shared host register skipped rank=" << pr
				        << ": " << cudaGetErrorString(reg));
				cudaGetLastError();
			}
		}
	}

	for (int peer = 0; peer < nproc; ++peer) {
		if (peer == rank) continue;
		if (check_host) {
			const bool same = on_my_node[static_cast<size_t>(peer)] != 0 &&
			                  shared_h_slots[static_cast<size_t>(peer)] != nullptr;
			if (!same) {
				cuda_ok(cudaMallocHost(&send_host[static_cast<size_t>(peer)], args.nbytes),
				        "cudaMallocHost(send)");
				cuda_ok(cudaMallocHost(&recv_host[static_cast<size_t>(peer)], args.nbytes),
				        "cudaMallocHost(recv)");
			}
		} else {
			cuda_ok(cudaMalloc(&recv_dev[static_cast<size_t>(peer)], args.nbytes),
			        "cudaMalloc(recv_dev)");
		}
	}

	// Per-source MPI samples (receiver collects, keyed by src_rank)
	std::vector<std::vector<double>> samples(static_cast<size_t>(nproc));
	for (int src = 0; src < nproc; ++src) {
		if (src == rank) continue;
		samples[static_cast<size_t>(src)].reserve(static_cast<size_t>(std::max(1, args.iters)));
	}

	auto do_one = [&](bool measure) {
		std::vector<MPI_Request> recv_req(static_cast<size_t>(2 * nproc), MPI_REQUEST_NULL);
		std::vector<MPI_Request> send_req(static_cast<size_t>(nproc),     MPI_REQUEST_NULL);

		// Post recvs
		for (int src = 0; src < nproc; ++src) {
			if (src == rank) continue;
			if (check_host) {
				const bool same = on_my_node[static_cast<size_t>(src)] != 0 &&
				                  shared_h_slots[static_cast<size_t>(src)] != nullptr;
				if (same) {
					mpi_ok(MPI_Irecv(nullptr, 0, MPI_BYTE, src,
					                 local_ready_tag(src, rank, nproc),
					                 MPI_COMM_WORLD,
					                 &recv_req[static_cast<size_t>(nproc + src)]),
					       "MPI_Irecv local_ready");
				} else {
					mpi_ok(MPI_Irecv(recv_host[static_cast<size_t>(src)], count,
					                 MPI_BYTE, src, pair_tag(src, rank, nproc),
					                 MPI_COMM_WORLD, &recv_req[static_cast<size_t>(src)]),
					       "MPI_Irecv host");
				}
			} else {
				mpi_ok(MPI_Irecv(recv_dev[static_cast<size_t>(src)], count,
				                 MPI_BYTE, src, pair_tag(src, rank, nproc),
				                 MPI_COMM_WORLD, &recv_req[static_cast<size_t>(src)]),
				       "MPI_Irecv device");
			}
		}

		// Receiver-side measurement starts once, immediately before the
		// exchange phase. It must not depend on src: indexing a per-dst t0
		// by src made the reported latency depend on rank traversal order.
		const double t0_mpi = measure ? MPI_Wtime() : 0.0;

		bool shared_copied = false;
		for (int dst = 0; dst < nproc; ++dst) {
			if (dst == rank) continue;
			if (check_host) {
				const bool same = on_my_node[static_cast<size_t>(dst)] != 0 &&
				                  shared_h_slots[static_cast<size_t>(rank)] != nullptr;
				if (same) {
					if (!shared_copied) {
						cuda_ok(cudaMemcpy(shared_h_slots[static_cast<size_t>(rank)],
						                   d_send, args.nbytes, cudaMemcpyDeviceToHost),
						        "D2H(shared)");
						if (shared_h_win != MPI_WIN_NULL)
							mpi_ok(MPI_Win_sync(shared_h_win), "MPI_Win_sync(sender)");
						shared_copied = true;
					}
					mpi_ok(MPI_Isend(nullptr, 0, MPI_BYTE, dst,
					                 local_ready_tag(rank, dst, nproc),
					                 MPI_COMM_WORLD, &send_req[static_cast<size_t>(dst)]),
					       "MPI_Isend local_ready");
				} else {
					cuda_ok(cudaMemcpy(send_host[static_cast<size_t>(dst)], d_send,
					                   args.nbytes, cudaMemcpyDeviceToHost), "D2H");
					mpi_ok(MPI_Isend(send_host[static_cast<size_t>(dst)], count,
					                 MPI_BYTE, dst, pair_tag(rank, dst, nproc),
					                 MPI_COMM_WORLD, &send_req[static_cast<size_t>(dst)]),
					       "MPI_Isend host");
				}
			} else {
				mpi_ok(MPI_Isend(d_send, count, MPI_BYTE, dst,
				                 pair_tag(rank, dst, nproc),
				                 MPI_COMM_WORLD, &send_req[static_cast<size_t>(dst)]),
				       "MPI_Isend device");
			}
		}

		// Receive completions
		for (int done = 0; done < nproc - 1; ++done) {
			int idx = MPI_UNDEFINED;
			mpi_ok(MPI_Waitany(static_cast<int>(recv_req.size()), recv_req.data(),
			                   &idx, MPI_STATUS_IGNORE), "MPI_Waitany");
			if (idx == MPI_UNDEFINED) break;

			int src = idx;
			if (check_host) {
				const bool same = idx >= nproc;
				src = same ? (idx - nproc) : idx;
				if (same) {
					if (shared_h_win != MPI_WIN_NULL)
						mpi_ok(MPI_Win_sync(shared_h_win), "MPI_Win_sync(receiver)");
					cuda_ok(cudaMemcpy(d_recv,
					                   shared_h_slots[static_cast<size_t>(src)],
					                   args.nbytes, cudaMemcpyHostToDevice), "H2D(shared)");
				} else {
					cuda_ok(cudaMemcpy(d_recv, recv_host[static_cast<size_t>(src)],
					                   args.nbytes, cudaMemcpyHostToDevice), "H2D");
				}
		} else {
			// Synchronize before D2D copy: UCX cuda_ipc may have written to
			// recv_dev[src] on a non-default stream; ensure it is complete.
			cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before D2D)");
			cuda_ok(cudaMemcpy(d_recv, recv_dev[static_cast<size_t>(src)],
			                   args.nbytes, cudaMemcpyDeviceToDevice), "D2D");
		}

			if (measure) {
				const double t1 = MPI_Wtime();
				samples[static_cast<size_t>(src)].push_back(
				    (t1 - t0_mpi) * 1e6);
			}
		}

		mpi_ok(MPI_Waitall(nproc, send_req.data(), MPI_STATUSES_IGNORE),
		       "MPI_Waitall sends");
	};

	for (int i = 0; i < args.warmup; ++i) do_one(false);
	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(iters)");
	for (int i = 0; i < args.iters;  ++i) do_one(true);

	// Save raw samples and compute acks
	for (int src = 0; src < nproc; ++src) {
		if (src == rank) continue;
		Task t{src, local_gpu, rank, local_gpu, 0};
		append_raw_samples(args, rank, t, rank_labels,
		                   samples[static_cast<size_t>(src)]);
	}

	// Gather results at rank 0
	std::vector<double> results;
	if (rank == 0)
		results.assign(static_cast<size_t>(nproc) * static_cast<size_t>(nproc) *
		               static_cast<size_t>(ACK_FIELDS), 0.0);

	auto result_ptr = [&](int src, int dst) -> double * {
		return results.data() +
		       (static_cast<size_t>(src) * static_cast<size_t>(nproc) +
		        static_cast<size_t>(dst)) * static_cast<size_t>(ACK_FIELDS);
	};

	for (int src = 0; src < nproc; ++src) {
		for (int dst = 0; dst < nproc; ++dst) {
			if (src == dst) {
				if (rank == 0) result_ptr(src, dst)[6] = 1.0;
				continue;
			}
			if (rank == dst) {
				fill_ack(samples[static_cast<size_t>(src)], ack.data());
				if (rank == 0) {
					std::copy(ack.begin(), ack.end(), result_ptr(src, dst));
				} else {
					mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0,
					                nproc * nproc + pair_tag(src, dst, nproc),
					                MPI_COMM_WORLD), "MPI_Send ack");
				}
			} else if (rank == 0) {
				mpi_ok(MPI_Recv(result_ptr(src, dst), ACK_FIELDS, MPI_DOUBLE, dst,
				                nproc * nproc + pair_tag(src, dst, nproc),
				                MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Recv ack");
			}
		}
	}

	// Cleanup
	for (int peer = 0; peer < nproc; ++peer) {
		if (peer == rank) continue;
		if (send_host[static_cast<size_t>(peer)]) cudaFreeHost(send_host[static_cast<size_t>(peer)]);
		if (recv_host[static_cast<size_t>(peer)]) cudaFreeHost(recv_host[static_cast<size_t>(peer)]);
		if (recv_dev[static_cast<size_t>(peer)])  cudaFree(recv_dev[static_cast<size_t>(peer)]);
	}
	for (char *p : shared_h_registered) cuda_ok(cudaHostUnregister(p), "cudaHostUnregister");
	if (shared_h_win != MPI_WIN_NULL) mpi_ok(MPI_Win_free(&shared_h_win), "MPI_Win_free");
	cudaFree(d_send);
	cudaFree(d_recv);

	DBG_LOG(rank, args, "all_to_all RUN_DONE");
	return results;
}

} // namespace gpu_benchmark
