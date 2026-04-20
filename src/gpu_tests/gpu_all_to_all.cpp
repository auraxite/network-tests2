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


std::vector<double> run_all_to_all(int rank, int nproc, const Args &args,
                                   bool check_host, int local_gpu,
                                   const std::vector<std::string> &rank_labels) {
    std::vector<double> ack(ACK_FIELDS, 0.0);
    const int count = static_cast<int>(args.nbytes);

    char *d_send = nullptr;
    char *d_recv = nullptr;
    std::vector<char *> send_host(static_cast<size_t>(nproc), nullptr);
    std::vector<char *> recv_host(static_cast<size_t>(nproc), nullptr);
    std::vector<char *> recv_dev(static_cast<size_t>(nproc), nullptr);

    DBG_LOG(rank, args,
            "all_to_all RUN_BEGIN local_gpu=" << local_gpu << " bytes=" << args.nbytes
            << " nproc=" << nproc << " env_path=" << (check_host ? "host" : "auto"));

    cuda_ok(cudaSetDevice(local_gpu), "cudaSetDevice(all_to_all)");
    cuda_ok(cudaMalloc(&d_send, args.nbytes), "cudaMalloc(d_send)");
    cuda_ok(cudaMalloc(&d_recv, args.nbytes), "cudaMalloc(d_recv)");
    cuda_ok(cudaMemset(d_send, 0xAA, args.nbytes), "cudaMemset(d_send init)");

    for (int peer_rank = 0; peer_rank < nproc; ++peer_rank) {
        if (peer_rank == rank)
            continue;
        if (check_host) {
            cuda_ok(cudaMallocHost(&send_host[static_cast<size_t>(peer_rank)], args.nbytes),
                    "cudaMallocHost(send_host)");
            cuda_ok(cudaMallocHost(&recv_host[static_cast<size_t>(peer_rank)], args.nbytes),
                    "cudaMallocHost(recv_host)");
        } else {
            cuda_ok(cudaMalloc(&recv_dev[static_cast<size_t>(peer_rank)], args.nbytes),
                    "cudaMalloc(recv_dev)");
        }
    }

    DBG_LOG(rank, args, "all_to_all ALLOC_DONE");

    std::vector<std::vector<double>> samples_mpi_us(static_cast<size_t>(nproc));
    std::vector<std::vector<double>> samples_cpu_us(static_cast<size_t>(nproc));
    std::vector<std::vector<double>> samples_gpu_us(static_cast<size_t>(nproc));
    for (int src_rank = 0; src_rank < nproc; ++src_rank) {
        if (src_rank == rank)
            continue;
        const size_t cap = static_cast<size_t>(std::max(1, args.iters));
        samples_mpi_us[static_cast<size_t>(src_rank)].reserve(cap);
        samples_cpu_us[static_cast<size_t>(src_rank)].reserve(cap);
        samples_gpu_us[static_cast<size_t>(src_rank)].reserve(cap);
    }

    auto do_one = [&](int iter_idx, bool measure) {
        DBG_LOG(rank, args,
                "all_to_all ITER_BEGIN idx=" << iter_idx
                << " measure=" << (measure ? 1 : 0));

        std::vector<MPI_Request> recv_req(static_cast<size_t>(nproc), MPI_REQUEST_NULL);
        std::vector<MPI_Request> send_req(static_cast<size_t>(nproc), MPI_REQUEST_NULL);
        std::vector<double> t0_mpi_s(static_cast<size_t>(nproc), 0.0);
        std::vector<double> t0_cpu_us(static_cast<size_t>(nproc), 0.0);

        cudaEvent_t ev_start = nullptr;
        std::vector<cudaEvent_t> ev_stop(static_cast<size_t>(nproc), nullptr);
        if (measure) {
            cuda_ok(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
            cuda_ok(cudaEventRecord(ev_start), "cudaEventRecord(start)");
        }

        DBG_LOG(rank, args, "all_to_all RECV_POST_BEGIN");
        for (int src_rank = 0; src_rank < nproc; ++src_rank) {
            if (src_rank == rank)
                continue;
            void *recv_ptr = check_host
                                 ? static_cast<void *>(recv_host[static_cast<size_t>(src_rank)])
                                 : static_cast<void *>(recv_dev[static_cast<size_t>(src_rank)]);
            mpi_ok(MPI_Irecv(recv_ptr, count, MPI_BYTE, src_rank,
                             alltoall_pair_tag(src_rank, rank, nproc),
                             MPI_COMM_WORLD,
                             &recv_req[static_cast<size_t>(src_rank)]),
                   "MPI_Irecv(all_to_all)");
        }
        DBG_LOG(rank, args, "all_to_all RECV_POST_DONE");

        DBG_LOG(rank, args, "all_to_all SEND_PHASE_BEGIN");
        for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
            if (dst_rank == rank)
                continue;
            if (measure) {
                t0_mpi_s[static_cast<size_t>(dst_rank)] = MPI_Wtime();
                t0_cpu_us[static_cast<size_t>(dst_rank)] = clock_gettime_wrapper();
            }

            if (check_host) {
                char *send_ptr = send_host[static_cast<size_t>(dst_rank)];
                cuda_ok(cudaMemcpy(send_ptr, d_send, args.nbytes, cudaMemcpyDeviceToHost),
                        "D2H(all_to_all)");
                mpi_ok(MPI_Isend(send_ptr, count, MPI_BYTE, dst_rank,
                                 alltoall_pair_tag(rank, dst_rank, nproc),
                                 MPI_COMM_WORLD,
                                 &send_req[static_cast<size_t>(dst_rank)]),
                       "MPI_Isend(all_to_all host)");
            } else {
                mpi_ok(MPI_Isend(d_send, count, MPI_BYTE, dst_rank,
                                 alltoall_pair_tag(rank, dst_rank, nproc),
                                 MPI_COMM_WORLD,
                                 &send_req[static_cast<size_t>(dst_rank)]),
                       "MPI_Isend(all_to_all device)");
            }
        }
        DBG_LOG(rank, args, "all_to_all SEND_PHASE_DONE");

        DBG_LOG(rank, args, "all_to_all RECV_PHASE_BEGIN");
        const int n_peers = nproc - 1;
        for (int done = 0; done < n_peers; ++done) {
            int idx = MPI_UNDEFINED;
            mpi_ok(MPI_Waitany(nproc, recv_req.data(), &idx, MPI_STATUS_IGNORE),
                   "MPI_Waitany(all_to_all)");
            if (idx == MPI_UNDEFINED)
                break;

            const int src_rank = idx;
            if (check_host) {
                cuda_ok(cudaMemcpy(d_recv, recv_host[static_cast<size_t>(src_rank)],
                                   args.nbytes, cudaMemcpyHostToDevice),
                        "H2D(all_to_all)");
            } else {
                cuda_ok(cudaMemcpy(d_recv, recv_dev[static_cast<size_t>(src_rank)],
                                   args.nbytes, cudaMemcpyDeviceToDevice),
                        "D2D(all_to_all)");
            }

            if (measure) {
                const double t1_mpi_s = MPI_Wtime();
                const double t1_cpu_us = clock_gettime_wrapper();
                const double mpi_sample_us =
                    (t1_mpi_s - t0_mpi_s[static_cast<size_t>(src_rank)]) * 1e6;
                const double cpu_sample_us =
                    t1_cpu_us - t0_cpu_us[static_cast<size_t>(src_rank)];

                samples_mpi_us[static_cast<size_t>(src_rank)].push_back(mpi_sample_us);
                samples_cpu_us[static_cast<size_t>(src_rank)].push_back(cpu_sample_us);

                cudaEvent_t &ev_stop_src = ev_stop[static_cast<size_t>(src_rank)];
                cuda_ok(cudaEventCreate(&ev_stop_src), "cudaEventCreate(stop[src])");
                cuda_ok(cudaEventRecord(ev_stop_src), "cudaEventRecord(stop[src])");
                cuda_ok(cudaEventSynchronize(ev_stop_src),
                        "cudaEventSynchronize(stop[src])");
                float elapsed_ms = 0.0f;
                cuda_ok(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop_src),
                        "cudaEventElapsedTime");
                samples_gpu_us[static_cast<size_t>(src_rank)].push_back(
                    static_cast<double>(elapsed_ms) * 1e3);
            }
        }
        DBG_LOG(rank, args, "all_to_all RECV_PHASE_DONE");

        mpi_ok(MPI_Waitall(nproc, send_req.data(), MPI_STATUSES_IGNORE),
               "MPI_Waitall(all_to_all sends)");

        if (measure) {
            if (ev_start)
                cuda_ok(cudaEventDestroy(ev_start), "cudaEventDestroy(start)");
            for (int src_rank = 0; src_rank < nproc; ++src_rank) {
                cudaEvent_t &stop = ev_stop[static_cast<size_t>(src_rank)];
                if (stop)
                    cuda_ok(cudaEventDestroy(stop), "cudaEventDestroy(stop[src])");
            }
        }

        DBG_LOG(rank, args,
                "all_to_all ITER_DONE idx=" << iter_idx
                << " measure=" << (measure ? 1 : 0));
    };

    DBG_LOG(rank, args, "all_to_all WARMUP_BEGIN");
    for (int i = 0; i < args.warmup; ++i)
        do_one(i, false);
    DBG_LOG(rank, args, "all_to_all WARMUP_DONE");

    mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(all_to_all iters)");

    DBG_LOG(rank, args, "all_to_all ITERS_BEGIN");
    for (int i = 0; i < args.iters; ++i)
        do_one(i, true);
    DBG_LOG(rank, args, "all_to_all ITERS_DONE");

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
                                    nproc * nproc +
                                        alltoall_pair_tag(src_rank, dst_rank, nproc),
                                    MPI_COMM_WORLD),
                           "MPI_Send ack(all_to_all)");
                    DBG_LOG(rank, args,
                            "all_to_all ACK_SENT src=" << src_rank << " dst=" << dst_rank);
                }
            } else if (rank == 0) {
                mpi_ok(MPI_Recv(result_ptr(src_rank, dst_rank), ACK_FIELDS, MPI_DOUBLE,
                                dst_rank,
                                nproc * nproc +
                                    alltoall_pair_tag(src_rank, dst_rank, nproc),
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
        if (send_host[static_cast<size_t>(peer_rank)])
            cudaFreeHost(send_host[static_cast<size_t>(peer_rank)]);
        if (recv_host[static_cast<size_t>(peer_rank)])
            cudaFreeHost(recv_host[static_cast<size_t>(peer_rank)]);
        if (recv_dev[static_cast<size_t>(peer_rank)])
            cudaFree(recv_dev[static_cast<size_t>(peer_rank)]);
    }
    if (d_send)
        cudaFree(d_send);
    if (d_recv)
        cudaFree(d_recv);

    DBG_LOG(rank, args, "all_to_all RUN_DONE");
    return results;
}

} // namespace gpu_benchmark
