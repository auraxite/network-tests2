#include "gpu_one_to_one.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <functional>
#include <iomanip>
#include <sstream>
#include <vector>

namespace gpu_benchmark {

static constexpr int RDV_TAG_FORWARD  = 42;
static constexpr int RDV_TAG_BACKWARD = 43;

std::vector<double> run_one_to_one(int rank, const Task &t, const Args &args,
                                    bool check_host, bool same_node_pair,
                                    char *d_send, char *d_recv, char *h_buf,
                                    char *shared_h_buf, MPI_Win shared_h_win,
                                    const std::vector<std::string> &rank_labels);

void schedule_one_to_one(
    int rank, int nproc, const Args &args, bool via_host,
    MPI_Comm node_comm, int node_rank,
    const std::vector<int> &on_my_node,
    const std::vector<std::string> &rank_labels,
    const std::function<void(const std::string &)> &mirror) {

	const int local_gpu = 0;
	cuda_ok(cudaSetDevice(local_gpu), "cudaSetDevice(schedule_one_to_one)");

	char   *d_send           = nullptr;
	char   *d_recv           = nullptr;
	char   *h_buf            = nullptr;
	char   *shared_h_buf     = nullptr;
	MPI_Win shared_h_win     = MPI_WIN_NULL;
	bool    shared_h_registered = false;
	const bool use_shared = via_host;

	cuda_ok(cudaMalloc(&d_send, args.nbytes), "cudaMalloc(d_send)");
	cuda_ok(cudaMalloc(&d_recv, args.nbytes), "cudaMalloc(d_recv)");
	cuda_ok(cudaMemset(d_send, 0xAA, args.nbytes), "cudaMemset(d_send init)");

	if (via_host)
		cuda_ok(cudaMallocHost(&h_buf, args.nbytes), "cudaMallocHost(h_buf)");

	if (use_shared) {
		MPI_Aint shared_bytes  = 0;
		int      disp_unit     = 0;
		void    *shared_base   = nullptr;
		const MPI_Aint my_bytes = (node_rank == 0)
		                              ? static_cast<MPI_Aint>(args.nbytes)
		                              : static_cast<MPI_Aint>(0);
		mpi_ok(MPI_Win_allocate_shared(my_bytes, sizeof(char), MPI_INFO_NULL,
		                               node_comm, &shared_base, &shared_h_win),
		       "MPI_Win_allocate_shared");
		mpi_ok(MPI_Win_shared_query(shared_h_win, 0, &shared_bytes, &disp_unit,
		                            &shared_base),
		       "MPI_Win_shared_query");
		shared_h_buf = static_cast<char *>(shared_base);
		if (shared_h_buf && shared_bytes >= static_cast<MPI_Aint>(args.nbytes)) {
			cudaError_t reg = cudaHostRegister(shared_h_buf, args.nbytes,
			                                   cudaHostRegisterPortable);
			if (reg == cudaSuccess || reg == cudaErrorHostMemoryAlreadyRegistered) {
				if (reg != cudaSuccess) cudaGetLastError();
				shared_h_registered = true;
			} else {
				DBG_LOG(rank, args, "shared host register skipped: "
				        << cudaGetErrorString(reg));
				cudaGetLastError();
			}
		}
	}

	if (rank != 0) {
		while (true) {
			Task tw{};
			mpi_ok(MPI_Recv(&tw, sizeof(Task), MPI_BYTE, 0, 1, MPI_COMM_WORLD,
			                MPI_STATUS_IGNORE),
			       "MPI_Recv Task (worker)");
			DBG_LOG(rank, args, "WORKER_TASK src=" << tw.src_rank
			        << " dst=" << tw.dst_rank << " stop=" << tw.stop);
			if (tw.stop) break;
			const bool same = on_my_node[static_cast<size_t>(tw.src_rank)] != 0 &&
			                  on_my_node[static_cast<size_t>(tw.dst_rank)] != 0;
			auto ack = run_one_to_one(rank, tw, args, via_host, same,
			                          d_send, d_recv, h_buf,
			                          shared_h_buf, shared_h_win, rank_labels);
			mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD),
			       "MPI_Send ack (worker)");
		}
		if (shared_h_registered)
			cuda_ok(cudaHostUnregister(shared_h_buf), "cudaHostUnregister");
		if (shared_h_win != MPI_WIN_NULL)
			mpi_ok(MPI_Win_free(&shared_h_win), "MPI_Win_free");
		if (h_buf) cudaFreeHost(h_buf);
		cudaFree(d_send);
		cudaFree(d_recv);
		return;
	}

	const double test_t0 = MPI_Wtime();
	Task t{};
	for (int src = 0; src < nproc; ++src) {
		for (int dst = 0; dst < nproc; ++dst) {
			t = {src, 0, dst, 0, 0};
			DBG_LOG(rank, args, "PAIR src=" << src << " dst=" << dst);

			if (src == dst) {
				mirror(print_pair_line("pair",
				       rank_labels[static_cast<size_t>(src)],
				       rank_labels[static_cast<size_t>(dst)],
				       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, args.stat_out));
				continue;
			}

			const bool same = on_my_node[static_cast<size_t>(src)] != 0 &&
			                  on_my_node[static_cast<size_t>(dst)] != 0;

			if (src != 0) mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, src, 1,
			                              MPI_COMM_WORLD), "MPI_Send Task src");
			if (dst != 0) mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, dst, 1,
			                              MPI_COMM_WORLD), "MPI_Send Task dst");

			std::vector<double> metric(ACK_FIELDS, 0.0);
			if (src == 0 || dst == 0) {
				auto ack = run_one_to_one(rank, t, args, via_host, same,
				                          d_send, d_recv, h_buf,
				                          shared_h_buf, shared_h_win, rank_labels);
				if (ack[6] == 1.0) metric = ack;
			}
			if (src != 0) {
				std::vector<double> ack(ACK_FIELDS, 0.0);
				mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, src, 2,
				                MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Recv ack src");
				if (ack[6] == 1.0) metric = ack;
			}
			if (dst != 0) {
				std::vector<double> ack(ACK_FIELDS, 0.0);
				mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, dst, 2,
				                MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Recv ack dst");
				if (ack[6] == 1.0) metric = ack;
			}

			mirror(print_pair_line("pair",
			       rank_labels[static_cast<size_t>(src)],
			       rank_labels[static_cast<size_t>(dst)],
			       metric[0], metric[1], metric[2], metric[3],
			       metric[4], metric[5], args.stat_out));
		}
	}

	{
		std::ostringstream oss;
		oss << "TotalTimeSec: " << std::fixed << std::setprecision(TOTAL_TIME_DIGITS)
		    << (MPI_Wtime() - test_t0) << "\n";
		mirror(oss.str());
	}
	t.stop = 1;
	for (int r = 1; r < nproc; ++r)
		mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, r, 1, MPI_COMM_WORLD),
		       "MPI_Send stop");

	if (shared_h_registered)
		cuda_ok(cudaHostUnregister(shared_h_buf), "cudaHostUnregister");
	if (shared_h_win != MPI_WIN_NULL)
		mpi_ok(MPI_Win_free(&shared_h_win), "MPI_Win_free");
	if (h_buf) cudaFreeHost(h_buf);
	cudaFree(d_send);
	cudaFree(d_recv);
}

std::vector<double> run_one_to_one(int rank, const Task &t, const Args &args,
                                    bool check_host, bool same_node_pair,
                                    char *d_send, char *d_recv, char *h_buf,
                                    char *shared_h_buf, MPI_Win shared_h_win,
                                    const std::vector<std::string> &rank_labels) {
	std::vector<double> ack(ACK_FIELDS, 0.0);
	const bool is_sender   = (rank == t.src_rank);
	const bool is_receiver = (rank == t.dst_rank);
	if (!is_sender && !is_receiver) return ack;
	if (t.src_rank == t.dst_rank) { ack[6] = 1.0; return ack; }

	const int count = static_cast<int>(args.nbytes);

	DBG_LOG(rank, args, "RUN src=" << t.src_rank << " dst=" << t.dst_rank
	        << " bytes=" << args.nbytes
	        << " path=" << (check_host ? "host" : "auto"));

	auto do_one = [&]() {
		// Intra-node: shared host memory path (avoids CUDA IPC unavailability)
		if (same_node_pair && shared_h_buf) {
			if (is_sender) {
				cuda_ok(cudaMemcpy(shared_h_buf, d_send, count,
				                   cudaMemcpyDeviceToHost), "D2H(shared)");
				if (shared_h_win != MPI_WIN_NULL)
					mpi_ok(MPI_Win_sync(shared_h_win), "MPI_Win_sync(sender)");
				char dummy = 0;
				mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, t.dst_rank, RDV_TAG_FORWARD,
				                    &dummy, 0, MPI_BYTE, t.dst_rank, RDV_TAG_BACKWARD,
				                    MPI_COMM_WORLD, MPI_STATUS_IGNORE),
				       "Sendrecv ready(sender)");
			}
			if (is_receiver) {
				char dummy = 0;
				mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, t.src_rank, RDV_TAG_BACKWARD,
				                    &dummy, 0, MPI_BYTE, t.src_rank, RDV_TAG_FORWARD,
				                    MPI_COMM_WORLD, MPI_STATUS_IGNORE),
				       "Sendrecv ready(receiver)");
				if (shared_h_win != MPI_WIN_NULL)
					mpi_ok(MPI_Win_sync(shared_h_win), "MPI_Win_sync(receiver)");
				cuda_ok(cudaMemcpy(d_recv, shared_h_buf, count,
				                   cudaMemcpyHostToDevice), "H2D(shared)");
			}
			return;
		}

		// Inter-node or intra-node without shared buffer
		if (is_sender) {
			if (check_host) {
				cuda_ok(cudaMemcpy(h_buf, d_send, count, cudaMemcpyDeviceToHost), "D2H");
				mpi_ok(MPI_Send(h_buf, count, MPI_BYTE, t.dst_rank, 0, MPI_COMM_WORLD),
				       "MPI_Send(host)");
			} else {
				mpi_ok(MPI_Send(d_send, count, MPI_BYTE, t.dst_rank, 0, MPI_COMM_WORLD),
				       "MPI_Send(device)");
			}
		}
		if (is_receiver) {
			MPI_Status st{};
			if (check_host) {
				mpi_ok(MPI_Recv(h_buf, count, MPI_BYTE, t.src_rank, 0, MPI_COMM_WORLD, &st),
				       "MPI_Recv(host)");
				cuda_ok(cudaMemcpy(d_recv, h_buf, count, cudaMemcpyHostToDevice), "H2D");
			} else {
				mpi_ok(MPI_Recv(d_recv, count, MPI_BYTE, t.src_rank, 0, MPI_COMM_WORLD, &st),
				       "MPI_Recv(device)");
				// Probe read of d_recv on the default stream: triggers any
				// CUDA-aware MPI cross-stream dependencies set up by UCX so
				// the timer reflects buffer visibility, not just MPI-level
				// completion. Reads only 1 byte, but is synchronous w.r.t.
				// the receive buffer.
				volatile char probe = 0;
				cuda_ok(cudaMemcpy((void *)&probe, d_recv, 1,
				                   cudaMemcpyDeviceToHost),
				        "probe(d_recv)");
			}
		}
	};

	std::vector<double> samples_us;
	samples_us.reserve(static_cast<size_t>(std::max(1, args.iters)));

	for (int i = 0; i < args.warmup; ++i) do_one();

	for (int i = 0; i < args.iters; ++i) {
		const double t0 = MPI_Wtime();
		do_one();
		const double t1 = MPI_Wtime();
		if (is_receiver)
			samples_us.push_back((t1 - t0) * 1e6);

		// Per-iteration rendezvous: keeps sender/receiver in lockstep
		char dummy = 0;
		if (is_sender)
			mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, t.dst_rank, RDV_TAG_FORWARD,
			                    &dummy, 0, MPI_BYTE, t.dst_rank, RDV_TAG_BACKWARD,
			                    MPI_COMM_WORLD, MPI_STATUS_IGNORE), "iter rendezvous(sender)");
		else
			mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, t.src_rank, RDV_TAG_BACKWARD,
			                    &dummy, 0, MPI_BYTE, t.src_rank, RDV_TAG_FORWARD,
			                    MPI_COMM_WORLD, MPI_STATUS_IGNORE), "iter rendezvous(receiver)");
	}

	if (is_receiver) {
		append_raw_samples(args, rank, t, rank_labels, samples_us);
		fill_ack(samples_us, ack.data());
	}
	return ack;
}

} // namespace gpu_benchmark
