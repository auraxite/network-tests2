#include "gpu_one_to_one.hpp"

#include <algorithm>    // std::max и др.
#include <cuda_runtime.h> // CUDA: устройство, события, cudaMalloc, ...
#include <functional>   // std::function — параметр mirror в schedule_one_to_one
#include <iomanip>      // std::setprecision, std::fixed
#include <sstream>      // std::ostringstream — строки TotalTimeSec и т.п.
#include <vector>       // буферы ack, samples

namespace gpu_benchmark {

std::vector<double> run_one_to_one(int rank, const Task &t, const Args &args,
									 bool check_host,
									 const std::vector<std::string> &rank_labels) {
	std::vector<double> ack(ACK_FIELDS, 0.0);
	const bool is_sender = (rank == t.src_rank);
	const bool is_receiver = (rank == t.dst_rank);
	const bool collect_raw_samples_here = is_sender;
	if (!is_sender && !is_receiver)
		return ack;
	if (t.src_rank == t.dst_rank) {
		ack[6] = 1.0;
		return ack;
	}

	char *d_send = nullptr;
	char *d_recv = nullptr;
	char *h_buf = nullptr;
	const int count = static_cast<int>(args.nbytes);
	{
		std::ostringstream oss;
		oss << "one_to_one RUN_BEGIN src=" << t.src_rank << "." << t.src_gpu
			<< " dst=" << t.dst_rank << "." << t.dst_gpu << " bytes=" << args.nbytes
			<< " role_sender=" << (is_sender ? 1 : 0)
			<< " role_receiver=" << (is_receiver ? 1 : 0)
			<< " env_path=" << (check_host ? "host" : "auto");
		debug_log(args.debug, rank, oss.str());
	}

	if (is_sender) {
		cuda_ok(cudaSetDevice(t.src_gpu), "cudaSetDevice(src)");
		cuda_ok(cudaMalloc(&d_send, args.nbytes), "cudaMalloc(src)");
		cuda_ok(cudaMemset(d_send, 0xAA, args.nbytes), "cudaMemset(src)");
	}
	if (is_receiver) {
		cuda_ok(cudaSetDevice(t.dst_gpu), "cudaSetDevice(dst)");
		cuda_ok(cudaMalloc(&d_recv, args.nbytes), "cudaMalloc(dst)");
	}
	if (check_host)
		cuda_ok(cudaMallocHost(&h_buf, args.nbytes), "cudaMallocHost");
	debug_log(args.debug, rank, "one_to_one ALLOC_DONE");

	int current_iter = -1;
	const char *current_phase = "none";
	auto do_one = [&]() {
		if (is_sender) {
			if (check_host) {
				{
					std::ostringstream oss;
					oss << "one_to_one D2H_BEGIN phase=" << current_phase
						<< " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
				cuda_ok(cudaMemcpy(h_buf, d_send, count, cudaMemcpyDeviceToHost), "D2H");
				{
					std::ostringstream oss;
					oss << "one_to_one D2H_DONE phase=" << current_phase
						<< " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
				{
					std::ostringstream oss;
					oss << "one_to_one MPI_SEND_BEGIN dst=" << t.dst_rank
						<< " tag=" << 0 << " bytes=" << count
						<< " path=host"
						<< " phase=" << current_phase << " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
				mpi_ok(MPI_Send(h_buf, count, MPI_BYTE, t.dst_rank, 0, MPI_COMM_WORLD),
					   "MPI_Send (host staging)");
				{
					std::ostringstream oss;
					oss << "one_to_one MPI_SEND_DONE path=host"
						<< " phase=" << current_phase << " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
			} else {
				{
					std::ostringstream oss;
					oss << "one_to_one MPI_SEND_BEGIN dst=" << t.dst_rank
						<< " tag=" << 0 << " bytes=" << count
						<< " path=device"
						<< " phase=" << current_phase << " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
				mpi_ok(MPI_Send(d_send, count, MPI_BYTE, t.dst_rank, 0, MPI_COMM_WORLD),
					   "MPI_Send (device buffer)");
				{
					std::ostringstream oss;
					oss << "one_to_one MPI_SEND_DONE path=device"
						<< " phase=" << current_phase << " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
			}
		}
		if (is_receiver) {
			MPI_Status st{};
			if (check_host) {
				{
					std::ostringstream oss;
					oss << "one_to_one MPI_RECV_BEGIN src=" << t.src_rank
						<< " tag=" << 0 << " bytes=" << count
						<< " path=host"
						<< " phase=" << current_phase << " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
				mpi_ok(MPI_Recv(h_buf, count, MPI_BYTE, t.src_rank, 0, MPI_COMM_WORLD,
								&st),
					   "MPI_Recv (host staging)");
				{
					std::ostringstream oss;
					oss << "one_to_one MPI_RECV_DONE path=host"
						<< " phase=" << current_phase << " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
				{
					std::ostringstream oss;
					oss << "one_to_one H2D_BEGIN phase=" << current_phase
						<< " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
				cuda_ok(cudaMemcpy(d_recv, h_buf, count, cudaMemcpyHostToDevice), "H2D");
				{
					std::ostringstream oss;
					oss << "one_to_one H2D_DONE phase=" << current_phase
						<< " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
			} else {
				{
					std::ostringstream oss;
					oss << "one_to_one MPI_RECV_BEGIN src=" << t.src_rank
						<< " tag=" << 0 << " bytes=" << count
						<< " path=device"
						<< " phase=" << current_phase << " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
				mpi_ok(MPI_Recv(d_recv, count, MPI_BYTE, t.src_rank, 0, MPI_COMM_WORLD,
								&st),
					   "MPI_Recv (device buffer)");
				{
					std::ostringstream oss;
					oss << "one_to_one MPI_RECV_DONE path=device"
						<< " phase=" << current_phase << " idx=" << current_iter;
					debug_log(args.debug, rank, oss.str());
				}
			}
		}
	};

	std::vector<double> samples_mpi_us;
	std::vector<double> samples_cpu_us;
	std::vector<double> samples_gpu_us;
	cudaEvent_t ev_start = nullptr;
	cudaEvent_t ev_stop = nullptr;
	if (collect_raw_samples_here) {
		cuda_ok(cudaSetDevice(t.src_gpu), "cudaSetDevice(timing)");
		cuda_ok(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
		cuda_ok(cudaEventCreate(&ev_stop), "cudaEventCreate(stop)");
	}
	samples_mpi_us.reserve(static_cast<size_t>(std::max(1, args.iters)));
	samples_cpu_us.reserve(static_cast<size_t>(std::max(1, args.iters)));
	samples_gpu_us.reserve(static_cast<size_t>(std::max(1, args.iters)));

	debug_log(args.debug, rank, "one_to_one WARMUP_BEGIN");
	current_phase = "warmup";
	for (int i = 0; i < args.warmup; ++i) {
		current_iter = i;
		do_one();
	}
	debug_log(args.debug, rank, "one_to_one WARMUP_DONE");
	
	debug_log(args.debug, rank, "one_to_one ITERS_BEGIN");
	current_phase = "iters";
	for (int i = 0; i < args.iters; ++i) {
		current_iter = i;
		{
			std::ostringstream oss;
			oss << "one_to_one ITER_BEGIN idx=" << i;
			debug_log(args.debug, rank, oss.str());
		}
		const double t0_mpi = MPI_Wtime();
		const double t0_clk = clock_gettime_wrapper();
		if (collect_raw_samples_here)
			cuda_ok(cudaEventRecord(ev_start), "cudaEventRecord(start)");
		do_one();
		if (collect_raw_samples_here) {
			cuda_ok(cudaEventRecord(ev_stop), "cudaEventRecord(stop)");
			cuda_ok(cudaEventSynchronize(ev_stop), "cudaEventSynchronize(stop)");
			float elapsed_ms = 0.0f;
			cuda_ok(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop),
					"cudaEventElapsedTime");
			samples_gpu_us.push_back(static_cast<double>(elapsed_ms) * 1e3);
			const double t1_mpi = MPI_Wtime();
			const double t1_clk = clock_gettime_wrapper();
			samples_mpi_us.push_back((t1_mpi - t0_mpi) * 1e6);
			samples_cpu_us.push_back(t1_clk - t0_clk);
		}
		{
			std::ostringstream oss;
			oss << "one_to_one ITER_DONE idx=" << i;
			debug_log(args.debug, rank, oss.str());
		}
	}
	debug_log(args.debug, rank, "one_to_one ITERS_DONE");
	if (collect_raw_samples_here) {
		switch (args.timer) {
		case Timer::All:
		case Timer::Mpi:
			append_raw_samples(args, rank, t, rank_labels, samples_mpi_us);
			break;
		case Timer::Cpu:
			append_raw_samples(args, rank, t, rank_labels, samples_cpu_us);
			break;
		case Timer::Cuda:
			append_raw_samples(args, rank, t, rank_labels, samples_gpu_us);
			break;
		}
	}

	if (!samples_gpu_us.empty() && !samples_mpi_us.empty() && !samples_cpu_us.empty())
		fill_ack(samples_mpi_us, samples_cpu_us, samples_gpu_us, args,
									 ack.data());
	debug_log(args.debug, rank, "one_to_one STATS_DONE");
	if (ev_start)
		cudaEventDestroy(ev_start);
	if (ev_stop)
		cudaEventDestroy(ev_stop);

	if (h_buf)
		cudaFreeHost(h_buf);
	if (d_send)
		cudaFree(d_send);
	if (d_recv)
		cudaFree(d_recv);
	debug_log(args.debug, rank, "one_to_one RUN_DONE");
	return ack;
}

void schedule_one_to_one(
	int rank, int nproc, const Args &args, bool via_host,
	const std::vector<std::string> &rank_labels,
	const std::function<void(const std::string &)> &mirror, NetcdfBundle *nc,
	int matrix_idx) {
		if (rank != 0) {
			while (true) {
				Task tw{};
				mpi_ok(MPI_Recv(&tw, sizeof(Task), MPI_BYTE, 0, 1, MPI_COMM_WORLD,
								MPI_STATUS_IGNORE),
					"MPI_Recv Task (worker)");
				{
					std::ostringstream oss;
					oss << "one_to_one WORKER_TASK_RECV src=" << tw.src_rank << "." << tw.src_gpu
						<< " dst=" << tw.dst_rank << "." << tw.dst_gpu << " stop=" << tw.stop;
					debug_log(args.debug, rank, oss.str());
				}
				if (tw.stop)
					break;
				auto ack = run_one_to_one(rank, tw, args, via_host, rank_labels);
				mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD),
					"MPI_Send ack (worker)");
				debug_log(args.debug, rank, "one_to_one WORKER_ACK_SENT");
			}
			debug_log(args.debug, rank, "one_to_one WORKER_STOP");
			return;
		}

		if (nc != nullptr)
			netcdf_reset_matrix(*nc);
		const double test_t0 = MPI_Wtime();
		Task t{};
		for (int src_rank = 0; src_rank < nproc; ++src_rank) {
			for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
				t.src_rank = src_rank;
				t.src_gpu = 0;
				t.dst_rank = dst_rank;
				t.dst_gpu = 0;
				t.stop = 0;
				{
					std::ostringstream oss;
					oss << "one_to_one PAIR_BEGIN src=" << t.src_rank << "." << t.src_gpu
						<< " dst=" << t.dst_rank << "." << t.dst_gpu;
					debug_log(args.debug, rank, oss.str());
				}
				if (src_rank == dst_rank) {
					std::vector<double> metric(ACK_FIELDS, 0.0);
					if (nc != nullptr)
						netcdf_store_pair(*nc, src_rank, dst_rank, metric);
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

				if (src_rank != 0)
					mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, src_rank, 1, MPI_COMM_WORLD),
						"MPI_Send Task (src)");
				if (dst_rank != 0)
					mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, dst_rank, 1, MPI_COMM_WORLD),
						"MPI_Send Task (dst)");
				debug_log(args.debug, rank, "one_to_one PAIR_DISPATCH_DONE");

				std::vector<double> metric(ACK_FIELDS, 0.0);
				if (src_rank == 0 || dst_rank == 0) {
					auto ack0 = run_one_to_one(rank, t, args, via_host, rank_labels);
					if (ack0[6] == 1.0)
						metric = ack0;
				}

				if (src_rank != 0) {
					std::vector<double> ack(ACK_FIELDS, 0.0);
					mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, src_rank, 2,
									MPI_COMM_WORLD, MPI_STATUS_IGNORE),
						"MPI_Recv ack (src)");
					if (ack[6] == 1.0)
						metric = ack;
					debug_log(args.debug, rank, "one_to_one ACK_SRC_RECV");
				}
				if (dst_rank != 0) {
					std::vector<double> ack(ACK_FIELDS, 0.0);
					mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, dst_rank, 2,
									MPI_COMM_WORLD, MPI_STATUS_IGNORE),
						"MPI_Recv ack (dst)");
					if (ack[6] == 1.0)
						metric = ack;
					debug_log(args.debug, rank, "one_to_one ACK_DST_RECV");
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
				if (nc != nullptr)
					netcdf_store_pair(*nc, src_rank, dst_rank, metric);
				{
					std::ostringstream oss;
					oss << "one_to_one PAIR_DONE src=" << t.src_rank << "." << t.src_gpu
						<< " dst=" << t.dst_rank << "." << t.dst_gpu;
					debug_log(args.debug, rank, oss.str());
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

		t.stop = 1;
		for (int r = 1; r < nproc; ++r)
			mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, r, 1, MPI_COMM_WORLD),
				"MPI_Send stop Task");
	}

} // namespace gpu_benchmark
