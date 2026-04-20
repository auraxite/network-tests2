#include "gpu_one_to_one.hpp"

#include <algorithm>      // std::max и др.
#include <cuda_runtime.h> // CUDA: устройство, события, cudaMalloc, ...
#include <functional>     // std::function — параметр mirror в schedule_one_to_one
#include <iomanip>        // std::setprecision, std::fixed
#include <sstream>        // std::ostringstream — строки TotalTimeSec и т.п.
#include <vector>         // буферы ack, samples

namespace gpu_benchmark {

static constexpr int RDV_TAG_FORWARD = 42;
static constexpr int RDV_TAG_BACKWARD = 43;

std::vector<double> run_one_to_one(int rank, const Task &t, const Args &args,
								   bool check_host,
								   bool same_node_pair,
								   char *d_send, char *d_recv, char *h_buf,
								   char *shared_h_buf, MPI_Win shared_h_win,
								   const std::vector<std::string> &rank_labels);

void schedule_one_to_one(
	int rank, int nproc, const Args &args, bool via_host,
	const std::vector<int> &rank_to_gpu,
	bool enable_local_shared_fallback,
	MPI_Comm node_comm, int node_rank,
	const std::vector<int> &on_my_node,
	const std::vector<std::string> &rank_labels,
	const std::function<void(const std::string &)> &mirror) {

	/* Один раз на весь прогон выделяем буферы текущего ранга на ЕГО GPU.
	   Раньше cudaMalloc / cudaMallocHost / cudaFree делались на КАЖДОЙ паре
	   (это nproc*(nproc-1) аллокаций), что давало большой накладной шум и
	   удлиняло общий прогон в разы. cudaMallocHost особенно дорогой —
	   pinning страниц через ядро. Теперь выделяем по одному d_send + d_recv
	   и при необходимости один h_buf. */
	const int my_gpu = rank_to_gpu[static_cast<size_t>(rank)];
	cuda_ok(cudaSetDevice(my_gpu), "cudaSetDevice(schedule_one_to_one)");

	char *d_send = nullptr;
	char *d_recv = nullptr;
	char *h_buf = nullptr;
	char *shared_h_buf = nullptr;
	MPI_Win shared_h_win = MPI_WIN_NULL;
	bool shared_h_registered = false;
	const bool use_shared_host_buffer = via_host || enable_local_shared_fallback;
	cuda_ok(cudaMalloc(&d_send, args.nbytes), "cudaMalloc(d_send)");
	cuda_ok(cudaMalloc(&d_recv, args.nbytes), "cudaMalloc(d_recv)");
	cuda_ok(cudaMemset(d_send, 0xAA, args.nbytes), "cudaMemset(d_send init)");
	if (via_host) {
		cuda_ok(cudaMallocHost(&h_buf, args.nbytes), "cudaMallocHost(h_buf)");
	}
	if (use_shared_host_buffer) {
		MPI_Aint shared_bytes = 0;
		int disp_unit = 0;
		void *shared_base = nullptr;
		const MPI_Aint my_shared_bytes = (node_rank == 0)
											 ? static_cast<MPI_Aint>(args.nbytes)
											 : static_cast<MPI_Aint>(0);
		mpi_ok(MPI_Win_allocate_shared(my_shared_bytes, sizeof(char), MPI_INFO_NULL,
									   node_comm, &shared_base, &shared_h_win),
			   "MPI_Win_allocate_shared(one_to_one host)");
		mpi_ok(MPI_Win_shared_query(shared_h_win, 0, &shared_bytes, &disp_unit,
									&shared_base),
			   "MPI_Win_shared_query(one_to_one host)");
		shared_h_buf = static_cast<char *>(shared_base);
		if (shared_h_buf != nullptr && shared_bytes >= static_cast<MPI_Aint>(args.nbytes)) {
			cudaError_t reg_err =
				cudaHostRegister(shared_h_buf, args.nbytes, cudaHostRegisterPortable);
			if (reg_err == cudaSuccess) {
				shared_h_registered = true;
			} else if (reg_err == cudaErrorHostMemoryAlreadyRegistered) {
				cudaGetLastError();
				shared_h_registered = true;
			} else {
				DBG_LOG(rank, args,
						"one_to_one shared host register skipped: "
						<< cudaGetErrorString(reg_err));
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
			DBG_LOG(rank, args,
					"one_to_one WORKER_TASK_RECV src=" << tw.src_rank << "." << tw.src_gpu
					<< " dst=" << tw.dst_rank << "." << tw.dst_gpu << " stop=" << tw.stop);
			if (tw.stop)
				break;
			const bool same_node_pair =
				on_my_node[static_cast<size_t>(tw.src_rank)] != 0 &&
				on_my_node[static_cast<size_t>(tw.dst_rank)] != 0;
			auto ack = run_one_to_one(rank, tw, args, via_host,
									  same_node_pair,
									  d_send, d_recv, h_buf,
									  shared_h_buf, shared_h_win, rank_labels);
			mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD),
				"MPI_Send ack (worker)");
			DBG_LOG(rank, args, "one_to_one WORKER_ACK_SENT");
		}
		DBG_LOG(rank, args, "one_to_one WORKER_STOP");
		if (shared_h_registered)
			cuda_ok(cudaHostUnregister(shared_h_buf), "cudaHostUnregister(shared host)");
		if (shared_h_win != MPI_WIN_NULL)
			mpi_ok(MPI_Win_free(&shared_h_win), "MPI_Win_free(one_to_one host)");
		if (h_buf)
			cudaFreeHost(h_buf);
		cudaFree(d_send);
		cudaFree(d_recv);
		return;
	}

	const double test_t0 = MPI_Wtime();
	Task t{};
	for (int src_rank = 0; src_rank < nproc; ++src_rank) {
		for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
			t.src_rank = src_rank;
			/* src_gpu / dst_gpu теперь берутся из глобального справочника
			   rank_to_gpu, заполненного в gpu_benchmark.cpp по локальному
			   рангу процесса на узле. Раньше тут стояло 0 у всех, и при
			   --gres=gpu:N (когда процессу видны все GPU узла) ВСЕ четыре
			   процесса узла начали бы бегать по физическому GPU 0. */
			t.src_gpu = rank_to_gpu[static_cast<size_t>(src_rank)];
			t.dst_rank = dst_rank;
			t.dst_gpu = rank_to_gpu[static_cast<size_t>(dst_rank)];
			t.stop = 0;
			DBG_LOG(rank, args,
					"one_to_one PAIR_BEGIN src=" << t.src_rank << "." << t.src_gpu
					<< " dst=" << t.dst_rank << "." << t.dst_gpu);
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

			const bool same_node_pair =
				on_my_node[static_cast<size_t>(src_rank)] != 0 &&
				on_my_node[static_cast<size_t>(dst_rank)] != 0;

			if (src_rank != 0)
				mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, src_rank, 1, MPI_COMM_WORLD),
					"MPI_Send Task (src)");
			if (dst_rank != 0)
				mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, dst_rank, 1, MPI_COMM_WORLD),
					"MPI_Send Task (dst)");
			DBG_LOG(rank, args, "one_to_one PAIR_DISPATCH_DONE");

			std::vector<double> metric(ACK_FIELDS, 0.0);
			if (src_rank == 0 || dst_rank == 0) {
				auto ack0 = run_one_to_one(rank, t, args, via_host,
										   same_node_pair,
										   d_send, d_recv, h_buf,
										   shared_h_buf, shared_h_win, rank_labels);
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
				DBG_LOG(rank, args, "one_to_one ACK_SRC_RECV");
			}
			if (dst_rank != 0) {
				std::vector<double> ack(ACK_FIELDS, 0.0);
				mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, dst_rank, 2,
								MPI_COMM_WORLD, MPI_STATUS_IGNORE),
					"MPI_Recv ack (dst)");
				if (ack[6] == 1.0)
					metric = ack;
				DBG_LOG(rank, args, "one_to_one ACK_DST_RECV");
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
			DBG_LOG(rank, args,
					"one_to_one PAIR_DONE src=" << t.src_rank << "." << t.src_gpu
					<< " dst=" << t.dst_rank << "." << t.dst_gpu);
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
			"MPI_Send stop Task");

	if (shared_h_registered)
		cuda_ok(cudaHostUnregister(shared_h_buf), "cudaHostUnregister(shared host)");
	if (shared_h_win != MPI_WIN_NULL)
		mpi_ok(MPI_Win_free(&shared_h_win), "MPI_Win_free(one_to_one host)");
	if (h_buf)
		cudaFreeHost(h_buf);
	cudaFree(d_send);
	cudaFree(d_recv);
}

std::vector<double> run_one_to_one(int rank, const Task &t, const Args &args,
								   bool check_host,
								   bool same_node_pair,
								   char *d_send, char *d_recv, char *h_buf,
								   char *shared_h_buf, MPI_Win shared_h_win,
								   const std::vector<std::string> &rank_labels) {
	std::vector<double> ack(ACK_FIELDS, 0.0);
	const bool is_sender = (rank == t.src_rank);
	const bool is_receiver = (rank == t.dst_rank);
	/* Таймер закреплён за получателем: сэмпл = MPI_Recv (ожидание сети + сам Recv)
	   плюс H2D у получателя. На одних часах, без межузловой синхронизации.
	   Близко по смыслу к «когда у получателя данные оказались на GPU». */
	const bool collect_raw_samples_here = is_receiver;
	if (!is_sender && !is_receiver)
		return ack;
	if (t.src_rank == t.dst_rank) {
		ack[6] = 1.0;
		return ack;
	}

	const int count = static_cast<int>(args.nbytes);
	DBG_LOG(rank, args,
			"one_to_one RUN_BEGIN src=" << t.src_rank << "." << t.src_gpu
			<< " dst=" << t.dst_rank << "." << t.dst_gpu << " bytes=" << args.nbytes
			<< " role_sender=" << (is_sender ? 1 : 0)
			<< " role_receiver=" << (is_receiver ? 1 : 0)
			<< " env_path=" << (check_host ? "host" : "auto"));

	DBG_LOG(rank, args, "one_to_one PREPARE_DONE");

	int current_iter = -1;
	const char *current_phase = "none";
	auto do_one = [&]() {
		if (same_node_pair && shared_h_buf != nullptr) {
			if (is_sender) {
				DBG_LOG(rank, args,
						"one_to_one LOCAL_D2H_BEGIN phase=" << current_phase
						<< " idx=" << current_iter);
				cuda_ok(cudaMemcpy(shared_h_buf, d_send, count, cudaMemcpyDeviceToHost),
						"D2H(shared host)");
				if (shared_h_win != MPI_WIN_NULL)
					mpi_ok(MPI_Win_sync(shared_h_win), "MPI_Win_sync(shared host sender)");
				DBG_LOG(rank, args,
						"one_to_one LOCAL_D2H_DONE phase=" << current_phase
						<< " idx=" << current_iter);
				char dummy = 0;
				mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, t.dst_rank, RDV_TAG_FORWARD,
									&dummy, 0, MPI_BYTE, t.dst_rank, RDV_TAG_BACKWARD,
									MPI_COMM_WORLD, MPI_STATUS_IGNORE),
					   "MPI_Sendrecv local host ready (sender)");
			}
			if (is_receiver) {
				char dummy = 0;
				mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, t.src_rank, RDV_TAG_BACKWARD,
									&dummy, 0, MPI_BYTE, t.src_rank, RDV_TAG_FORWARD,
									MPI_COMM_WORLD, MPI_STATUS_IGNORE),
					   "MPI_Sendrecv local host ready (receiver)");
				if (shared_h_win != MPI_WIN_NULL)
					mpi_ok(MPI_Win_sync(shared_h_win), "MPI_Win_sync(shared host receiver)");
				DBG_LOG(rank, args,
						"one_to_one LOCAL_H2D_BEGIN phase=" << current_phase
						<< " idx=" << current_iter);
				cuda_ok(cudaMemcpy(d_recv, shared_h_buf, count, cudaMemcpyHostToDevice),
						"H2D(shared host)");
				DBG_LOG(rank, args,
						"one_to_one LOCAL_H2D_DONE phase=" << current_phase
						<< " idx=" << current_iter);
			}
			return;
		}

		if (is_sender) {
			if (check_host) {
				DBG_LOG(rank, args,
						"one_to_one D2H_BEGIN phase=" << current_phase
						<< " idx=" << current_iter);
				cuda_ok(cudaMemcpy(h_buf, d_send, count, cudaMemcpyDeviceToHost), "D2H");
				DBG_LOG(rank, args,
						"one_to_one D2H_DONE phase=" << current_phase
						<< " idx=" << current_iter);
				DBG_LOG(rank, args,
						"one_to_one MPI_SEND_BEGIN dst=" << t.dst_rank
						<< " tag=0 bytes=" << count << " path=host"
						<< " phase=" << current_phase << " idx=" << current_iter);
				mpi_ok(MPI_Send(h_buf, count, MPI_BYTE, t.dst_rank, 0, MPI_COMM_WORLD),
					   "MPI_Send (host staging)");
				DBG_LOG(rank, args,
						"one_to_one MPI_SEND_DONE path=host"
						<< " phase=" << current_phase << " idx=" << current_iter);
			} else {
				DBG_LOG(rank, args,
						"one_to_one MPI_SEND_BEGIN dst=" << t.dst_rank
						<< " tag=0 bytes=" << count << " path=device"
						<< " phase=" << current_phase << " idx=" << current_iter);
				mpi_ok(MPI_Send(d_send, count, MPI_BYTE, t.dst_rank, 0, MPI_COMM_WORLD),
					   "MPI_Send (device buffer)");
				DBG_LOG(rank, args,
						"one_to_one MPI_SEND_DONE path=device"
						<< " phase=" << current_phase << " idx=" << current_iter);
			}
		}
		if (is_receiver) {
			MPI_Status st{};
			if (check_host) {
				DBG_LOG(rank, args,
						"one_to_one MPI_RECV_BEGIN src=" << t.src_rank
						<< " tag=0 bytes=" << count << " path=host"
						<< " phase=" << current_phase << " idx=" << current_iter);
				mpi_ok(MPI_Recv(h_buf, count, MPI_BYTE, t.src_rank, 0, MPI_COMM_WORLD,
								&st),
					   "MPI_Recv (host staging)");
				DBG_LOG(rank, args,
						"one_to_one MPI_RECV_DONE path=host"
						<< " phase=" << current_phase << " idx=" << current_iter);
				DBG_LOG(rank, args,
						"one_to_one H2D_BEGIN phase=" << current_phase
						<< " idx=" << current_iter);
				cuda_ok(cudaMemcpy(d_recv, h_buf, count, cudaMemcpyHostToDevice), "H2D");
				DBG_LOG(rank, args,
						"one_to_one H2D_DONE phase=" << current_phase
						<< " idx=" << current_iter);
			} else {
				DBG_LOG(rank, args,
						"one_to_one MPI_RECV_BEGIN src=" << t.src_rank
						<< " tag=0 bytes=" << count << " path=device"
						<< " phase=" << current_phase << " idx=" << current_iter);
				mpi_ok(MPI_Recv(d_recv, count, MPI_BYTE, t.src_rank, 0, MPI_COMM_WORLD,
								&st),
					   "MPI_Recv (device buffer)");
				DBG_LOG(rank, args,
						"one_to_one MPI_RECV_DONE path=device"
						<< " phase=" << current_phase << " idx=" << current_iter);
			}
		}
	};

	std::vector<double> samples_mpi_us;
	std::vector<double> samples_cpu_us;
	std::vector<double> samples_gpu_us;
	cudaEvent_t ev_start = nullptr;
	cudaEvent_t ev_stop = nullptr;
	if (collect_raw_samples_here) {
		cuda_ok(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
		cuda_ok(cudaEventCreate(&ev_stop), "cudaEventCreate(stop)");
	}
	samples_mpi_us.reserve(static_cast<size_t>(std::max(1, args.iters)));
	samples_cpu_us.reserve(static_cast<size_t>(std::max(1, args.iters)));
	samples_gpu_us.reserve(static_cast<size_t>(std::max(1, args.iters)));

	DBG_LOG(rank, args, "one_to_one WARMUP_BEGIN");
	current_phase = "warmup";
	for (int i = 0; i < args.warmup; ++i) {
		current_iter = i;
		do_one();
	}
	DBG_LOG(rank, args, "one_to_one WARMUP_DONE");

	DBG_LOG(rank, args, "one_to_one ITERS_BEGIN");
	current_phase = "iters";
	for (int i = 0; i < args.iters; ++i) {
		current_iter = i;
		DBG_LOG(rank, args, "one_to_one ITER_BEGIN idx=" << i);

		char dummy = 0;
		if (is_sender) {
			mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, t.dst_rank, RDV_TAG_FORWARD,
								&dummy, 0, MPI_BYTE, t.dst_rank, RDV_TAG_BACKWARD,
								MPI_COMM_WORLD, MPI_STATUS_IGNORE),
				   "MPI_Sendrecv iter rendezvous (sender)");
		} else {
			mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, t.src_rank, RDV_TAG_BACKWARD,
								&dummy, 0, MPI_BYTE, t.src_rank, RDV_TAG_FORWARD,
								MPI_COMM_WORLD, MPI_STATUS_IGNORE),
				   "MPI_Sendrecv iter rendezvous (receiver)");
		}

		/* CUDA-event start — ВНЕ интервала t0..t1: сам по себе cudaEventRecord
		   стоит несколько мкс, и эти мкс не должны попадать в samples_mpi_us. */
		if (collect_raw_samples_here)
			cuda_ok(cudaEventRecord(ev_start), "cudaEventRecord(start)");

		const double t0_mpi = MPI_Wtime();
		const double t0_clk = clock_gettime_wrapper();
		do_one();
		const double t1_mpi = MPI_Wtime();
		const double t1_clk = clock_gettime_wrapper();

		samples_mpi_us.push_back((t1_mpi - t0_mpi) * 1e6);
		samples_cpu_us.push_back(t1_clk - t0_clk);

		/* CUDA-event stop + sync + elapsed — тоже ВНЕ интервала. В env=auto
		   default stream обычно пуст, и samples_gpu_us здесь близок к нулю —
		   это нормально: реальная H2D происходит во внутреннем UCX-потоке,
		   который наш ev_stop не видит. В env=host default stream содержит
		   H2D, и samples_gpu_us показывает её время. */
		if (collect_raw_samples_here) {
			cuda_ok(cudaEventRecord(ev_stop), "cudaEventRecord(stop)");
			cuda_ok(cudaEventSynchronize(ev_stop), "cudaEventSynchronize(stop)");
			float elapsed_ms = 0.0f;
			cuda_ok(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop),
					"cudaEventElapsedTime");
			samples_gpu_us.push_back(static_cast<double>(elapsed_ms) * 1e3);
		}

		DBG_LOG(rank, args, "one_to_one ITER_DONE idx=" << i);
	}
	DBG_LOG(rank, args, "one_to_one ITERS_DONE");
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
	DBG_LOG(rank, args, "one_to_one STATS_DONE");
	if (ev_start)
		cudaEventDestroy(ev_start);
	if (ev_stop)
		cudaEventDestroy(ev_stop);

	DBG_LOG(rank, args, "one_to_one RUN_DONE");
	return ack;
}

} // namespace gpu_benchmark
