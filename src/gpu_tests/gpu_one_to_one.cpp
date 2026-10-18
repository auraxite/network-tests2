#include "gpu_one_to_one.hpp"
#include "gpu_one_to_one_detail.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <sstream>

namespace gpu_benchmark {

namespace one_to_one_detail {

namespace {

constexpr int TAG_DATA         = 0;
constexpr int RDV_TAG_FORWARD  = 42;
constexpr int RDV_TAG_BACKWARD = 43;

/* Пустое сообщение туда и обратно: держит отправителя и получателя в одной фазе. */
void rendezvous(bool is_sender, int peer) {
	char dummy = 0;
	if (is_sender)
		mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, peer, RDV_TAG_FORWARD,
		                    &dummy, 0, MPI_BYTE, peer, RDV_TAG_BACKWARD,
		                    MPI_COMM_WORLD, MPI_STATUS_IGNORE), "Sendrecv rendezvous(sender)");
	else
		mpi_ok(MPI_Sendrecv(&dummy, 0, MPI_BYTE, peer, RDV_TAG_BACKWARD,
		                    &dummy, 0, MPI_BYTE, peer, RDV_TAG_FORWARD,
		                    MPI_COMM_WORLD, MPI_STATUS_IGNORE), "Sendrecv rendezvous(receiver)");
}

/* Одна передача src -> dst. Вызывается и отправителем, и получателем. */
void transfer_once(const Task &t, bool is_sender, bool via_host, bool same_node,
                   size_t nbytes, Buffers &b) {
	const int count = static_cast<int>(nbytes);

	// Внутри узла через общую оперативную память: копия GPU -> память у
	// отправителя, сигнал готовности, копия память -> GPU у получателя.
	if (same_node && b.shared_buf) {
		if (is_sender) {
			cuda_ok(cudaMemcpy(b.shared_buf, b.d_send.get(), count, cudaMemcpyDeviceToHost),
			        "D2H(shared)");
			b.shared->sync();
			rendezvous(true, t.dst_rank);
		} else {
			rendezvous(false, t.src_rank);
			b.shared->sync();
			cuda_ok(cudaMemcpy(b.d_recv.get(), b.shared_buf, count, cudaMemcpyHostToDevice),
			        "H2D(shared)");
		}
		return;
	}

	// Через MPI: буфер в памяти GPU (auto) или копия через оперативную память (host).
	if (is_sender) {
		if (via_host) {
			cuda_ok(cudaMemcpy(b.h_buf.get(), b.d_send.get(), count, cudaMemcpyDeviceToHost), "D2H");
			mpi_ok(MPI_Send(b.h_buf.get(), count, MPI_BYTE, t.dst_rank, TAG_DATA, MPI_COMM_WORLD),
			       "MPI_Send(host)");
		} else {
			mpi_ok(MPI_Send(b.d_send.get(), count, MPI_BYTE, t.dst_rank, TAG_DATA, MPI_COMM_WORLD),
			       "MPI_Send(device)");
		}
	} else {
		if (via_host) {
			mpi_ok(MPI_Recv(b.h_buf.get(), count, MPI_BYTE, t.src_rank, TAG_DATA, MPI_COMM_WORLD,
			                MPI_STATUS_IGNORE), "MPI_Recv(host)");
			cuda_ok(cudaMemcpy(b.d_recv.get(), b.h_buf.get(), count, cudaMemcpyHostToDevice), "H2D");
		} else {
			mpi_ok(MPI_Recv(b.d_recv.get(), count, MPI_BYTE, t.src_rank, TAG_DATA, MPI_COMM_WORLD,
			                MPI_STATUS_IGNORE), "MPI_Recv(device)");
			// UCX cuda_ipc может сообщить о завершении MPI раньше, чем копия на
			// устройстве действительно закончится: ждём её внутри замера.
			cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize(recv device)");
		}
	}
}

} // namespace

double epoch_now() {
	using namespace std::chrono;
	return duration<double>(system_clock::now().time_since_epoch()).count();
}

bool same_node(const Task &t, const std::vector<int> &on_my_node) {
	return on_my_node[t.src_rank] != 0 && on_my_node[t.dst_rank] != 0;
}

Buffers make_buffers(const Args &args, bool via_host, MPI_Comm node_comm, int node_rank,
                     int rank) {
	Buffers b;
	b.d_send = DeviceBuffer(args.nbytes);
	b.d_recv = DeviceBuffer(args.nbytes);
	cuda_ok(cudaMemset(b.d_send.get(), 0xAA, args.nbytes), "cudaMemset(d_send)");
	if (via_host) {
		b.h_buf = PinnedBuffer(args.nbytes);
		// один общий буфер на узел: память выделяет ранг 0 узла
		b.shared = std::make_unique<SharedHostWindow>(
		    node_comm, node_rank == 0 ? args.nbytes : 0, rank, args);
		b.shared_buf = b.shared->segment(0, args.nbytes);
	}
	return b;
}

std::vector<double> measure_pair(int rank, const Task &t, const Args &args, bool via_host,
                                 bool same_node, Buffers &b,
                                 const std::vector<std::string> &rank_labels,
                                 bool with_times) {
	std::vector<double> ack(ACK_FIELDS, 0.0);
	const bool is_sender   = rank == t.src_rank;
	const bool is_receiver = rank == t.dst_rank;
	if (!is_sender && !is_receiver) return ack;

	DBG_LOG(rank, args, "RUN src=" << t.src_rank << " dst=" << t.dst_rank
	        << " bytes=" << args.nbytes << " path=" << (via_host ? "host" : "auto"));

	const bool probe = with_times && args.probe;
	ProbeResult before;
	if (probe) before = cpu_probe();

	for (int i = 0; i < args.warmup; ++i)
		transfer_once(t, is_sender, via_host, same_node, args.nbytes, b);

	std::vector<double> samples_us, times, cpus;
	samples_us.reserve(static_cast<size_t>(args.iters));
	const bool record_times = with_times && is_receiver;
	const bool record_cpus  = with_times && args.cpu_log;   // и отправитель, и получатель
	if (record_times) times.reserve(static_cast<size_t>(args.iters));
	if (record_cpus) cpus.reserve(static_cast<size_t>(args.iters));
	for (int i = 0; i < args.iters; ++i) {
		if (record_times) times.push_back(epoch_now());
		if (record_cpus) cpus.push_back(current_cpu());
		const double t0 = MPI_Wtime();
		transfer_once(t, is_sender, via_host, same_node, args.nbytes, b);
		const double t1 = MPI_Wtime();
		if (is_receiver)
			samples_us.push_back((t1 - t0) * 1e6);
		rendezvous(is_sender, is_sender ? t.dst_rank : t.src_rank);
	}

	if (record_cpus)
		append_raw_series(args, rank_labels[t.src_rank], rank_labels[t.dst_rank], cpus,
		                  is_sender ? ".scpu" : ".rcpu", 0);
	if (probe) {
		const ProbeResult after = cpu_probe();
		append_raw_series(args, rank_labels[t.src_rank], rank_labels[t.dst_rank],
		                  {before.core_ns, before.l3_ns, after.core_ns, after.l3_ns},
		                  is_sender ? ".sprobe" : ".rprobe", 4);
	}
	if (is_receiver) {
		append_raw_samples(args, rank_labels[t.src_rank], rank_labels[t.dst_rank], samples_us);
		if (record_times)
			append_raw_series(args, rank_labels[t.src_rank], rank_labels[t.dst_rank], times,
			                  ".ts", 6);
		fill_ack(samples_us, ack.data());
	}
	return ack;
}

void worker_loop(int rank, const Args &args, bool via_host, Buffers &b,
                 const std::vector<int> &on_my_node,
                 const std::vector<std::string> &rank_labels, bool with_times) {
	while (true) {
		Task t{};
		mpi_ok(MPI_Recv(&t, sizeof(Task), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD,
		                MPI_STATUS_IGNORE), "MPI_Recv Task");
		DBG_LOG(rank, args, "WORKER_TASK src=" << t.src_rank << " dst=" << t.dst_rank
		        << " stop=" << t.stop);
		if (t.stop) break;
		const auto ack = measure_pair(rank, t, args, via_host, same_node(t, on_my_node), b,
		                              rank_labels, with_times);
		mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0, TAG_ACK, MPI_COMM_WORLD),
		       "MPI_Send ack");
	}
}

} // namespace one_to_one_detail

using namespace one_to_one_detail;

void schedule_one_to_one(int rank, int nproc, const Args &args, bool via_host,
                         MPI_Comm node_comm, int node_rank,
                         const std::vector<int> &on_my_node,
                         const std::vector<std::string> &rank_labels,
                         const std::function<void(const std::string &)> &mirror) {
	Buffers b = make_buffers(args, via_host, node_comm, node_rank, rank);

	if (rank != 0) {
		worker_loop(rank, args, via_host, b, on_my_node, rank_labels, false);
		return;
	}

	// Ранг 0: перебирает пары, сам участвует в парах со своим номером.
	const double test_t0 = MPI_Wtime();
	for (int src = 0; src < nproc; ++src) {
		for (int dst = 0; dst < nproc; ++dst) {
			std::vector<double> metric(ACK_FIELDS, 0.0);
			if (src != dst) {
				const Task t{src, dst, 0};
				DBG_LOG(rank, args, "PAIR src=" << src << " dst=" << dst);
				for (int peer : {src, dst})
					if (peer != 0)
						mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, peer, TAG_TASK, MPI_COMM_WORLD),
						       "MPI_Send Task");
				if (src == 0 || dst == 0) {
					const auto ack = measure_pair(rank, t, args, via_host, same_node(t, on_my_node),
					                              b, rank_labels);
					if (ack[6] == 1.0) metric = ack;
				}
				for (int peer : {src, dst}) {
					if (peer == 0) continue;
					std::vector<double> ack(ACK_FIELDS, 0.0);
					mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, peer, TAG_ACK,
					                MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Recv ack");
					if (ack[6] == 1.0) metric = ack;
				}
			}
			mirror(print_pair_line("pair", rank_labels[src], rank_labels[dst], metric.data(),
			                       args.stat_out));
		}
	}

	std::ostringstream oss;
	oss << "TotalTimeSec: " << std::fixed << std::setprecision(TOTAL_TIME_DIGITS)
	    << (MPI_Wtime() - test_t0) << "\n";
	mirror(oss.str());

	const Task stop{0, 0, 1};
	for (int r = 1; r < nproc; ++r)
		mpi_ok(MPI_Send(&stop, sizeof(Task), MPI_BYTE, r, TAG_TASK, MPI_COMM_WORLD), "MPI_Send stop");
}

} // namespace gpu_benchmark
