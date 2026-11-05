#include "gpu_one_to_one.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <utility>

namespace gpu_benchmark {

namespace {

constexpr int TAG_TASK         = 1;
constexpr int TAG_ACK          = 2;
constexpr int TAG_DATA         = 0;
constexpr int RDV_TAG_FORWARD  = 42;
constexpr int RDV_TAG_BACKWARD = 43;

/* Буферы ранга: выделяются один раз на весь режим, а не на каждую пару. */
struct Buffers {
	DeviceBuffer                      d_send, d_recv;
	PinnedBuffer                      h_buf;    // только для пути через память
	std::unique_ptr<SharedHostWindow> shared;   // общий буфер узла, путь через память
	char                             *shared_buf = nullptr;
};

/* Секунды с начала эпохи по системным часам (общие для узлов при NTP). */
double epoch_now() {
	using namespace std::chrono;
	return duration<double>(system_clock::now().time_since_epoch()).count();
}

bool same_node(const Task &t, const std::vector<int> &on_my_node) {
	return on_my_node[t.src_rank] != 0 && on_my_node[t.dst_rank] != 0;
}

/* Идёт ли пара через RAM: всегда в среде host, а с --sys-via-host — ещё пары GPU разных
   процессоров одного узла (прямой путь между ними на G5500 в тысячи раз медленнее, NOTES.md). */
bool pair_via_host(const Task &t, const Args &args, bool via_host, bool same,
                   const std::vector<int> &gpu_numa) {
	if (via_host) return true;
	if (!args.sys_via_host || !same) return false;
	const int a = gpu_numa[t.src_rank], b = gpu_numa[t.dst_rank];
	return a >= 0 && b >= 0 && a != b;
}

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
	if (via_host && same_node && b.shared_buf) {
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

	// Через MPI: буфер в памяти GPU (custom, default) или копия через оперативную память (host).
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

/* Замер пары. Возвращает ack: заполнен у получателя, у отправителя valid = 0.
   Получатель пишет в .raw на каждую итерацию время её начала по системным
   часам (снимается вне замеряемого интервала) и задержку. */
std::vector<double> measure_pair(int rank, const Task &t, const Args &args, bool via_host,
                                 bool same_node, Buffers &b,
                                 const std::vector<std::string> &rank_labels) {
	std::vector<double> ack(ACK_FIELDS, 0.0);
	const bool is_sender   = rank == t.src_rank;
	const bool is_receiver = rank == t.dst_rank;
	if (!is_sender && !is_receiver) return ack;

	DBG_LOG(rank, args, "RUN src=" << t.src_rank << " dst=" << t.dst_rank
	        << " bytes=" << args.nbytes << " env=" << env_to_string(args.env));

	ProbeResult before;
	if (args.probe) before = cpu_probe();

	for (int i = 0; i < args.warmup; ++i)
		transfer_once(t, is_sender, via_host, same_node, args.nbytes, b);

	std::vector<double> samples_us, times, cpus;
	samples_us.reserve(static_cast<size_t>(args.iters));
	if (is_receiver) times.reserve(static_cast<size_t>(args.iters));
	if (args.cpu_log) cpus.reserve(static_cast<size_t>(args.iters));   // и отправитель, и получатель
	for (int i = 0; i < args.iters; ++i) {
		if (is_receiver) times.push_back(epoch_now());
		if (args.cpu_log) cpus.push_back(current_cpu());
		const double t0 = MPI_Wtime();
		transfer_once(t, is_sender, via_host, same_node, args.nbytes, b);
		const double t1 = MPI_Wtime();
		if (is_receiver)
			samples_us.push_back((t1 - t0) * 1e6);
		rendezvous(is_sender, is_sender ? t.dst_rank : t.src_rank);
	}

	if (args.cpu_log)
		append_raw_series(args, rank_labels[t.src_rank], rank_labels[t.dst_rank], cpus,
		                  is_sender ? ".scpu" : ".rcpu", 0);
	if (args.probe) {
		const ProbeResult after = cpu_probe();
		append_raw_series(args, rank_labels[t.src_rank], rank_labels[t.dst_rank],
		                  {before.core_ns, before.l3_ns, after.core_ns, after.l3_ns},
		                  is_sender ? ".sprobe" : ".rprobe", 4);
	}
	if (is_receiver) {
		append_raw_samples(args, rank_labels[t.src_rank], rank_labels[t.dst_rank], samples_us,
		                   times);
		fill_ack(samples_us, ack.data());
	}
	return ack;
}

/* Ранги, кроме 0: ждут задание, участвуют в замере, возвращают ack. */
void worker_loop(int rank, const Args &args, bool via_host, Buffers &b,
                 const std::vector<int> &on_my_node,
                 const std::vector<std::string> &rank_labels,
                 const std::vector<int> &gpu_numa) {
	while (true) {
		Task t{};
		mpi_ok(MPI_Recv(&t, sizeof(Task), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD,
		                MPI_STATUS_IGNORE), "MPI_Recv Task");
		DBG_LOG(rank, args, "WORKER_TASK src=" << t.src_rank << " dst=" << t.dst_rank
		        << " stop=" << t.stop);
		if (t.stop) break;
		const bool same = same_node(t, on_my_node);
		const auto ack = measure_pair(rank, t, args, pair_via_host(t, args, via_host, same, gpu_numa),
		                              same, b, rank_labels);
		mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0, TAG_ACK, MPI_COMM_WORLD),
		       "MPI_Send ack");
	}
}

} // namespace

void schedule_one_to_one(int rank, int nproc, const Args &args, bool via_host,
                         MPI_Comm node_comm, int node_rank,
                         const std::vector<int> &on_my_node,
                         const std::vector<std::string> &rank_labels,
                         const std::vector<int> &gpu_numa,
                         const std::function<void(const std::string &)> &mirror) {
	// буферы в RAM нужны и для --sys-via-host: часть пар пойдёт через них
	Buffers b = make_buffers(args, via_host || args.sys_via_host, node_comm, node_rank, rank);

	// Разгон CPU до первой пары: без него первая пара мерится на низкой
	// частоте ядер (см. NOTES.md).
	if (args.prespin_sec > 0) {
		const double t_spin = MPI_Wtime();
		while (MPI_Wtime() - t_spin < args.prespin_sec) {}
		mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(prespin)");
	}

	if (rank != 0) {
		worker_loop(rank, args, via_host, b, on_my_node, rank_labels, gpu_numa);
		return;
	}

	// Ранг 0 перебирает пары и сам участвует в парах со своим номером.
	// Порядок пар: по источникам или случайный с --seed.
	std::vector<std::pair<int, int>> order;
	for (int src = 0; src < nproc; ++src)
		for (int dst = 0; dst < nproc; ++dst)
			if (src != dst) order.emplace_back(src, dst);
	if (args.random_order) {
		std::mt19937 gen(args.seed);
		std::shuffle(order.begin(), order.end(), gen);
	}

	std::ostringstream head;
	head << "Order: " << (args.random_order ? "random" : "seq") << " seed=" << args.seed
	     << " prespin_sec=" << args.prespin_sec << " cpu_log=" << (args.cpu_log ? 1 : 0) << "\n";
	mirror(head.str());

	std::vector<std::vector<double>> metric(static_cast<size_t>(nproc) * nproc,
	                                        std::vector<double>(ACK_FIELDS, 0.0));
	std::ostringstream times;
	const double test_t0 = MPI_Wtime();
	for (size_t k = 0; k < order.size(); ++k) {
		const auto [src, dst] = order[k];
		const Task t{src, dst, 0};
		DBG_LOG(rank, args, "PAIR src=" << src << " dst=" << dst);
		const double start = epoch_now();
		for (int peer : {src, dst})
			if (peer != 0)
				mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, peer, TAG_TASK, MPI_COMM_WORLD),
				       "MPI_Send Task");
		auto &m = metric[static_cast<size_t>(src) * nproc + dst];
		if (src == 0 || dst == 0) {
			const bool same = same_node(t, on_my_node);
			const auto ack = measure_pair(rank, t, args, pair_via_host(t, args, via_host, same, gpu_numa), same, b,
			                              rank_labels);
			if (ack[6] == 1.0) m = ack;
		}
		for (int peer : {src, dst}) {
			if (peer == 0) continue;
			std::vector<double> ack(ACK_FIELDS, 0.0);
			mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, peer, TAG_ACK, MPI_COMM_WORLD,
			                MPI_STATUS_IGNORE), "MPI_Recv ack");
			if (ack[6] == 1.0) m = ack;
		}
		// Время по системным часам ранга 0: от рассылки задания до последнего ack.
		times << "pair_time " << rank_labels[src] << " -> " << rank_labels[dst]
		      << " index=" << k << std::fixed << std::setprecision(6)
		      << " start=" << start << " end=" << epoch_now() << "\n";
	}
	mirror(times.str());

	// Строки pair в обычном порядке, чтобы разбор не зависел от порядка замеров.
	for (int src = 0; src < nproc; ++src)
		for (int dst = 0; dst < nproc; ++dst)
			mirror(print_pair_line("pair", rank_labels[src], rank_labels[dst],
			                       metric[static_cast<size_t>(src) * nproc + dst].data(),
			                       args.stat_out));

	std::ostringstream oss;
	oss << "TotalTimeSec: " << std::fixed << std::setprecision(TOTAL_TIME_DIGITS)
	    << (MPI_Wtime() - test_t0) << "\n";
	mirror(oss.str());

	const Task stop{0, 0, 1};
	for (int r = 1; r < nproc; ++r)
		mpi_ok(MPI_Send(&stop, sizeof(Task), MPI_BYTE, r, TAG_TASK, MPI_COMM_WORLD), "MPI_Send stop");
}

} // namespace gpu_benchmark
