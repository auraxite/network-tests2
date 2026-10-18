#include "gpu_one_to_one_v2.hpp"
#include "gpu_one_to_one_detail.hpp"

#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>
#include <utility>

namespace gpu_benchmark {

using namespace one_to_one_detail;

void schedule_one_to_one_v2(int rank, int nproc, const Args &args, bool via_host,
                            MPI_Comm node_comm, int node_rank,
                            const std::vector<int> &on_my_node,
                            const std::vector<std::string> &rank_labels,
                            const std::function<void(const std::string &)> &mirror) {
	Buffers b = make_buffers(args, via_host, node_comm, node_rank, rank);

	// Разгон CPU до первой пары: без него первая пара мерится на низкой
	// частоте ядер (см. NOTES.md).
	if (args.prespin_sec > 0) {
		const double t_spin = MPI_Wtime();
		while (MPI_Wtime() - t_spin < args.prespin_sec) {}
		mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(prespin)");
	}

	if (rank != 0) {
		worker_loop(rank, args, via_host, b, on_my_node, rank_labels, true);
		return;
	}

	// Порядок пар: как в one_to_one (по источникам) или случайный с --seed.
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
			const auto ack = measure_pair(rank, t, args, via_host, same_node(t, on_my_node), b,
			                              rank_labels, true);
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
