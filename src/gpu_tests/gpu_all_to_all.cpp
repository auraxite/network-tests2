#include "gpu_all_to_all.hpp"

#include <iomanip>
#include <memory>
#include <sstream>

namespace gpu_benchmark {

namespace {

int pair_tag(int src, int dst, int nproc) { return src * nproc + dst; }
int ready_tag(int src, int dst, int nproc) { return nproc * nproc + pair_tag(src, dst, nproc); }

/* Замеры на этом ранге как на получателе. Возвращает строку ack по источникам:
   элемент src — статистики пары (src, rank), свой элемент помечен valid = 1. */
std::vector<double> run_all_to_all(int rank, int nproc, const Args &args, bool via_host,
                                   MPI_Comm node_comm, const std::vector<int> &on_my_node,
                                   const std::vector<int> &node_ranks,
                                   const std::vector<std::string> &rank_labels) {
	const int count = static_cast<int>(args.nbytes);
	DBG_LOG(rank, args, "all_to_all RUN_BEGIN bytes=" << args.nbytes
	        << " path=" << (via_host ? "host" : "auto"));

	DeviceBuffer d_send(args.nbytes), d_recv(args.nbytes);
	cuda_ok(cudaMemset(d_send.get(), 0xAA, args.nbytes), "cudaMemset");

	// Путь через память: внутри узла — общие сегменты (по одному на ранг),
	// между узлами — отдельные закреплённые буферы на каждого соседа.
	// Путь auto: отдельный буфер приёма в памяти GPU на каждого соседа.
	std::unique_ptr<SharedHostWindow> shared;
	std::vector<char *>       slot(nproc, nullptr);   // общий сегмент ранга r (если на этом узле)
	std::vector<PinnedBuffer> send_host(nproc), recv_host(nproc);
	std::vector<DeviceBuffer> recv_dev(nproc);

	if (via_host) {
		shared = std::make_unique<SharedHostWindow>(node_comm, args.nbytes, rank, args);
		for (size_t li = 0; li < node_ranks.size(); ++li)
			slot[node_ranks[li]] = shared->segment(static_cast<int>(li), args.nbytes);
	}
	for (int peer = 0; peer < nproc; ++peer) {
		if (peer == rank) continue;
		if (!via_host) {
			recv_dev[peer] = DeviceBuffer(args.nbytes);
		} else if (!(on_my_node[peer] && slot[peer])) {
			send_host[peer] = PinnedBuffer(args.nbytes);
			recv_host[peer] = PinnedBuffer(args.nbytes);
		}
	}

	std::vector<std::vector<double>> samples(nproc);
	for (int src = 0; src < nproc; ++src)
		if (src != rank) samples[src].reserve(static_cast<size_t>(args.iters));

	auto one_iteration = [&](bool measure) {
		// recv[src] — данные от src, recv[nproc + src] — сигнал готовности
		// общего сегмента src (путь через память внутри узла).
		std::vector<MPI_Request> recv(2 * nproc, MPI_REQUEST_NULL);
		std::vector<MPI_Request> send(nproc, MPI_REQUEST_NULL);

		for (int src = 0; src < nproc; ++src) {
			if (src == rank) continue;
			if (via_host && on_my_node[src] && slot[src])
				mpi_ok(MPI_Irecv(nullptr, 0, MPI_BYTE, src, ready_tag(src, rank, nproc),
				                 MPI_COMM_WORLD, &recv[nproc + src]), "MPI_Irecv ready");
			else
				mpi_ok(MPI_Irecv(via_host ? recv_host[src].get() : recv_dev[src].get(), count,
				                 MPI_BYTE, src, pair_tag(src, rank, nproc), MPI_COMM_WORLD,
				                 &recv[src]), "MPI_Irecv");
		}

		// Общий старт итерации. Без барьера ранги расходятся по фазе, и в замер
		// попадает ожидание старта отправителя (см. NOTES.md). Приёмы уже
		// выставлены, поэтому барьер не задерживает передачу.
		mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(iteration)");
		const double t0 = measure ? MPI_Wtime() : 0.0;

		bool own_slot_filled = false;
		for (int dst = 0; dst < nproc; ++dst) {
			if (dst == rank) continue;
			if (!via_host) {
				mpi_ok(MPI_Isend(d_send.get(), count, MPI_BYTE, dst, pair_tag(rank, dst, nproc),
				                 MPI_COMM_WORLD, &send[dst]), "MPI_Isend device");
			} else if (on_my_node[dst] && slot[rank]) {
				if (!own_slot_filled) {   // одна копия в свой сегмент на всех соседей по узлу
					cuda_ok(cudaMemcpy(slot[rank], d_send.get(), args.nbytes,
					                   cudaMemcpyDeviceToHost), "D2H(shared)");
					shared->sync();
					own_slot_filled = true;
				}
				mpi_ok(MPI_Isend(nullptr, 0, MPI_BYTE, dst, ready_tag(rank, dst, nproc),
				                 MPI_COMM_WORLD, &send[dst]), "MPI_Isend ready");
			} else {
				cuda_ok(cudaMemcpy(send_host[dst].get(), d_send.get(), args.nbytes,
				                   cudaMemcpyDeviceToHost), "D2H");
				mpi_ok(MPI_Isend(send_host[dst].get(), count, MPI_BYTE, dst,
				                 pair_tag(rank, dst, nproc), MPI_COMM_WORLD, &send[dst]),
				       "MPI_Isend host");
			}
		}

		for (int done = 0; done < nproc - 1; ++done) {
			int idx = MPI_UNDEFINED;
			mpi_ok(MPI_Waitany(static_cast<int>(recv.size()), recv.data(), &idx,
			                   MPI_STATUS_IGNORE), "MPI_Waitany");
			if (idx == MPI_UNDEFINED) break;
			const int src = idx % nproc;
			if (via_host) {
				if (idx >= nproc) {
					shared->sync();
					cuda_ok(cudaMemcpy(d_recv.get(), slot[src], args.nbytes,
					                   cudaMemcpyHostToDevice), "H2D(shared)");
				} else {
					cuda_ok(cudaMemcpy(d_recv.get(), recv_host[src].get(), args.nbytes,
					                   cudaMemcpyHostToDevice), "H2D");
				}
			}
			// Путь auto: данные уже в recv_dev[src], синхронизация устройства здесь
			// намеренно не делается: она ждала бы и чужие передачи (см. NOTES.md).
			if (measure)
				samples[src].push_back((MPI_Wtime() - t0) * 1e6);
		}

		// Одна синхронизация устройства на итерацию, уже после замеров.
		if (!via_host)
			cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after exchange)");
		mpi_ok(MPI_Waitall(nproc, send.data(), MPI_STATUSES_IGNORE), "MPI_Waitall sends");
	};

	for (int i = 0; i < args.warmup; ++i) one_iteration(false);
	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(iters)");
	for (int i = 0; i < args.iters; ++i) one_iteration(true);

	std::vector<double> row(static_cast<size_t>(nproc) * ACK_FIELDS, 0.0);
	for (int src = 0; src < nproc; ++src) {
		double *ack = row.data() + static_cast<size_t>(src) * ACK_FIELDS;
		if (src == rank) {
			ack[6] = 1.0;
			continue;
		}
		append_raw_samples(args, rank_labels[src], rank_labels[rank], samples[src]);
		fill_ack(samples[src], ack);
	}
	DBG_LOG(rank, args, "all_to_all RUN_DONE");
	return row;
}

} // namespace

void schedule_all_to_all(int rank, int nproc, const Args &args, bool via_host,
                         MPI_Comm node_comm,
                         const std::vector<int> &on_my_node,
                         const std::vector<int> &node_ranks,
                         const std::vector<std::string> &rank_labels,
                         const std::function<void(const std::string &)> &mirror) {
	const double t0 = MPI_Wtime();
	const std::vector<double> row =
	    run_all_to_all(rank, nproc, args, via_host, node_comm, on_my_node, node_ranks, rank_labels);

	// Сбор после всех замеров: строка ранга dst — его статистики по источникам.
	std::vector<double> all;
	if (rank == 0) all.resize(static_cast<size_t>(nproc) * row.size());
	mpi_ok(MPI_Gather(row.data(), static_cast<int>(row.size()), MPI_DOUBLE, all.data(),
	                  static_cast<int>(row.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD),
	       "MPI_Gather results");
	if (rank != 0) return;

	for (int src = 0; src < nproc; ++src)
		for (int dst = 0; dst < nproc; ++dst) {
			const double *ack = all.data() +
			    (static_cast<size_t>(dst) * nproc + src) * ACK_FIELDS;
			mirror(print_pair_line("pair", rank_labels[src], rank_labels[dst], ack,
			                       args.stat_out));
		}

	std::ostringstream oss;
	oss << "TotalTimeSec: " << std::fixed << std::setprecision(TOTAL_TIME_DIGITS)
	    << (MPI_Wtime() - t0) << "\n";
	mirror(oss.str());
}

} // namespace gpu_benchmark
