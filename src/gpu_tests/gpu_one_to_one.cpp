/**
 * gpu_one_to_one — перебор всех пар GPU в стиле «мастер + воркеры»
 *
 * Ранг 0 (мастер) формирует задания (src_rank, src_gpu, dst_rank, dst_gpu) для
 * всех пар: for src_rank in ranks for dst_rank in ranks
 *
 * Схема «1 MPI-процесс = 1 GPU»: каждый rank видит ровно один локальный GPU
 * (индекс 0).
 *
 * Режим передачи (--mode):
 *   host — всегда через ОЗУ (D2H -> MPI -> H2D)
 *   auto — указатели device при CUDA-aware MPI, иначе host
 */

#include <mpi.h>

#if defined(OPEN_MPI) && OPEN_MPI
	#include "mpi-ext.h"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void cuda_ok(cudaError_t e, const char *msg) {
	if (e != cudaSuccess) {
		std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
		std::abort();
	}
}

void mpi_ok(int err, const char *msg) {
	if (err == MPI_SUCCESS)
		return;
	char buf[MPI_MAX_ERROR_STRING];
	int len = 0;
	if (MPI_Error_string(err, buf, &len) == MPI_SUCCESS && len > 0)
		std::cerr << msg << ": " << buf << "\n";
	else
		std::cerr << msg << ": MPI error code " << err << "\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
}

bool mpi_cuda_aware() {
	#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
		return MPIX_Query_cuda_support() == 1;
	#else
		return false;
	#endif
}

enum class Mode {
	Auto,
	Host,
};

// Что печатать в строке pair ... (остальные величины всё равно считаются внутри).
enum class StatOut {
	All,
	Avg,
	Median,
	Min,
	Max,
	Var,
};

struct Args { // Параметры запуска бенчмарка (CLI-аргументы).
	size_t nbytes = 4u * 1000u * 1000u; // Размер сообщения в байтах (по умолчанию 4 MB).
	int warmup = 10;		// Прогревочные итерации (не в статистике).
	int iters = 50;			// Измеряемые итерации.
	Mode mode = Mode::Auto; // host или auto (device при CUDA-aware MPI).
	StatOut stat_out = StatOut::All;
	std::string out_path;
};

struct Task { // Задание мастера
	int src_rank = -1;
	int src_gpu = -1;
	int dst_rank = -1;
	int dst_gpu = -1;
	int stop = 0; // 1 — завершить воркер.
};

void help(int rank) {
	if (rank != 0)
		return;
	std::cout << "gpu_one_to_one — all GPU pairs, master-slave\n"
			  << "  --bytes N       size in bytes (default 4 MB)\n"
			  << "  --warmup N      warmup iterations per pair\n"
			  << "  --iters N       measured iterations per pair\n"
			  << "  --mode M        auto | host\n"
			  << "  --stat S        all | avg | median | min | max | var (pair line output)\n"
			  << "  --out FILE      also write the same output to FILE (rank 0 only)\n"
			  << "  -o FILE         same as --out\n"
			  << "Env (layout):\n"
			  << "  SLURM_NODEID   node key source\n";
}

Mode parse_mode(const std::string &s, int rank) {
	if (s == "auto")
		return Mode::Auto;
	if (s == "host")
		return Mode::Host;
	if (rank == 0)
		std::cerr << "unknown --mode: " << s << " (use auto|host)\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
	return Mode::Auto;
}

StatOut parse_stat_out(const std::string &s, int rank) {
	if (s == "all")
		return StatOut::All;
	if (s == "avg")
		return StatOut::Avg;
	if (s == "median")
		return StatOut::Median;
	if (s == "min")
		return StatOut::Min;
	if (s == "max")
		return StatOut::Max;
	if (s == "var")
		return StatOut::Var;
	if (rank == 0)
		std::cerr << "unknown --stat: " << s
				  << " (use all|avg|median|min|max|var)\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
	return StatOut::All;
}

Args parse_args(int argc, char **argv, int rank) {
	Args a;
	for (int i = 1; i < argc; ++i) {
		const std::string s = argv[i];
		auto next = [&](const char *name) -> const char * {
			if (i + 1 >= argc) {
				if (rank == 0)
					std::cerr << "missing value for " << name << "\n";
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			return argv[++i];
		};

		if (s == "--bytes")
			a.nbytes = static_cast<size_t>(
				std::strtoull(next("--bytes"), nullptr, 10));
		else if (s == "--warmup")
			a.warmup = std::atoi(next("--warmup"));
		else if (s == "--iters")
			a.iters = std::atoi(next("--iters"));
		else if (s == "--mode")
			a.mode = parse_mode(next("--mode"), rank);
		else if (s == "--stat")
			a.stat_out = parse_stat_out(next("--stat"), rank);
		else if (s == "--out" || s == "-o")
			a.out_path = next("--out");
		else if (s == "--help" || s == "-h") {
			help(rank);
			mpi_ok(MPI_Finalize(), "MPI_Finalize");
			std::exit(0);
		} else {
			if (rank == 0)
				std::cerr << "unknown arg: " << s << "\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	return a;
}

bool check_host(Mode mode, bool cuda_aware) {
	if (mode == Mode::Host)
		return true;
	return !cuda_aware;
}

/*
 * Одно задание (одна пара GPU). Возврат:
 * {avg_us, median_us, min_us, max_us, var_us, valid_metric}.
 * valid=1 выставляет отправитель (или единственный участник при src==dst на
 * одном rank).
 */
std::vector<double> run_task(int rank, const Task &t, const Args &args,
							 bool check_host) {
	std::vector<double> ack(6, 0.0);
	const bool is_sender = (rank == t.src_rank);
	const bool is_receiver = (rank == t.dst_rank);
	if (!is_sender && !is_receiver)
		return ack;
	if (t.src_rank == t.dst_rank && t.src_gpu == t.dst_gpu) {
		// Диагональ "сам в себя": по договоренности считаем метрики нулевыми.
		ack[5] = 1.0;
		return ack;
	}

	char *d_send = nullptr;
	char *d_recv = nullptr;
	char *h_buf = nullptr;
	const int count = static_cast<int>(args.nbytes);

	if (is_sender) {
		cuda_ok(cudaSetDevice(t.src_gpu), "cudaSetDevice(src)");
		cuda_ok(cudaMalloc(&d_send, args.nbytes), "cudaMalloc(src)");
		cuda_ok(cudaMemset(d_send, 0xAA, args.nbytes), "cudaMemset(src)");
	}
	if (is_receiver) {
		cuda_ok(cudaSetDevice(t.dst_gpu), "cudaSetDevice(dst)");
		cuda_ok(cudaMalloc(&d_recv, args.nbytes), "cudaMalloc(dst)");
		cuda_ok(cudaMemset(d_recv, 0, args.nbytes), "cudaMemset(dst)");
	}
	if (check_host && (is_sender || is_receiver))
		cuda_ok(cudaMallocHost(&h_buf, args.nbytes), "cudaMallocHost");

	MPI_Status st{};

	auto do_one = [&](int tag) {
		if (t.src_rank == t.dst_rank) { // На одном узле
			if (t.src_gpu == t.dst_gpu) // Сам себе
				return;
			cuda_ok(cudaSetDevice(t.src_gpu), "cudaSetDevice(src local)");
			if (check_host) { // копирование через ОЗУ
				cuda_ok(cudaMemcpy(h_buf, d_send, args.nbytes, cudaMemcpyDeviceToHost),
					"D2H local");
				cuda_ok(cudaSetDevice(t.dst_gpu), "cudaSetDevice(dst local)");
				cuda_ok(cudaMemcpy(d_recv, h_buf, args.nbytes, cudaMemcpyHostToDevice),
					"H2D local");
			} else { // копирование напрямую между GPU
				cuda_ok(cudaMemcpyPeer(d_recv, t.dst_gpu, d_send, t.src_gpu, args.nbytes),
						"cudaMemcpyPeer");
			}
			return;
		}
		if (is_sender) {
			if (check_host) {
				cuda_ok(cudaMemcpy(h_buf, d_send, args.nbytes, cudaMemcpyDeviceToHost),
					"D2H");
				mpi_ok(MPI_Send(h_buf, count, MPI_BYTE, t.dst_rank, tag, MPI_COMM_WORLD),
					"MPI_Send (host staging)");
			} else {
				mpi_ok(MPI_Send(d_send, count, MPI_BYTE, t.dst_rank, tag, MPI_COMM_WORLD),
					"MPI_Send (device buffer)");
			}
		}
		if (is_receiver) {
			if (check_host) {
				mpi_ok(MPI_Recv(h_buf, count, MPI_BYTE, t.src_rank, tag, MPI_COMM_WORLD, &st),
					"MPI_Recv (host staging)");
				cuda_ok(cudaMemcpy(d_recv, h_buf, args.nbytes, cudaMemcpyHostToDevice),
					"H2D");
			} else {
				mpi_ok(MPI_Recv(d_recv, count, MPI_BYTE, t.src_rank, tag, MPI_COMM_WORLD, &st),
					"MPI_Recv (device buffer)");
			}
		}
	};

	for (int i = 0; i < args.warmup; ++i)
		do_one(i);

	std::vector<double> samples_us; // длительности итераций (мкс) для статистики ниже
	samples_us.reserve(static_cast<size_t>(std::max(1, args.iters))); // ёмкость под iters
	for (int i = 0; i < args.iters; ++i) {
		const double t0 = MPI_Wtime();
		do_one(1000 + i);
		const double t1 = MPI_Wtime();
		if (is_sender || (t.src_rank == t.dst_rank && is_receiver))
			samples_us.push_back((t1 - t0) * 1e6); // сек -> мкс
	}

	if (!samples_us.empty()) {
		const double n = static_cast<double>(samples_us.size());
		const double sum = std::accumulate(samples_us.begin(), samples_us.end(), 0.0); // сумма всех значений
		const double mean = sum / n;

		auto [it_min, it_max] = std::minmax_element(
			samples_us.begin(),
			samples_us.end()); // находит мин и макс значения
		const double min_v = *it_min;
		const double max_v = *it_max;

		std::vector<double> sorted = samples_us;
		std::sort(sorted.begin(), sorted.end());
		const size_t m = sorted.size() / 2;
		const double median = (sorted.size() % 2 == 0)
								? (sorted[m - 1] + sorted[m]) * 0.5
								: sorted[m];

		double var = 0.0;
		if (samples_us.size() > 1) {
			for (double x : samples_us) {
				const double d = x - mean;
				var += d * d;
			}
			var /= static_cast<double>(samples_us.size() - 1);
		}

		ack[0] = mean;
		ack[1] = median;
		ack[2] = min_v;
		ack[3] = max_v;
		ack[4] = var;
		ack[5] = 1.0;
	}

	if (h_buf)
		cudaFreeHost(h_buf);
	if (d_send)
		cudaFree(d_send);
	if (d_recv)
		cudaFree(d_recv);
	return ack;
}

static std::string host_name() {
	const char *env_host = std::getenv("HOSTNAME");
	if (env_host && *env_host)
		return std::string(env_host);
	char buf[256];
	buf[0] = '\0';
	if (gethostname(buf, sizeof(buf)) == 0) {
		buf[sizeof(buf) - 1] = '\0';
		return std::string(buf);
	}
	return std::string();
}

static int slurm_node_id() {
	const char *s = std::getenv("SLURM_NODEID");
	if (s && *s)
		return std::atoi(s);
	return -1;
}

static std::string format_pair_line(int src_rank, int dst_rank, double avg_us,
									double med_us, double min_us, double max_us,
									double var_us, StatOut stat) {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(3);
	oss << "pair " << src_rank << " -> " << dst_rank << " (r" << src_rank << " -> r"
		<< dst_rank << ") ";
	switch (stat) {
	case StatOut::All:
		oss << "avg_us=" << avg_us << " median_us=" << med_us
			<< " min_us=" << min_us << " max_us=" << max_us << " var_us=" << var_us;
		break;
	case StatOut::Avg:
		oss << "avg_us=" << avg_us;
		break;
	case StatOut::Median:
		oss << "median_us=" << med_us;
		break;
	case StatOut::Min:
		oss << "min_us=" << min_us;
		break;
	case StatOut::Max:
		oss << "max_us=" << max_us;
		break;
	case StatOut::Var:
		oss << "var_us=" << var_us;
		break;
	}
	oss << "\n";
	return oss.str();
}

} // namespace

int main(int argc, char **argv) {
	mpi_ok(MPI_Init(&argc, &argv), "MPI_Init");

	int rank = 0;
	int nproc = 1;
	mpi_ok(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
	mpi_ok(MPI_Comm_size(MPI_COMM_WORLD, &nproc), "MPI_Comm_size");

	const Args args = parse_args(argc, argv, rank);
	const bool cuda_aware = mpi_cuda_aware();
	const bool via_host = check_host(args.mode, cuda_aware);

	std::unique_ptr<std::ofstream> out_file;
	if (rank == 0 && !args.out_path.empty()) {
		out_file = std::make_unique<std::ofstream>(args.out_path);
		if (!out_file->is_open()) {
			std::cerr << "gpu_one_to_one: cannot open --out " << args.out_path
					  << "\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	auto mirror = [&](const std::string &s) {
		if (rank != 0)
			return;
		std::cout << s;
		if (out_file)
			*out_file << s;
	};

	constexpr int HOST_LEN = 64;
	char my_host[HOST_LEN];
	std::vector<char> hosts_recv;
	std::snprintf(my_host, sizeof(my_host), "%s", host_name().c_str());

	constexpr int PCI_LEN = 32;
	char my_pci[PCI_LEN];
	std::vector<char> pci_recv;
	std::snprintf(my_pci, sizeof(my_pci), "n/a");

	int local_gpu_count = 0;
	cudaError_t cnt_err = cudaGetDeviceCount(&local_gpu_count); // сколько GPU видит именно этот процесс
	if (cnt_err != cudaSuccess)
		local_gpu_count = 0;
	std::vector<int> gpu_counts(nproc, 0);
	mpi_ok(MPI_Gather(&local_gpu_count, 1, MPI_INT,
		gpu_counts.data(), 1, MPI_INT,
		0, MPI_COMM_WORLD),
		   "MPI_Gather");
	if (local_gpu_count > 0) {
		cudaError_t set_err = cudaSetDevice(0);
		if (set_err == cudaSuccess) {
			cudaError_t bus_err = cudaDeviceGetPCIBusId(my_pci, sizeof(my_pci), 0);
			if (bus_err != cudaSuccess)
				std::snprintf(my_pci, sizeof(my_pci), "n/a");
		}
	}

	int my_node = slurm_node_id();
	std::vector<int> node_recv;
	
	if (rank == 0) {
		hosts_recv.resize(static_cast<size_t>(nproc) * HOST_LEN);
		pci_recv.resize(static_cast<size_t>(nproc) * PCI_LEN);
		node_recv.resize(static_cast<size_t>(nproc));
	}
	mpi_ok(MPI_Gather(my_host, HOST_LEN, MPI_CHAR,
					  rank == 0 ? hosts_recv.data() : nullptr, HOST_LEN, MPI_CHAR, 0,
					  MPI_COMM_WORLD),
		   "MPI_Gather hostnames");
	mpi_ok(MPI_Gather(my_pci, PCI_LEN, MPI_CHAR,
					  rank == 0 ? pci_recv.data() : nullptr, PCI_LEN, MPI_CHAR, 0,
					  MPI_COMM_WORLD),
		   "MPI_Gather pci bus ids");
	mpi_ok(MPI_Gather(&my_node, 1, MPI_INT,
					  rank == 0 ? node_recv.data() : nullptr, 1, MPI_INT, 0,
					  MPI_COMM_WORLD),
		   "MPI_Gather local node");

	if (rank == 0) {
		{
			std::ostringstream oss;
			oss << "Mode: " << (via_host ? "host" : "GPUDirect") << "\n";
			mirror(oss.str());
		}
		{
			std::ostringstream oss;
			oss << "Hostname: " << host_name() << "\n";
			mirror(oss.str());
		}
		{
			std::ostringstream oss;
			oss << "Bytes: " << args.nbytes << "\n"
				<< "Warmup: " << args.warmup << "\n"
				<< "Iters: " << args.iters << "\n";
			mirror(oss.str());
		}

		for (int r = 0; r < nproc; ++r) {
			if (gpu_counts[r] != 1) {
				std::cerr
					<< "This benchmark expects 1 MPI rank = 1 visible GPU.\n"
					<< "Rank " << r << " sees " << gpu_counts[r]
					<< " GPU(s).\n";
				MPI_Abort(MPI_COMM_WORLD, 1); // errorcode 1 — ненулевой код выхода процесса
			}
		}
	}

	if (rank == 0) {
		{
			std::ostringstream oss;
			oss << "Ranks: " << nproc
				<< ", total GPUs (1 per rank): " << nproc << "\n";
			mirror(oss.str());
		}
		mirror("Rank map:\n");
		for (int r = 0; r < nproc; ++r) {
			const char *h = hosts_recv.data() + static_cast<size_t>(r) * HOST_LEN;
			const char *p = pci_recv.data() + static_cast<size_t>(r) * PCI_LEN;
			std::ostringstream oss;
			oss << "  r" << r << " host=" << h
				<< " local_node=" << node_recv[static_cast<size_t>(r)]
				<< " local_gpu=0"
				<< " pci=" << p << "\n";
			mirror(oss.str());
		}
		std::cout << std::fixed << std::setprecision(3);
		if (out_file)
			*out_file << std::fixed << std::setprecision(3);

		// Реальное суммарное время прогона: от первой пары до последней.
		const double test_t0 = MPI_Wtime();
		Task t{};
		for (int src_rank = 0; src_rank < nproc; ++src_rank) {
			for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
				t.src_rank = src_rank;
				t.src_gpu = 0; // локальный индекс GPU на rank (схема 1 rank = 1 GPU)
				t.dst_rank = dst_rank;
				t.dst_gpu = 0;
				t.stop = 0;
				/* Локальная пара (один MPI-процесс, два GPU): мастер либо сам
				 * run_task на rank 0, либо шлёт Task на rank и получает ack. */
				if (src_rank == dst_rank) {
					if (src_rank == 0) {
						auto ack = run_task(rank, t, args, via_host);
						if (ack[5] == 1.0) { // ack[5] - код валидности результата
							mirror(format_pair_line(src_rank, dst_rank,
													ack[0], ack[1], ack[2], ack[3],
													ack[4], args.stat_out));
						}
					} else {
						mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, src_rank, 1, MPI_COMM_WORLD),
							   "MPI_Send Task (same-rank non-root)");
						std::vector<double> ack(6, 0.0);
						mpi_ok(MPI_Recv(ack.data(), 6, MPI_DOUBLE, src_rank, 2,
										MPI_COMM_WORLD, MPI_STATUS_IGNORE),
							   "MPI_Recv ack (same-rank non-root)");
						if (ack[5] == 1.0) {
							mirror(format_pair_line(src_rank, dst_rank,
													ack[0], ack[1], ack[2], ack[3],
													ack[4], args.stat_out));
						}
					}
					continue;
				}

				/* Разные ранги: копия Task отправителю и получателю (если это
				 * не мастер). */
				else {
					if (src_rank != 0)
						mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, src_rank, 1,
										MPI_COMM_WORLD),
							"MPI_Send Task (src)");
					if (dst_rank != 0)
						mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, dst_rank, 1,
										MPI_COMM_WORLD),
							"MPI_Send Task (dst)");

					std::vector<double> metric(6, 0.0);
					if (src_rank == 0 || dst_rank == 0) {
						auto ack0 = run_task(rank, t, args, via_host);
						if (ack0[5] == 1.0)
							metric = ack0;
					}

					if (src_rank != 0) {
						std::vector<double> ack(6, 0.0);
						mpi_ok(MPI_Recv(ack.data(), 6, MPI_DOUBLE, src_rank, 2,
										MPI_COMM_WORLD, MPI_STATUS_IGNORE),
							"MPI_Recv ack (src)");
						if (ack[5] == 1.0)
							metric = ack;
					}
					if (dst_rank != 0) {
						std::vector<double> ack(6, 0.0);
						mpi_ok(MPI_Recv(ack.data(), 6, MPI_DOUBLE, dst_rank, 2,
										MPI_COMM_WORLD, MPI_STATUS_IGNORE),
							"MPI_Recv ack (dst)");
						if (ack[5] == 1.0)
							metric = ack;
					}

					mirror(format_pair_line(src_rank, dst_rank, metric[0],
											metric[1], metric[2], metric[3],
											metric[4], args.stat_out));
				}
			}
		}
		const double total_elapsed_s = MPI_Wtime() - test_t0;
		{
			std::ostringstream oss;
			oss << "TotalElapsedSec: " << std::fixed << std::setprecision(6)
				<< total_elapsed_s << "\n";
			mirror(oss.str());
		}

		t.stop = 1;
		for (int r = 1; r < nproc; ++r)
			mpi_ok(MPI_Send(&t, sizeof(Task), MPI_BYTE, r, 1, MPI_COMM_WORLD),
				   "MPI_Send stop Task");
	} else {
		while (true) {
			Task t{};
			mpi_ok(MPI_Recv(&t, sizeof(Task), MPI_BYTE, 0, 1, MPI_COMM_WORLD,
							MPI_STATUS_IGNORE),
				   "MPI_Recv Task (worker)");
			if (t.stop)
				break;
			auto ack = run_task(rank, t, args, via_host);
			mpi_ok(MPI_Send(ack.data(), 6, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD),
				   "MPI_Send ack (worker)");
		}
	}

	mpi_ok(MPI_Finalize(), "MPI_Finalize");
	return 0;
}
