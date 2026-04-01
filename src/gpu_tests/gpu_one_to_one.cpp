/**
 * gpu_one_to_one — перебор всех пар GPU в стиле «мастер + воркеры» (как у
 * Бегаева).
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
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

void cuda_ok(cudaError_t e, const char *msg) {
	if (e != cudaSuccess) {
		std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
		std::abort();
	}
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

struct Args { /* Параметры запуска бенчмарка (CLI-аргументы). */
	size_t nbytes =
		4u * 1000u * 1000u; /* Размер сообщения в байтах (по умолчанию 4 MB). */
	int warmup = 10;		/* Прогревочные итерации (не в статистике). */
	int iters = 50;			/* Измеряемые итерации. */
	Mode mode = Mode::Auto; /* host или auto (device при CUDA-aware MPI). */
	std::string out_path;	/* пусто — только stdout; иначе дублирование в файл (rank 0). */
};

struct Task { /* Задание мастера: пара source -> destination GPU. */
	int src_rank = -1;
	int src_gpu = -1;
	int dst_rank = -1;
	int dst_gpu = -1;
	int stop = 0; /* 1 — завершить воркер. */
};

void help(int rank) {
	if (rank != 0)
		return;
	std::cout << "gpu_one_to_one — all GPU pairs, master-slave\n"
			  << "  --bytes N       size in bytes (default 4 MiB)\n"
			  << "  --warmup N      warmup iterations per pair\n"
			  << "  --iters N       measured iterations per pair\n"
			  << "  --mode M        auto | host\n"
			  << "  --out FILE      also write the same output to FILE (rank 0 only)\n"
			  << "  -o FILE         same as --out\n";
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
		else if (s == "--out" || s == "-o")
			a.out_path = next("--out");
		else if (s == "--help" || s == "-h") {
			help(rank);
			MPI_Finalize();
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
				cuda_ok(cudaMemcpy(h_buf, d_send, args.nbytes,
								   cudaMemcpyDeviceToHost),
						"D2H local");
				cuda_ok(cudaSetDevice(t.dst_gpu), "cudaSetDevice(dst local)");
				cuda_ok(cudaMemcpy(d_recv, h_buf, args.nbytes,
								   cudaMemcpyHostToDevice),
						"H2D local");
			} else { // копирование напрямую между GPU
				cuda_ok(cudaMemcpyPeer(d_recv, t.dst_gpu, d_send, t.src_gpu,
									   args.nbytes),
						"cudaMemcpyPeer");
			}
			return;
		}
		if (is_sender) {
			if (check_host) {
				cuda_ok(cudaMemcpy(h_buf, d_send, args.nbytes,
								   cudaMemcpyDeviceToHost),
						"D2H");
				MPI_Send(h_buf, count, MPI_BYTE, t.dst_rank, tag,
						 MPI_COMM_WORLD);
			} else {
				MPI_Send(d_send, count, MPI_BYTE, t.dst_rank, tag,
						 MPI_COMM_WORLD);
			}
		}
		if (is_receiver) {
			if (check_host) {
				MPI_Recv(h_buf, count, MPI_BYTE, t.src_rank, tag,
						 MPI_COMM_WORLD, &st);
				cuda_ok(cudaMemcpy(d_recv, h_buf, args.nbytes,
								   cudaMemcpyHostToDevice),
						"H2D");
			} else {
				MPI_Recv(d_recv, count, MPI_BYTE, t.src_rank, tag,
						 MPI_COMM_WORLD, &st);
			}
		}
	};

	for (int i = 0; i < args.warmup; ++i)
		do_one(i);

	std::vector<double>
		samples_us; /* длительности итераций (мкс) для статистики ниже */
	samples_us.reserve(
		static_cast<size_t>(std::max(1, args.iters))); /* ёмкость под iters */
	for (int i = 0; i < args.iters; ++i) {
		const double t0 = MPI_Wtime(); /* секунды */
		do_one(1000 + i); /* одна передача; tag не пересекается с прогревом
							 do_one(0..warmup-1) */
		const double t1 = MPI_Wtime(); /* конец интервала, сек */
		if (is_sender || (t.src_rank == t.dst_rank &&
						  is_receiver))			   /* одна выборка/итерация */
			samples_us.push_back((t1 - t0) * 1e6); /* сек -> мкс */
	}

	if (!samples_us.empty()) {
		const double n = static_cast<double>(samples_us.size());
		const double sum = std::accumulate(samples_us.begin(), samples_us.end(),
										   0.0); // сумма всех значений
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

} // namespace

static std::string format_pair_line(int src_rank, int dst_rank, double avg_us,
									double med_us, double min_us, double max_us,
									double var_us) {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(3);
	oss << "pair g" << src_rank << " -> g" << dst_rank << " (r" << src_rank
		<< ":0 -> r" << dst_rank << ":0"
		<< ") avg_us=" << avg_us << " median_us=" << med_us
		<< " min_us=" << min_us << " max_us=" << max_us << " var_us=" << var_us
		<< "\n";
	return oss.str();
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);

	int rank = 0;
	int nproc = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // присваиваем ранг MPI-процесса
	MPI_Comm_size(MPI_COMM_WORLD, &nproc); // присваиваем кол-во MPI-процессов

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

	int local_gpu_count = 0;
	cudaError_t cnt_err = cudaGetDeviceCount(
		&local_gpu_count); // сколько GPU видит именно этот процесс
	if (cnt_err != cudaSuccess)
		local_gpu_count = 0;

	std::vector<int> gpu_counts(nproc, 0);
	/* Каждый rank шлёт 1×MPI_INT (local_gpu_count); root=0 получает массив
	 * gpu_counts[0..nproc-1] по порядку рангов. */
	MPI_Gather(&local_gpu_count, 1, MPI_INT, gpu_counts.data(), 1, MPI_INT, 0,
			   MPI_COMM_WORLD);

	/* Мастер: печать режима и проверка схемы «1 MPI rank = 1 видимый GPU». */
	if (rank == 0) {
		mirror(std::string("CUDA-aware MPI: ") + (cuda_aware ? "yes" : "no") +
			   "\n");
		{
			std::ostringstream oss;
			oss << "Mode: "
				<< (via_host ? "host (through RAM)"
								  : "device (direct GPU pointers)")
				<< "\n";
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

	/* Мастер: полный перебор пар (src_rank, dst_rank), печать метрик; воркеры в
	 * ветке else. Сообщения: тег 1 — структура Task, тег 2 — вектор из 6 double
	 * (метрики). */
	if (rank == 0) {
		{
			std::ostringstream oss;
			oss << "Ranks: " << nproc
				<< ", total GPUs (1 per rank): " << nproc << "\n";
			mirror(oss.str());
		}
		std::cout << std::fixed << std::setprecision(3);
		if (out_file)
			*out_file << std::fixed << std::setprecision(3);

		Task t{};
		for (int src_rank = 0; src_rank < nproc; ++src_rank) {
			for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
				t.src_rank = src_rank;
				t.src_gpu = 0; /* локальный индекс GPU на rank (схема 1 rank = 1 GPU) */
				t.dst_rank = dst_rank;
				t.dst_gpu = 0;
				t.stop = 0;

				/* Локальная пара (один MPI-процесс, два GPU): мастер либо сам
				 * run_task на rank 0, либо шлёт Task на rank и получает ack. */
				if (src_rank == dst_rank) {
					if (src_rank == 0) {
						auto ack = run_task(rank, t, args, via_host);
						if (ack[5] > 0.5) {
							mirror(format_pair_line(src_rank, dst_rank, ack[0],
													ack[1], ack[2], ack[3],
													ack[4]));
						}
					} else {
						MPI_Send(&t, sizeof(Task), MPI_BYTE, src_rank, 1,
								 MPI_COMM_WORLD);
						std::vector<double> ack(6, 0.0);
						MPI_Recv(ack.data(), 6, MPI_DOUBLE, src_rank, 2,
								 MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
						if (ack[5] > 0.5) {
							mirror(format_pair_line(src_rank, dst_rank, ack[0],
													ack[1], ack[2], ack[3],
													ack[4]));
						}
					}
					continue;
				}

				/* Разные ранги: копия Task отправителю и получателю (если это
				 * не мастер). */
				if (src_rank != 0)
					MPI_Send(&t, sizeof(Task), MPI_BYTE, src_rank, 1,
							 MPI_COMM_WORLD);
				if (dst_rank != 0)
					MPI_Send(&t, sizeof(Task), MPI_BYTE, dst_rank, 1,
							 MPI_COMM_WORLD);

				std::vector<double> metric(6, 0.0);
				if (src_rank == 0 || dst_rank == 0) {
					auto ack0 = run_task(rank, t, args, via_host);
					if (ack0[5] > 0.5)
						metric = ack0;
				}

				if (src_rank != 0) {
					std::vector<double> ack(6, 0.0);
					MPI_Recv(ack.data(), 6, MPI_DOUBLE, src_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if (ack[5] > 0.5)
						metric = ack;
				}
				if (dst_rank != 0) {
					std::vector<double> ack(6, 0.0);
					MPI_Recv(ack.data(), 6, MPI_DOUBLE, dst_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if (ack[5] > 0.5)
						metric = ack;
				}

				mirror(format_pair_line(src_rank, dst_rank, metric[0],
										metric[1], metric[2], metric[3],
										metric[4]));
			}
		}

		t.stop = 1;
		for (int r = 1; r < nproc; ++r)
			MPI_Send(&t, sizeof(Task), MPI_BYTE, r, 1, MPI_COMM_WORLD);
	} else {
		while (true) {
			Task t{};
			MPI_Recv(&t, sizeof(Task), MPI_BYTE, 0, 1, MPI_COMM_WORLD,
					 MPI_STATUS_IGNORE);
			if (t.stop)
				break;
			auto ack = run_task(rank, t, args, via_host);
			MPI_Send(ack.data(), 6, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
		}
	}

	MPI_Finalize();
	return 0;
}
