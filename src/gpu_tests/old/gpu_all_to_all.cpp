/**
 * gpu_all_to_all — измерение MPI_Alltoall на GPU-буферах (1 rank = 1 GPU).
 *
 * Каждый процесс держит send/recv размером nproc × nbytes: блок i уходит рангу i,
 * от ранга j приходит в блок j (стандартная раскладка MPI_Alltoall).
 *
 * Режим передачи (--mode):
 *   host — staging: D2H → MPI_Alltoall на host → H2D
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

struct Args { // Параметры запуска бенчмарка (CLI-аргументы).
	size_t nbytes = 4u * 1000u * 1000u; // Размер сообщения к каждому рангу (байт), не общий объём.
	int warmup = 10;		// Прогревочные итерации.
	int iters = 50;			// Измеряемые итерации.
	Mode mode = Mode::Auto; // host или auto (device при CUDA-aware MPI).
	std::string out_path;
};

void help(int rank) {
	if (rank != 0)
		return;
	std::cout << "gpu_all_to_all — MPI_Alltoall, 1 rank = 1 GPU\n"
			  << "  --bytes N       message size per peer in bytes (default 4 MB)\n"
			  << "  --warmup N      warmup iterations\n"
			  << "  --iters N       measured iterations\n"
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
 * Одна измеряемая операция: MPI_Alltoall на буферах размера nproc × block.
 * Возврат: {avg_us, median_us, min_us, max_us, var_us, valid}; valid=1 если rank
 * участвовал (всегда 1 для nproc>=1).
 */
std::vector<double> run_alltoall_series(int rank, int nproc, const Args &args,
										bool use_host) {
	std::vector<double> out(6, 0.0);
	const size_t total = static_cast<size_t>(nproc) * args.nbytes;
	const int block = static_cast<int>(args.nbytes);

	cuda_ok(cudaSetDevice(0), "cudaSetDevice");

	char *d_send = nullptr;
	char *d_recv = nullptr;
	cuda_ok(cudaMalloc(&d_send, total), "cudaMalloc(send)");
	cuda_ok(cudaMalloc(&d_recv, total), "cudaMalloc(recv)");
	cuda_ok(cudaMemset(d_send, static_cast<int>(0xA0 + (rank & 0x0F)), total),
		"cudaMemset(send)");
	cuda_ok(cudaMemset(d_recv, 0, total), "cudaMemset(recv)");

	char *h_send = nullptr;
	char *h_recv = nullptr;
	if (use_host) {
		cuda_ok(cudaMallocHost(&h_send, total), "cudaMallocHost(send)");
		cuda_ok(cudaMallocHost(&h_recv, total), "cudaMallocHost(recv)");
	}

	auto do_one = [&]() {
		if (use_host) {
			cuda_ok(cudaMemcpy(h_send, d_send, total, cudaMemcpyDeviceToHost),
				"D2H alltoall send");
			mpi_ok(MPI_Alltoall(h_send, block, MPI_BYTE, h_recv, block, MPI_BYTE,
					MPI_COMM_WORLD),
				"MPI_Alltoall (host)");
			cuda_ok(cudaMemcpy(d_recv, h_recv, total, cudaMemcpyHostToDevice),
				"H2D alltoall recv");
		} else {
			mpi_ok(MPI_Alltoall(d_send, block, MPI_BYTE, d_recv, block, MPI_BYTE,
					MPI_COMM_WORLD),
				"MPI_Alltoall (device)");
		}
	};

	for (int i = 0; i < args.warmup; ++i)
		do_one();

	std::vector<double> samples_us;
	samples_us.reserve(static_cast<size_t>(std::max(1, args.iters)));
	for (int i = 0; i < args.iters; ++i) {
		const double t0 = MPI_Wtime();
		do_one();
		const double t1 = MPI_Wtime();
		samples_us.push_back((t1 - t0) * 1e6); // сек -> мкс
	}

	if (!samples_us.empty()) {
		const double n = static_cast<double>(samples_us.size());
		const double sum = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
		const double mean = sum / n;

		auto [it_min, it_max] = std::minmax_element(
			samples_us.begin(), samples_us.end());
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

		out[0] = mean;
		out[1] = median;
		out[2] = min_v;
		out[3] = max_v;
		out[4] = var;
		out[5] = 1.0;
	}

	if (h_send)
		cudaFreeHost(h_send);
	if (h_recv)
		cudaFreeHost(h_recv);
	cudaFree(d_send);
	cudaFree(d_recv);
	return out;
}

} // namespace

static std::string format_alltoall_line(double avg_us, double med_us, double min_us,
										double max_us, double var_us) {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(3);
	oss << "alltoall collective avg_us=" << avg_us << " median_us=" << med_us
		<< " min_us=" << min_us << " max_us=" << max_us << " var_us=" << var_us
		<< "\n";
	return oss.str();
}

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
			std::cerr << "gpu_all_to_all: cannot open --out " << args.out_path
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
	cudaError_t cnt_err = cudaGetDeviceCount(&local_gpu_count);
	if (cnt_err != cudaSuccess)
		local_gpu_count = 0;

	std::vector<int> gpu_counts(nproc, 0);
	/* Каждый rank шлёт 1×MPI_INT (local_gpu_count); root=0 получает массив
	 * gpu_counts[0..nproc-1] по порядку рангов. */
	mpi_ok(MPI_Gather(&local_gpu_count, 1, MPI_INT, gpu_counts.data(), 1, MPI_INT,
			0, MPI_COMM_WORLD),
		   "MPI_Gather");

	// Мастер: печать режима и проверка схемы «1 MPI rank = 1 видимый GPU».
	if (rank == 0) {
		mirror(std::string("CUDA-aware MPI: ") + (cuda_aware ? "yes" : "no") +
			   "\n");
		{
			std::ostringstream oss;
			oss << "Mode: " << (via_host ? "host" : "GPUDirect") << "\n";
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
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
	}

	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");

	if (rank == 0) {
		{
			std::ostringstream oss;
			oss << "Ranks: " << nproc
				<< ", total GPUs (1 per rank): " << nproc << "\n";
			mirror(oss.str());
		}
		mirror("Scheme: GPU all-to-all (MPI_Alltoall)\n");
		std::cout << std::fixed << std::setprecision(3);
		if (out_file)
			*out_file << std::fixed << std::setprecision(3);
	}

	auto stats = run_alltoall_series(rank, nproc, args, via_host);

	if (rank == 0 && stats[5] > 0.5)
		mirror(format_alltoall_line(stats[0], stats[1], stats[2], stats[3],
				stats[4]));

	mpi_ok(MPI_Finalize(), "MPI_Finalize");
	return 0;
}
