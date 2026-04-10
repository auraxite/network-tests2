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

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
	#include "mpi-ext.h"
#endif
#include <numeric>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "netcdf_writer.h"


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

enum class Timer {
	All,
	Mpi,
	Cpu,
	Cuda,
};

enum class StatOut {
	All,
	Avg,
	Med,
	Min,
	Max,
	Std,
	Var,
};

constexpr int ACK_FIELDS = 25; // 6 selected + valid + 6 CPU + 6 MPI + 6 GPU

struct Args { // Параметры запуска бенчмарка (CLI-аргументы).
	size_t nbytes = 4u * 1000u * 1000u; // Размер сообщения в байтах (по умолчанию 4 MB).
	int warmup = 10;		// Прогревочные итерации (не в статистике).
	int iters = 50;			// Измеряемые итерации.
	Mode mode = Mode::Auto; // auto (device при CUDA-aware MPI) или host
	Timer timer = Timer::All; // источник тайминга для строки pair
	StatOut stat_out = StatOut::All;
	std::string out_path;
	bool dump_raw_samples = true; // Сохранять сырые выборки samples_us по парам.
};

struct Task { // Задание мастера
	int src_rank = -1;
	int src_gpu = -1;
	int dst_rank = -1;
	int dst_gpu = -1;
	int stop = 0; // 1 — завершить воркер
};

void help(int rank) {
	if (rank != 0)
		return;
	std::cout << "gpu_one_to_one — all GPU pairs, master-slave\n"
			  << "  --bytes N       size in bytes (default 4 MB)\n"
			  << "  --warmup N      warmup iterations per pair\n"
			  << "  --iters N       measured iterations per pair\n"
			  << "  --mode M        auto | host\n"
			  << "  --timer T       all | mpi | cpu | cuda (default all)\n"
			  << "  --stat S        all | avg | med | min | max | var | std (pair line output)\n"
			  << "  --out FILE, -o FILE  also write the same output to FILE (rank 0 only)\n";
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

Timer parse_timer(const std::string &s, int rank) {
	if (s == "all")
		return Timer::All;
	if (s == "mpi")
		return Timer::Mpi;
	if (s == "cpu")
		return Timer::Cpu;
	if (s == "cuda" || s == "gpu")
		return Timer::Cuda;
	if (rank == 0)
		std::cerr << "unknown --timer: " << s << " (use all|mpi|cpu|cuda)\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
	return Timer::All;
}

static const char *timer_to_string(Timer t) {
	switch (t) {
	case Timer::All: return "all";
	case Timer::Mpi: return "mpi";
	case Timer::Cpu: return "cpu";
	case Timer::Cuda: return "cuda";
	}
	return "all";
}

StatOut parse_stat_out(const std::string &s, int rank) {
	if (s == "all")
		return StatOut::All;
	if (s == "avg")
		return StatOut::Avg;
	if (s == "med")
		return StatOut::Med;
	if (s == "min")
		return StatOut::Min;
	if (s == "max")
		return StatOut::Max;
	if (s == "var")
		return StatOut::Var;
	if (s == "std")
		return StatOut::Std;
	if (rank == 0)
		std::cerr << "unknown --stat: " << s
				  << " (use all|avg|med|min|max|var|std)\n";
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
		else if (s == "--timer")
			a.timer = parse_timer(next("--timer"), rank);
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
	if (a.warmup < 0) {
		if (rank == 0)
			std::cerr << "--warmup must be >= 0\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (a.iters <= 0) {
		if (rank == 0)
			std::cerr << "--iters must be > 0\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	return a;
}

bool check_host(Mode mode, bool cuda_aware) {
	if (mode == Mode::Host)
		return true;
	return !cuda_aware;
}

static void append_raw_samples(const Args &args, int rank, const Task &t,
							   const std::vector<double> &samples_us);

static double monotonic_now_us() {
	timespec ts{};
	#if defined(CLOCK_MONOTONIC_RAW)
		const clockid_t clk_id = CLOCK_MONOTONIC_RAW;
	#else
		const clockid_t clk_id = CLOCK_MONOTONIC;
	#endif
	if (clock_gettime(clk_id, &ts) != 0)
		return 0.0;
	return static_cast<double>(ts.tv_sec) * 1e6 +
		   static_cast<double>(ts.tv_nsec) * 1e-3;
}

static void fill_stats6(const std::vector<double> &samples, double *out6) {
	const double n = static_cast<double>(samples.size());
	const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
	const double avg = sum / n;
	auto [it_min, it_max] = std::minmax_element(samples.begin(), samples.end());
	const double min_v = *it_min;
	const double max_v = *it_max;
	std::vector<double> sorted = samples;
	std::sort(sorted.begin(), sorted.end());
	const size_t m = sorted.size() / 2;
	const double med = (sorted.size() % 2 == 0)
							? (sorted[m - 1] + sorted[m]) * 0.5
							: sorted[m];
	double var = 0.0;
	if (samples.size() > 1) {
		for (double x : samples) {
			const double d = x - avg;
			var += d * d;
		}
		var /= static_cast<double>(samples.size() - 1);
	}
	const double stddev = std::sqrt(var);
	out6[0] = avg;
	out6[1] = med;
	out6[2] = min_v;
	out6[3] = max_v;
	out6[4] = var;
	out6[5] = stddev;
}

static std::string short_host(const std::string &host) {
	const size_t dot = host.find('.');
	if (dot == std::string::npos)
		return host;
	return host.substr(0, dot);
}

static std::string host_node_token(const std::string &host) {
	const std::string sh = short_host(host);
	size_t pos = sh.size();
	while (pos > 0 && std::isdigit(static_cast<unsigned char>(sh[pos - 1])))
		--pos;
	if (pos < sh.size())
		return sh.substr(pos);
	return sh;
}

static std::vector<std::string> build_rank_labels(const std::vector<char> &hosts_recv,
										   int nproc, int host_len) {
	std::vector<std::string> labels(static_cast<size_t>(nproc));
	std::vector<std::string> seen_hosts;
	std::vector<int> seen_counts;
	for (int r = 0; r < nproc; ++r) {
		const char *h = hosts_recv.data() + static_cast<size_t>(r) * static_cast<size_t>(host_len);
		const std::string sh = short_host(std::string(h));
		int local_idx = 0;
		bool found = false;
		for (size_t i = 0; i < seen_hosts.size(); ++i) {
			if (seen_hosts[i] == sh) {
				local_idx = seen_counts[i];
				seen_counts[i] += 1;
				found = true;
				break;
			}
		}
		if (!found) {
			seen_hosts.push_back(sh);
			seen_counts.push_back(1);
			local_idx = 0;
		}
		std::ostringstream oss;
		oss << host_node_token(sh) << "." << local_idx;
		labels[static_cast<size_t>(r)] = oss.str();
	}
	return labels;
}

/*
 * Одно задание (одна пара GPU). Возврат:
 * {avg_us, med_us, min_us, max_us, var_us, std_us, valid_metric}.
 * valid=1 выставляет отправитель (или единственный участник при src==dst на
 * одном rank).
 */
std::vector<double> run_task(int rank, const Task &t, const Args &args,
							 bool check_host) {
	std::vector<double> ack(ACK_FIELDS, 0.0);
	const bool is_sender = (rank == t.src_rank);
	const bool is_receiver = (rank == t.dst_rank);
	if (!is_sender && !is_receiver)
		return ack;
	if (t.src_rank == t.dst_rank && t.src_gpu == t.dst_gpu) {
		// Диагональ "сам в себя": по договоренности считаем метрики нулевыми.
		ack[6] = 1.0;
		return ack;
	}

	char *d_send = nullptr; // буфер в памяти GPU для отправки 
	char *d_recv = nullptr; // буфер в памяти GPU для приёма
	char *h_buf = nullptr; // буфер в host RAM для режима через CPU
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
	if (check_host)
		cuda_ok(cudaMallocHost(&h_buf, args.nbytes), "cudaMallocHost");

	MPI_Status st{};

	auto do_one = [&](int tag) {
		if (t.src_rank == t.dst_rank) { // На одном узле
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

	std::vector<double> samples_mpi_us;
	std::vector<double> samples_cpu_us;
	std::vector<double> samples_gpu_us;
	cudaEvent_t ev_start = nullptr;
	cudaEvent_t ev_stop = nullptr;
	if (is_sender) {
		cuda_ok(cudaSetDevice(t.src_gpu), "cudaSetDevice(src timing)");
		cuda_ok(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
		cuda_ok(cudaEventCreate(&ev_stop), "cudaEventCreate(stop)");
	}
	for (int i = 0; i < args.warmup; ++i)
		do_one(i);
	samples_mpi_us.reserve(static_cast<size_t>(std::max(1, args.iters)));
	samples_cpu_us.reserve(static_cast<size_t>(std::max(1, args.iters)));
	samples_gpu_us.reserve(static_cast<size_t>(std::max(1, args.iters)));
	for (int i = 0; i < args.iters; ++i) {
		const double t0_mpi = MPI_Wtime();
		const double t0_clk = monotonic_now_us();
		if (is_sender)
			cuda_ok(cudaEventRecord(ev_start), "cudaEventRecord(start)");
		do_one(1000 + i);
		if (is_sender) {
			cuda_ok(cudaEventRecord(ev_stop), "cudaEventRecord(stop)");
			cuda_ok(cudaEventSynchronize(ev_stop), "cudaEventSynchronize(stop)");
			float elapsed_ms = 0.0f;
			cuda_ok(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop),
					"cudaEventElapsedTime");
			samples_gpu_us.push_back(static_cast<double>(elapsed_ms) * 1e3); // ms -> us
			const double t1_mpi = MPI_Wtime();
			const double t1_clk = monotonic_now_us();
			samples_mpi_us.push_back((t1_mpi - t0_mpi) * 1e6); // s -> us
			samples_cpu_us.push_back(t1_clk - t0_clk);
		}
	}
	if (is_sender) {
		switch (args.timer) {
		case Timer::All:
			append_raw_samples(args, rank, t, samples_mpi_us);
			break;
		case Timer::Mpi:
			append_raw_samples(args, rank, t, samples_mpi_us);
			break;
		case Timer::Cpu:
			append_raw_samples(args, rank, t, samples_cpu_us);
			break;
		case Timer::Cuda:
			append_raw_samples(args, rank, t, samples_gpu_us);
			break;
		}
	}

	if (!samples_gpu_us.empty() && !samples_mpi_us.empty() && !samples_cpu_us.empty()) {
		switch (args.timer) {
		case Timer::All:
			fill_stats6(samples_mpi_us, ack.data());
			break;
		case Timer::Mpi:
			fill_stats6(samples_mpi_us, ack.data());
			break;
		case Timer::Cpu:
			fill_stats6(samples_cpu_us, ack.data());
			break;
		case Timer::Cuda:
			fill_stats6(samples_gpu_us, ack.data());
			break;
		}
		fill_stats6(samples_cpu_us, ack.data() + 7);
		fill_stats6(samples_mpi_us, ack.data() + 13);
		fill_stats6(samples_gpu_us, ack.data() + 19);
		ack[6] = 1.0;
	}
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
	return ack;
}

static std::string format_pair_line(const char *line_name,
									const std::string &src_label, const std::string &dst_label,
									double avg_us,
									double med_us, double min_us, double max_us,
									double var_us, double std_us, StatOut stat) {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(3);
	oss << line_name << " " << src_label << " -> " << dst_label << " ";
	switch (stat) {
	case StatOut::All:
		oss << "avg_us=" << avg_us << " med_us=" << med_us
			<< " min_us=" << min_us << " max_us=" << max_us
			<< " var_us=" << var_us << " std_us=" << std_us;
		break;
	case StatOut::Avg:
		oss << "avg_us=" << avg_us;
		break;
	case StatOut::Med:
		oss << "med_us=" << med_us;
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
	case StatOut::Std:
		oss << "std_us=" << std_us;
		break;
	}
	oss << "\n";
	return oss.str();
}

static void append_raw_samples(const Args &args, int rank, const Task &t,
							   const std::vector<double> &samples_us) {
	if (!args.dump_raw_samples || args.out_path.empty() || samples_us.empty())
		return;
	(void)rank;
	std::string base = args.out_path;
	std::string dir = ".";
	const size_t slash = base.find_last_of('/');
	if (slash != std::string::npos) {
		dir = base.substr(0, slash);
		base = base.substr(slash + 1);
	}
	const std::string suffix = ".txt";
	if (base.size() >= suffix.size() &&
		base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
		base.erase(base.size() - suffix.size());
	}
	const std::string raw_dir = dir + "/raw";
	if (mkdir(raw_dir.c_str(), 0775) != 0 && errno != EEXIST)
		return;
	std::ostringstream path;
	path << raw_dir << "/" << base
		 << "_src" << t.src_rank << "_dst" << t.dst_rank << ".raw";
	std::ofstream out(path.str(), std::ios::app);
	if (!out.is_open())
		return;
	for (size_t i = 0; i < samples_us.size(); ++i) {
		out << std::fixed << std::setprecision(3) << samples_us[i] << "\n";
	}
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

	constexpr int HOST_LEN = MPI_MAX_PROCESSOR_NAME;
	char my_host[HOST_LEN];
	std::vector<char> hosts_recv;
	{
		int name_len = 0;
		mpi_ok(MPI_Get_processor_name(my_host, &name_len),
			   "MPI_Get_processor_name");
		my_host[HOST_LEN - 1] = '\0';
	}

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

	if (rank == 0) {
		hosts_recv.resize(static_cast<size_t>(nproc) * HOST_LEN);
		pci_recv.resize(static_cast<size_t>(nproc) * PCI_LEN);
	}
	mpi_ok(MPI_Gather(my_host, HOST_LEN, MPI_CHAR,
					  rank == 0 ? hosts_recv.data() : nullptr, HOST_LEN, MPI_CHAR, 0,
					  MPI_COMM_WORLD),
		   "MPI_Gather hostnames");
	mpi_ok(MPI_Gather(my_pci, PCI_LEN, MPI_CHAR,
					  rank == 0 ? pci_recv.data() : nullptr, PCI_LEN, MPI_CHAR, 0,
					  MPI_COMM_WORLD),
		   "MPI_Gather pci bus ids");

	if (rank == 0) {
		{
			std::ostringstream oss;
			oss << "Mode: " << (via_host ? "host" : "auto") << "\n";
			mirror(oss.str());
		}
		{
			std::ostringstream oss;
			oss << "Timer: " << timer_to_string(args.timer) << "\n";
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
			if (gpu_counts[r] <= 0) {
				std::cerr
					<< "This benchmark expects at least 1 visible GPU per MPI rank.\n"
					<< "Rank " << r << " sees " << gpu_counts[r]
					<< " GPU(s).\n";
				MPI_Abort(MPI_COMM_WORLD, 1); // errorcode 1 — ненулевой код выхода процесса
			}
		}

		std::ostringstream oss;
		int total_visible_gpus = 0;
		for (int v : gpu_counts)
			total_visible_gpus += v;
		oss << "Ranks: " << nproc
			<< ", total visible GPUs: " << total_visible_gpus << "\n";
		mirror(oss.str());

		mirror("Rank map:\n");
		for (int r = 0; r < nproc; ++r) {
			const char *h = hosts_recv.data() + static_cast<size_t>(r) * HOST_LEN;
			const char *p = pci_recv.data() + static_cast<size_t>(r) * PCI_LEN;
			std::ostringstream oss;
			oss << "  r" << r << " hostname=" << h
				<< " local_gpu=0"
				<< " visible_gpus=" << gpu_counts[static_cast<size_t>(r)]
				<< " pci=" << p << "\n";
			mirror(oss.str());
		}
		const std::vector<std::string> rank_labels = build_rank_labels(hosts_recv, nproc, HOST_LEN);
		std::cout << std::fixed << std::setprecision(3);
		if (out_file)
			*out_file << std::fixed << std::setprecision(3);
		NetcdfBundle nc = netcdf_open_bundle(args.out_path, args.nbytes, args.iters, nproc);

		// Реальное суммарное время прогона: от первой пары до последней.
		const double test_t0 = MPI_Wtime();
		Task t{};
		for (int src_rank = 0; src_rank < nproc; ++src_rank) {
			for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
				t.src_rank = src_rank;
				t.src_gpu = 0; // локальный GPU для rank (в режиме 1 rank = 1 node используем GPU 0)
				t.dst_rank = dst_rank;
				t.dst_gpu = 0; // локальный GPU для rank (в режиме 1 rank = 1 node используем GPU 0)
				t.stop = 0;
				// Диагональ src==dst не измеряем: сразу печатаем нулевые метрики.
				if (src_rank == dst_rank) {
					std::vector<double> metric(ACK_FIELDS, 0.0);
					netcdf_store_pair(nc, src_rank, dst_rank, metric);
					if (args.timer == Timer::All) {
						mirror(format_pair_line("pair_mpi",
												rank_labels[static_cast<size_t>(src_rank)],
												rank_labels[static_cast<size_t>(dst_rank)],
												0.0, 0.0, 0.0, 0.0,
												0.0, 0.0, args.stat_out));
						mirror(format_pair_line("pair_cpu",
												rank_labels[static_cast<size_t>(src_rank)],
												rank_labels[static_cast<size_t>(dst_rank)],
												0.0, 0.0, 0.0, 0.0,
												0.0, 0.0, args.stat_out));
						mirror(format_pair_line("pair_cuda",
												rank_labels[static_cast<size_t>(src_rank)],
												rank_labels[static_cast<size_t>(dst_rank)],
												0.0, 0.0, 0.0, 0.0,
												0.0, 0.0, args.stat_out));
					} else {
						mirror(format_pair_line("pair",
												rank_labels[static_cast<size_t>(src_rank)],
												rank_labels[static_cast<size_t>(dst_rank)],
												0.0, 0.0, 0.0, 0.0,
												0.0, 0.0, args.stat_out));
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

					std::vector<double> metric(ACK_FIELDS, 0.0);
					if (src_rank == 0 || dst_rank == 0) {
						auto ack0 = run_task(rank, t, args, via_host);
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
					}
					if (dst_rank != 0) {
						std::vector<double> ack(ACK_FIELDS, 0.0);
						mpi_ok(MPI_Recv(ack.data(), ACK_FIELDS, MPI_DOUBLE, dst_rank, 2,
										MPI_COMM_WORLD, MPI_STATUS_IGNORE),
							"MPI_Recv ack (dst)");
						if (ack[6] == 1.0)
							metric = ack;
					}

					if (args.timer == Timer::All) {
						mirror(format_pair_line("pair_mpi",
												rank_labels[static_cast<size_t>(src_rank)],
												rank_labels[static_cast<size_t>(dst_rank)], metric[13],
												metric[14], metric[15], metric[16],
												metric[17], metric[18], args.stat_out));
						mirror(format_pair_line("pair_cpu",
												rank_labels[static_cast<size_t>(src_rank)],
												rank_labels[static_cast<size_t>(dst_rank)], metric[7],
												metric[8], metric[9], metric[10],
												metric[11], metric[12], args.stat_out));
						mirror(format_pair_line("pair_cuda",
												rank_labels[static_cast<size_t>(src_rank)],
												rank_labels[static_cast<size_t>(dst_rank)], metric[19],
												metric[20], metric[21], metric[22],
												metric[23], metric[24], args.stat_out));
					} else {
						mirror(format_pair_line("pair",
												rank_labels[static_cast<size_t>(src_rank)],
												rank_labels[static_cast<size_t>(dst_rank)], metric[0],
												metric[1], metric[2], metric[3],
												metric[4], metric[5], args.stat_out));
					}
					netcdf_store_pair(nc, src_rank, dst_rank, metric);
				}
			}
		}
		const double total_elapsed_s = MPI_Wtime() - test_t0;
		{
			std::ostringstream oss;
			oss << "TotalTimeSec: " << std::fixed << std::setprecision(6)
				<< total_elapsed_s << "\n";
			mirror(oss.str());
		}
		netcdf_flush_and_close(nc);

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
			mpi_ok(MPI_Send(ack.data(), ACK_FIELDS, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD),
				   "MPI_Send ack (worker)");
		}
	}

	mpi_ok(MPI_Finalize(), "MPI_Finalize");
	return 0;
}
