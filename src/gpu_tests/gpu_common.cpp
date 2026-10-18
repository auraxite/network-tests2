#include "gpu_common.hpp"

#include <algorithm>  // std::sort, std::minmax_element, std::fill
#include <cctype>     // std::isdigit, std::isalnum
#include <cerrno>     // errno после mkdir
#include <cmath>      // std::sqrt
#include <cstdlib>    // std::strtoull, std::atoi, std::exit, std::abort
#include <fstream>
#include <iomanip>    // std::setprecision, std::fixed
#include <iostream>
#include <numeric>    // std::accumulate
#include <pthread.h>  // pthread_setaffinity_np
#include <sched.h>    // sched_getaffinity, sched_getcpu
#include <sys/stat.h> // mkdir

#if defined(OPEN_MPI) && OPEN_MPI
#include "mpi-ext.h"
#endif

namespace gpu_benchmark {

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

// ------------------------------------------------------------------ //
// Аргументы командной строки                                          //
// ------------------------------------------------------------------ //

void help(int rank) {
	if (rank != 0)
		return;
	std::cout
		<< "gpu — GPU pair latency (one_to_one | one_to_one_v2 | all_to_all | one_to_all | all_to_one)\n"
		<< "  --bytes N       message size in bytes (default 4 MB)\n"
		<< "  --warmup N      warmup iterations per pair\n"
		<< "  --iters N       measured iterations per pair\n"
		<< "  --env E         auto | host\n"
		<< "  --mode M        one_to_one | one_to_one_v2 | all_to_all | one_to_all | all_to_one\n"
		<< "                  (default one_to_one)\n"
		<< "  --order O       one_to_one_v2: seq | random (default seq)\n"
		<< "  --seed N        one_to_one_v2: seed for --order random (default 1)\n"
		<< "  --cpu-log       one_to_one_v2: CPU of sender/receiver per iteration (raw/*.scpu, *.rcpu)\n"
		<< "  --prespin-sec S one_to_one_v2: busy-spin S seconds on every rank before the first pair\n"
		<< "  --pin-main K    pin the main thread to the K-th CPU (from 0) of its mask after init\n"
		<< "  --stat S        all | avg | med | min | max | var | std\n"
		<< "  --out FILE, -o FILE  write output to FILE (rank 0 only)\n"
		<< "  --debug, -d     verbose debug logs to stderr\n"
		<< "\n"
		<< "UCX defaults applied before MPI_Init for --env auto "
		   "(override via env vars):\n"
		<< "  UCX_IB_GPU_DIRECT_RDMA=y   enable GPUDirect RDMA\n"
		<< "  UCX_RNDV_SCHEME=auto       let UCX pick get/put zcopy or pipeline\n"
		<< "  UCX_MEMTYPE_CACHE          not set — unused in UCX 1.15, on by default\n"
		<< "  UCX_TLS=cuda_copy,cuda_ipc,<ib>,cma,sm,self\n"
		<< "  UCX_RNDV_THRESH            not set — UCX picks eager/rendezvous\n"
		<< "                             by size (forcing 0 hurts small msgs)\n";
}

[[noreturn]] static void bad_arg(int rank, const std::string &msg) {
	if (rank == 0)
		std::cerr << msg << "\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
	std::exit(1);
}

Mode parse_mode(const std::string &s, int rank) {
	if (s == "one_to_one" || s == "sequential" || s == "1to1")
		return Mode::OneToOne;
	if (s == "all_to_all" || s == "alltoall" || s == "parallel")
		return Mode::AllToAll;
	if (s == "one_to_all") return Mode::OneToAll;
	if (s == "all_to_one") return Mode::AllToOne;
	if (s == "one_to_one_v2") return Mode::OneToOneV2;
	bad_arg(rank, "unknown --mode: " + s +
	              " (use one_to_one|one_to_one_v2|all_to_all|one_to_all|all_to_one)");
}

const char *mode_to_string(Mode mode) {
	switch (mode) {
	case Mode::AllToAll: return "all_to_all";
	case Mode::OneToAll: return "one_to_all";
	case Mode::AllToOne: return "all_to_one";
	case Mode::OneToOneV2: return "one_to_one_v2";
	default:             return "one_to_one";
	}
}

Env parse_env(const std::string &s, int rank) {
	if (s == "auto") return Env::Auto;
	if (s == "host") return Env::Host;
	bad_arg(rank, "unknown --env: " + s + " (use auto|host)");
}

StatOut parse_stat_out(const std::string &s, int rank) {
	if (s == "all") return StatOut::All;
	if (s == "avg") return StatOut::Avg;
	if (s == "med") return StatOut::Med;
	if (s == "min") return StatOut::Min;
	if (s == "max") return StatOut::Max;
	if (s == "var") return StatOut::Var;
	if (s == "std") return StatOut::Std;
	bad_arg(rank, "unknown --stat: " + s + " (use all|avg|med|min|max|var|std)");
}

Args parse_args(int argc, char **argv, int rank) {
	Args a;
	for (int i = 1; i < argc; ++i) {
		const std::string s = argv[i];
		auto next = [&](const char *name) -> const char * {
			if (i + 1 >= argc)
				bad_arg(rank, std::string("missing value for ") + name);
			return argv[++i];
		};

		if (s == "--bytes")
			a.nbytes = static_cast<size_t>(std::strtoull(next("--bytes"), nullptr, 10));
		else if (s == "--warmup")
			a.warmup = std::atoi(next("--warmup"));
		else if (s == "--iters")
			a.iters = std::atoi(next("--iters"));
		else if (s == "--env")
			a.env = parse_env(next("--env"), rank);
		else if (s == "--mode")
			a.mode = parse_mode(next("--mode"), rank);
		else if (s == "--stat")
			a.stat_out = parse_stat_out(next("--stat"), rank);
		else if (s == "--out" || s == "-o")
			a.out_path = next("--out");
		else if (s == "--debug" || s == "-d")
			a.debug = true;
		else if (s == "--order") {
			const std::string o = next("--order");
			if (o == "random") a.random_order = true;
			else if (o == "seq") a.random_order = false;
			else bad_arg(rank, "unknown --order: " + o + " (use seq|random)");
		} else if (s == "--seed")
			a.seed = static_cast<unsigned>(std::strtoul(next("--seed"), nullptr, 10));
		else if (s == "--cpu-log")
			a.cpu_log = true;
		else if (s == "--prespin-sec")
			a.prespin_sec = std::atof(next("--prespin-sec"));
		else if (s == "--pin-main")
			a.pin_main = std::atoi(next("--pin-main"));
		else if (s == "--help" || s == "-h") {
			help(rank);
			mpi_ok(MPI_Finalize(), "MPI_Finalize");
			std::exit(0);
		} else
			bad_arg(rank, "unknown arg: " + s);
	}
	if (a.warmup < 0) bad_arg(rank, "--warmup must be >= 0");
	if (a.iters <= 0) bad_arg(rank, "--iters must be > 0");
	if (a.nbytes == 0) bad_arg(rank, "--bytes must be > 0");
	return a;
}

bool use_host_path(Env env, bool cuda_aware) {
	return env == Env::Host || !cuda_aware;
}

void debug_log(bool enabled, int rank, const std::string &msg) {
	if (enabled)
		std::cerr << "[DBG r" << rank << "] " << msg << "\n";
}

// ------------------------------------------------------------------ //
// Метки, статистики, вывод                                            //
// ------------------------------------------------------------------ //

std::vector<std::string> build_rank_labels(const std::vector<char> &hosts_recv,
                                           int nproc, int host_len) {
	std::vector<std::string> labels;
	std::vector<std::string> seen_hosts;
	std::vector<int>         seen_counts;
	for (int r = 0; r < nproc; ++r) {
		std::string host(hosts_recv.data() + static_cast<size_t>(r) * host_len);
		host = host.substr(0, host.find('.'));                  // короткое имя узла

		size_t k = 0;
		while (k < seen_hosts.size() && seen_hosts[k] != host) ++k;
		if (k == seen_hosts.size()) {
			seen_hosts.push_back(host);
			seen_counts.push_back(0);
		}
		const int local_idx = seen_counts[k]++;

		size_t pos = host.size();                               // цифры в конце имени
		while (pos > 0 && std::isdigit(static_cast<unsigned char>(host[pos - 1]))) --pos;
		const std::string token = pos < host.size() ? host.substr(pos) : host;

		labels.push_back(token + "." + std::to_string(local_idx));
	}
	return labels;
}

void fill_ack(const std::vector<double> &samples, double *ack) {
	std::fill(ack, ack + ACK_FIELDS, 0.0);
	if (samples.empty()) return;

	const double n   = static_cast<double>(samples.size());
	const double avg = std::accumulate(samples.begin(), samples.end(), 0.0) / n;
	const auto [it_min, it_max] = std::minmax_element(samples.begin(), samples.end());

	std::vector<double> sorted = samples;
	std::sort(sorted.begin(), sorted.end());
	const size_t m   = sorted.size() / 2;
	const double med = sorted.size() % 2 == 0 ? (sorted[m - 1] + sorted[m]) * 0.5 : sorted[m];

	double var = 0.0;
	if (samples.size() > 1) {
		for (double x : samples) var += (x - avg) * (x - avg);
		var /= static_cast<double>(samples.size() - 1);
	}

	ack[0] = avg;
	ack[1] = med;
	ack[2] = *it_min;
	ack[3] = *it_max;
	ack[4] = var;
	ack[5] = std::sqrt(var);
	ack[6] = 1.0; // valid
}

std::string print_pair_line(const char *line_name, const std::string &src_label,
                            const std::string &dst_label, const double *ack,
                            StatOut stat) {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(REPORT_DIGITS);
	oss << line_name << " " << src_label << " -> " << dst_label << " ";
	switch (stat) {
	case StatOut::All:
		oss << "avg_us=" << ack[0] << " med_us=" << ack[1]
		    << " min_us=" << ack[2] << " max_us=" << ack[3]
		    << " var_us=" << ack[4] << " std_us=" << ack[5];
		break;
	case StatOut::Avg: oss << "avg_us=" << ack[0]; break;
	case StatOut::Med: oss << "med_us=" << ack[1]; break;
	case StatOut::Min: oss << "min_us=" << ack[2]; break;
	case StatOut::Max: oss << "max_us=" << ack[3]; break;
	case StatOut::Var: oss << "var_us=" << ack[4]; break;
	case StatOut::Std: oss << "std_us=" << ack[5]; break;
	}
	oss << "\n";
	return oss.str();
}

int pin_thread_to_mask_cpu(int k) {
	cpu_set_t mask;
	CPU_ZERO(&mask);
	if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return -1;
	for (int c = 0; c < CPU_SETSIZE; ++c) {
		if (!CPU_ISSET(c, &mask) || k-- > 0) continue;
		cpu_set_t one;
		CPU_ZERO(&one);
		CPU_SET(c, &one);
		return pthread_setaffinity_np(pthread_self(), sizeof(one), &one) == 0 ? c : -1;
	}
	return -1;
}

int current_cpu() { return sched_getcpu(); }

void append_raw_samples(const Args &args, const std::string &src_label,
                        const std::string &dst_label,
                        const std::vector<double> &samples_us) {
	append_raw_series(args, src_label, dst_label, samples_us, ".raw", REPORT_DIGITS);
}

void append_raw_series(const Args &args, const std::string &src_label,
                       const std::string &dst_label, const std::vector<double> &values,
                       const std::string &ext, int digits) {
	if (!args.save_raw_samples || args.out_path.empty() || values.empty())
		return;

	const auto sanitize = [](const std::string &lab) {
		std::string out;
		for (unsigned char c : lab)
			out.push_back(std::isalnum(c) || c == '.' || c == '-' || c == '_'
			                  ? static_cast<char>(c) : '_');
		return out.empty() ? std::string("nolabel") : out;
	};

	std::string base = args.out_path;
	std::string dir  = ".";
	const size_t slash = base.find_last_of('/');
	if (slash != std::string::npos) {
		dir  = base.substr(0, slash);
		base = base.substr(slash + 1);
	}
	const std::string suffix = ".txt";
	if (base.size() >= suffix.size() &&
	    base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0)
		base.erase(base.size() - suffix.size());

	const std::string raw_dir = dir + "/raw";
	if (mkdir(raw_dir.c_str(), 0775) != 0 && errno != EEXIST) return;

	const std::string path = raw_dir + "/" + base + "_src" + sanitize(src_label) +
	                         "_dst" + sanitize(dst_label) + ext;
	std::ofstream out(path, std::ios::app);
	if (!out.is_open()) return;

	out << "# src=" << src_label << " dst=" << dst_label << "\n";
	for (double v : values)
		out << std::fixed << std::setprecision(digits) << v << "\n";
}

// ------------------------------------------------------------------ //
// Окно общей оперативной памяти узла                                  //
// ------------------------------------------------------------------ //

SharedHostWindow::SharedHostWindow(MPI_Comm node_comm, size_t my_bytes, int rank,
                                   const Args &args)
	: rank_(rank), args_(args) {
	void *base = nullptr;
	mpi_ok(MPI_Win_allocate_shared(static_cast<MPI_Aint>(my_bytes), sizeof(char),
	                               MPI_INFO_NULL, node_comm, &base, &win_),
	       "MPI_Win_allocate_shared");
}

char *SharedHostWindow::segment(int local_rank, size_t need) {
	MPI_Aint size = 0;
	int      disp_unit = 0;
	void    *base = nullptr;
	mpi_ok(MPI_Win_shared_query(win_, local_rank, &size, &disp_unit, &base),
	       "MPI_Win_shared_query");
	char *p = static_cast<char *>(base);
	if (!p || size < static_cast<MPI_Aint>(need))
		return nullptr;

	const cudaError_t reg = cudaHostRegister(p, need, cudaHostRegisterPortable);
	if (reg == cudaSuccess || reg == cudaErrorHostMemoryAlreadyRegistered) {
		registered_.push_back(p);
	} else {
		DBG_LOG(rank_, args_, "shared host register skipped (local rank " << local_rank
		        << "): " << cudaGetErrorString(reg));
	}
	if (reg != cudaSuccess)
		cudaGetLastError();   // сбросить ошибку CUDA, чтобы она не всплыла позже
	return p;
}

SharedHostWindow::~SharedHostWindow() {
	for (char *p : registered_)
		cuda_ok(cudaHostUnregister(p), "cudaHostUnregister");
	if (win_ != MPI_WIN_NULL)
		mpi_ok(MPI_Win_free(&win_), "MPI_Win_free");
}

} // namespace gpu_benchmark
