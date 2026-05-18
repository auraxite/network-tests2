#include "gpu_common.hpp"

#include <algorithm> // std::sort, std::minmax_element, …
#include <cctype>    // std::tolower
#include <cerrno>    // errno после mkdir
#include <cmath>     // std::sqrt
#include <cstdlib>   // std::strtoull, std::atoi, std::exit
#include <cstring>   // std::memcpy
#include <fstream>   // std::ofstream для raw-файлов
#include <iomanip>   // std::setprecision, std::fixed
#include <iostream>  // std::cout, std::cerr
#include <numeric>   // std::accumulate
#include <sstream>   // std::ostringstream
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

void help(int rank) {
	if (rank != 0)
		return;
	std::cout
		<< "gpu — GPU pair latency (one_to_one | all_to_all)\n"
		<< "  --bytes N       message size in bytes (default 4 MB)\n"
		<< "  --warmup N      warmup iterations per pair\n"
		<< "  --iters N       measured iterations per pair\n"
		<< "  --env E         auto | host\n"
		<< "  --mode M        one_to_one | all_to_all (default one_to_one)\n"
		<< "  --stat S        all | avg | med | min | max | var | std\n"
		<< "  --out FILE, -o FILE  write output to FILE (rank 0 only)\n"
		<< "  --debug, -d     verbose debug logs to stderr\n"
		<< "\n"
		<< "UCX defaults applied before MPI_Init (override via env vars):\n"
		<< "  UCX_RNDV_THRESH=0          force rendezvous for all sizes\n"
		<< "  UCX_IB_GPU_DIRECT_RDMA=yes enable GPUDirect RDMA\n"
		<< "  UCX_RNDV_SCHEME=get_zcopy  zero-copy rendezvous\n";
}

Mode parse_mode(const std::string &s, int rank) {
	if (s == "one_to_one" || s == "sequential" || s == "1to1")
		return Mode::OneToOne;
	if (s == "all_to_all" || s == "alltoall" || s == "parallel")
		return Mode::AllToAll;
	if (rank == 0)
		std::cerr << "unknown --mode: " << s << " (use one_to_one|all_to_all)\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
	return Mode::OneToOne;
}

const char *mode_to_string(Mode mode) {
	switch (mode) {
	case Mode::OneToOne: return "one_to_one";
	case Mode::AllToAll: return "all_to_all";
	}
	return "one_to_one";
}

Env parse_env(const std::string &s, int rank) {
	if (s == "auto") return Env::Auto;
	if (s == "host") return Env::Host;
	if (rank == 0)
		std::cerr << "unknown --env: " << s << " (use auto|host)\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
	return Env::Auto;
}

StatOut parse_stat_out(const std::string &s, int rank) {
	if (s == "all") return StatOut::All;
	if (s == "avg") return StatOut::Avg;
	if (s == "med") return StatOut::Med;
	if (s == "min") return StatOut::Min;
	if (s == "max") return StatOut::Max;
	if (s == "var") return StatOut::Var;
	if (s == "std") return StatOut::Std;
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
		if (rank == 0) std::cerr << "--warmup must be >= 0\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (a.iters <= 0) {
		if (rank == 0) std::cerr << "--iters must be > 0\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (a.nbytes == 0) {
		if (rank == 0) std::cerr << "--bytes must be > 0\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	return a;
}

void debug_log(bool enabled, int rank, const std::string &msg) {
	if (!enabled) return;
	std::cerr << "[DBG r" << rank << "] " << msg << "\n";
}

bool check_host(Env env, bool cuda_aware) {
	if (env == Env::Host) return true;
	return !cuda_aware;
}

std::vector<std::string> build_rank_labels(const std::vector<char> &hosts_recv,
                                            int nproc, int host_len) {
	const auto short_host = [](const std::string &host) {
		const size_t dot = host.find('.');
		return dot == std::string::npos ? host : host.substr(0, dot);
	};
	const auto node_token = [&](const std::string &host) {
		const std::string sh = short_host(host);
		size_t pos = sh.size();
		while (pos > 0 && std::isdigit(static_cast<unsigned char>(sh[pos - 1])))
			--pos;
		return pos < sh.size() ? sh.substr(pos) : sh;
	};

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
				local_idx = seen_counts[i]++;
				found = true;
				break;
			}
		}
		if (!found) {
			seen_hosts.push_back(sh);
			seen_counts.push_back(1);
		}
		std::ostringstream oss;
		oss << node_token(sh) << "." << local_idx;
		labels[static_cast<size_t>(r)] = oss.str();
	}
	return labels;
}

std::vector<std::string> build_global_gpu_labels(const std::vector<char> &hosts_recv,
                                                   int nproc, int host_len,
                                                   const std::vector<int> &gpu_counts) {
	const auto short_host = [](const std::string &host) {
		const size_t dot = host.find('.');
		return dot == std::string::npos ? host : host.substr(0, dot);
	};
	const auto node_token = [&](const std::string &host) {
		const std::string sh = short_host(host);
		size_t pos = sh.size();
		while (pos > 0 && std::isdigit(static_cast<unsigned char>(sh[pos - 1])))
			--pos;
		return pos < sh.size() ? sh.substr(pos) : sh;
	};

	std::vector<std::string> labels;
	for (int r = 0; r < nproc; ++r) {
		const char *h = hosts_recv.data() + static_cast<size_t>(r) * static_cast<size_t>(host_len);
		const std::string token = node_token(std::string(h));
		const int gc = (r < static_cast<int>(gpu_counts.size()) && gpu_counts[static_cast<size_t>(r)] > 0)
		                   ? gpu_counts[static_cast<size_t>(r)] : 0;
		for (int g = 0; g < gc; ++g) {
			std::ostringstream oss;
			oss << token << "." << g;
			labels.push_back(oss.str());
		}
	}
	return labels;
}

void fill_ack(const std::vector<double> &samples, double *ack) {
	std::fill(ack, ack + ACK_FIELDS, 0.0);
	if (samples.empty()) return;

	const double n   = static_cast<double>(samples.size());
	const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
	const double avg = sum / n;
	auto [it_min, it_max] = std::minmax_element(samples.begin(), samples.end());

	std::vector<double> sorted = samples;
	std::sort(sorted.begin(), sorted.end());
	const size_t m = sorted.size() / 2;
	const double med = (sorted.size() % 2 == 0)
	                       ? (sorted[m - 1] + sorted[m]) * 0.5
	                       : sorted[m];

	double var = 0.0;
	if (samples.size() > 1) {
		for (double x : samples) { double d = x - avg; var += d * d; }
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
                             const std::string &dst_label, double avg_us, double med_us,
                             double min_us, double max_us, double var_us, double std_us,
                             StatOut stat) {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(REPORT_DIGITS);
	oss << line_name << " " << src_label << " -> " << dst_label << " ";
	switch (stat) {
	case StatOut::All:
		oss << "avg_us=" << avg_us << " med_us=" << med_us
		    << " min_us=" << min_us << " max_us=" << max_us
		    << " var_us=" << var_us << " std_us=" << std_us;
		break;
	case StatOut::Avg: oss << "avg_us=" << avg_us; break;
	case StatOut::Med: oss << "med_us=" << med_us; break;
	case StatOut::Min: oss << "min_us=" << min_us; break;
	case StatOut::Max: oss << "max_us=" << max_us; break;
	case StatOut::Var: oss << "var_us=" << var_us; break;
	case StatOut::Std: oss << "std_us=" << std_us; break;
	}
	oss << "\n";
	return oss.str();
}

void append_raw_samples_named(const Args &args, const std::string &src_label,
                               const std::string &dst_label,
                               const std::vector<double> &samples_us) {
	const auto sanitize = [](const std::string &lab) {
		std::string out;
		for (unsigned char uc : lab) {
			const char c = static_cast<char>(uc);
			out.push_back((std::isalnum(uc) || c == '.' || c == '-' || c == '_') ? c : '_');
		}
		return out.empty() ? "nolabel" : out;
	};

	if (!args.save_raw_samples || args.out_path.empty() || samples_us.empty())
		return;

	std::string base = args.out_path;
	std::string dir  = ".";
	const size_t slash = base.find_last_of('/');
	if (slash != std::string::npos) { dir = base.substr(0, slash); base = base.substr(slash + 1); }
	const std::string suffix = ".txt";
	if (base.size() >= suffix.size() &&
	    base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0)
		base.erase(base.size() - suffix.size());

	const std::string raw_dir = dir + "/raw";
	if (mkdir(raw_dir.c_str(), 0775) != 0 && errno != EEXIST) return;

	std::ostringstream path;
	path << raw_dir << "/" << base << "_src" << sanitize(src_label)
	     << "_dst" << sanitize(dst_label) << ".raw";
	std::ofstream out(path.str(), std::ios::app);
	if (!out.is_open()) return;

	out << "# src=" << src_label << " dst=" << dst_label << "\n";
	for (double v : samples_us)
		out << std::fixed << std::setprecision(REPORT_DIGITS) << v << "\n";
}

void append_raw_samples(const Args &args, int rank, const Task &t,
                        const std::vector<std::string> &rank_labels,
                        const std::vector<double> &samples_us) {
	(void)rank;
	if (t.src_rank < 0 || t.dst_rank < 0 ||
	    static_cast<size_t>(t.src_rank) >= rank_labels.size() ||
	    static_cast<size_t>(t.dst_rank) >= rank_labels.size())
		return;
	append_raw_samples_named(args, rank_labels[static_cast<size_t>(t.src_rank)],
	                         rank_labels[static_cast<size_t>(t.dst_rank)], samples_us);
}

} // namespace gpu_benchmark
