#include "gpu_common.hpp"

#include <algorithm> // std::max, std::sort, …
#include <cctype>    // std::tolower и т.п. при разборе CLI
#include <cerrno>    // errno после mkdir
#include <cmath>     // std::sqrt для дисперсии/СКО
#include <cstdlib>   // std::strtoull, std::atoi, std::exit
#include <cstring>   // std::memcpy / строковые C API при необходимости
#include <fstream>   // std::ofstream для raw-файлов
#include <iomanip>   // std::setprecision, std::fixed (pair_*, raw)
#include <iostream>  // std::cout, std::cerr — help и ошибки CLI
#include <numeric>   // std::accumulate и др.
#include <sstream>   // std::ostringstream — print_pair_line и пути
#include <sys/stat.h> // mkdir для каталога raw/
#include <time.h>    // clock_gettime — clock_gettime_wrapper

#if defined(OPEN_MPI) && OPEN_MPI
#include "mpi-ext.h" // MPIX_Query_cuda_support (CUDA-aware MPI)
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
	std::cout << "gpu — GPU pair latency (one_to_one | all_to_all)\n"
			  << "  --bytes N       single-size text benchmark (default 4 MB)\n"
			  << "  --bytes-begin N first message size for NetCDF-only sweep\n"
			  << "  --bytes-end N   last message size for NetCDF-only sweep\n"
			  << "  --bytes-step N  message size step for NetCDF-only sweep\n"
			  << "  --warmup N      warmup iterations per pair\n"
			  << "  --iters N       measured iterations per pair\n"
			  << "  --env E         auto | host\n"
			  << "  --mode M        one_to_one | all_to_all (default one_to_one)\n"
			  << "  --timer T       all | mpi | cpu | cuda (default cuda)\n"
			  << "  --stat S        all | avg | med | min | max | var | std (pair line output)\n"
			  << "  --out FILE, -o FILE  also write the same output to FILE (rank 0 only)\n"
			  << "  --debug, -d     verbose debug logs to stderr\n";
}

Mode parse_mode(const std::string &s, int rank) {
	if (s == "one_to_one" || s == "sequential" || s == "1to1")
		return Mode::OneToOne;
	if (s == "all_to_all" || s == "alltoall" || s == "parallel")
		return Mode::AllToAll;
	if (rank == 0)
		std::cerr << "unknown --mode: " << s
				  << " (use one_to_one|all_to_all)\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
	return Mode::OneToOne;
}

const char *mode_to_string(Mode mode) {
	switch (mode) {
	case Mode::OneToOne:
		return "one_to_one";
	case Mode::AllToAll:
		return "all_to_all";
	}
	return "one_to_one";
}

Env parse_env(const std::string &s, int rank) {
	if (s == "auto")
		return Env::Auto;
	if (s == "host")
		return Env::Host;
	if (rank == 0)
		std::cerr << "unknown --env: " << s << " (use auto|host)\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
	return Env::Auto;
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

const char *timer_to_string(Timer t) {
	switch (t) {
	case Timer::All:
		return "all";
	case Timer::Mpi:
		return "mpi";
	case Timer::Cpu:
		return "cpu";
	case Timer::Cuda:
		return "cuda";
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
	bool bytes_was_set = false;
	bool begin_was_set = false;
	bool end_was_set = false;
	bool step_was_set = false;
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

		if (s == "--bytes") {
			a.nbytes = static_cast<size_t>(std::strtoull(next("--bytes"), nullptr, 10));
			bytes_was_set = true;
		}
		else if (s == "--bytes-begin") {
			a.begin_nbytes =
				static_cast<size_t>(std::strtoull(next("--bytes-begin"), nullptr, 10));
			begin_was_set = true;
		}
		else if (s == "--bytes-end") {
			a.end_nbytes =
				static_cast<size_t>(std::strtoull(next("--bytes-end"), nullptr, 10));
			end_was_set = true;
		}
		else if (s == "--bytes-step") {
			a.step_nbytes =
				static_cast<size_t>(std::strtoull(next("--bytes-step"), nullptr, 10));
			step_was_set = true;
		}
		else if (s == "--warmup")
			a.warmup = std::atoi(next("--warmup"));
		else if (s == "--iters")
			a.iters = std::atoi(next("--iters"));
		else if (s == "--env")
			a.env = parse_env(next("--env"), rank);
		else if (s == "--mode")
			a.mode = parse_mode(next("--mode"), rank);
		else if (s == "--timer")
			a.timer = parse_timer(next("--timer"), rank);
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
		if (rank == 0)
			std::cerr << "--warmup must be >= 0\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (a.iters <= 0) {
		if (rank == 0)
			std::cerr << "--iters must be > 0\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	const bool any_range_flag = begin_was_set || end_was_set || step_was_set;
	const bool all_range_flags = begin_was_set && end_was_set && step_was_set;
	if (bytes_was_set && any_range_flag) {
		if (rank == 0)
			std::cerr << "use either --bytes or --bytes-begin/--bytes-end/--bytes-step\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (any_range_flag && !all_range_flags) {
		if (rank == 0)
			std::cerr << "NetCDF sweep requires all of --bytes-begin, --bytes-end, --bytes-step\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	a.sweep_sizes = all_range_flags;
	if (a.sweep_sizes) {
		if (a.out_path.empty()) {
			if (rank == 0)
				std::cerr << "NetCDF sweep requires --out prefix\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	} else {
		a.begin_nbytes = a.nbytes;
		a.end_nbytes = a.nbytes;
		a.step_nbytes = 1;
	}
	if (!a.sweep_sizes && a.nbytes == 0) {
		if (rank == 0)
			std::cerr << "--bytes must be > 0\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (a.sweep_sizes && (a.begin_nbytes == 0 || a.end_nbytes == 0 || a.step_nbytes == 0)) {
		if (rank == 0)
			std::cerr << "--bytes-begin, --bytes-end and --bytes-step must be > 0\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (a.begin_nbytes > a.end_nbytes) {
		if (rank == 0)
			std::cerr << "--bytes-begin must be <= --bytes-end\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	a.nbytes = a.begin_nbytes;
	return a;
}

void debug_log(bool enabled, int rank, const std::string &msg) {
	if (!enabled)
		return;
	std::cerr << "[DBG r" << rank << "] " << msg << "\n";
}

bool check_host(Env env, bool cuda_aware) {
	if (env == Env::Host)
		return true;
	return !cuda_aware;
}

std::vector<size_t> build_message_sizes(const Args &args, int rank) {
	std::vector<size_t> sizes;
	if (!args.sweep_sizes) {
		sizes.push_back(args.nbytes);
		return sizes;
	}
	for (size_t nbytes = args.begin_nbytes; nbytes <= args.end_nbytes;
		 nbytes += args.step_nbytes) {
		sizes.push_back(nbytes);
		if (args.end_nbytes - nbytes < args.step_nbytes)
			break;
	}
	if (sizes.empty()) {
		if (rank == 0)
			std::cerr << "gpu: empty message size sweep\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	return sizes;
}

double clock_gettime_wrapper() {
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

std::vector<std::string> build_rank_labels(const std::vector<char> &hosts_recv,
										   int nproc, int host_len) {
	const auto short_host = [](const std::string &host) {
		const size_t dot = host.find('.');
		if (dot == std::string::npos)
			return host;
		return host.substr(0, dot);
	};
	const auto host_node_token = [&](const std::string &host) {
		const std::string sh = short_host(host);
		size_t pos = sh.size();
		while (pos > 0 && std::isdigit(static_cast<unsigned char>(sh[pos - 1])))
			--pos;
		if (pos < sh.size())
			return sh.substr(pos);
		return sh;
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

void fill_ack(const std::vector<double> &samples_mpi_us,
								  const std::vector<double> &samples_cpu_us,
								  const std::vector<double> &samples_gpu_us,
								  const Args &args, double *ack) {
	const auto fill_stats = [](const std::vector<double> &samples, double *out6) {
		const double n = static_cast<double>(samples.size());
		const double _sum = std::accumulate(samples.begin(), samples.end(), 0.0);
		const double _avg = _sum / n;
		auto [it_min, it_max] = std::minmax_element(samples.begin(), samples.end());
		const double _min = *it_min;
		const double _max = *it_max;
		std::vector<double> sorted = samples;
		std::sort(sorted.begin(), sorted.end());
		const size_t m = sorted.size() / 2;
		const double _med = (sorted.size() % 2 == 0)
								? (sorted[m - 1] + sorted[m]) * 0.5
								: sorted[m];
		double _var = 0.0;
		if (samples.size() > 1) {
			for (double x : samples) {
				const double d = x - _avg;
				_var += d * d;
			}
			_var /= static_cast<double>(samples.size() - 1);
		}
		const double _std = std::sqrt(_var);
		out6[0] = _avg;
		out6[1] = _med;
		out6[2] = _min;
		out6[3] = _max;
		out6[4] = _var;
		out6[5] = _std;
	};

	std::fill(ack, ack + ACK_FIELDS, 0.0);
	if (samples_mpi_us.empty() || samples_cpu_us.empty() || samples_gpu_us.empty())
		return;
	switch (args.timer) {
	case Timer::All:
		fill_stats(samples_mpi_us, ack);
		break;
	case Timer::Mpi:
		fill_stats(samples_mpi_us, ack);
		break;
	case Timer::Cpu:
		fill_stats(samples_cpu_us, ack);
		break;
	case Timer::Cuda:
		fill_stats(samples_gpu_us, ack);
		break;
	default:
		fill_stats(samples_mpi_us, ack);
		break;
	}
	fill_stats(samples_cpu_us, ack + 7);
	fill_stats(samples_mpi_us, ack + 13);
	fill_stats(samples_gpu_us, ack + 19);
	ack[6] = 1.0;
}

std::string print_pair_line(const char *line_name,
							 const std::string &src_label, const std::string &dst_label,
							 double avg_us, double med_us, double min_us, double max_us,
							 double var_us, double std_us, StatOut stat) {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(REPORT_DIGITS);
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

void append_raw_samples(const Args &args, int rank, const Task &t,
						const std::vector<std::string> &rank_labels,
						const std::vector<double> &samples_us) {
	const auto sanitize_label_for_raw_path = [](const std::string &lab) {
		std::string out;
		out.reserve(lab.size());
		for (unsigned char uc : lab) {
			const char c = static_cast<char>(uc);
			if (std::isalnum(uc) || c == '.' || c == '-' || c == '_')
				out.push_back(c);
			else
				out.push_back('_');
		}
		return out.empty() ? "nolabel" : out;
	};

	if (!args.save_raw_samples || args.out_path.empty() || samples_us.empty())
		return;
	(void)rank;
	if (t.src_rank < 0 || t.dst_rank < 0 ||
		static_cast<size_t>(t.src_rank) >= rank_labels.size() ||
		static_cast<size_t>(t.dst_rank) >= rank_labels.size())
		return;
	const std::string src_tok =
		sanitize_label_for_raw_path(rank_labels[static_cast<size_t>(t.src_rank)]);
	const std::string dst_tok =
		sanitize_label_for_raw_path(rank_labels[static_cast<size_t>(t.dst_rank)]);
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
	path << raw_dir << "/" << base << "_src" << src_tok << "_dst" << dst_tok << ".raw";
	std::ofstream out(path.str(), std::ios::app);
	if (!out.is_open())
		return;
	out << "# src=" << rank_labels[static_cast<size_t>(t.src_rank)] << " dst="
		<< rank_labels[static_cast<size_t>(t.dst_rank)] << "\n";
	for (size_t i = 0; i < samples_us.size(); ++i) {
		out << std::fixed << std::setprecision(REPORT_DIGITS) << samples_us[i]
			<< "\n";
	}
}

} // namespace gpu_benchmark
