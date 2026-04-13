#pragma once

#include <cstddef>       // std::size_t
#include <cuda_runtime.h> // cudaError_t, cudaSetDevice, … (объявления в API)
#include <mpi.h>           // MPI типы и коллективы
#include <string>          // std::string в Args, метках рангов
#include <vector>          // std::vector в интерфейсе

#include "netcdf_writer.h" // NetcdfBundle, запись матриц метрик

namespace gpu_benchmark {

/** Знаков после запятой в pair_*, raw .raw, iostream перед бенчмарком. */
inline constexpr int REPORT_DIGITS = 3;
/** Знаков после запятой для TotalTimeSec (секунды). */
inline constexpr int TOTAL_TIME_DIGITS = 6;

void cuda_ok(cudaError_t e, const char *msg);
void mpi_ok(int err, const char *msg);
bool mpi_cuda_aware();

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

enum class Scheme {
	OneToOne,
	AllToAll,
};

/* Полный набор статистик одной пары src->dst; используют обе схемы. */
constexpr int ACK_FIELDS = 25;

struct Args {
	size_t nbytes = 4u * 1000u * 1000u;
	size_t begin_nbytes = 4u * 1000u * 1000u;
	size_t end_nbytes = 4u * 1000u * 1000u;
	size_t step_nbytes = 1;
	bool sweep_sizes = false;
	int warmup = 10;
	int iters = 50;
	Mode mode = Mode::Auto;
	Timer timer = Timer::All;
	StatOut stat_out = StatOut::All;
	Scheme scheme = Scheme::OneToOne;
	std::string out_path;
	bool save_raw_samples = true;
};

/* Описание одной пары src/dst; активно используется в one-to-one. */
struct Task {
	int src_rank = -1;
	int src_gpu = -1;
	int dst_rank = -1;
	int dst_gpu = -1;
	int stop = 0;
};

void help(int rank);
Scheme parse_scheme(const std::string &s, int rank);
const char *scheme_to_string(Scheme sch);
Mode parse_mode(const std::string &s, int rank);
Timer parse_timer(const std::string &s, int rank);
const char *timer_to_string(Timer t);
StatOut parse_stat_out(const std::string &s, int rank);
Args parse_args(int argc, char **argv, int rank);
bool check_host(Mode mode, bool cuda_aware);
std::vector<size_t> build_message_sizes(const Args &args, int rank);

/* Обёртка над clock_gettime: возвращает время в микросекундах. */
double clock_gettime_wrapper();
/* Общие подписи рангов для pair/raw в one-to-one и all-to-all. */
std::vector<std::string> build_rank_labels(const std::vector<char> &hosts_recv,
											 int nproc, int host_len);

/* Заполняет ack по трем наборам samples; используют обе схемы. */
void fill_ack(const std::vector<double> &samples_mpi_us,
			  const std::vector<double> &samples_cpu_us,
			  const std::vector<double> &samples_gpu_us, const Args &args,
			  double *ack);

/* Сохраняет raw samples одной пары src->dst; используют обе схемы. */
void append_raw_samples(const Args &args, int rank, const Task &t,
						const std::vector<std::string> &rank_labels,
						const std::vector<double> &samples_us);
/* Печатает строку pair/pair_mpi/pair_cpu/pair_cuda; используют обе схемы. */
std::string print_pair_line(const char *line_name, const std::string &src_label,
							const std::string &dst_label, double avg_us, double med_us,
							double min_us, double max_us, double var_us, double std_us,
							StatOut stat);

} // namespace gpu_benchmark
