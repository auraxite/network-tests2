#pragma once

#include <cstddef>       // std::size_t
#include <cuda_runtime.h> // cudaError_t, cudaSetDevice, … (объявления в API)
#include <mpi.h>           // MPI типы и коллективы
#include <string>          // std::string в Args, метках рангов
#include <vector>          // std::vector в интерфейсе

namespace gpu_benchmark {

/** Знаков после запятой в pair_*, raw .raw, iostream перед бенчмарком. */
inline constexpr int REPORT_DIGITS = 3;
/** Знаков после запятой для TotalTimeSec (секунды). */
inline constexpr int TOTAL_TIME_DIGITS = 3;

void cuda_ok(cudaError_t e, const char *msg);
void mpi_ok(int err, const char *msg);
bool mpi_cuda_aware();

enum class Env {
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

enum class Mode {
	OneToOne,
	AllToAll,
};

/* Полный набор статистик одной пары src->dst; используют оба режима. */
constexpr int ACK_FIELDS = 25;

struct Args {
	size_t nbytes = 4u * 1000u * 1000u;
	int warmup = 10;
	int iters = 50;
	Env env = Env::Auto;
	Timer timer = Timer::Mpi;
	StatOut stat_out = StatOut::All;
	Mode mode = Mode::OneToOne;
	std::string out_path;
	bool save_raw_samples = true;
	bool debug = false;
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
Mode parse_mode(const std::string &s, int rank);
const char *mode_to_string(Mode mode);
Env parse_env(const std::string &s, int rank);
Timer parse_timer(const std::string &s, int rank);
const char *timer_to_string(Timer t);
StatOut parse_stat_out(const std::string &s, int rank);
Args parse_args(int argc, char **argv, int rank);
bool check_host(Env env, bool cuda_aware);
void debug_log(bool enabled, int rank, const std::string &msg);

/* Обёртка над clock_gettime: возвращает время в микросекундах. */
double clock_gettime_wrapper();
/* Общие подписи рангов для pair/raw в one-to-one и all-to-all. */
std::vector<std::string> build_rank_labels(const std::vector<char> &hosts_recv,
											 int nproc, int host_len);

/* Заполняет ack по трем наборам samples; используют оба режима. */
void fill_ack(const std::vector<double> &samples_mpi_us,
			  const std::vector<double> &samples_cpu_us,
			  const std::vector<double> &samples_gpu_us, const Args &args,
			  double *ack);

/* Сохраняет raw samples одной пары src->dst; используют оба режима. */
void append_raw_samples(const Args &args, int rank, const Task &t,
						const std::vector<std::string> &rank_labels,
						const std::vector<double> &samples_us);
/* Печатает строку pair/pair_mpi/pair_cpu/pair_cuda; используют оба режима. */
std::string print_pair_line(const char *line_name, const std::string &src_label,
							const std::string &dst_label, double avg_us, double med_us,
							double min_us, double max_us, double var_us, double std_us,
							StatOut stat);

} // namespace gpu_benchmark
