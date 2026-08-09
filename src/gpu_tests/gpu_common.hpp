#pragma once

#include <cstddef>        // std::size_t
#include <cuda_runtime.h> // cudaError_t, cudaSetDevice, …
#include <functional>
#include <mpi.h>          // MPI типы и коллективы
#include <sstream>        // std::ostringstream — для макроса DBG_LOG
#include <string>         // std::string в Args, метках рангов
#include <vector>         // std::vector в интерфейсе

namespace gpu_benchmark {

inline constexpr int REPORT_DIGITS     = 3;
inline constexpr int TOTAL_TIME_DIGITS = 3;

void cuda_ok(cudaError_t e, const char *msg);
void mpi_ok(int err, const char *msg);
bool mpi_cuda_aware();

enum class Env { Auto, Host };

enum class StatOut { All, Avg, Med, Min, Max, Std, Var };

enum class Mode { OneToOne, AllToAll, CudaOneToOne, CudaAllToAll };

/* ack layout (7 doubles):
     [0] avg_us  [1] med_us  [2] min_us  [3] max_us
     [4] var_us  [5] std_us  [6] valid (1.0 = data present) */
constexpr int ACK_FIELDS = 7;

struct Args {
	size_t      nbytes          = 4u * 1000u * 1000u;
	int         warmup          = 10;
	int         iters           = 50;
	Env         env             = Env::Auto;
	StatOut     stat_out        = StatOut::All;
	Mode        mode            = Mode::OneToOne;
	std::string out_path;
	bool        save_raw_samples = true;
	bool        debug            = false;
};

struct Task {
	int src_rank = -1;
	int src_gpu  = -1;
	int dst_rank = -1;
	int dst_gpu  = -1;
	int stop     = 0;
};

void        help(int rank);
Mode        parse_mode(const std::string &s, int rank);
const char *mode_to_string(Mode mode);
Env         parse_env(const std::string &s, int rank);
StatOut     parse_stat_out(const std::string &s, int rank);
Args        parse_args(int argc, char **argv, int rank);
bool        check_host(Env env, bool cuda_aware);
void        debug_log(bool enabled, int rank, const std::string &msg);

#define DBG_LOG(rank_, args_, expr_)                                \
	do {                                                            \
		if ((args_).debug) {                                        \
			std::ostringstream _dbg_oss;                            \
			_dbg_oss << expr_;                                      \
			::gpu_benchmark::debug_log(true, (rank_), _dbg_oss.str()); \
		}                                                           \
	} while (0)

std::vector<std::string> build_rank_labels(const std::vector<char> &hosts_recv,
                                            int nproc, int host_len);
std::vector<std::string> build_global_gpu_labels(const std::vector<char> &hosts_recv,
                                                  int nproc, int host_len,
                                                  const std::vector<int> &gpu_counts);

/* Заполняет ack[0..6] по MPI-сэмплам. */
void fill_ack(const std::vector<double> &samples_mpi_us, double *ack);

void append_raw_samples(const Args &args, int rank, const Task &t,
                        const std::vector<std::string> &rank_labels,
                        const std::vector<double> &samples_us);
void append_raw_samples_named(const Args &args, const std::string &src_label,
                               const std::string &dst_label,
                               const std::vector<double> &samples_us);

std::string print_pair_line(const char *line_name, const std::string &src_label,
                             const std::string &dst_label, double avg_us, double med_us,
                             double min_us, double max_us, double var_us, double std_us,
                             StatOut stat);

// CUDA IPC / P2P payload modes. MPI is used only for control and results.
void schedule_cuda_one_to_one(
    int rank, int nproc, const Args &args, MPI_Comm node_comm,
    const std::vector<int> &node_ranks,
    const std::vector<std::string> &rank_labels,
    const std::function<void(const std::string &)> &mirror);

void schedule_cuda_all_to_all(
    int rank, int nproc, const Args &args, MPI_Comm node_comm,
    const std::vector<int> &node_ranks,
    const std::vector<std::string> &rank_labels,
    const std::function<void(const std::string &)> &mirror);

} // namespace gpu_benchmark
