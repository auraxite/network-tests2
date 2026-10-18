#pragma once

#include <cstddef>        // std::size_t
#include <cuda_runtime.h>
#include <mpi.h>
#include <sstream>        // std::ostringstream — для макроса DBG_LOG
#include <string>
#include <utility>        // std::swap
#include <vector>

namespace gpu_benchmark {

inline constexpr int REPORT_DIGITS     = 3;
inline constexpr int TOTAL_TIME_DIGITS = 3;

void cuda_ok(cudaError_t e, const char *msg);
void mpi_ok(int err, const char *msg);
bool mpi_cuda_aware();

enum class Env { Auto, Host };

enum class StatOut { All, Avg, Med, Min, Max, Std, Var };

enum class Mode { OneToOne, AllToAll, OneToAll, AllToOne, OneToOneV2 };

/* ack layout (7 doubles):
     [0] avg_us  [1] med_us  [2] min_us  [3] max_us
     [4] var_us  [5] std_us  [6] valid (1.0 = data present) */
constexpr int ACK_FIELDS = 7;

struct Args {
	size_t      nbytes           = 4u * 1000u * 1000u;
	int         warmup           = 10;
	int         iters            = 50;
	Env         env              = Env::Auto;
	StatOut     stat_out         = StatOut::All;
	Mode        mode             = Mode::OneToOne;
	std::string out_path;
	bool        save_raw_samples = true;
	bool        debug            = false;
	bool        random_order     = false;   // one_to_one_v2: --order random
	unsigned    seed             = 1;       // one_to_one_v2: --seed
};

/* Задание пары для one_to_one: рассылается рангом 0 как MPI_BYTE. */
struct Task {
	int src_rank = -1;
	int dst_rank = -1;
	int stop     = 0;
};

void        help(int rank);
Mode        parse_mode(const std::string &s, int rank);
const char *mode_to_string(Mode mode);
Env         parse_env(const std::string &s, int rank);
StatOut     parse_stat_out(const std::string &s, int rank);
Args        parse_args(int argc, char **argv, int rank);
/* true — данные идут через оперативную память (среда host или MPI без CUDA). */
bool        use_host_path(Env env, bool cuda_aware);
void        debug_log(bool enabled, int rank, const std::string &msg);

#define DBG_LOG(rank_, args_, expr_)                                    \
	do {                                                                \
		if ((args_).debug) {                                            \
			std::ostringstream _dbg_oss;                                \
			_dbg_oss << expr_;                                          \
			::gpu_benchmark::debug_log(true, (rank_), _dbg_oss.str());  \
		}                                                               \
	} while (0)

/* Метки рангов "<цифры имени узла>.<номер ранга на узле>". */
std::vector<std::string> build_rank_labels(const std::vector<char> &hosts_recv,
                                           int nproc, int host_len);

/* Заполняет ack[0..6] по сериям замеров. */
void fill_ack(const std::vector<double> &samples_us, double *ack);

/* Дописывает серию в raw/<base>_src<src>_dst<dst>.raw рядом с --out. */
void append_raw_samples(const Args &args, const std::string &src_label,
                        const std::string &dst_label,
                        const std::vector<double> &samples_us);

/* То же с другим расширением и точностью (например, .ts — отметки времени). */
void append_raw_series(const Args &args, const std::string &src_label,
                       const std::string &dst_label, const std::vector<double> &values,
                       const std::string &ext, int digits);

std::string print_pair_line(const char *line_name, const std::string &src_label,
                            const std::string &dst_label, const double *ack,
                            StatOut stat);

// ------------------------------------------------------------------ //
// Буферы: выделяются один раз до замеров, освобождаются при выходе    //
// из области видимости.                                               //
// ------------------------------------------------------------------ //

class DeviceBuffer {
public:
	DeviceBuffer() = default;
	explicit DeviceBuffer(size_t n) { cuda_ok(cudaMalloc(&p_, n), "cudaMalloc"); }
	~DeviceBuffer() { if (p_) cudaFree(p_); }
	DeviceBuffer(DeviceBuffer &&o) noexcept : p_(o.p_) { o.p_ = nullptr; }
	DeviceBuffer &operator=(DeviceBuffer &&o) noexcept { std::swap(p_, o.p_); return *this; }
	DeviceBuffer(const DeviceBuffer &) = delete;
	DeviceBuffer &operator=(const DeviceBuffer &) = delete;
	char *get() const { return p_; }
private:
	char *p_ = nullptr;
};

/* Закреплённая (pinned) оперативная память. */
class PinnedBuffer {
public:
	PinnedBuffer() = default;
	explicit PinnedBuffer(size_t n) { cuda_ok(cudaMallocHost(&p_, n), "cudaMallocHost"); }
	~PinnedBuffer() { if (p_) cudaFreeHost(p_); }
	PinnedBuffer(PinnedBuffer &&o) noexcept : p_(o.p_) { o.p_ = nullptr; }
	PinnedBuffer &operator=(PinnedBuffer &&o) noexcept { std::swap(p_, o.p_); return *this; }
	PinnedBuffer(const PinnedBuffer &) = delete;
	PinnedBuffer &operator=(const PinnedBuffer &) = delete;
	char *get() const { return p_; }
private:
	char *p_ = nullptr;
};

/* Окно общей оперативной памяти узла (MPI_Win_allocate_shared). Каждый ранг
   выделяет my_bytes, сегменты соседей доступны через segment(). Сегменты
   регистрируются в CUDA (cudaHostRegister), чтобы копирования GPU <-> память
   шли без промежуточного буфера драйвера. Создание и освобождение коллективны
   по node_comm. */
class SharedHostWindow {
public:
	SharedHostWindow(MPI_Comm node_comm, size_t my_bytes, int rank, const Args &args);
	~SharedHostWindow();
	SharedHostWindow(const SharedHostWindow &) = delete;
	SharedHostWindow &operator=(const SharedHostWindow &) = delete;

	/* Сегмент ранга local_rank узла, если он не меньше need байт, иначе nullptr. */
	char *segment(int local_rank, size_t need);
	MPI_Win win() const { return win_; }
	void sync() const { if (win_ != MPI_WIN_NULL) mpi_ok(MPI_Win_sync(win_), "MPI_Win_sync"); }

private:
	MPI_Win             win_ = MPI_WIN_NULL;
	int                 rank_;
	const Args         &args_;
	std::vector<char *> registered_;
};

} // namespace gpu_benchmark
