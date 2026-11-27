/* gpu_tests2: задержки передачи между GPU (C, MPI + CUDA). Общие типы и функции.
   Отличия от gpu_tests (C++) — README.md этого каталога. */
#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <stddef.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <mpi.h>

#define REPORT_DIGITS     3
#define TOTAL_TIME_DIGITS 3
#define MAX_GPUS_ARG      64

void cuda_ok(cudaError_t e, const char *msg);
void mpi_ok(int err, const char *msg);
int  mpi_cuda_aware(void);

typedef enum { ENV_CUSTOM, ENV_HOST, ENV_DEFAULT } Env;
typedef enum { STAT_ALL, STAT_AVG, STAT_MED, STAT_MIN, STAT_MAX, STAT_STD, STAT_VAR } StatOut;
typedef enum { MODE_ONE_TO_ONE, MODE_ALL_TO_ALL, MODE_ONE_TO_ALL, MODE_ALL_TO_ONE } Mode;

/* ack (10 чисел):
     [0] avg_us  [1] med_us  [2] min_us  [3] max_us
     [4] var_us  [5] std_us  [6] valid (1.0 = данные есть)
     --verify:   [7] checked — проверено итераций
                 [8] bad     — после полного завершения приёма длина или содержимое неверны
                 [9] early   — в момент отметки времени края сообщения ещё не те
                               (только одновременные режимы) */
#define ACK_FIELDS 10

typedef struct {
	size_t      nbytes;
	int         warmup;
	int         iters;
	Env         env;
	StatOut     stat_out;
	Mode        mode;
	const char *out_path;            /* NULL — без файла */
	int         save_raw_samples;
	int         debug;
	int         random_order;        /* --order random */
	unsigned    seed;                /* --seed */
	int         cpu_log;             /* one_to_one: --cpu-log */
	double      prespin_sec;         /* --prespin-sec */
	int         pin_main;            /* --pin-main K: K-й CPU маски, -1 — нет */
	int         probe;               /* --probe */
	int         gpus[MAX_GPUS_ARG];  /* --gpus 0,2: GPU для локальных рангов узла по порядку */
	int         n_gpus;              /* 0 — ключа нет */
	int         sys_via_host;        /* one_to_one: пары GPU разных процессоров узла — через RAM */
	int         sys_via_host_def;    /* включено по умолчанию (custom, one_to_one), а не ключом */
	int         p2p_nonblocking;     /* one_to_one: --p2p nonblocking */
	int         verify;              /* --verify */
} Args;

/* Задание пары для one_to_one: рассылается рангом 0 как MPI_BYTE. */
typedef struct {
	int src_rank;
	int dst_rank;
	int stop;
} Task;

void        help(int rank);
Args        parse_args(int argc, char **argv, int rank);
/* "0,2" -> {0, 2}; возвращает число номеров, 0 при ошибке. */
int         parse_gpu_list(const char *s, int *out, int max);
const char *mode_to_string(Mode mode);
const char *env_to_string(Env env);

#define DBG_LOG(rank_, args_, ...)                                   \
	do {                                                             \
		if ((args_)->debug) {                                        \
			fprintf(stderr, "[DBG r%d] ", (rank_));                  \
			fprintf(stderr, __VA_ARGS__);                            \
			fputc('\n', stderr);                                     \
		}                                                            \
	} while (0)

/* Вывод результатов: ранг 0 пишет в stdout и, если задан --out, в файл. */
void out_open(int rank, const char *path);
void out_printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void out_close(void);

/* Метки рангов "<цифры имени узла>.<номер ранга на узле>": labels[r] указывает на
   строку длиной до LABEL_LEN. Память освобождается free_rank_labels. */
#define LABEL_LEN 64
char **build_rank_labels(const char *hosts_recv, int nproc, int host_len);
void   free_rank_labels(char **labels, int nproc);

/* Заполняет ack[0..6] по серии замеров (n штук). */
void fill_ack(const double *samples_us, size_t n, double *ack);

/* Дописывает серию в raw/<base>_src<src>_dst<dst>.raw рядом с --out.
   С start_times (по одному на замер, может быть NULL) строка — «время_начала_с задержка_мкс»,
   без них — только задержка. */
void append_raw_samples(const Args *args, const char *src_label, const char *dst_label,
                        const double *samples_us, size_t n, const double *start_times);
/* Ряд чисел в файл с другим расширением и точностью (например, .scpu, .sprobe). */
void append_raw_series(const Args *args, const char *src_label, const char *dst_label,
                       const double *values, size_t n, const char *ext, int digits);

/* Закрепляет вызывающий поток за k-м (с 0) CPU его текущей маски. Номер CPU или -1. */
int pin_thread_to_mask_cpu(int k);
int current_cpu(void);

/* Секунды с начала эпохи по системным часам (общие для узлов при NTP). */
double epoch_now(void);

/* Генератор для случайного порядка (splitmix64) и тасование Фишера — Йейтса. */
typedef struct { unsigned long long s; } Rng;
void               rng_seed(Rng *r, unsigned long long seed);
unsigned long long rng_next(Rng *r);
size_t             rng_below(Rng *r, size_t n);   /* равномерно в [0, n) */
void               shuffle_ints(int *a, size_t n, Rng *r);

/* Тесты CPU для --probe, нс на шаг: core_ns — цепочка зависимых умножений (частота ядра);
   l3_ns — переходы по указателям по 8 МБ (больше L2, меньше L3). */
typedef struct {
	double core_ns;
	double l3_ns;
} ProbeResult;
ProbeResult cpu_probe(void);

/* --verify: шаблон зависит от позиции байта и метки tag из 1..255; метки соседних итераций
   различны. Всё — вне замеряемого интервала. */
#define VERIFY_EDGE 4096
unsigned verify_tag(long it, int src, int dst);
void     verify_fill(void *dev, size_t n, unsigned tag);
int      verify_ok(const void *dev, size_t n, unsigned tag);
int      verify_host_ok(const void *host, size_t n, unsigned tag);
/* Только первые и последние VERIFY_EDGE байт, чтение на своём неблокирующем потоке. */
int      verify_edges_ok(const void *dev, size_t n, unsigned tag, cudaStream_t stream);
void     print_verify_line(const char *src_label, const char *dst_label, const double *ack);
void     print_pair_line(const char *line_name, const char *src_label, const char *dst_label,
                         const double *ack, StatOut stat);

/* Окно общей оперативной памяти узла (MPI_Win_allocate_shared) для --sys-via-host.
   Сегменты регистрируются в CUDA (cudaHostRegister). Создание и освобождение коллективны
   по node_comm. */
typedef struct {
	MPI_Win win;
	int     rank;
	int     debug;
	char   *registered[MAX_GPUS_ARG];
	int     n_registered;
} SharedHostWindow;

void  shared_window_create(SharedHostWindow *w, MPI_Comm node_comm, size_t my_bytes, int rank,
                           const Args *args);
/* Сегмент ранга local_rank узла, если он не меньше need байт, иначе NULL. */
char *shared_window_segment(SharedHostWindow *w, int local_rank, size_t need);
void  shared_window_sync(const SharedHostWindow *w);
void  shared_window_free(SharedHostWindow *w);

/* Выделение памяти с аварийным выходом при нехватке. */
void *xmalloc(size_t n);
void *xcalloc(size_t n, size_t size);

#endif
