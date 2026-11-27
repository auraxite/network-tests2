#define _GNU_SOURCE   /* sched_getcpu, pthread_setaffinity_np, CPU_* */

#include "gpu_common.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#if defined(OPEN_MPI) && OPEN_MPI
#include "mpi-ext.h"
#endif

void cuda_ok(cudaError_t e, const char *msg) {
	if (e != cudaSuccess) {
		fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
		abort();
	}
}

void mpi_ok(int err, const char *msg) {
	if (err == MPI_SUCCESS)
		return;
	char buf[MPI_MAX_ERROR_STRING];
	int  len = 0;
	if (MPI_Error_string(err, buf, &len) == MPI_SUCCESS && len > 0)
		fprintf(stderr, "%s: %s\n", msg, buf);
	else
		fprintf(stderr, "%s: MPI error code %d\n", msg, err);
	MPI_Abort(MPI_COMM_WORLD, 1);
}

int mpi_cuda_aware(void) {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
	return MPIX_Query_cuda_support() == 1;
#else
	return 0;
#endif
}

void *xmalloc(size_t n) {
	void *p = malloc(n ? n : 1);
	if (!p) {
		fprintf(stderr, "out of memory (%zu bytes)\n", n);
		abort();
	}
	return p;
}

void *xcalloc(size_t n, size_t size) {
	void *p = calloc(n ? n : 1, size ? size : 1);
	if (!p) {
		fprintf(stderr, "out of memory (%zu x %zu bytes)\n", n, size);
		abort();
	}
	return p;
}

/* ------------------------------------------------------------------ */
/* Аргументы командной строки                                          */
/* ------------------------------------------------------------------ */

void help(int rank) {
	if (rank != 0)
		return;
	printf(
		"gpu2 — GPU pair latency (one_to_one | all_to_all | one_to_all | all_to_one), C version\n"
		"  --bytes N       message size in bytes (default 4 MB)\n"
		"  --warmup N      warmup iterations per pair\n"
		"  --iters N       measured iterations per pair\n"
		"  --env E         custom | host | default (auto = custom, old name)\n"
		"                  host: buffers in VRAM are given to MPI, UCX copies through RAM itself\n"
		"                  (no cuda_ipc, no GPUDirect RDMA, pipeline fragments in RAM)\n"
		"                  default: only UCX_TLS=^tcp, no adapter choice; Open MPI\n"
		"                  and UCX pick the path themselves\n"
		"  --gpus L        GPU for local ranks in order, e.g. 0,2 (default: local rank)\n"
		"  --sys-via-host  one_to_one: pairs of GPUs of different CPUs on one node go via RAM\n"
		"                  (the program copies through a shared RAM buffer); other pairs as the env says.\n"
		"                  On by default for --env custom; --no-sys-via-host turns it off\n"
		"  --mode M        one_to_one | all_to_all | one_to_all | all_to_one (default one_to_one)\n"
		"  --order O       seq | random (default seq): one_to_one — order of pairs, other modes — order of rounds\n"
		"  --seed N        seed for --order random (default 1)\n"
		"  --cpu-log       one_to_one: CPU of sender/receiver per iteration (raw/*.scpu, *.rcpu)\n"
		"  --prespin-sec S busy-spin S seconds on every rank before the first pair (round)\n"
		"  --pin-main K    pin the main thread to the K-th CPU (from 0) of its mask after init\n"
		"  --p2p P         one_to_one: blocking (MPI_Send/MPI_Recv, default) |\n"
		"                  nonblocking (MPI_Isend/MPI_Irecv + MPI_Wait)\n"
		"  --probe         core and L3 latency probes before/after each pair (round in other modes)\n"
		"                  (raw/*.sprobe, *.rprobe: core_ns l3_ns before, then after)\n"
		"  --verify        new pattern in every message, receiver checks length and content outside\n"
		"                  the timed interval; lines \"verify src -> dst checked= bad= early=\"\n"
		"  --stat S        all | avg | med | min | max | var | std\n"
		"  --out FILE, -o FILE  write output to FILE (rank 0 only)\n"
		"  --debug, -d     verbose debug logs to stderr\n"
		"\n"
		"UCX variables are set before MPI_Init with overwrite=0 (an explicit export wins):\n"
		"  custom: UCX_TLS=cuda_copy,cuda_ipc,<ib>,cma,sm,self UCX_IB_GPU_DIRECT_RDMA=y\n"
		"          UCX_RNDV_SCHEME=auto UCX_NET_DEVICES=<IB adapter closest to the GPU>\n"
		"  host:   UCX_TLS=cuda_copy,<ib>,cma,sm,self UCX_IB_GPU_DIRECT_RDMA=n\n"
		"          UCX_RNDV_SCHEME=auto <fragment memory type>=host UCX_NET_DEVICES=<closest IB>\n"
		"  default: UCX_TLS=^tcp\n");
}

static void bad_arg(int rank, const char *fmt, ...) __attribute__((noreturn, format(printf, 2, 3)));
static void bad_arg(int rank, const char *fmt, ...) {
	if (rank == 0) {
		va_list ap;
		va_start(ap, fmt);
		vfprintf(stderr, fmt, ap);
		va_end(ap);
		fputc('\n', stderr);
	}
	MPI_Abort(MPI_COMM_WORLD, 1);
	exit(1);
}

static Mode parse_mode(const char *s, int rank) {
	if (!strcmp(s, "one_to_one") || !strcmp(s, "sequential") || !strcmp(s, "1to1") ||
	    !strcmp(s, "one_to_one_v2"))
		return MODE_ONE_TO_ONE;
	if (!strcmp(s, "all_to_all") || !strcmp(s, "alltoall") || !strcmp(s, "parallel"))
		return MODE_ALL_TO_ALL;
	if (!strcmp(s, "one_to_all")) return MODE_ONE_TO_ALL;
	if (!strcmp(s, "all_to_one")) return MODE_ALL_TO_ONE;
	bad_arg(rank, "unknown --mode: %s (use one_to_one|all_to_all|one_to_all|all_to_one)", s);
}

const char *mode_to_string(Mode mode) {
	switch (mode) {
	case MODE_ALL_TO_ALL: return "all_to_all";
	case MODE_ONE_TO_ALL: return "one_to_all";
	case MODE_ALL_TO_ONE: return "all_to_one";
	default:              return "one_to_one";
	}
}

static Env parse_env(const char *s, int rank) {
	if (!strcmp(s, "custom") || !strcmp(s, "auto")) return ENV_CUSTOM;   /* auto — прежнее имя */
	if (!strcmp(s, "host")) return ENV_HOST;
	if (!strcmp(s, "default")) return ENV_DEFAULT;
	bad_arg(rank, "unknown --env: %s (use custom|host|default)", s);
}

const char *env_to_string(Env env) {
	switch (env) {
	case ENV_HOST:    return "host";
	case ENV_DEFAULT: return "default";
	default:          return "custom";
	}
}

static StatOut parse_stat_out(const char *s, int rank) {
	if (!strcmp(s, "all")) return STAT_ALL;
	if (!strcmp(s, "avg")) return STAT_AVG;
	if (!strcmp(s, "med")) return STAT_MED;
	if (!strcmp(s, "min")) return STAT_MIN;
	if (!strcmp(s, "max")) return STAT_MAX;
	if (!strcmp(s, "var")) return STAT_VAR;
	if (!strcmp(s, "std")) return STAT_STD;
	bad_arg(rank, "unknown --stat: %s (use all|avg|med|min|max|var|std)", s);
}

int parse_gpu_list(const char *s, int *out, int max) {
	int n = 0;
	const char *p = s;
	if (!s || !*s) return 0;
	for (;;) {
		const char *q = p;
		while (*q && *q != ',') {
			if (!isdigit((unsigned char)*q)) return 0;
			++q;
		}
		if (q == p || n >= max) return 0;
		out[n++] = atoi(p);
		if (!*q) break;
		p = q + 1;
	}
	return n;
}

static const char *next_arg(int argc, char **argv, int *i, const char *name, int rank) {
	if (*i + 1 >= argc)
		bad_arg(rank, "missing value for %s", name);
	return argv[++*i];
}

Args parse_args(int argc, char **argv, int rank) {
	int  sys_flag = -1;   /* --sys-via-host: 1, --no-sys-via-host: 0, не задано: -1 */
	Args a;
	memset(&a, 0, sizeof(a));
	a.nbytes           = 4u * 1000u * 1000u;
	a.warmup           = 10;
	a.iters            = 50;
	a.env              = ENV_CUSTOM;
	a.stat_out         = STAT_ALL;
	a.mode             = MODE_ONE_TO_ONE;
	a.save_raw_samples = 1;
	a.seed             = 1;
	a.pin_main         = -1;

	for (int i = 1; i < argc; ++i) {
		const char *s = argv[i];
		if (!strcmp(s, "--bytes"))
			a.nbytes = (size_t)strtoull(next_arg(argc, argv, &i, s, rank), NULL, 10);
		else if (!strcmp(s, "--warmup"))
			a.warmup = atoi(next_arg(argc, argv, &i, s, rank));
		else if (!strcmp(s, "--iters"))
			a.iters = atoi(next_arg(argc, argv, &i, s, rank));
		else if (!strcmp(s, "--env"))
			a.env = parse_env(next_arg(argc, argv, &i, s, rank), rank);
		else if (!strcmp(s, "--mode"))
			a.mode = parse_mode(next_arg(argc, argv, &i, s, rank), rank);
		else if (!strcmp(s, "--stat"))
			a.stat_out = parse_stat_out(next_arg(argc, argv, &i, s, rank), rank);
		else if (!strcmp(s, "--out") || !strcmp(s, "-o"))
			a.out_path = next_arg(argc, argv, &i, "--out", rank);
		else if (!strcmp(s, "--debug") || !strcmp(s, "-d"))
			a.debug = 1;
		else if (!strcmp(s, "--order")) {
			const char *o = next_arg(argc, argv, &i, s, rank);
			if (!strcmp(o, "random")) a.random_order = 1;
			else if (!strcmp(o, "seq")) a.random_order = 0;
			else bad_arg(rank, "unknown --order: %s (use seq|random)", o);
		} else if (!strcmp(s, "--seed"))
			a.seed = (unsigned)strtoul(next_arg(argc, argv, &i, s, rank), NULL, 10);
		else if (!strcmp(s, "--cpu-log"))
			a.cpu_log = 1;
		else if (!strcmp(s, "--prespin-sec"))
			a.prespin_sec = atof(next_arg(argc, argv, &i, s, rank));
		else if (!strcmp(s, "--pin-main"))
			a.pin_main = atoi(next_arg(argc, argv, &i, s, rank));
		else if (!strcmp(s, "--probe"))
			a.probe = 1;
		else if (!strcmp(s, "--verify"))
			a.verify = 1;
		else if (!strcmp(s, "--p2p")) {
			const char *o = next_arg(argc, argv, &i, s, rank);
			if (!strcmp(o, "nonblocking")) a.p2p_nonblocking = 1;
			else if (!strcmp(o, "blocking")) a.p2p_nonblocking = 0;
			else bad_arg(rank, "unknown --p2p: %s (use blocking|nonblocking)", o);
		} else if (!strcmp(s, "--sys-via-host"))
			sys_flag = 1;
		else if (!strcmp(s, "--no-sys-via-host"))
			sys_flag = 0;
		else if (!strcmp(s, "--gpus")) {
			a.n_gpus = parse_gpu_list(next_arg(argc, argv, &i, s, rank), a.gpus, MAX_GPUS_ARG);
			if (a.n_gpus == 0)
				bad_arg(rank, "--gpus: список номеров через запятую, например 0,2");
		} else if (!strcmp(s, "--help") || !strcmp(s, "-h")) {
			help(rank);
			mpi_ok(MPI_Finalize(), "MPI_Finalize");
			exit(0);
		} else
			bad_arg(rank, "unknown arg: %s", s);
	}
	if (a.warmup < 0) bad_arg(rank, "--warmup must be >= 0");
	if (a.iters <= 0) bad_arg(rank, "--iters must be > 0");
	if (a.nbytes == 0) bad_arg(rank, "--bytes must be > 0");
	if (a.nbytes > (size_t)0x7fffffff) bad_arg(rank, "--bytes must fit in int (MPI count)");
	/* custom по умолчанию обходит прямой путь между GPU разных процессоров (NOTES.md gpu_tests);
	   ключ работает только в one_to_one, по умолчанию в других режимах выключается молча,
	   заданный явно — ошибка */
	if (a.p2p_nonblocking && a.mode != MODE_ONE_TO_ONE)
		bad_arg(rank, "--p2p nonblocking только для --mode one_to_one (остальные режимы и так неблокирующие)");
	if (sys_flag == 1 && a.mode != MODE_ONE_TO_ONE)
		bad_arg(rank, "--sys-via-host пока только для --mode one_to_one");
	if (sys_flag == -1) {
		a.sys_via_host     = a.env == ENV_CUSTOM && a.mode == MODE_ONE_TO_ONE;
		a.sys_via_host_def = a.sys_via_host;
	} else {
		a.sys_via_host = sys_flag == 1;
	}
	return a;
}

/* ------------------------------------------------------------------ */
/* Вывод                                                               */
/* ------------------------------------------------------------------ */

static int   g_out_rank = -1;
static FILE *g_out_file = NULL;

void out_open(int rank, const char *path) {
	g_out_rank = rank;
	if (rank != 0 || !path) return;
	g_out_file = fopen(path, "w");
	if (!g_out_file) {
		fprintf(stderr, "gpu2: cannot open --out %s\n", path);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
}

void out_printf(const char *fmt, ...) {
	if (g_out_rank != 0) return;
	va_list ap;
	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
	if (g_out_file) {
		va_start(ap, fmt);
		vfprintf(g_out_file, fmt, ap);
		va_end(ap);
	}
}

void out_close(void) {
	fflush(stdout);
	if (g_out_file) fclose(g_out_file);
	g_out_file = NULL;
}

/* ------------------------------------------------------------------ */
/* Метки, статистики                                                   */
/* ------------------------------------------------------------------ */

char **build_rank_labels(const char *hosts_recv, int nproc, int host_len) {
	char **labels = xcalloc((size_t)nproc, sizeof(char *));
	char  (*seen)[MPI_MAX_PROCESSOR_NAME] = xcalloc((size_t)nproc, MPI_MAX_PROCESSOR_NAME);
	int   *counts = xcalloc((size_t)nproc, sizeof(int));
	int    n_seen = 0;

	for (int r = 0; r < nproc; ++r) {
		char host[MPI_MAX_PROCESSOR_NAME];
		snprintf(host, sizeof(host), "%s", hosts_recv + (size_t)r * (size_t)host_len);
		char *dot = strchr(host, '.');                       /* короткое имя узла */
		if (dot) *dot = '\0';

		int k = 0;
		while (k < n_seen && strcmp(seen[k], host) != 0) ++k;
		if (k == n_seen) {
			snprintf(seen[n_seen], MPI_MAX_PROCESSOR_NAME, "%s", host);
			counts[n_seen++] = 0;
		}
		const int local_idx = counts[k]++;

		size_t pos = strlen(host);                           /* цифры в конце имени */
		while (pos > 0 && isdigit((unsigned char)host[pos - 1])) --pos;
		const char *token = host[pos] ? host + pos : host;

		labels[r] = xmalloc(LABEL_LEN);
		snprintf(labels[r], LABEL_LEN, "%s.%d", token, local_idx);
	}
	free(seen);
	free(counts);
	return labels;
}

void free_rank_labels(char **labels, int nproc) {
	if (!labels) return;
	for (int r = 0; r < nproc; ++r) free(labels[r]);
	free(labels);
}

static int cmp_double(const void *a, const void *b) {
	const double x = *(const double *)a, y = *(const double *)b;
	return (x > y) - (x < y);
}

void fill_ack(const double *samples, size_t n, double *ack) {
	for (int i = 0; i < ACK_FIELDS; ++i) ack[i] = 0.0;
	if (n == 0) return;

	double sum = 0.0, mn = samples[0], mx = samples[0];
	for (size_t i = 0; i < n; ++i) {
		sum += samples[i];
		if (samples[i] < mn) mn = samples[i];
		if (samples[i] > mx) mx = samples[i];
	}
	const double avg = sum / (double)n;

	double *sorted = xmalloc(n * sizeof(double));
	memcpy(sorted, samples, n * sizeof(double));
	qsort(sorted, n, sizeof(double), cmp_double);
	const size_t m   = n / 2;
	const double med = n % 2 == 0 ? (sorted[m - 1] + sorted[m]) * 0.5 : sorted[m];
	free(sorted);

	double var = 0.0;
	if (n > 1) {
		for (size_t i = 0; i < n; ++i) var += (samples[i] - avg) * (samples[i] - avg);
		var /= (double)(n - 1);
	}

	ack[0] = avg;
	ack[1] = med;
	ack[2] = mn;
	ack[3] = mx;
	ack[4] = var;
	ack[5] = sqrt(var);
	ack[6] = 1.0; /* valid */
}

void print_verify_line(const char *src_label, const char *dst_label, const double *ack) {
	out_printf("verify %s -> %s checked=%.0f bad=%.0f early=%.0f\n", src_label, dst_label,
	           ack[7], ack[8], ack[9]);
}

void print_pair_line(const char *line_name, const char *src_label, const char *dst_label,
                     const double *ack, StatOut stat) {
	const int d = REPORT_DIGITS;
	out_printf("%s %s -> %s ", line_name, src_label, dst_label);
	switch (stat) {
	case STAT_ALL:
		out_printf("avg_us=%.*f med_us=%.*f min_us=%.*f max_us=%.*f var_us=%.*f std_us=%.*f",
		           d, ack[0], d, ack[1], d, ack[2], d, ack[3], d, ack[4], d, ack[5]);
		break;
	case STAT_AVG: out_printf("avg_us=%.*f", d, ack[0]); break;
	case STAT_MED: out_printf("med_us=%.*f", d, ack[1]); break;
	case STAT_MIN: out_printf("min_us=%.*f", d, ack[2]); break;
	case STAT_MAX: out_printf("max_us=%.*f", d, ack[3]); break;
	case STAT_VAR: out_printf("var_us=%.*f", d, ack[4]); break;
	case STAT_STD: out_printf("std_us=%.*f", d, ack[5]); break;
	}
	out_printf("\n");
}

/* ------------------------------------------------------------------ */
/* --verify                                                            */
/* ------------------------------------------------------------------ */

unsigned verify_tag(long it, int src, int dst) {
	const unsigned long h = (unsigned long)it * 37ul + (unsigned long)src * 11ul +
	                        (unsigned long)dst * 5ul;
	return 1u + (unsigned)(h % 255ul);   /* 37 не кратно 255: соседние итерации различны */
}

static inline unsigned char verify_byte(size_t k, unsigned tag) {
	return (unsigned char)((((uint32_t)k * 2654435761u) >> 24) ^ tag);
}

static unsigned char *verify_scratch(size_t n) {
	static unsigned char *h = NULL;
	static size_t         cap = 0;
	if (cap < n) {
		free(h);
		h   = xmalloc(n);
		cap = n;
	}
	return h;
}

static int verify_range(const unsigned char *h, size_t first, size_t len, unsigned tag) {
	for (size_t k = 0; k < len; ++k)
		if (h[k] != verify_byte(first + k, tag)) return 0;
	return 1;
}

void verify_fill(void *dev, size_t n, unsigned tag) {
	unsigned char *h = verify_scratch(n);
	for (size_t k = 0; k < n; ++k) h[k] = verify_byte(k, tag);
	cuda_ok(cudaMemcpy(dev, h, n, cudaMemcpyHostToDevice), "verify_fill");
}

int verify_ok(const void *dev, size_t n, unsigned tag) {
	unsigned char *h = verify_scratch(n);
	cuda_ok(cudaMemcpy(h, dev, n, cudaMemcpyDeviceToHost), "verify_ok");
	return verify_range(h, 0, n, tag);
}

int verify_host_ok(const void *host, size_t n, unsigned tag) {
	return verify_range((const unsigned char *)host, 0, n, tag);
}

int verify_edges_ok(const void *dev, size_t n, unsigned tag, cudaStream_t stream) {
	static unsigned char *h = NULL;
	if (!h) cuda_ok(cudaMallocHost((void **)&h, 2 * VERIFY_EDGE), "cudaMallocHost(verify edges)");
	const char  *d        = (const char *)dev;
	const size_t head     = n < VERIFY_EDGE ? n : VERIFY_EDGE;
	size_t       tail_off = n;
	if (n > VERIFY_EDGE) tail_off = n - VERIFY_EDGE > head ? n - VERIFY_EDGE : head;
	const size_t tail = n - tail_off;
	cuda_ok(cudaMemcpyAsync(h, d, head, cudaMemcpyDeviceToHost, stream), "verify_edges head");
	if (tail)
		cuda_ok(cudaMemcpyAsync(h + VERIFY_EDGE, d + tail_off, tail, cudaMemcpyDeviceToHost, stream),
		        "verify_edges tail");
	cuda_ok(cudaStreamSynchronize(stream), "verify_edges sync");
	return verify_range(h, 0, head, tag) && verify_range(h + VERIFY_EDGE, tail_off, tail, tag);
}

/* ------------------------------------------------------------------ */
/* CPU: привязка, время, генератор, --probe                            */
/* ------------------------------------------------------------------ */

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

int current_cpu(void) { return sched_getcpu(); }

double epoch_now(void) {
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double mono_ns(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

void rng_seed(Rng *r, unsigned long long seed) { r->s = seed; }

unsigned long long rng_next(Rng *r) {   /* splitmix64 */
	unsigned long long z = (r->s += 0x9E3779B97F4A7C15ull);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

size_t rng_below(Rng *r, size_t n) {   /* без смещения: отбрасываем хвост */
	const unsigned long long lim = ~0ull - (~0ull % n);
	unsigned long long x;
	do x = rng_next(r); while (x >= lim);
	return (size_t)(x % n);
}

void shuffle_ints(int *a, size_t n, Rng *r) {
	for (size_t i = n; i > 1; --i) {
		const size_t j = rng_below(r, i);
		const int t = a[i - 1];
		a[i - 1] = a[j];
		a[j] = t;
	}
}

/* Строка кэша с номером следующей строки: один переход — один промах L2. */
typedef struct {
	uint32_t next;
	char     pad[60];
} __attribute__((aligned(64))) ProbeLine;

#define PROBE_LINES ((8u << 20) / sizeof(ProbeLine))

/* Случайный цикл по всем строкам (алгоритм Саттоло): аппаратная предвыборка
   не угадывает следующий адрес. Строится один раз. */
static const ProbeLine *probe_lines(void) {
	static ProbeLine *lines = NULL;
	if (lines) return lines;
	uint32_t *order = xmalloc(PROBE_LINES * sizeof(uint32_t));
	for (size_t i = 0; i < PROBE_LINES; ++i) order[i] = (uint32_t)i;
	Rng r;
	rng_seed(&r, 12345);
	for (size_t i = PROBE_LINES - 1; i > 0; --i) {
		const size_t j = rng_below(&r, i);   /* j в [0, i): цикл Саттоло */
		const uint32_t t = order[i];
		order[i] = order[j];
		order[j] = t;
	}
	if (posix_memalign((void **)&lines, 64, PROBE_LINES * sizeof(ProbeLine)) != 0) {
		fprintf(stderr, "probe: out of memory\n");
		abort();
	}
	for (size_t i = 0; i < PROBE_LINES; ++i)
		lines[order[i]].next = order[(i + 1) % PROBE_LINES];
	free(order);
	return lines;
}

static volatile uint64_t g_probe_sink = 0;
/* Константы читаются из volatile, чтобы компилятор не свернул цепочку в формулу. */
static volatile uint64_t g_probe_mul = 6364136223846793005ull;
static volatile uint64_t g_probe_add = 1442695040888963407ull;

ProbeResult cpu_probe(void) {
	const ProbeLine *lines = probe_lines();
	ProbeResult r = {0.0, 0.0};

	enum { CORE_STEPS = 200000, CHASE_STEPS = 50000 };
	uint64_t x = g_probe_sink | 1;
	const uint64_t mul = g_probe_mul, add = g_probe_add;
	double t0 = mono_ns();
	for (int i = 0; i < CORE_STEPS; ++i)
		x = x * mul + add;
	double t1 = mono_ns();
	r.core_ns = (t1 - t0) / CORE_STEPS;

	uint32_t p = (uint32_t)(x % PROBE_LINES);
	for (int i = 0; i < 2000; ++i) p = lines[p].next;   /* разогрев TLB и кэша */
	t0 = mono_ns();
	for (int i = 0; i < CHASE_STEPS; ++i) p = lines[p].next;
	t1 = mono_ns();
	r.l3_ns = (t1 - t0) / CHASE_STEPS;

	g_probe_sink = x + p;
	return r;
}

/* ------------------------------------------------------------------ */
/* Файлы raw/                                                          */
/* ------------------------------------------------------------------ */

static void sanitize(const char *lab, char *out, size_t cap) {
	size_t n = 0;
	for (const unsigned char *c = (const unsigned char *)lab; *c && n + 1 < cap; ++c)
		out[n++] = (isalnum(*c) || *c == '.' || *c == '-' || *c == '_') ? (char)*c : '_';
	out[n] = '\0';
	if (n == 0) snprintf(out, cap, "nolabel");
}

/* Открыть на дозапись raw/<base>_src<src>_dst<dst><ext> рядом с --out и записать
   строку-заголовок. NULL — писать не нужно или не получилось. */
static FILE *open_raw_file(const Args *args, const char *src_label, const char *dst_label,
                           const char *ext) {
	if (!args->save_raw_samples || !args->out_path || !*args->out_path)
		return NULL;

	char dir[4096], base[4096];
	const char *slash = strrchr(args->out_path, '/');
	if (slash) {
		snprintf(dir, sizeof(dir), "%.*s", (int)(slash - args->out_path), args->out_path);
		snprintf(base, sizeof(base), "%s", slash + 1);
	} else {
		snprintf(dir, sizeof(dir), ".");
		snprintf(base, sizeof(base), "%s", args->out_path);
	}
	const size_t bl = strlen(base);
	if (bl >= 4 && strcmp(base + bl - 4, ".txt") == 0) base[bl - 4] = '\0';

	char raw_dir[4200];
	snprintf(raw_dir, sizeof(raw_dir), "%s/raw", dir);
	if (mkdir(raw_dir, 0775) != 0 && errno != EEXIST) return NULL;

	char s[LABEL_LEN + 8], d[LABEL_LEN + 8], path[8400];
	sanitize(src_label, s, sizeof(s));
	sanitize(dst_label, d, sizeof(d));
	snprintf(path, sizeof(path), "%s/%s_src%s_dst%s%s", raw_dir, base, s, d, ext);
	FILE *f = fopen(path, "a");
	if (!f) return NULL;
	fprintf(f, "# src=%s dst=%s\n", src_label, dst_label);
	return f;
}

void append_raw_samples(const Args *args, const char *src_label, const char *dst_label,
                        const double *samples_us, size_t n, const double *start_times) {
	if (n == 0) return;
	FILE *f = open_raw_file(args, src_label, dst_label, ".raw");
	if (!f) return;
	/* Два столбца — время начала итерации (с, системные часы) и задержка (мкс);
	   без времени — только задержка, как в прежнем формате. */
	for (size_t i = 0; i < n; ++i) {
		if (start_times) fprintf(f, "%.6f ", start_times[i]);
		fprintf(f, "%.*f\n", REPORT_DIGITS, samples_us[i]);
	}
	fclose(f);
}

void append_raw_series(const Args *args, const char *src_label, const char *dst_label,
                       const double *values, size_t n, const char *ext, int digits) {
	if (n == 0) return;
	FILE *f = open_raw_file(args, src_label, dst_label, ext);
	if (!f) return;
	for (size_t i = 0; i < n; ++i) fprintf(f, "%.*f\n", digits, values[i]);
	fclose(f);
}

/* ------------------------------------------------------------------ */
/* Окно общей оперативной памяти узла                                  */
/* ------------------------------------------------------------------ */

void shared_window_create(SharedHostWindow *w, MPI_Comm node_comm, size_t my_bytes, int rank,
                          const Args *args) {
	memset(w, 0, sizeof(*w));
	w->win   = MPI_WIN_NULL;
	w->rank  = rank;
	w->debug = args->debug;
	void *base = NULL;
	mpi_ok(MPI_Win_allocate_shared((MPI_Aint)my_bytes, sizeof(char), MPI_INFO_NULL, node_comm,
	                               &base, &w->win),
	       "MPI_Win_allocate_shared");
}

char *shared_window_segment(SharedHostWindow *w, int local_rank, size_t need) {
	MPI_Aint size = 0;
	int      disp_unit = 0;
	void    *base = NULL;
	mpi_ok(MPI_Win_shared_query(w->win, local_rank, &size, &disp_unit, &base),
	       "MPI_Win_shared_query");
	char *p = (char *)base;
	if (!p || size < (MPI_Aint)need)
		return NULL;

	const cudaError_t reg = cudaHostRegister(p, need, cudaHostRegisterPortable);
	if ((reg == cudaSuccess || reg == cudaErrorHostMemoryAlreadyRegistered) &&
	    w->n_registered < MAX_GPUS_ARG) {
		w->registered[w->n_registered++] = p;
	} else if (w->debug) {
		fprintf(stderr, "[DBG r%d] shared host register skipped (local rank %d): %s\n", w->rank,
		        local_rank, cudaGetErrorString(reg));
	}
	if (reg != cudaSuccess)
		cudaGetLastError();   /* сбросить ошибку CUDA, чтобы она не всплыла позже */
	return p;
}

void shared_window_sync(const SharedHostWindow *w) {
	if (w->win != MPI_WIN_NULL) mpi_ok(MPI_Win_sync(w->win), "MPI_Win_sync");
}

void shared_window_free(SharedHostWindow *w) {
	for (int i = 0; i < w->n_registered; ++i)
		cuda_ok(cudaHostUnregister(w->registered[i]), "cudaHostUnregister");
	w->n_registered = 0;
	if (w->win != MPI_WIN_NULL)
		mpi_ok(MPI_Win_free(&w->win), "MPI_Win_free");
}
