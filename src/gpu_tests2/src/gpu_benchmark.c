/* gpu_tests2: главная программа. До MPI_Init выбирает GPU и сетевой адаптер и выставляет
   переменные UCX для среды; после — собирает сведения о рангах, печатает шапку и запускает режим. */
#define _GNU_SOURCE   /* realpath, setenv */

#include "gpu_common.h"
#include "gpu_one_to_one.h"
#include "gpu_simultaneous.h"

#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* Имя переменной UCX для типа памяти промежуточных буферов передачи частями (pipeline).
   Подставляет build/build.sh по выводу `ucx_info -c` той сборки UCX, с которой собирается
   программа: в разных версиях UCX имя разное. Пусто — переменная в этой сборке не найдена. */
#ifndef UCX_FRAG_MEM_VAR
#define UCX_FRAG_MEM_VAR ""
#endif

/* ------------------------------------------------------------------ */
/* До MPI_Init: GPU, сетевой адаптер, переменные UCX                   */
/* ------------------------------------------------------------------ */

/* IB-транспорты, допустимые и для Verbs, и для MLX5. */
static const char *IB_TLS = "rc_verbs,rc_mlx5,ud_verbs,ud_mlx5,dc_mlx5";

/* Значение ключа до разбора аргументов (NULL, если ключа нет). */
static const char *pre_scan_arg(int argc, char **argv, const char *key) {
	for (int i = 1; i < argc - 1; ++i)
		if (!strcmp(argv[i], key)) return argv[i + 1];
	return NULL;
}

static void lower(char *s) {
	for (; *s; ++s) *s = (char)tolower((unsigned char)*s);
}

/* NUMA-узел (процессор) GPU по PCI-адресу из /sys, -1 если не узнать. */
static int gpu_numa_node(const char *pci) {
	char p[64], path[160];
	snprintf(p, sizeof(p), "%s", pci);
	lower(p);
	snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", p);
	FILE *f = fopen(path, "r");
	if (!f) return -1;
	int n = -1;
	if (fscanf(f, "%d", &n) != 1) n = -1;
	fclose(f);
	return n;
}

/* Номер ранга на узле по переменным лаунчера, -1 если не нашли. OMPI_* первым: под mpirun
   внутри задания Slurm процессы наследуют SLURM_LOCALID служебного процесса. */
static int env_local_rank(void) {
	static const char *vars[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "PMI_LOCAL_RANK",
	                             "MV2_COMM_WORLD_LOCAL_RANK"};
	for (size_t i = 0; i < sizeof(vars) / sizeof(vars[0]); ++i) {
		const char *s = getenv(vars[i]);
		if (!s || !*s) continue;
		char *end = NULL;
		const long n = strtol(s, &end, 10);
		if (end != s && n >= 0) return (int)n;
	}
	return -1;
}

static int read_first_line(const char *path, char *out, size_t cap) {
	FILE *f = fopen(path, "r");
	if (!f) return 0;
	if (!fgets(out, (int)cap, f)) {
		fclose(f);
		return 0;
	}
	fclose(f);
	size_t n = strlen(out);
	while (n > 0 && (out[n - 1] == '\n' || out[n - 1] == '\r' || out[n - 1] == ' ')) out[--n] = '\0';
	return 1;
}

/* Есть ли у карты порт InfiniBand (Ethernet-карты пропускаем). */
static int is_infiniband(const char *dev_dir) {
	char ports[PATH_MAX];
	snprintf(ports, sizeof(ports), "%s/ports", dev_dir);
	DIR *d = opendir(ports);
	if (!d) return 0;
	int ib = 0;
	const struct dirent *e;
	while ((e = readdir(d)) != NULL) {
		if (e->d_name[0] == '.') continue;
		char path[PATH_MAX + 300], line[64];
		snprintf(path, sizeof(path), "%s/%s/link_layer", ports, e->d_name);
		if (read_first_line(path, line, sizeof(line)) && !strcmp(line, "InfiniBand")) {
			ib = 1;
			break;
		}
	}
	closedir(d);
	return ib;
}

/* Ближайшая к GPU IB-карта: самый длинный общий префикс путей в /sys (общий коммутатор PCIe).
   Пишет имя в out, 0 — не нашли. */
static int closest_ib_device(const char *gpu_pci, char *out, size_t cap) {
	char gdev[PATH_MAX], gpath[PATH_MAX];
	snprintf(gdev, sizeof(gdev), "/sys/bus/pci/devices/%s", gpu_pci);
	if (!realpath(gdev, gpath)) return 0;
	DIR *d = opendir("/sys/class/infiniband");
	if (!d) return 0;

	size_t best_len = 0;
	int    found = 0;
	const struct dirent *e;
	while ((e = readdir(d)) != NULL) {
		if (e->d_name[0] == '.') continue;
		char dev_dir[PATH_MAX], dev_link[PATH_MAX + 16], ipath[PATH_MAX];
		snprintf(dev_dir, sizeof(dev_dir), "/sys/class/infiniband/%s", e->d_name);
		if (!is_infiniband(dev_dir)) continue;
		snprintf(dev_link, sizeof(dev_link), "%s/device", dev_dir);
		if (!realpath(dev_link, ipath)) continue;
		size_t n = 0;
		while (gpath[n] && ipath[n] && gpath[n] == ipath[n]) ++n;
		if (!found || n > best_len) {
			snprintf(out, cap, "%s", e->d_name);
			best_len = n;
			found = 1;
		}
	}
	closedir(d);
	return found;
}

/* Выбрать GPU по локальному номеру и карту под него, до MPI_Init. Возвращает номер GPU или -1
   (тогда диагностика будет после MPI_Init). nic_out — имя карты или пустая строка. */
static int bind_gpu_and_nic_early(char *nic_out, size_t cap, int pick_nic, const int *gpus,
                                  int n_gpus) {
	nic_out[0] = '\0';
	const int lrank = env_local_rank();
	if (lrank < 0) return -1;

	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return -1;
	int dev = count > 1 ? lrank : 0;
	if (n_gpus > 0) {   /* --gpus: номер GPU берётся из списка по локальному рангу */
		if (lrank >= n_gpus) return -1;
		dev = gpus[lrank];
	}
	if (dev >= count || cudaSetDevice(dev) != cudaSuccess) return -1;

	char pci[32] = {0};
	if (cudaDeviceGetPCIBusId(pci, sizeof(pci), dev) != cudaSuccess) return dev;
	lower(pci);   /* в /sys путь в нижнем регистре */

	if (!pick_nic) return dev;   /* --env default: адаптер выбирает UCX */
	if (closest_ib_device(pci, nic_out, cap)) {
		char val[96];
		snprintf(val, sizeof(val), "%s:1", nic_out);
		setenv("UCX_NET_DEVICES", val, 0);
	}
	return dev;
}

/* Переменные UCX для среды. overwrite=0: явный export в скрипте важнее.
   custom — прямые пути к памяти GPU (cuda_ipc, GPUDirect RDMA).
   host   — буферы те же (в памяти GPU), но UCX запрещены прямые пути: cuda_ipc исключён из
            UCX_TLS, GPUDirect RDMA выключен, промежуточные буферы передачи частями — в RAM.
            UCX сам копирует GPU -> RAM (cuda_copy), передаёт (sm/cma или IB) и копирует RAM -> GPU.
   UCX_RNDV_THRESH и UCX_MEMTYPE_CACHE намеренно не выставляются. */
static void set_ucx_for_env(const char *env) {
	char tls[256];
	if (!strcmp(env, "default")) {
		/* default: только исключить TCP. Без этого на узлах с Kubernetes UCX берёт TCP через
		   flannel.1, соединение между узлами отклоняется и не создаётся целиком (NOTES.md gpu_tests) */
		setenv("UCX_TLS", "^tcp", 0);
		return;
	}
	if (!strcmp(env, "host")) {
		snprintf(tls, sizeof(tls), "cuda_copy,%s,cma,sm,self", IB_TLS);
		setenv("UCX_TLS", tls, 0);
		setenv("UCX_IB_GPU_DIRECT_RDMA", "n", 0);
		setenv("UCX_RNDV_SCHEME", "auto", 0);
		if (UCX_FRAG_MEM_VAR[0]) setenv(UCX_FRAG_MEM_VAR, "host", 0);
		return;
	}
	snprintf(tls, sizeof(tls), "cuda_copy,cuda_ipc,%s,cma,sm,self", IB_TLS);
	setenv("UCX_TLS", tls, 0);
	setenv("UCX_IB_GPU_DIRECT_RDMA", "y", 0);
	setenv("UCX_RNDV_SCHEME", "auto", 0);
}

/* ------------------------------------------------------------------ */
/* После MPI_Init                                                      */
/* ------------------------------------------------------------------ */

#define HOST_LEN MPI_MAX_PROCESSOR_NAME
#define PCI_LEN  32
#define NIC_LEN  32

/* Сведения обо всех рангах, собранные через MPI_Allgather. */
typedef struct {
	char *hosts, *pcis, *nics;
	int  *gpu_counts, *local_gpus;
} RankInfo;

static const char *ri_host(const RankInfo *ri, int r) { return ri->hosts + (size_t)r * HOST_LEN; }
static const char *ri_pci(const RankInfo *ri, int r)  { return ri->pcis + (size_t)r * PCI_LEN; }
static const char *ri_nic(const RankInfo *ri, int r)  { return ri->nics + (size_t)r * NIC_LEN; }

static RankInfo gather_rank_info(int nproc, int local_gpu_count, int local_gpu,
                                 const char *early_nic) {
	char my_host[HOST_LEN] = {0}, my_pci[PCI_LEN] = {0}, my_nic[NIC_LEN] = {0};
	int  len = 0;
	mpi_ok(MPI_Get_processor_name(my_host, &len), "MPI_Get_processor_name");
	if (cudaDeviceGetPCIBusId(my_pci, sizeof(my_pci), local_gpu) != cudaSuccess)
		snprintf(my_pci, sizeof(my_pci), "n/a");
	snprintf(my_nic, sizeof(my_nic), "%s", early_nic[0] ? early_nic : "ucx-auto");

	RankInfo ri;
	ri.hosts      = xcalloc((size_t)nproc, HOST_LEN);
	ri.pcis       = xcalloc((size_t)nproc, PCI_LEN);
	ri.nics       = xcalloc((size_t)nproc, NIC_LEN);
	ri.gpu_counts = xcalloc((size_t)nproc, sizeof(int));
	ri.local_gpus = xcalloc((size_t)nproc, sizeof(int));
	mpi_ok(MPI_Allgather(my_host, HOST_LEN, MPI_CHAR, ri.hosts, HOST_LEN, MPI_CHAR, MPI_COMM_WORLD),
	       "MPI_Allgather hostnames");
	mpi_ok(MPI_Allgather(&local_gpu_count, 1, MPI_INT, ri.gpu_counts, 1, MPI_INT, MPI_COMM_WORLD),
	       "MPI_Allgather gpu_counts");
	mpi_ok(MPI_Allgather(&local_gpu, 1, MPI_INT, ri.local_gpus, 1, MPI_INT, MPI_COMM_WORLD),
	       "MPI_Allgather local_gpus");
	mpi_ok(MPI_Allgather(my_pci, PCI_LEN, MPI_CHAR, ri.pcis, PCI_LEN, MPI_CHAR, MPI_COMM_WORLD),
	       "MPI_Allgather pci");
	/* Карту собираем со всех рангов: ранги выбирают разные карты. */
	mpi_ok(MPI_Allgather(my_nic, NIC_LEN, MPI_CHAR, ri.nics, NIC_LEN, MPI_CHAR, MPI_COMM_WORLD),
	       "MPI_Allgather nic");
	return ri;
}

static void free_rank_info(RankInfo *ri) {
	free(ri->hosts); free(ri->pcis); free(ri->nics); free(ri->gpu_counts); free(ri->local_gpus);
}

static const char *ONE_RANK_PER_GPU_HINT =
	"  Launch exactly one rank per GPU, e.g.:\n"
	"    srun --ntasks-per-node=<gpus_per_node> --gpus-per-node=<gpus_per_node> --gpu-bind=none ...\n";

/* Ранг 0: у каждого ранга есть GPU, и два ранга одного узла не делят один GPU. */
static void check_gpu_assignment(int nproc, const RankInfo *ri) {
	for (int r = 0; r < nproc; ++r)
		if (ri->gpu_counts[r] <= 0) {
			fprintf(stderr, "rank %d sees no GPUs\n", r);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	for (int a = 0; a < nproc; ++a)
		for (int b = a + 1; b < nproc; ++b)
			if (!strcmp(ri_host(ri, a), ri_host(ri, b)) && !strcmp(ri_pci(ri, a), ri_pci(ri, b))) {
				fprintf(stderr,
				        "error: ranks %d and %d share GPU %s on %s (GPU oversubscription).\n%s"
				        "  Если стоит --gpu-bind=closest — это и есть причина: на этих узлах все GPU\n"
				        "  равноудалены по CPU-аффинности. Бери --gpu-bind=none либо map_gpu:0,1,2,3.\n",
				        a, b, ri_pci(ri, a), ri_host(ri, a), ONE_RANK_PER_GPU_HINT);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
}

static void print_header(const Args *args, int nproc, const RankInfo *ri) {
	out_printf("Env: %s\nMode: %s\nTransport: mpi_ucx\nTimer: mpi\nBytes: %zu\nWarmup: %d\nIters: %d\n",
	           env_to_string(args->env), mode_to_string(args->mode), args->nbytes, args->warmup,
	           args->iters);
	out_printf("Implementation: C (gpu_tests2)\n");
	if (args->env == ENV_HOST)
		out_printf("Host staging: ucx (buffers in VRAM, UCX copies through RAM)\n");
	int total_gpus = 0;
	for (int r = 0; r < nproc; ++r) total_gpus += ri->gpu_counts[r];
	out_printf("Ranks: %d, total visible GPUs: %d\nRank map:\n", nproc, total_gpus);
	for (int r = 0; r < nproc; ++r)
		out_printf("  r%d hostname=%s local_gpu=%d visible_gpus=%d pci=%s nic=%s\n", r, ri_host(ri, r),
		           ri->local_gpus[r], ri->gpu_counts[r], ri_pci(ri, r), ri_nic(ri, r));
}

/* Действующие переменные UCX и словесное описание пути. */
static void print_ucx_config(int cuda_aware, Env env) {
	out_printf("CUDA-aware MPI: %s\n", cuda_aware ? "yes" : "no");
	{
		char lib[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
		int  len = 0;
		MPI_Get_library_version(lib, &len);
		lib[strcspn(lib, "\n,")] = '\0';                      /* «Open MPI v5.0.10» */
		const char *ucx = getenv("CLUSTBENCH_UCX_VERSION");   /* выставляет task.sbatch */
		out_printf("MPI library: %s\nUCX version: %s\n", lib, ucx ? ucx : "unknown");
	}
	out_printf("UCX config (effective after MPI_Init):\n");
	const char *keys[] = {"UCX_TLS", "UCX_RNDV_THRESH", "UCX_IB_GPU_DIRECT_RDMA", "UCX_RNDV_SCHEME",
	                      "UCX_MEMTYPE_CACHE", "UCX_NET_DEVICES", "UCX_LOG_LEVEL", UCX_FRAG_MEM_VAR};
	for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
		if (!keys[i][0]) continue;
		const char *v = getenv(keys[i]);
		out_printf("  %s = %s\n", keys[i], v ? v : "<not set>");
	}
	if (env == ENV_HOST && !UCX_FRAG_MEM_VAR[0])
		out_printf("  WARNING: build found no UCX variable for pipeline fragment memory type;\n"
		           "           UCX may keep pipeline fragments in VRAM (see README.md)\n");

	out_printf("Route:\n");
	switch (env) {
	case ENV_DEFAULT:
		out_printf("  library default except UCX_TLS=^tcp (adapter chosen by UCX)\n");
		break;
	case ENV_HOST:
		out_printf("  intra-node: UCX via RAM   (cuda_copy + sm/cma, no cuda_ipc)\n"
		           "  inter-node: UCX via RAM   (cuda_copy + IB, no GPUDirect RDMA; large msgs in "
		           "fragments)\n");
		break;
	default: {
		out_printf("  intra-node: custom           (UCX may use CUDA IPC or NVLink if available)\n");
		const char *rndv = getenv("UCX_RNDV_THRESH");
		if (rndv && !strcmp(rndv, "0"))
			out_printf("  inter-node: UCX rendezvous   (forced for all sizes by UCX_RNDV_THRESH=0)\n");
		else
			out_printf("  inter-node: UCX auto         (small msgs: eager; large msgs: rendezvous, "
			           "GPU Direct if GDR available)\n");
	}
	}
}

int main(int argc, char **argv) {
	/* Оба шага обязаны идти до MPI_Init: UCX читает окружение при инициализации. */
	const char *env_arg  = pre_scan_arg(argc, argv, "--env");
	const char *gpus_arg = pre_scan_arg(argc, argv, "--gpus");
	if (!env_arg || !strcmp(env_arg, "auto")) env_arg = "custom";
	int  early_gpus[MAX_GPUS_ARG];
	const int n_early_gpus = gpus_arg ? parse_gpu_list(gpus_arg, early_gpus, MAX_GPUS_ARG) : 0;
	char early_nic[NIC_LEN];
	const int early_gpu = bind_gpu_and_nic_early(early_nic, sizeof(early_nic),
	                                             strcmp(env_arg, "default") != 0, early_gpus,
	                                             n_early_gpus);
	set_ucx_for_env(env_arg);

	mpi_ok(MPI_Init(&argc, &argv), "MPI_Init");
	int rank = 0, nproc = 1;
	mpi_ok(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
	mpi_ok(MPI_Comm_size(MPI_COMM_WORLD, &nproc), "MPI_Comm_size");

	const Args args       = parse_args(argc, argv, rank);
	const int  cuda_aware = mpi_cuda_aware();
	/* Буферы всегда в памяти GPU (в том числе в host): без CUDA-aware MPI мерить нечего. */
	if (!cuda_aware) {
		if (rank == 0)
			fprintf(stderr, "gpu2: MPI без поддержки CUDA (MPIX_Query_cuda_support != 1): буферы в "
			                "памяти GPU передать нельзя. Нужен Open MPI, собранный с CUDA и UCX.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	/* Ранги этого узла */
	MPI_Comm node_comm = MPI_COMM_NULL;
	mpi_ok(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm),
	       "MPI_Comm_split_type");
	int local_rank = 0, node_size = 1;
	mpi_ok(MPI_Comm_rank(node_comm, &local_rank), "MPI_Comm_rank(node)");
	mpi_ok(MPI_Comm_size(node_comm, &node_size), "MPI_Comm_size(node)");
	int *node_ranks = xcalloc((size_t)node_size, sizeof(int));
	mpi_ok(MPI_Allgather(&rank, 1, MPI_INT, node_ranks, 1, MPI_INT, node_comm),
	       "MPI_Allgather node_ranks");
	int *on_my_node = xcalloc((size_t)nproc, sizeof(int));
	for (int i = 0; i < node_size; ++i) on_my_node[node_ranks[i]] = 1;

	/* GPU ранга: номер на узле среди видимых устройств (один GPU — номер 0). Остаток от
	   деления не используется: двум рангам нельзя давать один GPU. */
	int local_gpu_count = 0;
	cudaGetDeviceCount(&local_gpu_count);
	if (local_gpu_count <= 0) {
		fprintf(stderr, "rank %d: no visible CUDA devices\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	int local_gpu = 0;
	if (local_gpu_count > 1) {
		if (local_rank >= local_gpu_count) {
			fprintf(stderr, "rank %d (local_rank %d) has no dedicated GPU: node exposes only %d devices.\n%s",
			        rank, local_rank, local_gpu_count, ONE_RANK_PER_GPU_HINT);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		local_gpu = local_rank;
	}
	if (args.n_gpus > 0) {   /* --gpus 0,2: GPU по списку, а не по локальному рангу */
		if (local_rank >= args.n_gpus || args.gpus[local_rank] >= local_gpu_count) {
			fprintf(stderr, "rank %d (local_rank %d): в --gpus нет GPU для этого ранга или номер больше "
			                "числа видимых GPU (%d)\n", rank, local_rank, local_gpu_count);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		local_gpu = args.gpus[local_rank];
	}
	if (early_gpu >= 0 && early_gpu != local_gpu)
		fprintf(stderr, "rank %d: ранний выбор GPU (%d, из окружения лаунчера) разошёлся с MPI "
		                "local_rank (%d). Беру %d, но UCX_NET_DEVICES уже зафиксирован под %d — выбор "
		                "карты может быть неоптимальным.\n",
		        rank, early_gpu, local_gpu, local_gpu, early_gpu);
	cuda_ok(cudaSetDevice(local_gpu), "cudaSetDevice");

	out_open(rank, args.out_path);

	RankInfo ri = gather_rank_info(nproc, local_gpu_count, local_gpu, early_nic);
	if (rank == 0) check_gpu_assignment(nproc, &ri);
	char **rank_labels = build_rank_labels(ri.hosts, nproc, HOST_LEN);

	print_header(&args, nproc, &ri);
	print_ucx_config(cuda_aware, args.env);

	/* --pin-main: главный поток (он крутит опрос в замерах) — на один CPU. После MPI_Init и
	   создания контекста CUDA, поэтому их служебные потоки остаются на всей маске ранга. */
	int pinned = -1;
	if (args.pin_main >= 0) {
		cuda_ok(cudaFree(NULL), "cudaFree(0) context");
		pinned = pin_thread_to_mask_cpu(args.pin_main);
	}
	{
		int *all = rank == 0 ? xmalloc((size_t)nproc * sizeof(int)) : NULL;
		mpi_ok(MPI_Gather(&pinned, 1, MPI_INT, all, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Gather pin");
		if (rank == 0) {
			out_printf("Pin:");
			for (int r = 0; r < nproc; ++r) out_printf(" r%d=%d", r, all[r]);
			if (args.pin_main < 0) out_printf(" (off)\n");
			else out_printf(" (K=%d)\n", args.pin_main);
		}
		free(all);
	}

	/* Процессор каждого GPU — для --sys-via-host и для шапки */
	int *gpu_numa = xcalloc((size_t)nproc, sizeof(int));
	{
		int my_numa = gpu_numa_node(ri_pci(&ri, rank));
		mpi_ok(MPI_Allgather(&my_numa, 1, MPI_INT, gpu_numa, 1, MPI_INT, MPI_COMM_WORLD),
		       "MPI_Allgather gpu_numa");
		out_printf("GPU NUMA:");
		for (int r = 0; r < nproc; ++r) out_printf(" r%d=%d", r, gpu_numa[r]);
		out_printf("\nSYS pairs via host: %s%s\n", args.sys_via_host ? "on" : "off",
		           args.sys_via_host_def ? " (default for custom)" : "");
	}

	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier before benchmark");
	if (args.mode == MODE_ONE_TO_ONE)
		schedule_one_to_one(rank, nproc, &args, node_comm, local_rank, on_my_node, rank_labels,
		                    gpu_numa);
	else
		schedule_simultaneous(rank, nproc, &args, rank_labels);

	out_close();
	free(gpu_numa);
	free_rank_labels(rank_labels, nproc);
	free_rank_info(&ri);
	free(on_my_node);
	free(node_ranks);
	mpi_ok(MPI_Comm_free(&node_comm), "MPI_Comm_free");
	mpi_ok(MPI_Finalize(), "MPI_Finalize");
	return 0;
}
