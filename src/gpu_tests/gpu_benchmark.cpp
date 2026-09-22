#include "gpu_all_to_all.hpp"
#include "gpu_common.hpp"
#include "gpu_one_to_one.hpp"

#include <cctype>    // std::tolower
#include <climits>   // PATH_MAX
#include <cstdlib>   // setenv, getenv, realpath
#include <cstring>
#include <dirent.h>  // opendir/readdir — обход /sys
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

// ------------------------------------------------------------------ //
// UCX helpers (called before and after MPI_Init)                      //
// ------------------------------------------------------------------ //

/* Scan argv for --env before MPI_Init (args aren't parsed yet). */
static bool pre_scan_host_env(int argc, char **argv) {
	for (int i = 1; i < argc - 1; ++i)
		if (std::string(argv[i]) == "--env" && std::string(argv[i + 1]) == "host")
			return true;
	return false;
}

/* IB transports valid for both Verbs and MLX5 drivers. */
static const char *IB_TLS = "rc_verbs,rc_mlx5,ud_verbs,ud_mlx5,dc_mlx5";

// ------------------------------------------------------------------ //
// Раннее связывание: GPU и NIC выбираются ДО MPI_Init                  //
// ------------------------------------------------------------------ //
//
// Порядок принципиален. UCX читает UCX_NET_DEVICES только при своей
// инициализации, то есть внутри MPI_Init — выставлять её позже бесполезно.
// А чтобы узнать, какая карта «своя», нужно сначала знать свой GPU.
// Отсюда цепочка: local_rank → GPU → ближайшая IB-карта → MPI_Init.
//
// local_rank здесь нельзя брать из MPI_Comm_split_type (он требует уже
// поднятого MPI), поэтому читаем его из окружения лаунчера — та же
// информация, доступная раньше.

/* Номер ранга внутри узла по переменным лаунчера. -1, если не нашли. */
static int env_local_rank() {
	static const char *const vars[] = {
		"SLURM_LOCALID",
		"OMPI_COMM_WORLD_LOCAL_RANK",
		"PMI_LOCAL_RANK",
		"MV2_COMM_WORLD_LOCAL_RANK",
	};
	for (const char *v : vars) {
		const char *s = getenv(v);
		if (!s || !*s) continue;
		char *end = nullptr;
		const long n = std::strtol(s, &end, 10);
		if (end != s && n >= 0) return static_cast<int>(n);
	}
	return -1;
}

static std::string read_first_line(const std::string &path) {
	std::ifstream f(path);
	std::string line;
	if (f && std::getline(f, line)) {
		while (!line.empty() && (line.back() == '\n' || line.back() == '\r' ||
		                         line.back() == ' '))
			line.pop_back();
		return line;
	}
	return std::string();
}

static std::string real_path(const std::string &p) {
	char buf[PATH_MAX];
	return realpath(p.c_str(), buf) ? std::string(buf) : std::string();
}

/* Только настоящие InfiniBand-порты. Фильтр обязателен: на g5500-* карты
   mlx5_1/2/3 — это Ethernet на 25/10/40 Гбит/с, и выбрать одну из них
   вместо стогигабитного IB было бы хуже, чем автовыбор UCX. */
static bool is_infiniband(const std::string &dev_dir) {
	DIR *d = opendir((dev_dir + "/ports").c_str());
	if (!d) return false;
	bool ib = false;
	while (const dirent *e = readdir(d)) {
		if (e->d_name[0] == '.') continue;
		if (read_first_line(dev_dir + "/ports/" + e->d_name + "/link_layer") ==
		    "InfiniBand") {
			ib = true;
			break;
		}
	}
	closedir(d);
	return ib;
}

/* Ближайшая к GPU IB-карта: у кого длиннее общий префикс пути в /sys, тот
   ближе по дереву PCIe (общий префикс = общий свитч). Это именно расстояние
   GPU↔NIC, которое и определяет скорость при GPUDirect. Собственная
   эвристика UCX оценивает расстояние до ПРОЦЕССА, то есть другую метрику:
   при GPUDirect данные идут GPU→NIC мимо CPU.
   На g8600-* это различает mlx5_8..11 (PXB к GPU0/1) и mlx5_12..15
   (PXB к GPU2/3) от остальных, которые от GPU через NODE. */
static std::string closest_ib_device(const std::string &gpu_pci) {
	const std::string gpath = real_path("/sys/bus/pci/devices/" + gpu_pci);
	if (gpath.empty()) return std::string();

	DIR *d = opendir("/sys/class/infiniband");
	if (!d) return std::string();

	std::string best;
	size_t best_len = 0;
	while (const dirent *e = readdir(d)) {
		if (e->d_name[0] == '.') continue;
		const std::string dev_dir = std::string("/sys/class/infiniband/") + e->d_name;
		if (!is_infiniband(dev_dir)) continue;
		const std::string ipath = real_path(dev_dir + "/device");
		if (ipath.empty()) continue;
		size_t n = 0;
		while (n < gpath.size() && n < ipath.size() && gpath[n] == ipath[n]) ++n;
		if (best.empty() || n > best_len) {
			best = e->d_name;
			best_len = n;
		}
	}
	closedir(d);
	return best;
}

/* Привязать GPU этого ранга и выбрать под него карту — до MPI_Init.
   Возвращает индекс устройства, либо -1, если связать не удалось: тогда
   всё отработает по-старому, а проверки после MPI_Init выдадут нормальную
   диагностику. В nic_out кладётся имя выбранной карты (пусто — не нашли). */
static int bind_gpu_and_nic_early(std::string &nic_out) {
	nic_out.clear();

	const int lrank = env_local_rank();
	if (lrank < 0) return -1;              // лаунчер неизвестен — откат

	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return -1;

	const int dev = (count > 1) ? lrank : 0;
	if (dev >= count) return -1;           // рангов больше, чем GPU: см. ниже
	if (cudaSetDevice(dev) != cudaSuccess) return -1;

	char pci[32] = {};
	if (cudaDeviceGetPCIBusId(pci, sizeof(pci), dev) != cudaSuccess) return dev;

	// cudaDeviceGetPCIBusId отдаёт "0000:D3:00.0", в /sys путь в нижнем регистре
	std::string pci_lower(pci);
	for (char &c : pci_lower)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

	const std::string nic = closest_ib_device(pci_lower);
	if (!nic.empty()) {
		nic_out = nic;
		// overwrite=0: явный export в job-скрипте всегда важнее
		setenv("UCX_NET_DEVICES", (nic + ":1").c_str(), 0);
	}
	return dev;
}

/* Set UCX flags before MPI_Init based on the requested env.
   overwrite=0: explicit exports in the job script always take precedence.

   env=auto  →  include cuda_copy + cuda_ipc so UCX can use CUDA IPC for
               intra-node and GPU Direct RDMA for inter-node.

   env=host  →  exclude cuda_* transports; the code already does explicit
               D2H+MPI(host_buf)+H2D, UCX only ever sees host pointers.

   ВАЖНО про настройки ниже. Прошлый прогон (results/*_auto_*) упирался
   в ~0.8 ГБ/с на ЛЮБОМ размере и ЛЮБОЙ паре — и внутри узла, и между
   узлами, тогда как host-режим давал 13 ГБ/с внутри узла и 6 ГБ/с между.
   Одинаковый потолок для intra и inter означает, что GPU-путь ни разу не
   попал ни в CUDA IPC, ни в настоящий GPUDirect RDMA, а всё время шёл
   через непайплайненное staging-копирование. Виновники были такие:

     UCX_RNDV_THRESH=0    форсировал rendezvous даже для 1 КБ — из-за
                          этого auto проигрывал host на малых размерах
                          (31 мкс против 20 мкс). Теперь не трогаем порог:
                          пусть UCX сам выбирает eager/rendezvous.
     UCX_MEMTYPE_CACHE=n  отключал кэш типа памяти. Снят через unset
                          в task.sbatch; сами её не выставляем — в UCX 1.15
                          она не потребляется (см. ниже).
     UCX_RNDV_SCHEME=     форсированный put_zcopy мешал UCX выбрать
       put_zcopy          get_zcopy/пайплайн. Теперь auto.
   (gdr_copy сюда не добавляем — на этом кластере нет gdrcopy, см. ниже.)

   Если после этого auto всё ещё ~0.8 ГБ/с — значит GPUDirect RDMA не
   работает на уровне системы. Проверь на узле (см. task.sbatch):
     lsmod | grep -E 'nvidia_peermem|nv_peer_mem'   — модуль должен быть загружен,
                                                      иначе NIC физически не может
                                                      читать память GPU;
     nvidia-smi topo -m                             — GPU и HCA должны висеть на
                                                      одном PCIe-свитче/NUMA-узле.
                                                      GDR через UPI между сокетами
                                                      как раз и даёт ~0.8 ГБ/с. */
static void set_ucx_for_env(bool host_env) {
	if (host_env) {
		// No CUDA-aware transports needed — MPI receives host pointers only
		std::string tls = std::string(IB_TLS) + ",cma,sm,self";
		setenv("UCX_TLS", tls.c_str(), /*overwrite=*/0);
		setenv("UCX_IB_GPU_DIRECT_RDMA", "n", /*overwrite=*/0);
	} else {
		// auto: CUDA IPC for intra-node, GPU Direct RDMA for inter-node.
		//
		// gdr_copy в список НЕ включён: на этом кластере нет gdrapi.h, UCX
		// собирается без gdrcopy (см. _ucx_build_job.sh), и упоминание
		// недоступного транспорта только плодит warning'и на каждом ранге.
		// Если gdrcopy появится — пересобери UCX и добавь транспорт руками:
		//   export UCX_TLS=gdr_copy,cuda_copy,cuda_ipc,<ib>,cma,sm,self
		// На крупные размеры это не влияет: gdr_copy ускоряет мелкие
		// сообщения, а полосу GPU→NIC→GPU определяет GPUDirect RDMA.
		std::string tls = std::string("cuda_copy,cuda_ipc,") + IB_TLS +
		                  ",cma,sm,self";
		setenv("UCX_TLS",                tls.c_str(), 0);
		setenv("UCX_IB_GPU_DIRECT_RDMA", "y",         0);
		// auto — UCX сам выберет get_zcopy/put_zcopy/пайплайн под размер и
		// доступность GDR. Форсировать put_zcopy нельзя: без рабочего GDR он
		// скатывается в медленный staging.
		setenv("UCX_RNDV_SCHEME",        "auto",      0);
		// UCX_MEMTYPE_CACHE здесь НЕ выставляем. В UCX 1.15 эта переменная не
		// потребляется (проверено: "UCX WARN unused environment variable:
		// UCX_MEMTYPE_CACHE" на каждом ранге), а её включённое состояние и так
		// является дефолтом. Выставлять её — значит только сорить варнингами.
		// Выключенный кэш (=n) по-прежнему вреден, поэтому task.sbatch делает
		// unset, если он пришёл из окружения.
	}
	// UCX_RNDV_THRESH намеренно НЕ выставляем: дефолт UCX (~8–16 КБ) сам
	// переключает eager→rendezvous по размеру. Форсированный 0 душил малые
	// сообщения. Чтобы вернуть старое поведение: export UCX_RNDV_THRESH=0.
}

/* Print the UCX knobs that matter for GPU-direct transfers. */
static void print_ucx_config(const std::function<void(const std::string &)> &mirror,
                              bool cuda_aware, bool via_host) {
	std::ostringstream o;

	o << "CUDA-aware MPI: " << (cuda_aware ? "yes" : "no") << "\n";
	o << "UCX config (effective after MPI_Init):\n";

	const char *keys[] = {
		"UCX_TLS",
		"UCX_RNDV_THRESH",
		"UCX_IB_GPU_DIRECT_RDMA",
		"UCX_RNDV_SCHEME",
		"UCX_MEMTYPE_CACHE",
		"UCX_NET_DEVICES",
		"UCX_LOG_LEVEL",
	};
	for (const char *k : keys) {
		const char *v = getenv(k);
		o << "  " << k << " = ";
		if (v) o << v << "\n";
		else   o << "<not set>\n";
	}

	// Derived route description
	o << "Route:\n";
	o << "  intra-node: ";
	if (via_host)
		o << "host-staging     (explicit D2H + MPI + H2D)\n";
	else
		o << "UCX-auto         (UCX may use CUDA IPC or NVLink if available)\n";

	o << "  inter-node: ";
	if (via_host) {
		o << "host-staging     (explicit D2H + MPI + H2D)\n";
	} else {
		const char *rndv = getenv("UCX_RNDV_THRESH");
		const bool rndv_zero = rndv && (std::string(rndv) == "0");
		if (rndv_zero)
			o << "UCX rendezvous   (forced for all sizes by UCX_RNDV_THRESH=0)\n"
			  << "                 WARNING: rendezvous для мелких сообщений даёт\n"
			     "                 лишнюю латентность — auto проиграет host на\n"
			     "                 малых размерах. Убери UCX_RNDV_THRESH=0.\n";
		else
			o << "UCX auto         (small msgs: eager; large msgs: rendezvous, "
			     "GPU Direct if GDR available)\n";
	}

	mirror(o.str());
}

// ------------------------------------------------------------------ //
// main                                                                 //
// ------------------------------------------------------------------ //

int main(int argc, char **argv) {
	using namespace gpu_benchmark;

	// Оба вызова обязаны идти до MPI_Init — UCX читает окружение при init.
	// Сначала GPU и карта (карта выводится из GPU), потом остальные флаги.
	std::string early_nic;
	const int early_gpu = bind_gpu_and_nic_early(early_nic);

	set_ucx_for_env(pre_scan_host_env(argc, argv));

	mpi_ok(MPI_Init(&argc, &argv), "MPI_Init");
	int rank = 0, nproc = 1;
	mpi_ok(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
	mpi_ok(MPI_Comm_size(MPI_COMM_WORLD, &nproc), "MPI_Comm_size");

	const Args args         = parse_args(argc, argv, rank);
	const bool cuda_aware   = mpi_cuda_aware();
	const bool via_host     = check_host(args.env, cuda_aware);
	const bool cuda_ipc_mode =
		args.mode == Mode::CudaOneToOne || args.mode == Mode::CudaAllToAll;
	if (cuda_ipc_mode && args.env == Env::Host) {
		if (rank == 0)
			std::cerr << "CUDA IPC modes require --env auto: their payload path is "
			             "direct GPU-to-GPU P2P, not host staging.\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	// Node communicator
	MPI_Comm node_comm = MPI_COMM_NULL;
	mpi_ok(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
	                           MPI_INFO_NULL, &node_comm),
	       "MPI_Comm_split_type");
	int local_rank = 0, node_size = 1;
	mpi_ok(MPI_Comm_rank(node_comm, &local_rank), "MPI_Comm_rank(node)");
	mpi_ok(MPI_Comm_size(node_comm, &node_size),  "MPI_Comm_size(node)");

	std::vector<int> node_ranks(static_cast<size_t>(node_size));
	mpi_ok(MPI_Allgather(&rank, 1, MPI_INT, node_ranks.data(), 1, MPI_INT, node_comm),
	       "MPI_Allgather node_ranks");
	std::vector<int> on_my_node(static_cast<size_t>(nproc), 0);
	for (int r : node_ranks) on_my_node[static_cast<size_t>(r)] = 1;

	int local_gpu_count = 0;
	cudaGetDeviceCount(&local_gpu_count);
	if (local_gpu_count <= 0) {
		std::cerr << "rank " << rank << ": no visible CUDA devices\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	/* GPU index within this process's CUDA_VISIBLE_DEVICES.
	   --gpu-bind=none (all node GPUs visible)  →  local_rank.
	   Один видимый GPU (map_gpu/per_task)      →  0.
	   Never use modulo when ranks_on_node > visible_gpus.

	   НЕ советуй здесь --gpu-bind=closest. Он ищет для каждой задачи
	   ближайший GPU независимо от других задач и не гарантирует
	   уникальности. На узлах этого кластера все GPU имеют ОДИНАКОВУЮ
	   CPU-аффинность (g5500: "0,32" у обоих; g8600: "32,96" у всех
	   четырёх), поэтому "ближайший" — ничья для каждой задачи, и Slurm
	   выдаёт всем один и тот же GPU. Это уже ломалось: коммит 4e4c763
	   от 27.05.2026 добавил проверку на дубликаты ниже именно из-за
	   этого, а 6f59951 от 10.08.2026 вернул --gpu-bind=none.
	   Детерминированные альтернативы: map_gpu:0,1,2,3 или per_task:1. */
	int local_gpu = 0;
	if (local_gpu_count > 1) {
		if (local_rank >= local_gpu_count) {
			std::cerr << "rank " << rank << " (local_rank " << local_rank
			          << ") has no dedicated GPU: node exposes only "
			          << local_gpu_count << " devices.\n"
			          << "  Launch exactly one rank per GPU, e.g.:\n"
			          << "    srun --ntasks-per-node=<gpus_per_node>"
			          << " --gpus-per-node=<gpus_per_node> --gpu-bind=none ...\n"
			          << "  (не используй --gpu-bind=closest: на этих узлах все\n"
			          << "   GPU равноудалены по CPU-аффинности и несколько\n"
			          << "   рангов получат один и тот же GPU)\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		local_gpu = local_rank;
	}
	// Сверка раннего связывания с тем, что даёт MPI. Расходиться они не
	// должны: SLURM_LOCALID и ранг в MPI_COMM_TYPE_SHARED — одно и то же
	// число. Если расходятся, лаунчер разложил ранги не так, как мы решили
	// до MPI_Init, и UCX_NET_DEVICES указывает на карту чужого GPU.
	if (early_gpu >= 0 && early_gpu != local_gpu) {
		std::cerr << "rank " << rank << ": ранний выбор GPU (" << early_gpu
		          << ", из окружения лаунчера) разошёлся с MPI local_rank ("
		          << local_gpu << "). Беру " << local_gpu
		          << ", но UCX_NET_DEVICES уже зафиксирован под "
		          << early_gpu << " — выбор карты может быть неоптимальным.\n";
	}
	cuda_ok(cudaSetDevice(local_gpu), "cudaSetDevice");

	// Optional output file
	std::unique_ptr<std::ofstream> out_file;
	if (rank == 0 && !args.out_path.empty()) {
		out_file = std::make_unique<std::ofstream>(args.out_path);
		if (!out_file->is_open()) {
			std::cerr << "gpu: cannot open --out " << args.out_path << "\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	auto mirror = [&](const std::string &s) {
		if (rank != 0) return;
		std::cout << s;
		if (out_file) *out_file << s;
	};

	// Gather hostnames and GPU info
	constexpr int HOST_LEN = MPI_MAX_PROCESSOR_NAME;
	constexpr int PCI_LEN  = 32;
	char my_host[HOST_LEN] = {};
	char my_pci[PCI_LEN]   = {};
	{ int n = 0; mpi_ok(MPI_Get_processor_name(my_host, &n), "MPI_Get_processor_name"); }
	if (cudaDeviceGetPCIBusId(my_pci, sizeof(my_pci), local_gpu) != cudaSuccess)
		std::snprintf(my_pci, sizeof(my_pci), "n/a");

	std::vector<char> hosts_recv(static_cast<size_t>(nproc) * HOST_LEN);
	std::vector<char> pci_recv(static_cast<size_t>(nproc) * PCI_LEN);
	std::vector<int>  gpu_counts(static_cast<size_t>(nproc), 0);
	std::vector<int>  local_gpus(static_cast<size_t>(nproc), 0);
	mpi_ok(MPI_Allgather(my_host, HOST_LEN, MPI_CHAR,
	                     hosts_recv.data(), HOST_LEN, MPI_CHAR, MPI_COMM_WORLD),
	       "MPI_Allgather hostnames");
	mpi_ok(MPI_Allgather(&local_gpu_count, 1, MPI_INT,
	                     gpu_counts.data(), 1, MPI_INT, MPI_COMM_WORLD),
	       "MPI_Allgather gpu_counts");
	mpi_ok(MPI_Allgather(&local_gpu, 1, MPI_INT,
	                     local_gpus.data(), 1, MPI_INT, MPI_COMM_WORLD),
	       "MPI_Allgather local_gpus");
	mpi_ok(MPI_Allgather(my_pci, PCI_LEN, MPI_CHAR,
	                     pci_recv.data(), PCI_LEN, MPI_CHAR, MPI_COMM_WORLD),
	       "MPI_Allgather pci");

	// Выбранная карта у каждого ранга. Собираем со всех, потому что ранги
	// выбирают РАЗНЫЕ карты (в этом весь смысл), и печатать одно значение
	// с ранга 0 было бы обманом.
	constexpr int NIC_LEN = 32;
	char my_nic[NIC_LEN] = {};
	std::snprintf(my_nic, sizeof(my_nic), "%s",
	              early_nic.empty() ? "ucx-auto" : early_nic.c_str());
	std::vector<char> nic_recv(static_cast<size_t>(nproc) * NIC_LEN);
	mpi_ok(MPI_Allgather(my_nic, NIC_LEN, MPI_CHAR,
	                     nic_recv.data(), NIC_LEN, MPI_CHAR, MPI_COMM_WORLD),
	       "MPI_Allgather nic");

	if (rank == 0) {
		for (int r = 0; r < nproc; ++r) {
			if (gpu_counts[static_cast<size_t>(r)] <= 0) {
				std::cerr << "rank " << r << " sees no GPUs\n";
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
		// Two ranks on the same host must not point to one physical GPU.
		for (int a = 0; a < nproc; ++a) {
			for (int b = a + 1; b < nproc; ++b) {
				const char *ha = hosts_recv.data() + static_cast<size_t>(a) * HOST_LEN;
				const char *hb = hosts_recv.data() + static_cast<size_t>(b) * HOST_LEN;
				const char *pa = pci_recv.data()   + static_cast<size_t>(a) * PCI_LEN;
				const char *pb = pci_recv.data()   + static_cast<size_t>(b) * PCI_LEN;
				if (std::strcmp(ha, hb) == 0 && std::strcmp(pa, pb) == 0) {
					std::cerr << "error: ranks " << a << " and " << b
					          << " share GPU " << pa << " on " << ha
					          << " (GPU oversubscription).\n"
					          << "  Launch exactly one rank per GPU, e.g.:\n"
					          << "    srun --ntasks-per-node=<gpus_per_node>"
					          << " --gpus-per-node=<gpus_per_node> --gpu-bind=none ...\n"
					          << "  Если сейчас стоит --gpu-bind=closest — это и есть\n"
					          << "  причина: на этих узлах все GPU равноудалены по\n"
					          << "  CPU-аффинности, ничья разрешается в пользу одного\n"
					          << "  и того же устройства. Бери --gpu-bind=none (ранг\n"
					          << "  выбирает GPU по local_rank) либо map_gpu:0,1,2,3.\n";
					MPI_Abort(MPI_COMM_WORLD, 1);
				}
			}
		}
	}

	const auto rank_labels      = build_rank_labels(hosts_recv, nproc, HOST_LEN);
	const auto global_gpu_labels =
		build_global_gpu_labels(hosts_recv, nproc, HOST_LEN, gpu_counts);

	if (rank == 0) {
		// Standard header fields (parsed by gpu_heatmap.py)
		{
			std::ostringstream o;
			o << "Env: "    << (via_host ? "host" : "auto") << "\n"
			  << "Mode: "   << mode_to_string(args.mode)    << "\n"
			  << "Transport: " << (cuda_ipc_mode ? "cuda_ipc_p2p" : "mpi_ucx") << "\n"
			  << "Timer: mpi\n"
			  << "Bytes: "  << args.nbytes  << "\n"
			  << "Warmup: " << args.warmup  << "\n"
			  << "Iters: "  << args.iters   << "\n";
			int total_gpus = 0;
			for (int v : gpu_counts) total_gpus += v;
			o << "Ranks: " << nproc
			  << ", total visible GPUs: " << total_gpus << "\n";
			o << "Rank map:\n";
			for (int r = 0; r < nproc; ++r) {
				o << "  r" << r
				  << " hostname=" << (hosts_recv.data() + static_cast<size_t>(r) * HOST_LEN)
				  << " local_gpu=" << local_gpus[static_cast<size_t>(r)]
				  << " visible_gpus=" << gpu_counts[static_cast<size_t>(r)]
				  << " pci=" << (pci_recv.data() + static_cast<size_t>(r) * PCI_LEN)
				  << " nic=" << (nic_recv.data() + static_cast<size_t>(r) * NIC_LEN)
				  << "\n";
			}
			mirror(o.str());
		}

		// UCX / route diagnostics
		print_ucx_config(mirror, cuda_aware, via_host);

		std::cout << std::fixed << std::setprecision(REPORT_DIGITS);
		if (out_file) *out_file << std::fixed << std::setprecision(REPORT_DIGITS);
	}

	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier before benchmark");

	if (args.mode == Mode::OneToOne)
		schedule_one_to_one(rank, nproc, args, via_host,
		                    node_comm, local_rank, on_my_node,
		                    rank_labels, mirror);
	else if (args.mode == Mode::AllToAll)
		schedule_all_to_all(rank, nproc, args, via_host,
		                    node_comm, local_rank, on_my_node, node_ranks,
		                    rank_labels, mirror);
	// else if (args.mode == Mode::CudaOneToOne)
	// 	schedule_cuda_one_to_one(rank, nproc, args, node_comm, node_ranks,
	// 	                         rank_labels, mirror);
	// else
	// 	schedule_cuda_all_to_all(rank, nproc, args, node_comm, node_ranks,
	// 	                         rank_labels, mirror);

	mpi_ok(MPI_Comm_free(&node_comm), "MPI_Comm_free");
	mpi_ok(MPI_Finalize(), "MPI_Finalize");
	return 0;
}
