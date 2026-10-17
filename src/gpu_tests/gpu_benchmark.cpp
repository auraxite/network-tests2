#include "gpu_all_to_all.hpp"
#include "gpu_common.hpp"
#include "gpu_one_to_one.hpp"

#include <cctype>
#include <climits>   // PATH_MAX
#include <cstdio>
#include <cstdlib>   // setenv, getenv, realpath, strtol
#include <cstring>
#include <dirent.h>  // обход /sys
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using namespace gpu_benchmark;

namespace {

// ------------------------------------------------------------------ //
// До MPI_Init: GPU, сетевой адаптер, переменные UCX (см. NOTES.md)    //
// ------------------------------------------------------------------ //

/* IB-транспорты, допустимые и для Verbs, и для MLX5. */
const char *IB_TLS = "rc_verbs,rc_mlx5,ud_verbs,ud_mlx5,dc_mlx5";

/* --env host до разбора аргументов: переменные UCX нужны раньше MPI_Init. */
bool pre_scan_host_env(int argc, char **argv) {
	for (int i = 1; i < argc - 1; ++i)
		if (std::string(argv[i]) == "--env" && std::string(argv[i + 1]) == "host")
			return true;
	return false;
}

/* Номер ранга на узле по переменным лаунчера, -1 если не нашли. */
int env_local_rank() {
	for (const char *v : {"OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID",
	                      "PMI_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"}) {
		const char *s = getenv(v);
		if (!s || !*s) continue;
		char *end = nullptr;
		const long n = std::strtol(s, &end, 10);
		if (end != s && n >= 0) return static_cast<int>(n);
	}
	return -1;
}

std::string read_first_line(const std::string &path) {
	std::ifstream f(path);
	std::string line;
	if (f && std::getline(f, line))
		while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
			line.pop_back();
	return line;
}

std::string real_path(const std::string &p) {
	char buf[PATH_MAX];
	return realpath(p.c_str(), buf) ? std::string(buf) : std::string();
}

/* Есть ли у карты порт InfiniBand (Ethernet-карты пропускаем). */
bool is_infiniband(const std::string &dev_dir) {
	DIR *d = opendir((dev_dir + "/ports").c_str());
	if (!d) return false;
	bool ib = false;
	while (const dirent *e = readdir(d))
		if (e->d_name[0] != '.' &&
		    read_first_line(dev_dir + "/ports/" + e->d_name + "/link_layer") == "InfiniBand") {
			ib = true;
			break;
		}
	closedir(d);
	return ib;
}

/* Ближайшая к GPU IB-карта: самый длинный общий префикс путей в /sys. */
std::string closest_ib_device(const std::string &gpu_pci) {
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

/* Выбрать GPU по локальному номеру и карту под него, до MPI_Init. Возвращает
   номер GPU или -1, если не получилось (тогда диагностика будет после MPI_Init). */
int bind_gpu_and_nic_early(std::string &nic_out) {
	nic_out.clear();
	const int lrank = env_local_rank();
	if (lrank < 0) return -1;

	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return -1;
	const int dev = count > 1 ? lrank : 0;
	if (dev >= count || cudaSetDevice(dev) != cudaSuccess) return -1;

	char pci[32] = {};
	if (cudaDeviceGetPCIBusId(pci, sizeof(pci), dev) != cudaSuccess) return dev;
	std::string pci_lower(pci);   // в /sys путь в нижнем регистре
	for (char &c : pci_lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

	nic_out = closest_ib_device(pci_lower);
	if (!nic_out.empty())
		setenv("UCX_NET_DEVICES", (nic_out + ":1").c_str(), 0);
	return dev;
}

/* Переменные UCX для среды. overwrite=0: явный export в скрипте важнее. */
void set_ucx_for_env(bool host_env) {
	if (host_env) {
		setenv("UCX_TLS", (std::string(IB_TLS) + ",cma,sm,self").c_str(), 0);
		setenv("UCX_IB_GPU_DIRECT_RDMA", "n", 0);
	} else {
		setenv("UCX_TLS", (std::string("cuda_copy,cuda_ipc,") + IB_TLS + ",cma,sm,self").c_str(), 0);
		setenv("UCX_IB_GPU_DIRECT_RDMA", "y", 0);
		setenv("UCX_RNDV_SCHEME", "auto", 0);
	}
	// UCX_RNDV_THRESH и UCX_MEMTYPE_CACHE намеренно не выставляются.
}

// ------------------------------------------------------------------ //
// После MPI_Init                                                      //
// ------------------------------------------------------------------ //

/* Сведения обо всех рангах, собранные через MPI_Allgather. */
struct RankInfo {
	static constexpr int HOST_LEN = MPI_MAX_PROCESSOR_NAME;
	static constexpr int PCI_LEN  = 32;
	static constexpr int NIC_LEN  = 32;
	std::vector<char> hosts, pcis, nics;
	std::vector<int>  gpu_counts, local_gpus;

	const char *host(int r) const { return hosts.data() + static_cast<size_t>(r) * HOST_LEN; }
	const char *pci(int r)  const { return pcis.data()  + static_cast<size_t>(r) * PCI_LEN; }
	const char *nic(int r)  const { return nics.data()  + static_cast<size_t>(r) * NIC_LEN; }
};

RankInfo gather_rank_info(int nproc, int local_gpu_count, int local_gpu,
                          const std::string &early_nic) {
	char my_host[RankInfo::HOST_LEN] = {};
	char my_pci[RankInfo::PCI_LEN]   = {};
	char my_nic[RankInfo::NIC_LEN]   = {};
	int  len = 0;
	mpi_ok(MPI_Get_processor_name(my_host, &len), "MPI_Get_processor_name");
	if (cudaDeviceGetPCIBusId(my_pci, sizeof(my_pci), local_gpu) != cudaSuccess)
		std::snprintf(my_pci, sizeof(my_pci), "n/a");
	std::snprintf(my_nic, sizeof(my_nic), "%s", early_nic.empty() ? "ucx-auto" : early_nic.c_str());

	RankInfo ri;
	ri.hosts.resize(static_cast<size_t>(nproc) * RankInfo::HOST_LEN);
	ri.pcis.resize(static_cast<size_t>(nproc) * RankInfo::PCI_LEN);
	ri.nics.resize(static_cast<size_t>(nproc) * RankInfo::NIC_LEN);
	ri.gpu_counts.resize(nproc);
	ri.local_gpus.resize(nproc);
	mpi_ok(MPI_Allgather(my_host, RankInfo::HOST_LEN, MPI_CHAR, ri.hosts.data(), RankInfo::HOST_LEN,
	                     MPI_CHAR, MPI_COMM_WORLD), "MPI_Allgather hostnames");
	mpi_ok(MPI_Allgather(&local_gpu_count, 1, MPI_INT, ri.gpu_counts.data(), 1, MPI_INT,
	                     MPI_COMM_WORLD), "MPI_Allgather gpu_counts");
	mpi_ok(MPI_Allgather(&local_gpu, 1, MPI_INT, ri.local_gpus.data(), 1, MPI_INT,
	                     MPI_COMM_WORLD), "MPI_Allgather local_gpus");
	mpi_ok(MPI_Allgather(my_pci, RankInfo::PCI_LEN, MPI_CHAR, ri.pcis.data(), RankInfo::PCI_LEN,
	                     MPI_CHAR, MPI_COMM_WORLD), "MPI_Allgather pci");
	// Карту собираем со всех рангов: ранги выбирают разные карты.
	mpi_ok(MPI_Allgather(my_nic, RankInfo::NIC_LEN, MPI_CHAR, ri.nics.data(), RankInfo::NIC_LEN,
	                     MPI_CHAR, MPI_COMM_WORLD), "MPI_Allgather nic");
	return ri;
}

const char *ONE_RANK_PER_GPU_HINT =
	"  Launch exactly one rank per GPU, e.g.:\n"
	"    srun --ntasks-per-node=<gpus_per_node> --gpus-per-node=<gpus_per_node> --gpu-bind=none ...\n";

/* Ранг 0: у каждого ранга есть GPU, и два ранга одного узла не делят один GPU. */
void check_gpu_assignment(int nproc, const RankInfo &ri) {
	for (int r = 0; r < nproc; ++r)
		if (ri.gpu_counts[r] <= 0) {
			std::cerr << "rank " << r << " sees no GPUs\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	for (int a = 0; a < nproc; ++a)
		for (int b = a + 1; b < nproc; ++b)
			if (std::strcmp(ri.host(a), ri.host(b)) == 0 && std::strcmp(ri.pci(a), ri.pci(b)) == 0) {
				std::cerr << "error: ranks " << a << " and " << b << " share GPU " << ri.pci(a)
				          << " on " << ri.host(a) << " (GPU oversubscription).\n"
				          << ONE_RANK_PER_GPU_HINT
				          << "  Если сейчас стоит --gpu-bind=closest — это и есть\n"
				          << "  причина: на этих узлах все GPU равноудалены по\n"
				          << "  CPU-аффинности, ничья разрешается в пользу одного\n"
				          << "  и того же устройства. Бери --gpu-bind=none (ранг\n"
				          << "  выбирает GPU по local_rank) либо map_gpu:0,1,2,3.\n";
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
}

std::string header_text(const Args &args, bool via_host, int nproc, const RankInfo &ri) {
	std::ostringstream o;
	o << "Env: "    << (via_host ? "host" : "auto") << "\n"
	  << "Mode: "   << mode_to_string(args.mode)    << "\n"
	  << "Transport: mpi_ucx\n"
	  << "Timer: mpi\n"
	  << "Bytes: "  << args.nbytes  << "\n"
	  << "Warmup: " << args.warmup  << "\n"
	  << "Iters: "  << args.iters   << "\n";
	int total_gpus = 0;
	for (int v : ri.gpu_counts) total_gpus += v;
	o << "Ranks: " << nproc << ", total visible GPUs: " << total_gpus << "\n";
	o << "Rank map:\n";
	for (int r = 0; r < nproc; ++r)
		o << "  r" << r << " hostname=" << ri.host(r) << " local_gpu=" << ri.local_gpus[r]
		  << " visible_gpus=" << ri.gpu_counts[r] << " pci=" << ri.pci(r)
		  << " nic=" << ri.nic(r) << "\n";
	return o.str();
}

/* Действующие переменные UCX и словесное описание пути. */
std::string ucx_config_text(bool cuda_aware, bool via_host) {
	std::ostringstream o;
	o << "CUDA-aware MPI: " << (cuda_aware ? "yes" : "no") << "\n";
	o << "UCX config (effective after MPI_Init):\n";
	for (const char *k : {"UCX_TLS", "UCX_RNDV_THRESH", "UCX_IB_GPU_DIRECT_RDMA", "UCX_RNDV_SCHEME",
	                      "UCX_MEMTYPE_CACHE", "UCX_NET_DEVICES", "UCX_LOG_LEVEL"}) {
		const char *v = getenv(k);
		o << "  " << k << " = " << (v ? v : "<not set>") << "\n";
	}

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
		if (rndv && std::string(rndv) == "0")
			o << "UCX rendezvous   (forced for all sizes by UCX_RNDV_THRESH=0)\n"
			  << "                 WARNING: rendezvous для мелких сообщений даёт\n"
			     "                 лишнюю латентность — auto проиграет host на\n"
			     "                 малых размерах. Убери UCX_RNDV_THRESH=0.\n";
		else
			o << "UCX auto         (small msgs: eager; large msgs: rendezvous, "
			     "GPU Direct if GDR available)\n";
	}
	return o.str();
}

} // namespace

int main(int argc, char **argv) {
	// Оба вызова обязаны идти до MPI_Init: UCX читает окружение при инициализации.
	std::string early_nic;
	const int early_gpu = bind_gpu_and_nic_early(early_nic);
	set_ucx_for_env(pre_scan_host_env(argc, argv));

	mpi_ok(MPI_Init(&argc, &argv), "MPI_Init");
	int rank = 0, nproc = 1;
	mpi_ok(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
	mpi_ok(MPI_Comm_size(MPI_COMM_WORLD, &nproc), "MPI_Comm_size");

	const Args args       = parse_args(argc, argv, rank);
	const bool cuda_aware = mpi_cuda_aware();
	const bool via_host   = use_host_path(args.env, cuda_aware);

	// Ранги этого узла
	MPI_Comm node_comm = MPI_COMM_NULL;
	mpi_ok(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm),
	       "MPI_Comm_split_type");
	int local_rank = 0, node_size = 1;
	mpi_ok(MPI_Comm_rank(node_comm, &local_rank), "MPI_Comm_rank(node)");
	mpi_ok(MPI_Comm_size(node_comm, &node_size), "MPI_Comm_size(node)");
	std::vector<int> node_ranks(node_size);
	mpi_ok(MPI_Allgather(&rank, 1, MPI_INT, node_ranks.data(), 1, MPI_INT, node_comm),
	       "MPI_Allgather node_ranks");
	std::vector<int> on_my_node(nproc, 0);
	for (int r : node_ranks) on_my_node[r] = 1;

	// GPU ранга: номер на узле среди видимых устройств (один GPU — номер 0).
	// Остаток от деления не используется: двум рангам нельзя давать один GPU.
	int local_gpu_count = 0;
	cudaGetDeviceCount(&local_gpu_count);
	if (local_gpu_count <= 0) {
		std::cerr << "rank " << rank << ": no visible CUDA devices\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	int local_gpu = 0;
	if (local_gpu_count > 1) {
		if (local_rank >= local_gpu_count) {
			std::cerr << "rank " << rank << " (local_rank " << local_rank
			          << ") has no dedicated GPU: node exposes only "
			          << local_gpu_count << " devices.\n" << ONE_RANK_PER_GPU_HINT
			          << "  (не используй --gpu-bind=closest: на этих узлах все\n"
			          << "   GPU равноудалены по CPU-аффинности и несколько\n"
			          << "   рангов получат один и тот же GPU)\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		local_gpu = local_rank;
	}
	if (early_gpu >= 0 && early_gpu != local_gpu)
		std::cerr << "rank " << rank << ": ранний выбор GPU (" << early_gpu
		          << ", из окружения лаунчера) разошёлся с MPI local_rank ("
		          << local_gpu << "). Беру " << local_gpu
		          << ", но UCX_NET_DEVICES уже зафиксирован под "
		          << early_gpu << " — выбор карты может быть неоптимальным.\n";
	cuda_ok(cudaSetDevice(local_gpu), "cudaSetDevice");

	// Вывод: ранг 0 пишет в stdout и, если задан --out, в файл.
	std::unique_ptr<std::ofstream> out_file;
	if (rank == 0 && !args.out_path.empty()) {
		out_file = std::make_unique<std::ofstream>(args.out_path);
		if (!out_file->is_open()) {
			std::cerr << "gpu: cannot open --out " << args.out_path << "\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	const auto mirror = [&](const std::string &s) {
		if (rank != 0) return;
		std::cout << s;
		if (out_file) *out_file << s;
	};

	const RankInfo ri = gather_rank_info(nproc, local_gpu_count, local_gpu, early_nic);
	if (rank == 0) check_gpu_assignment(nproc, ri);
	const auto rank_labels = build_rank_labels(ri.hosts, nproc, RankInfo::HOST_LEN);

	mirror(header_text(args, via_host, nproc, ri));
	mirror(ucx_config_text(cuda_aware, via_host));

	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier before benchmark");
	if (args.mode == Mode::OneToOne)
		schedule_one_to_one(rank, nproc, args, via_host, node_comm, local_rank, on_my_node,
		                    rank_labels, mirror);
	else
		schedule_all_to_all(rank, nproc, args, via_host, node_comm, on_my_node, node_ranks,
		                    rank_labels, mirror);

	mpi_ok(MPI_Comm_free(&node_comm), "MPI_Comm_free");
	mpi_ok(MPI_Finalize(), "MPI_Finalize");
	return 0;
}
