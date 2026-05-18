#include "gpu_all_to_all.hpp"
#include "gpu_common.hpp"
#include "gpu_one_to_one.hpp"

#include <cstdlib>   // setenv, getenv
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

/* Set UCX flags before MPI_Init based on the requested env.
   overwrite=0: explicit exports in the job script always take precedence.

   env=auto  →  include cuda_copy + cuda_ipc so UCX can use CUDA IPC for
               intra-node and GPU Direct RDMA for inter-node.
               UCX_RNDV_THRESH is intentionally left at the UCX default so
               the natural eager→rendezvous transition remains visible.

   env=host  →  exclude cuda_* transports; the code already does explicit
               D2H+MPI(host_buf)+H2D, UCX only ever sees host pointers. */
static void set_ucx_for_env(bool host_env) {
	if (host_env) {
		// No CUDA-aware transports needed — MPI receives host pointers only
		std::string tls = std::string(IB_TLS) + ",cma,sm,self";
		setenv("UCX_TLS", tls.c_str(), /*overwrite=*/0);
	} else {
		// auto: CUDA IPC for intra-node, GPU Direct RDMA for inter-node
		std::string tls = std::string("cuda_copy,cuda_ipc,") + IB_TLS + ",cma,sm,self";
		setenv("UCX_TLS",             tls.c_str(),   0);
		setenv("UCX_IB_GPU_DIRECT_RDMA", "yes",      0);
		setenv("UCX_RNDV_SCHEME",    "get_zcopy",    0);
		// UCX_RNDV_THRESH: NOT set — preserve the natural eager/rendezvous
		// threshold so the protocol-switch transition remains visible in data.
	}
}

/* Print the UCX knobs that matter for GPU-direct transfers. */
static void print_ucx_config(const std::function<void(const std::string &)> &mirror,
                              bool cuda_aware, bool via_host,
                              bool local_shared_fallback) {
	std::ostringstream o;

	o << "CUDA-aware MPI: " << (cuda_aware ? "yes" : "no") << "\n";
	o << "UCX config (effective after MPI_Init):\n";

	const char *keys[] = {
		"UCX_TLS",
		"UCX_RNDV_THRESH",
		"UCX_IB_GPU_DIRECT_RDMA",
		"UCX_RNDV_SCHEME",
		"UCX_TLS",
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
	if (local_shared_fallback)
		o << "host-shared-mem  (D2H → shared DRAM → H2D, CUDA IPC bypassed)\n";
	else if (via_host)
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
			o << "UCX rendezvous   (GPU→NIC→GPU via GPUDirect RDMA, all sizes)\n";
		else
			o << "UCX auto         (small msgs: eager+host staging; "
			     "large msgs: GPU Direct if GDR available)\n"
			  << "                 WARNING: set UCX_RNDV_THRESH=0 to force "
			     "GPU Direct for all sizes\n";
	}

	mirror(o.str());
}

// ------------------------------------------------------------------ //
// main                                                                 //
// ------------------------------------------------------------------ //

int main(int argc, char **argv) {
	using namespace gpu_benchmark;

	// Must happen before MPI_Init — UCX reads env at init time
	set_ucx_for_env(pre_scan_host_env(argc, argv));

	mpi_ok(MPI_Init(&argc, &argv), "MPI_Init");
	int rank = 0, nproc = 1;
	mpi_ok(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
	mpi_ok(MPI_Comm_size(MPI_COMM_WORLD, &nproc), "MPI_Comm_size");

	const Args args         = parse_args(argc, argv, rank);
	const bool cuda_aware   = mpi_cuda_aware();
	const bool via_host     = check_host(args.env, cuda_aware);

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

	/* In auto mode with 1 GPU per rank: proactively use shared host memory
	   for intra-node pairs because CUDA IPC context sharing is typically
	   unavailable under cgroup/--gpus-per-task=1 isolation. */
	const bool enable_local_shared_fallback =
		(args.env == Env::Auto) && cuda_aware && (local_gpu_count <= 1);

	const int local_gpu = 0;
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
	mpi_ok(MPI_Allgather(my_host, HOST_LEN, MPI_CHAR,
	                     hosts_recv.data(), HOST_LEN, MPI_CHAR, MPI_COMM_WORLD),
	       "MPI_Allgather hostnames");
	mpi_ok(MPI_Allgather(&local_gpu_count, 1, MPI_INT,
	                     gpu_counts.data(), 1, MPI_INT, MPI_COMM_WORLD),
	       "MPI_Allgather gpu_counts");
	mpi_ok(MPI_Allgather(my_pci, PCI_LEN, MPI_CHAR,
	                     pci_recv.data(), PCI_LEN, MPI_CHAR, MPI_COMM_WORLD),
	       "MPI_Allgather pci");

	if (rank == 0) {
		for (int r = 0; r < nproc; ++r) {
			if (gpu_counts[static_cast<size_t>(r)] <= 0) {
				std::cerr << "rank " << r << " sees no GPUs\n";
				MPI_Abort(MPI_COMM_WORLD, 1);
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
				  << " local_gpu=0"
				  << " visible_gpus=" << gpu_counts[static_cast<size_t>(r)]
				  << " pci=" << (pci_recv.data() + static_cast<size_t>(r) * PCI_LEN)
				  << "\n";
			}
			mirror(o.str());
		}

		// UCX / route diagnostics
		print_ucx_config(mirror, cuda_aware, via_host, enable_local_shared_fallback);

		std::cout << std::fixed << std::setprecision(REPORT_DIGITS);
		if (out_file) *out_file << std::fixed << std::setprecision(REPORT_DIGITS);
	}

	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier before benchmark");

	if (args.mode == Mode::OneToOne)
		schedule_one_to_one(rank, nproc, args, via_host,
		                    enable_local_shared_fallback,
		                    node_comm, local_rank, on_my_node,
		                    rank_labels, mirror);
	else
		schedule_all_to_all(rank, nproc, args, via_host,
		                    node_comm, local_rank, on_my_node, node_ranks,
		                    rank_labels, mirror);

	mpi_ok(MPI_Comm_free(&node_comm), "MPI_Comm_free");
	mpi_ok(MPI_Finalize(), "MPI_Finalize");
	return 0;
}
