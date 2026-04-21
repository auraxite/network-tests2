#include "gpu_all_to_all.hpp"
#include "gpu_common.hpp"
#include "gpu_one_to_one.hpp"

#include <fstream>   // std::ofstream — запись текстового вывода в --out
#include <iomanip>   // std::setprecision, std::fixed — формат чисел в выводе
#include <iostream>  // std::cout, std::cerr
#include <memory>    // std::unique_ptr, std::make_unique — файл открываем опционально
#include <sstream>   // std::ostringstream — собрать строку перед mirror(...)
#include <vector>    // std::vector — буферы Gather/Bcast имён хостов

int main(int argc, char **argv) {
	using namespace gpu_benchmark;

	mpi_ok(MPI_Init(&argc, &argv), "MPI_Init");
	int rank = 0;
	int nproc = 1;
	mpi_ok(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
	mpi_ok(MPI_Comm_size(MPI_COMM_WORLD, &nproc), "MPI_Comm_size");

	const Args args = parse_args(argc, argv, rank);
	const bool cuda_aware = mpi_cuda_aware();
	const bool via_host = check_host(args.env, cuda_aware);

	/* === Узловой коммуникатор ===
	   local_rank используется дальше для node-local логики (shared host fallback).
	   В текущем режиме запуска (1 видимый GPU на MPI-процесс) CUDA device всегда
	   выбирается как индекс 0 в видимом пространстве процесса. */
	MPI_Comm node_comm = MPI_COMM_NULL;
	mpi_ok(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
							   MPI_INFO_NULL, &node_comm),
		   "MPI_Comm_split_type(node)");
	int local_rank = 0;
	mpi_ok(MPI_Comm_rank(node_comm, &local_rank), "MPI_Comm_rank(node)");
	int node_size = 1;
	mpi_ok(MPI_Comm_size(node_comm, &node_size), "MPI_Comm_size(node)");
	std::vector<int> node_ranks(static_cast<size_t>(node_size), 0);
	mpi_ok(MPI_Allgather(&rank, 1, MPI_INT, node_ranks.data(), 1, MPI_INT, node_comm),
		   "MPI_Allgather(node ranks)");
	std::vector<int> on_my_node(static_cast<size_t>(nproc), 0);
	for (int r : node_ranks)
		on_my_node[static_cast<size_t>(r)] = 1;

	int local_gpu_count = 0;
	cudaGetDeviceCount(&local_gpu_count);
	if (local_gpu_count <= 0) {
		std::cerr << "rank " << rank << ": no visible CUDA devices\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	/* При env=auto и ровно одном видимом GPU на процесс часто работает схема
	   с cgroup-изоляцией (--gpus-per-task=1): MPI может быть CUDA-aware, но
	   локальный CUDA IPC между процессами узла недоступен и UCX тихо уходит в
	   host staging. Для intra-node пар заранее включаем явный shared-host path,
	   чтобы fallback был управляемым и одинаковым на sender/receiver. */
	const bool enable_local_shared_fallback =
		(args.env == Env::Auto) && cuda_aware && (local_gpu_count <= 1);
	const int local_gpu = 0;
	cuda_ok(cudaSetDevice(local_gpu), "cudaSetDevice(initial)");

	// Текстовый отчёт дублируется в stdout и (опционально) в --out на rank 0.
	std::unique_ptr<std::ofstream> out_file;
	if (rank == 0 && !args.out_path.empty()) {
		out_file = std::make_unique<std::ofstream>(args.out_path);
		if (!out_file->is_open()) {
			std::cerr << "gpu: cannot open --out " << args.out_path << "\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	auto mirror = [&](const std::string &s) {
		if (rank != 0)
			return;
		std::cout << s;
		if (out_file)
			*out_file << s;
	};

	constexpr int HOST_LEN = MPI_MAX_PROCESSOR_NAME; // минимальный размер буфера для MPI_Get_processor_name
	char my_host[HOST_LEN];
	{
		int name_len = 0;
		mpi_ok(MPI_Get_processor_name(my_host, &name_len), "MPI_Get_processor_name");
		my_host[HOST_LEN - 1] = '\0';
	}

	constexpr int PCI_LEN = 32;
	char my_pci[PCI_LEN];
	std::snprintf(my_pci, sizeof(my_pci), "n/a");
	/* В режиме 1 visible GPU per rank индекс устройства внутри процесса всегда 0. */
	if (cudaDeviceGetPCIBusId(my_pci, sizeof(my_pci), local_gpu) != cudaSuccess) {
		std::snprintf(my_pci, sizeof(my_pci), "n/a");
	}

	std::vector<char> hosts_recv(static_cast<size_t>(nproc) * HOST_LEN);
	std::vector<char> pci_recv(static_cast<size_t>(nproc) * PCI_LEN);
	std::vector<int> gpu_counts(nproc, 0);
	mpi_ok(MPI_Allgather(my_host, HOST_LEN, MPI_CHAR, hosts_recv.data(), HOST_LEN,
						 MPI_CHAR, MPI_COMM_WORLD),
		   "MPI_Allgather hostnames");
	mpi_ok(MPI_Allgather(&local_gpu_count, 1, MPI_INT, gpu_counts.data(), 1, MPI_INT,
						 MPI_COMM_WORLD),
		   "MPI_Allgather gpu counts");
	mpi_ok(MPI_Allgather(my_pci, PCI_LEN, MPI_CHAR, pci_recv.data(), PCI_LEN,
						 MPI_CHAR, MPI_COMM_WORLD),
		   "MPI_Allgather pci bus ids");

	if (rank == 0) {
		for (int r = 0; r < nproc; ++r) {
			if (gpu_counts[static_cast<size_t>(r)] <= 0) {
				std::cerr
					<< "This benchmark expects at least 1 visible GPU per MPI rank.\n"
					<< "Rank " << r << " sees " << gpu_counts[static_cast<size_t>(r)]
					<< " GPU(s).\n";
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
	}

	const std::vector<std::string> rank_labels = build_rank_labels(hosts_recv, nproc, HOST_LEN);
	const std::vector<std::string> global_gpu_labels =
		build_global_gpu_labels(hosts_recv, nproc, HOST_LEN, gpu_counts);

	if (rank == 0) {
		{
			std::ostringstream oss;
			oss << "Env: " << (via_host ? "host" : "auto") << "\n";
			mirror(oss.str());
		}
		{
			std::ostringstream oss;
			oss << "Mode: " << mode_to_string(args.mode) << "\n";
			mirror(oss.str());
		}
		{
			std::ostringstream oss;
			oss << "Timer: " << timer_to_string(args.timer) << "\n";
			mirror(oss.str());
		}
		{
			std::ostringstream oss;
			oss << "Bytes: " << args.nbytes << "\n"
				<< "Warmup: " << args.warmup << "\n"
				<< "Iters: " << args.iters << "\n";
			mirror(oss.str());
		}

		int total_visible_gpus = 0;
		for (int v : gpu_counts)
			total_visible_gpus += v;
		std::ostringstream oss;
		oss << "Ranks: " << nproc << ", total visible GPUs: " << total_visible_gpus << "\n";
		mirror(oss.str());

		mirror("Rank map:\n");
		for (int r = 0; r < nproc; ++r) {
			const char *h = hosts_recv.data() + static_cast<size_t>(r) * HOST_LEN;
			const char *p = pci_recv.data() + static_cast<size_t>(r) * PCI_LEN;
			std::ostringstream oss;
			oss << "  r" << r << " hostname=" << h
				<< " local_gpu=0"
				<< " visible_gpus=" << gpu_counts[static_cast<size_t>(r)] << " pci=" << p
				<< "\n";
			mirror(oss.str());
		}
		std::cout << std::fixed << std::setprecision(REPORT_DIGITS);
		if (out_file)
			*out_file << std::fixed << std::setprecision(REPORT_DIGITS);
	}

	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier before benchmark");

	if (args.mode == Mode::OneToOne)
		schedule_one_to_one(rank, nproc, args, via_host,
							enable_local_shared_fallback,
							node_comm, local_rank, on_my_node,
							rank_labels, mirror);
	else if (args.mode == Mode::AllToAll)
		schedule_all_to_all(rank, nproc, args, via_host,
							rank_labels, mirror);

	mpi_ok(MPI_Comm_free(&node_comm), "MPI_Comm_free(node)");

	mpi_ok(MPI_Finalize(), "MPI_Finalize");
	return 0;
}
