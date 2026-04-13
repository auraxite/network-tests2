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
	const bool via_host = check_host(args.mode, cuda_aware);
	const std::vector<size_t> message_sizes = build_message_sizes(args, rank);

	// В single-size режиме дублируем текстовый отчёт в stdout и (опционально) в --out.
	std::unique_ptr<std::ofstream> out_file;
	if (rank == 0 && !args.sweep_sizes && !args.out_path.empty()) {
		out_file = std::make_unique<std::ofstream>(args.out_path);
		if (!out_file->is_open()) {
			std::cerr << "gpu: cannot open --out " << args.out_path << "\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	auto mirror = [&](const std::string &s) {
		if (rank != 0)
			return;
		if (args.sweep_sizes)
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

	int local_gpu_count = 0;
	cudaGetDeviceCount(&local_gpu_count);
	if (local_gpu_count > 0) {
		cudaSetDevice(0);
		if (cudaDeviceGetPCIBusId(my_pci, sizeof(my_pci), 0) != cudaSuccess) {
			std::snprintf(my_pci, sizeof(my_pci), "n/a");
		}
	}

	std::vector<char> hosts_recv;
	std::vector<char> pci_recv;
	// Нужно на root всегда: Gather host/pci и Bcast имён — и в режиме sweep (--sweep_sizes).
	if (rank == 0) {
		hosts_recv.resize(static_cast<size_t>(nproc) * HOST_LEN);
		pci_recv.resize(static_cast<size_t>(nproc) * PCI_LEN);
	}
	std::vector<int> gpu_counts(nproc, 0);

	mpi_ok(MPI_Gather(&local_gpu_count, 1, MPI_INT, gpu_counts.data(), 1, MPI_INT, 0,
					  MPI_COMM_WORLD),
		   "MPI_Gather gpu counts");
	mpi_ok(MPI_Gather(my_host, HOST_LEN, MPI_CHAR,
					  rank == 0 ? hosts_recv.data() : nullptr, HOST_LEN, MPI_CHAR, 0,
					  MPI_COMM_WORLD),
		   "MPI_Gather hostnames");
	mpi_ok(MPI_Gather(my_pci, PCI_LEN, MPI_CHAR,
					  rank == 0 ? pci_recv.data() : nullptr, PCI_LEN, MPI_CHAR, 0,
					  MPI_COMM_WORLD),
		   "MPI_Gather pci bus ids");

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

	if (rank != 0)
		hosts_recv.resize(static_cast<size_t>(nproc) * HOST_LEN);
	mpi_ok(MPI_Bcast(hosts_recv.data(), nproc * HOST_LEN, MPI_CHAR, 0, MPI_COMM_WORLD),
		   "MPI_Bcast hostnames");

	const std::vector<std::string> rank_labels = build_rank_labels(hosts_recv, nproc, HOST_LEN);

	if (rank == 0) {
		{
			std::ostringstream oss;
			oss << "Mode: " << (via_host ? "host" : "auto") << "\n";
			mirror(oss.str());
		}
		{
			std::ostringstream oss;
			oss << "Scheme: " << scheme_to_string(args.scheme) << "\n";
			mirror(oss.str());
		}
		{
			std::ostringstream oss;
			oss << "Timer: " << timer_to_string(args.timer) << "\n";
			mirror(oss.str());
		}
		{
			std::ostringstream oss;
			if (message_sizes.size() == 1) {
				oss << "Bytes: " << message_sizes.front() << "\n";
			} else {
				oss << "BytesBegin: " << args.begin_nbytes << "\n"
					<< "BytesEnd: " << args.end_nbytes << "\n"
					<< "BytesStep: " << args.step_nbytes << "\n"
					<< "NumSizes: " << message_sizes.size() << "\n";
			}
			oss << "Warmup: " << args.warmup << "\n"
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
			oss << "  r" << r << " hostname=" << h << " local_gpu=0"
				<< " visible_gpus=" << gpu_counts[static_cast<size_t>(r)] << " pci=" << p
				<< "\n";
			mirror(oss.str());
		}
		std::cout << std::fixed << std::setprecision(REPORT_DIGITS);
		if (out_file)
			*out_file << std::fixed << std::setprecision(REPORT_DIGITS);
	}

	mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier before benchmark");

	
	NetcdfBundle nc{};
	if (rank == 0 && args.sweep_sizes) {
		nc = netcdf_open_bundle(args.out_path, args.begin_nbytes, args.end_nbytes,
								args.step_nbytes, args.iters, nproc);
	}

	for (size_t size_idx = 0; size_idx < message_sizes.size(); ++size_idx) {
		Args run_args = args;
		run_args.nbytes = message_sizes[size_idx];
		if (args.scheme == Scheme::OneToOne)
			schedule_one_to_one(rank, nproc, run_args, via_host, rank_labels, mirror,
								rank == 0 && args.sweep_sizes ? &nc : nullptr,
								static_cast<int>(size_idx));
		else if (args.scheme == Scheme::AllToAll)
			schedule_all_to_all(rank, nproc, run_args, via_host, rank_labels, mirror,
								rank == 0 && args.sweep_sizes ? &nc : nullptr,
								static_cast<int>(size_idx));
		mpi_ok(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier between message sizes");
	}

	if (rank == 0 && args.sweep_sizes)
		netcdf_flush_and_close(nc);

	mpi_ok(MPI_Finalize(), "MPI_Finalize");
	return 0;
}
