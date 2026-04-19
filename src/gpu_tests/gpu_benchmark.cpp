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

	/* === Подбор GPU по локальному рангу на узле ===
	   Раньше каждый процесс делал cudaSetDevice(0). Это работало только при
	   --gpus-per-task=1 (когда Slurm-cgroup делал каждому процессу видимым
	   ровно один GPU как индекс 0). Но такая cgroup-изоляция выключает
	   cuda_ipc между процессами одного узла — UCX не может разделить
	   IPC-handle между двумя cgroup'ами. Это и давало плоские ~745 мкс
	   в env=auto: UCX делал fallback на cuda_copy (host staging).
	   При --gres=gpu:N все GPU узла видны всем процессам, и нам нужно
	   самим раздать кому какой device. Базируемся на ранге внутри
	   shared-memory-коммуникатора (MPI_COMM_TYPE_SHARED == «один узел»). */
	MPI_Comm node_comm = MPI_COMM_NULL;
	mpi_ok(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
							   MPI_INFO_NULL, &node_comm),
		   "MPI_Comm_split_type(node)");
	int local_rank = 0;
	mpi_ok(MPI_Comm_rank(node_comm, &local_rank), "MPI_Comm_rank(node)");
	MPI_Comm_free(&node_comm);

	int local_gpu_count = 0;
	cudaGetDeviceCount(&local_gpu_count);
	if (local_gpu_count <= 0) {
		std::cerr << "rank " << rank << ": no visible CUDA devices\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	const int my_gpu = local_rank % local_gpu_count;
	cuda_ok(cudaSetDevice(my_gpu), "cudaSetDevice(initial)");

	/* Глобальный справочник «ранг → его локальный device id». Дальше
	   schedule_one_to_one / schedule_all_to_all берут отсюда t.src_gpu /
	   t.dst_gpu и сами выбирают свой буфер — больше нигде не нужно
	   догадываться, какой GPU у соседнего ранга. */
	std::vector<int> rank_to_gpu(static_cast<size_t>(nproc), 0);
	mpi_ok(MPI_Allgather(&my_gpu, 1, MPI_INT, rank_to_gpu.data(), 1, MPI_INT,
						 MPI_COMM_WORLD),
		   "MPI_Allgather rank_to_gpu");

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
	/* PCI bus id берём для my_gpu (того, который выбрали по local_rank выше), а
	   не для GPU 0 — иначе в Rank map все 4 ранга узла рапортуют один и тот же
	   PCI и понять, кто на каком device, нельзя. */
	if (cudaDeviceGetPCIBusId(my_pci, sizeof(my_pci), my_gpu) != cudaSuccess) {
		std::snprintf(my_pci, sizeof(my_pci), "n/a");
	}

	std::vector<char> hosts_recv;
	std::vector<char> pci_recv;
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
				<< " local_gpu=" << rank_to_gpu[static_cast<size_t>(r)]
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
		schedule_one_to_one(rank, nproc, args, via_host, rank_to_gpu,
							rank_labels, mirror);
	else if (args.mode == Mode::AllToAll)
		schedule_all_to_all(rank, nproc, args, via_host, rank_to_gpu,
							rank_labels, mirror);

	mpi_ok(MPI_Finalize(), "MPI_Finalize");
	return 0;
}
