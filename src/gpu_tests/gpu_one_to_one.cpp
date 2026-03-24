/**
 * gpu_one_to_one — перебор ВСЕХ пар GPU в стиле "мастер + воркеры" (как у Бегаева).
 *
 * Ранг 0 (мастер) формирует задания (src_rank, src_gpu, dst_rank, dst_gpu) для всех пар:
 *   for src_rank in ranks
 *     for dst_rank in ranks
 *       for src_gpu on src_rank
 *         for dst_gpu on dst_rank
 *
 * Режим передачи задаётся аргументом --mode:
 *   --mode host   : всегда через ОЗУ (D2H -> MPI -> H2D)
 *   --mode device : передача с device-указателями (нужен CUDA-aware MPI)
 *   --mode auto   : device если CUDA-aware, иначе host
 */

#include <mpi.h>

#if defined(OPEN_MPI) && OPEN_MPI
#include "mpi-ext.h"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void cuda_ok(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
    std::abort();
  }
}

bool mpi_cuda_aware() {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  return MPIX_Query_cuda_support() == 1;
#else
  return false;
#endif
}

enum class Mode {
  kAuto,
  kHost,
  kDevice,
};

struct Args {
  size_t nbytes = 4u * 1024u * 1024u;
  int warmup = 10;
  int iters = 50;
  Mode mode = Mode::kAuto;
};

struct Task {
  int src_rank = -1;
  int src_gpu = -1;
  int dst_rank = -1;
  int dst_gpu = -1;
  int stop = 0;
};

void usage(int rank) {
  if (rank != 0) return;
  std::cout << "gpu_one_to_one — all GPU pairs, master-worker\n"
            << "  --bytes N       size in bytes (default 4MiB)\n"
            << "  --warmup N      warmup iterations per pair\n"
            << "  --iters N       measured iterations per pair\n"
            << "  --mode M        auto | host | device\n";
}

Mode parse_mode(const std::string &s, int rank) {
  if (s == "auto") return Mode::kAuto;
  if (s == "host") return Mode::kHost;
  if (s == "device") return Mode::kDevice;
  if (rank == 0) std::cerr << "unknown --mode: " << s << " (use auto|host|device)\n";
  MPI_Abort(MPI_COMM_WORLD, 1);
  return Mode::kAuto;
}

Args parse_args(int argc, char **argv, int rank) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        if (rank == 0) std::cerr << "missing value for " << name << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      return argv[++i];
    };

    if (s == "--bytes") a.nbytes = static_cast<size_t>(std::strtoull(next("--bytes"), nullptr, 10));
    else if (s == "--warmup") a.warmup = std::atoi(next("--warmup"));
    else if (s == "--iters") a.iters = std::atoi(next("--iters"));
    else if (s == "--mode") a.mode = parse_mode(next("--mode"), rank);
    else if (s == "--help" || s == "-h") {
      usage(rank);
      MPI_Finalize();
      std::exit(0);
    } else {
      if (rank == 0) std::cerr << "unknown arg: " << s << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  return a;
}

bool use_host_staging(Mode mode, bool cuda_aware) {
  if (mode == Mode::kHost) return true;
  if (mode == Mode::kDevice) return false;
  return !cuda_aware;
}

int global_gpu_index(const std::vector<int> &prefix, int rank, int local_gpu) {
  return prefix[rank] + local_gpu;
}

/**
 * Выполнить одно задание (одна пара GPU), вернуть {avg_us, gbps, valid_metric}.
 * Метрику (valid=1) заполняет отправитель, либо единственный участник при src==dst.
 */
std::vector<double> run_task(int rank,
                             const Task &t,
                             const Args &args,
                             bool host_staging) {
  std::vector<double> ack(3, 0.0);
  const bool is_sender = (rank == t.src_rank);
  const bool is_receiver = (rank == t.dst_rank);
  if (!is_sender && !is_receiver) {
    return ack;
  }

  char *d_send = nullptr;
  char *d_recv = nullptr;
  char *h_buf = nullptr;
  const int count = static_cast<int>(args.nbytes);

  if (is_sender) {
    cuda_ok(cudaSetDevice(t.src_gpu), "cudaSetDevice(src)");
    cuda_ok(cudaMalloc(&d_send, args.nbytes), "cudaMalloc(src)");
    cuda_ok(cudaMemset(d_send, 0x5A, args.nbytes), "cudaMemset(src)");
  }
  if (is_receiver) {
    cuda_ok(cudaSetDevice(t.dst_gpu), "cudaSetDevice(dst)");
    cuda_ok(cudaMalloc(&d_recv, args.nbytes), "cudaMalloc(dst)");
    cuda_ok(cudaMemset(d_recv, 0, args.nbytes), "cudaMemset(dst)");
  }
  if (host_staging && (is_sender || is_receiver)) {
    cuda_ok(cudaMallocHost(&h_buf, args.nbytes), "cudaMallocHost");
  }

  MPI_Status st{};

  auto do_one = [&](int tag) {
    if (t.src_rank == t.dst_rank) {
      if (t.src_gpu == t.dst_gpu) return;

      cuda_ok(cudaSetDevice(t.src_gpu), "cudaSetDevice(src local)");
      if (host_staging) {
        cuda_ok(cudaMemcpy(h_buf, d_send, args.nbytes, cudaMemcpyDeviceToHost), "D2H local");
        cuda_ok(cudaSetDevice(t.dst_gpu), "cudaSetDevice(dst local)");
        cuda_ok(cudaMemcpy(d_recv, h_buf, args.nbytes, cudaMemcpyHostToDevice), "H2D local");
      } else {
        cuda_ok(cudaMemcpyPeer(d_recv, t.dst_gpu, d_send, t.src_gpu, args.nbytes), "cudaMemcpyPeer");
      }
      return;
    }

    if (is_sender) {
      if (host_staging) {
        cuda_ok(cudaMemcpy(h_buf, d_send, args.nbytes, cudaMemcpyDeviceToHost), "D2H");
        MPI_Send(h_buf, count, MPI_BYTE, t.dst_rank, tag, MPI_COMM_WORLD);
      } else {
        MPI_Send(d_send, count, MPI_BYTE, t.dst_rank, tag, MPI_COMM_WORLD);
      }
    }
    if (is_receiver) {
      if (host_staging) {
        MPI_Recv(h_buf, count, MPI_BYTE, t.src_rank, tag, MPI_COMM_WORLD, &st);
        cuda_ok(cudaMemcpy(d_recv, h_buf, args.nbytes, cudaMemcpyHostToDevice), "H2D");
      } else {
        MPI_Recv(d_recv, count, MPI_BYTE, t.src_rank, tag, MPI_COMM_WORLD, &st);
      }
    }
  };

  for (int i = 0; i < args.warmup; ++i) {
    do_one(i);
  }
  const double t0 = MPI_Wtime();
  for (int i = 0; i < args.iters; ++i) {
    do_one(1000 + i);
  }
  const double t1 = MPI_Wtime();

  if (is_sender || (t.src_rank == t.dst_rank && is_receiver)) {
    const double one_sec = (t1 - t0) / static_cast<double>(std::max(1, args.iters));
    ack[0] = one_sec * 1e6;
    ack[1] = (one_sec > 0.0) ? (static_cast<double>(args.nbytes) / one_sec / 1e9) : 0.0;
    ack[2] = 1.0;
  }

  if (h_buf) cudaFreeHost(h_buf);
  if (d_send) cudaFree(d_send);
  if (d_recv) cudaFree(d_recv);
  return ack;
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const Args args = parse_args(argc, argv, rank);
  const bool cuda_aware = mpi_cuda_aware();
  const bool host_staging = use_host_staging(args.mode, cuda_aware);

  if (args.mode == Mode::kDevice && !cuda_aware) {
    if (rank == 0) {
      std::cerr << "Requested --mode device, but MPI is not CUDA-aware.\n";
    }
    MPI_Finalize();
    return 2;
  }

  int local_gpu_count = 0;
  cudaError_t cnt_err = cudaGetDeviceCount(&local_gpu_count);
  if (cnt_err != cudaSuccess) local_gpu_count = 0;

  std::vector<int> gpu_counts(nproc, 0);
  MPI_Gather(&local_gpu_count, 1, MPI_INT, gpu_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "CUDA-aware MPI: " << (cuda_aware ? "yes" : "no") << "\n";
    std::cout << "Mode: " << (host_staging ? "host (through RAM)" : "device (direct GPU pointers)") << "\n";
  }

  if (rank == 0) {
    std::vector<int> prefix(nproc + 1, 0);
    for (int r = 0; r < nproc; ++r) prefix[r + 1] = prefix[r] + gpu_counts[r];
    const int total_gpu = prefix[nproc];

    std::cout << "Ranks: " << nproc << ", total GPUs: " << total_gpu << "\n";
    std::cout << std::fixed << std::setprecision(3);

    Task t{};
    for (int src_rank = 0; src_rank < nproc; ++src_rank) {
      for (int dst_rank = 0; dst_rank < nproc; ++dst_rank) {
        for (int src_gpu = 0; src_gpu < gpu_counts[src_rank]; ++src_gpu) {
          for (int dst_gpu = 0; dst_gpu < gpu_counts[dst_rank]; ++dst_gpu) {
            t.src_rank = src_rank;
            t.src_gpu = src_gpu;
            t.dst_rank = dst_rank;
            t.dst_gpu = dst_gpu;
            t.stop = 0;

            if (src_rank == dst_rank) {
              if (src_rank == 0) {
                auto ack = run_task(rank, t, args, host_staging);
                if (ack[2] > 0.5) {
                  std::cout << "pair g" << static_cast<int>(global_gpu_index(prefix, src_rank, src_gpu))
                            << " -> g" << static_cast<int>(global_gpu_index(prefix, dst_rank, dst_gpu))
                            << " (r" << src_rank << ":" << src_gpu << " -> r" << dst_rank << ":" << dst_gpu
                            << ") avg_us=" << ack[0] << " gbps=" << ack[1] << "\n";
                }
              } else {
                MPI_Send(&t, sizeof(Task), MPI_BYTE, src_rank, 1, MPI_COMM_WORLD);
                std::vector<double> ack(3, 0.0);
                MPI_Recv(ack.data(), 3, MPI_DOUBLE, src_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (ack[2] > 0.5) {
                  std::cout << "pair g" << static_cast<int>(global_gpu_index(prefix, src_rank, src_gpu))
                            << " -> g" << static_cast<int>(global_gpu_index(prefix, dst_rank, dst_gpu))
                            << " (r" << src_rank << ":" << src_gpu << " -> r" << dst_rank << ":" << dst_gpu
                            << ") avg_us=" << ack[0] << " gbps=" << ack[1] << "\n";
                }
              }
              continue;
            }

            if (src_rank != 0) MPI_Send(&t, sizeof(Task), MPI_BYTE, src_rank, 1, MPI_COMM_WORLD);
            if (dst_rank != 0) MPI_Send(&t, sizeof(Task), MPI_BYTE, dst_rank, 1, MPI_COMM_WORLD);

            std::vector<double> metric(3, 0.0);
            if (src_rank == 0 || dst_rank == 0) {
              auto ack0 = run_task(rank, t, args, host_staging);
              if (ack0[2] > 0.5) metric = ack0;
            }

            if (src_rank != 0) {
              std::vector<double> ack(3, 0.0);
              MPI_Recv(ack.data(), 3, MPI_DOUBLE, src_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              if (ack[2] > 0.5) metric = ack;
            }
            if (dst_rank != 0) {
              std::vector<double> ack(3, 0.0);
              MPI_Recv(ack.data(), 3, MPI_DOUBLE, dst_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              if (ack[2] > 0.5) metric = ack;
            }

            std::cout << "pair g" << static_cast<int>(global_gpu_index(prefix, src_rank, src_gpu))
                      << " -> g" << static_cast<int>(global_gpu_index(prefix, dst_rank, dst_gpu))
                      << " (r" << src_rank << ":" << src_gpu << " -> r" << dst_rank << ":" << dst_gpu
                      << ") avg_us=" << metric[0] << " gbps=" << metric[1] << "\n";
          }
        }
      }
    }

    t.stop = 1;
    for (int r = 1; r < nproc; ++r) {
      MPI_Send(&t, sizeof(Task), MPI_BYTE, r, 1, MPI_COMM_WORLD);
    }
  } else {
    while (true) {
      Task t{};
      MPI_Recv(&t, sizeof(Task), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (t.stop) break;

      auto ack = run_task(rank, t, args, host_staging);
      MPI_Send(ack.data(), 3, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();
  return 0;
}
