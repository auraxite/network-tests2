#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Config {
  std::size_t message_bytes = 4 * 1024 * 1024;
  int warmup_iters = 10;
  int iters = 50;
  bool all_to_all = false;
  bool pairwise_pingpong = true;
};

void parse_args(int argc, char **argv, Config &cfg, int rank) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need_val = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        if (rank == 0) {
          std::cerr << "Missing value for " << name << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
      }
      return argv[++i];
    };
    if (a == "--bytes") {
      cfg.message_bytes = static_cast<std::size_t>(std::strtoull(need_val("--bytes"), nullptr, 10));
    } else if (a == "--warmup") {
      cfg.warmup_iters = std::atoi(need_val("--warmup"));
    } else if (a == "--iters") {
      cfg.iters = std::atoi(need_val("--iters"));
    } else if (a == "--all-to-all") {
      cfg.all_to_all = true;
    } else if (a == "--pairwise") {
      cfg.pairwise_pingpong = true;
    } else if (a == "--help") {
      if (rank == 0) {
        std::cout
            << "gpu_load options:\n"
            << "  --bytes N       message size in bytes (default 4194304)\n"
            << "  --warmup N      warmup iterations (default 10)\n"
            << "  --iters N       measured iterations (default 50)\n"
            << "  --all-to-all    run MPI_Alltoall load phase\n"
            << "  --pairwise      run pairwise ping-pong matrix phase (default on)\n";
      }
      MPI_Finalize();
      std::exit(0);
    } else {
      if (rank == 0) {
        std::cerr << "Unknown argument: " << a << "\n";
      }
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }
}

double pingpong_pair_seconds(int rank, int peer, std::vector<char> &buf, int warmup, int iters) {
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < warmup; ++i) {
    if (rank == peer) {
      MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, 0, 101, MPI_COMM_WORLD);
    } else if (rank == 0) {
      MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, peer, 100, MPI_COMM_WORLD);
      MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, peer, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  for (int i = 0; i < iters; ++i) {
    if (rank == peer) {
      MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, 0, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, 0, 201, MPI_COMM_WORLD);
    } else if (rank == 0) {
      MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, peer, 200, MPI_COMM_WORLD);
      MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_CHAR, peer, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  double t1 = MPI_Wtime();
  return t1 - t0;
}

void print_pairwise_matrix(int rank, int world, const Config &cfg) {
  std::vector<char> buf(cfg.message_bytes, static_cast<char>(rank & 0xFF));
  std::vector<double> latency_us(world, 0.0);
  std::vector<double> bandwidth_gbps(world, 0.0);

  if (rank == 0) {
    std::cout << "\n=== Pairwise Ping-Pong Matrix (rank0 <-> rankN) ===\n";
    std::cout << "Message bytes: " << cfg.message_bytes << ", warmup: " << cfg.warmup_iters << ", iters: " << cfg.iters
              << "\n";
  }

  for (int peer = 1; peer < world; ++peer) {
    double sec = pingpong_pair_seconds(rank, peer, buf, cfg.warmup_iters, cfg.iters);
    if (rank == 0) {
      const double rtt_s = sec / static_cast<double>(cfg.iters);
      const double one_way_us = (rtt_s * 1e6) / 2.0;
      const double full_bytes = static_cast<double>(cfg.message_bytes) * 2.0;  // send + recv
      const double gbps = (full_bytes / rtt_s) / 1e9;
      latency_us[peer] = one_way_us;
      bandwidth_gbps[peer] = gbps;
    }
  }

  if (rank == 0) {
    std::cout << std::left << std::setw(10) << "peer" << std::setw(18) << "latency_us(one-way)"
              << std::setw(18) << "RTT_GBps" << "\n";
    for (int peer = 1; peer < world; ++peer) {
      std::cout << std::left << std::setw(10) << peer << std::setw(18) << std::fixed << std::setprecision(3)
                << latency_us[peer] << std::setw(18) << std::fixed << std::setprecision(3) << bandwidth_gbps[peer]
                << "\n";
    }
  }
}

void run_all_to_all(int rank, int world, const Config &cfg) {
  std::vector<char> sendbuf(cfg.message_bytes * static_cast<std::size_t>(world), 1);
  std::vector<char> recvbuf(cfg.message_bytes * static_cast<std::size_t>(world), 0);

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < cfg.warmup_iters; ++i) {
    MPI_Alltoall(sendbuf.data(), static_cast<int>(cfg.message_bytes), MPI_CHAR, recvbuf.data(),
                 static_cast<int>(cfg.message_bytes), MPI_CHAR, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();
  for (int i = 0; i < cfg.iters; ++i) {
    MPI_Alltoall(sendbuf.data(), static_cast<int>(cfg.message_bytes), MPI_CHAR, recvbuf.data(),
                 static_cast<int>(cfg.message_bytes), MPI_CHAR, MPI_COMM_WORLD);
  }
  double t1 = MPI_Wtime();
  double local_sec = (t1 - t0) / static_cast<double>(cfg.iters);
  double worst_sec = 0.0;
  MPI_Reduce(&local_sec, &worst_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const double bytes_per_rank = static_cast<double>(cfg.message_bytes) * static_cast<double>(world);
    const double agg_bytes = bytes_per_rank * static_cast<double>(world);
    const double agg_gbps = (agg_bytes / worst_sec) / 1e9;
    std::cout << "\n=== MPI_Alltoall Load Phase ===\n";
    std::cout << "Message bytes per peer: " << cfg.message_bytes << ", world size: " << world << "\n";
    std::cout << "Worst-rank time per iter: " << std::fixed << std::setprecision(6) << worst_sec << " s\n";
    std::cout << "Approx aggregate throughput: " << std::fixed << std::setprecision(3) << agg_gbps << " GB/s\n";
  }
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  Config cfg;
  parse_args(argc, argv, cfg, rank);

  if (world < 2) {
    if (rank == 0) {
      std::cerr << "Need at least 2 MPI ranks.\n";
    }
    MPI_Finalize();
    return 1;
  }

  if (cfg.pairwise_pingpong) {
    print_pairwise_matrix(rank, world, cfg);
  }
  if (cfg.all_to_all) {
    run_all_to_all(rank, world, cfg);
  }

  MPI_Finalize();
  return 0;
}
