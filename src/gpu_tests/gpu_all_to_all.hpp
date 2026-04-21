#pragma once

#include "gpu_common.hpp"

#include <functional> // std::function — колбэк mirror в schedule_all_to_all
#include <vector>     // std::vector в сигнатурах

namespace gpu_benchmark {

std::vector<double> run_all_to_all(int rank, int nproc, const Args &args,
								   bool check_host, int local_gpu,
								   MPI_Comm node_comm,
								   const std::vector<int> &on_my_node,
								   const std::vector<int> &node_ranks,
								   const std::vector<std::string> &rank_labels);

void schedule_all_to_all(int rank, int nproc, const Args &args, bool via_host,
						 MPI_Comm node_comm, int node_rank,
						 const std::vector<int> &on_my_node,
						 const std::vector<int> &node_ranks,
						 const std::vector<std::string> &rank_labels,
						 const std::function<void(const std::string &)> &mirror);

} // namespace gpu_benchmark
