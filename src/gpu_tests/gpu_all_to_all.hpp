#pragma once

#include "gpu_common.hpp"

#include <functional> // std::function — колбэк mirror в schedule_all_to_all
#include <vector>     // std::vector в сигнатурах

namespace gpu_benchmark {

std::vector<double> run_all_to_all(int rank, int nproc, const Args &args,
								   bool check_host, int local_gpu,
								   const std::vector<std::string> &rank_labels);

void schedule_all_to_all(int rank, int nproc, const Args &args, bool via_host,
						 const std::vector<std::string> &rank_labels,
						 const std::function<void(const std::string &)> &mirror,
						 NetcdfBundle *nc, int matrix_idx);

} // namespace gpu_benchmark
