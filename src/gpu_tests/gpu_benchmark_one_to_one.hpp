#pragma once

#include "gpu_benchmark_common.hpp"

#include <functional> // std::function — колбэк mirror в schedule_one_to_one
#include <vector>     // std::vector в параметрах и возвращаемых типах

namespace gpu_benchmark {

std::vector<double> run_one_to_one(int rank, const Task &t, const Args &args,
								   bool check_host,
								   const std::vector<std::string> &rank_labels);

void schedule_one_to_one(int rank, int nproc, const Args &args, bool via_host,
						 const std::vector<std::string> &rank_labels,
						 const std::function<void(const std::string &)> &mirror,
						 NetcdfBundle *nc, int matrix_idx);

} // namespace gpu_benchmark
