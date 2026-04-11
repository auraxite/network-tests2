#pragma once

#include "gpu_benchmark_common.hpp"

#include <functional> // std::function — колбэк mirror в schedule_all_to_all
#include <vector>     // std::vector в сигнатурах

namespace gpu_benchmark {

/** Ядро измерения all-to-all (аналог run_one_to_one в gpu_benchmark_one_to_one.hpp). */
std::vector<double> run_all_to_all(int rank, int nproc, const Args &args,
								   bool check_host, int local_gpu,
								   const std::vector<std::string> &rank_labels);

void schedule_all_to_all(int rank, int nproc, const Args &args, bool via_host,
						 const std::vector<std::string> &rank_labels,
						 const std::function<void(const std::string &)> &mirror);

} // namespace gpu_benchmark
