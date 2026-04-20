#pragma once

#include "gpu_common.hpp"

#include <functional>
#include <string>
#include <vector>

namespace gpu_benchmark {

void schedule_one_to_one_node(int rank, int nproc, int local_gpu_count,
							  const std::vector<int> &gpu_counts, const Args &args,
							  bool check_host,
							  const std::vector<std::string> &global_gpu_labels,
							  const std::function<void(const std::string &)> &mirror);

} // namespace gpu_benchmark
