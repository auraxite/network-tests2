#pragma once

#include "gpu_common.hpp"

#include <functional>
#include <string>
#include <vector>

namespace gpu_benchmark {

/* Режим one_to_one: ранг 0 по очереди назначает пары (src, dst), остальные
   пары в это время простаивают. Строки результата отдаются в mirror. */
void schedule_one_to_one(int rank, int nproc, const Args &args, bool via_host,
                         MPI_Comm node_comm, int node_rank,
                         const std::vector<int> &on_my_node,
                         const std::vector<std::string> &rank_labels,
                         const std::function<void(const std::string &)> &mirror);

} // namespace gpu_benchmark
