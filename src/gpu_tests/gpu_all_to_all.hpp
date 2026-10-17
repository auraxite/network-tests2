#pragma once

#include "gpu_common.hpp"

#include <functional>
#include <string>
#include <vector>

namespace gpu_benchmark {

/* Режим all_to_all: в каждой итерации все ранги одновременно передают всем.
   Задержка пары (src, dst) — время от общего старта итерации до завершения
   приёма от src на dst. Строки результата отдаются в mirror. */
void schedule_all_to_all(int rank, int nproc, const Args &args, bool via_host,
                         MPI_Comm node_comm,
                         const std::vector<int> &on_my_node,
                         const std::vector<int> &node_ranks,
                         const std::vector<std::string> &rank_labels,
                         const std::function<void(const std::string &)> &mirror);

} // namespace gpu_benchmark
