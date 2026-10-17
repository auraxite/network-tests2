#pragma once

#include "gpu_common.hpp"

#include <functional>
#include <string>
#include <vector>

namespace gpu_benchmark {

/* Одновременные передачи в каждой итерации:
   all_to_all — все ранги передают всем;
   one_to_all — корень передаёт всем остальным;
   all_to_one — все остальные передают корню.
   В one_to_all и all_to_one корнем по очереди становится каждый ранг.
   Задержка пары (src, dst) — время от общего старта итерации до завершения
   приёма от src на dst. Строки результата отдаются в mirror. */
void schedule_concurrent(int rank, int nproc, const Args &args, bool via_host,
                         MPI_Comm node_comm,
                         const std::vector<int> &on_my_node,
                         const std::vector<int> &node_ranks,
                         const std::vector<std::string> &rank_labels,
                         const std::function<void(const std::string &)> &mirror);

} // namespace gpu_benchmark
