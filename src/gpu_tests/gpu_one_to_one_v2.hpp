#pragma once

#include "gpu_common.hpp"

#include <functional>
#include <string>
#include <vector>

namespace gpu_benchmark {

/* Режим one_to_one_v2: замер тот же, что в one_to_one, плюс
   --order random --seed N — случайный порядок пар;
   строки pair_time — время начала и конца каждой пары по системным часам;
   файл .ts рядом с .raw — время начала каждой итерации. */
void schedule_one_to_one_v2(int rank, int nproc, const Args &args, bool via_host,
                            MPI_Comm node_comm, int node_rank,
                            const std::vector<int> &on_my_node,
                            const std::vector<std::string> &rank_labels,
                            const std::function<void(const std::string &)> &mirror);

} // namespace gpu_benchmark
