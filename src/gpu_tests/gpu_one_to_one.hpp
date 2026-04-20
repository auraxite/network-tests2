#pragma once

#include "gpu_common.hpp"

#include <functional> // std::function — колбэк mirror в schedule_one_to_one
#include <vector>     // std::vector в параметрах и возвращаемых типах

namespace gpu_benchmark {

/* Запускает один замер пары (t.src_rank → t.dst_rank). Буферы d_send / d_recv
   / h_buf принадлежат вызывающему ранку и выделены ОДИН раз в schedule_*; это
   убирает per-pair cudaMalloc/cudaMallocHost из горячего пути. h_buf может
   быть nullptr, если check_host == false. */
std::vector<double> run_one_to_one(int rank, const Task &t, const Args &args,
								   bool check_host,
								   bool same_node_pair,
								   char *d_send, char *d_recv, char *h_buf,
								   char *shared_h_buf, MPI_Win shared_h_win,
								   const std::vector<std::string> &rank_labels);

/* rank_to_gpu[r] — локальный (внутри узла процесса r) device id, выбранный в
   gpu_benchmark.cpp по локальному рангу. Используется здесь для проставления
   t.src_gpu / t.dst_gpu корректно, чтобы каждый процесс работал именно с тем
   GPU, который ему выделил Slurm/CUDA, а не с GPU 0 у всех. */
void schedule_one_to_one(int rank, int nproc, const Args &args, bool via_host,
						 const std::vector<int> &rank_to_gpu,
						 const std::vector<std::string> &rank_labels,
						 const std::function<void(const std::string &)> &mirror);

} // namespace gpu_benchmark
