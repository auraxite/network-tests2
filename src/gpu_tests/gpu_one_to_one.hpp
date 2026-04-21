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

/* node_comm / node_rank / on_my_node готовятся в gpu_benchmark.cpp один раз и
   передаются сюда, чтобы не делать повторный
   MPI_Comm_split_type/MPI_Allgather внутри schedule_one_to_one. */
void schedule_one_to_one(int rank, int nproc, const Args &args, bool via_host,
						 bool enable_local_shared_fallback,
						 MPI_Comm node_comm, int node_rank,
						 const std::vector<int> &on_my_node,
						 const std::vector<std::string> &rank_labels,
						 const std::function<void(const std::string &)> &mirror);

} // namespace gpu_benchmark
