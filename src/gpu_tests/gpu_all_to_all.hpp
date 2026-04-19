#pragma once

#include "gpu_common.hpp"

#include <functional> // std::function — колбэк mirror в schedule_all_to_all
#include <vector>     // std::vector в сигнатурах

namespace gpu_benchmark {

std::vector<double> run_all_to_all(int rank, int nproc, const Args &args,
								   bool check_host, int local_gpu,
								   const std::vector<std::string> &rank_labels);

/* rank_to_gpu[r] — локальный device id процесса r на его узле; для all_to_all
   нужен только rank_to_gpu[rank] (свой), но прокидываем весь вектор, чтобы
   сигнатура совпадала с schedule_one_to_one и для записи в Rank map был
   одинаковый источник истины. */
void schedule_all_to_all(int rank, int nproc, const Args &args, bool via_host,
						 const std::vector<int> &rank_to_gpu,
						 const std::vector<std::string> &rank_labels,
						 const std::function<void(const std::string &)> &mirror);

} // namespace gpu_benchmark
