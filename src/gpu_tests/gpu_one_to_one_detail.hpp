#pragma once

/* Общее для one_to_one и one_to_one_v2: буферы, замер пары, цикл рабочих рангов. */

#include "gpu_common.hpp"

#include <memory>
#include <string>
#include <vector>

namespace gpu_benchmark::one_to_one_detail {

constexpr int TAG_TASK = 1;
constexpr int TAG_ACK  = 2;

/* Буферы ранга: выделяются один раз на весь режим, а не на каждую пару. */
struct Buffers {
	DeviceBuffer                      d_send, d_recv;
	PinnedBuffer                      h_buf;    // только для пути через память
	std::unique_ptr<SharedHostWindow> shared;   // общий буфер узла, путь через память
	char                             *shared_buf = nullptr;
};

Buffers make_buffers(const Args &args, bool via_host, MPI_Comm node_comm, int node_rank,
                     int rank);

/* Замер пары. Возвращает ack: заполнен у получателя, у отправителя valid = 0.
   with_times: получатель пишет рядом с .raw файл .ts — время начала каждой
   итерации по системным часам (снимается вне замеряемого интервала). */
std::vector<double> measure_pair(int rank, const Task &t, const Args &args, bool via_host,
                                 bool same_node, Buffers &b,
                                 const std::vector<std::string> &rank_labels,
                                 bool with_times = false);

/* Ранги, кроме 0: ждут задание, участвуют в замере, возвращают ack. */
void worker_loop(int rank, const Args &args, bool via_host, Buffers &b,
                 const std::vector<int> &on_my_node,
                 const std::vector<std::string> &rank_labels, bool with_times);

bool same_node(const Task &t, const std::vector<int> &on_my_node);

/* Секунды с начала эпохи по системным часам (общие для узлов при NTP). */
double epoch_now();

} // namespace gpu_benchmark::one_to_one_detail
