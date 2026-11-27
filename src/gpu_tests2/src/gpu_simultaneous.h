#ifndef GPU_SIMULTANEOUS_H
#define GPU_SIMULTANEOUS_H

#include "gpu_common.h"

/* Одновременные передачи в каждой итерации:
   all_to_all — все ранги передают всем;
   one_to_all — корень передаёт всем остальным;
   all_to_one — все остальные передают корню.
   В one_to_all и all_to_one корнем по очереди становится каждый ранг.
   Задержка пары (src, dst) — время от общего старта итерации до завершения приёма от src на dst.
   Буферы всегда в памяти GPU; путь (в том числе через RAM в среде host) выбирает UCX. */
void schedule_simultaneous(int rank, int nproc, const Args *args, char **rank_labels);

#endif
