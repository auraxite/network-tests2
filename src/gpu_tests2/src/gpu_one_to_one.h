#ifndef GPU_ONE_TO_ONE_H
#define GPU_ONE_TO_ONE_H

#include "gpu_common.h"

/* Режим one_to_one: ранг 0 по очереди (или в случайном порядке, --order random) назначает
   пары (src, dst), остальные ранги в это время простаивают. Буферы всегда в памяти GPU и
   отдаются MPI; путь задают переменные UCX среды. Исключение — --sys-via-host: пары GPU
   разных процессоров одного узла программа копирует через общий буфер в RAM сама.
   on_my_node[r] — ранг r на этом узле; gpu_numa[r] — процессор GPU ранга r (-1 — неизвестно). */
void schedule_one_to_one(int rank, int nproc, const Args *args, MPI_Comm node_comm, int node_rank,
                         const int *on_my_node, char **rank_labels, const int *gpu_numa);

#endif
