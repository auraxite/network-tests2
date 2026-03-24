#include "my_time.h"
#include "my_malloc.h"
#include "tests_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

extern int comm_rank;
extern int comm_size;
extern int* gpu_count;
extern int total_gpu;

/*
 * one_to_one_cuda_normal.c — измерение задержки/времени передачи между парами GPU.
 *
 * Схема работы:
 *   - Ранг 0 (мастер) перебирает все пары MPI-процессов (n,m) и все пары локальных GPU
 *     на отправителе и приёмнике.
 *   - Для каждой пары мастер рассылает задание в gpu_sr[4]: [send_proc, send_gpu, recv_proc, recv_gpu].
 *   - Участники с соответствующими рангами вызывают real_one_to_one_cuda и шлют подтверждение conf.
 *   - После всех тестов мастер шлёт send_proc = -1 — сигнал «завершить цикл» на ненулевых рангах.
 *
 * Два режима в real_one_to_one_cuda:
 *   - Один узел (source_proc == dest_proc): cudaMemcpyPeerAsync между GPU + события CUDA (время в секундах * 0.0001 от ms).
 *   - Разные узлы: GPU→CPU (D2H), MPI_Send/MPI_Recv, CPU→GPU (H2D); время считается на CPU-таймере
 *     у отправителя и у приёмника отдельно; в матрицу результатов пишет только процесс-приёмник (dest_proc).
 *
 * Индексация times: dest_gpu * total_gpu + stride + source_gpu, где stride — сумма gpu_count[0..source_proc-1]
 * (смещение «глобального» индекса GPU отправителя в линейной нумерации всех GPU кластера).
 */

Test_time_result_type calc_stats( px_my_time_type* all_times, int num_repeats );

void real_one_to_one_cuda( Test_time_result_type *times, int mes_length, int num_repeats, int source_proc, int dest_proc,
                           int source_gpu, int dest_gpu );

/* Оркестратор: только ранг 0 управляет порядком тестов; остальные крутятся в while до «стоп-сообщения». */
int one_to_one_cuda( Test_time_result_type * times, int mes_length, int num_repeats )
{

    int i, j, k, n, m;

    int gpu_sr[4];  /* Задание для воркеров: [0]=send_proc, [1]=send_gpu, [2]=recv_proc, [3]=recv_gpu */
    int conf = 1; /* «тест на моём ранге выполнен» — отсылается мастеру тегом 2 */
    int send_proc, recv_proc;
    int send_gpu, recv_gpu;
    MPI_Status status;
    cudaError_t cuda_error;
    if ( comm_rank == 0 )
    {
        /* Внешние циклы: все упорядоченные пары (отправитель-процесс, приёмник-процесс). */
        for ( n = 0; n < comm_size; n++ )
        {
            for ( m = 0; m < comm_size; m++ )
            {
                send_proc = n;
                recv_proc = m;

                gpu_sr[0] = send_proc;
                gpu_sr[2] = recv_proc;

                /* Внутренние циклы: все пары локальных GPU на этих двух процессах. */
                for ( i = 0; i < gpu_count[send_proc]; i++ )
                    for ( j = 0; j < gpu_count[recv_proc]; j++ )
                {
                    send_gpu = i;
                    recv_gpu = j;
                    gpu_sr[1] = i;
                    gpu_sr[3] = j;
                    printf("Test between %d, GPU %d and %d, GPU %d began\n", send_proc,
                    i, recv_proc, j);
                    /* Тот же MPI-ранг: либо только он сам делает peer-копирование, либо мастер шлёт ему задание. */
                    if ( send_proc == recv_proc ) 
                    {
                        if ( send_proc != 0 ) 
                        {
                            /* Мастер не участвует в CUDA на чужом узле: шлём задание рангу send_proc, ждём conf. */
                            MPI_Send( gpu_sr, 4, MPI_INT, send_proc, 1, MPI_COMM_WORLD );

                            MPI_Recv( &conf, 1, MPI_INT, send_proc, 2, MPI_COMM_WORLD, &status );
                            printf("Test between %d, GPU %d and %d, GPU %d finished\n", send_proc,
                            i, recv_proc, j);
                        
                        }
                        else
                        {
                            /* send_proc==recv_proc==0: мастер сам выполняет тест на своих GPU. */
                            real_one_to_one_cuda( times, mes_length, num_repeats, send_proc, recv_proc,
                                                  send_gpu, recv_gpu );
                        }
                        continue;
                    }

                    /* Разные процессы: обоим участникам нужно одинаковое задание (тег 1). */
                    if ( send_proc )
                        MPI_Send( gpu_sr, 4, MPI_INT, send_proc, 1, MPI_COMM_WORLD );
                    if ( recv_proc )
                        MPI_Send( gpu_sr, 4, MPI_INT, recv_proc, 1, MPI_COMM_WORLD );
                    
                    /* Межузловой обмен: CUDA+MPI делают только ранги, у которых comm_rank совпадает с send или recv;
                     * при этом один из них часто ранг 0 — он же мастер, поэтому условие «0 в паре». */
                    if ( recv_proc == 0 || send_proc == 0)
                        real_one_to_one_cuda( times, mes_length, num_repeats, send_proc, recv_proc,
                                              send_gpu, recv_gpu );

                    /* Синхронизация с воркерами: каждый из задействованных рангов шлёт conf (тег 2). */
                    if ( send_proc )
                        MPI_Recv( &conf, 1, MPI_INT, send_proc, 2, MPI_COMM_WORLD, &status );
                    if ( recv_proc )
                        MPI_Recv( &conf, 1, MPI_INT, recv_proc, 2, MPI_COMM_WORLD, &status );
                    printf("Test between %d, GPU %d and %d, GPU %d finished\n", send_proc,
                    i, recv_proc, j);
                }

            }
        }
        /* Стоп-сигнал: send_proc == -1 заставляет воркеров выйти из while. */
        gpu_sr[0] = -1;
        for ( i = 1; i < comm_size; i++ )
            MPI_Send( gpu_sr, 4, MPI_INT, i, 1, MPI_COMM_WORLD );
    }
    else 
    {
        /* Ненулевой ранг: бесконечно принимает задания от мастера (ранг 0), тег 1. */
        while( 1 )
        {
            MPI_Recv( gpu_sr, 4, MPI_INT, 0, 1, MPI_COMM_WORLD, &status );
            send_proc = gpu_sr[0];
            send_gpu = gpu_sr[1];
            recv_proc = gpu_sr[2];
            recv_gpu = gpu_sr[3];

            if ( send_proc == -1 )
                break;

            /* Локальный случай на воркере: оба конца на этом же MPI-процессе. */
            if (send_proc == comm_rank && recv_proc == comm_rank) {
                real_one_to_one_cuda( times, mes_length, num_repeats, send_proc, recv_proc,
                                      send_gpu, recv_gpu );
                MPI_Send( &conf, 1, MPI_INT, 0, 2, MPI_COMM_WORLD );
                continue;
            }

            /* Иначе этот ранг может быть только отправителем, только приёмником или обоими (редко при разнесённых GPU). */
            if ( send_proc == comm_rank )
                real_one_to_one_cuda( times, mes_length, num_repeats, send_proc, recv_proc,
                                      send_gpu, recv_gpu );
            if ( recv_proc == comm_rank ) 
                real_one_to_one_cuda( times, mes_length, num_repeats, send_proc, recv_proc,
                                      send_gpu, recv_gpu );
            /* Один conf на задание: если ранг и send и recv, real_one_to_one вызовется дважды, conf один — особенность схемы. */
            MPI_Send( &conf, 1, MPI_INT, 0, 2, MPI_COMM_WORLD );
        }
    }
    return 0;
}

/*
 * Реальное измерение для одной пары (source_proc:source_gpu) -> (dest_proc:dest_gpu).
 * На разных рангах выполняются разные ветки (отправитель / приёмник); матрица times заполняется
 * на приёмнике (или при intra-node — на том ранге, где вызывается и есть доступ к обоим GPU).
 */
void real_one_to_one_cuda( Test_time_result_type *times, int mes_length, int num_repeats, int source_proc, int dest_proc,
                           int source_gpu, int dest_gpu )
{
    px_my_time_type time_beg,time_end; /* метки времени на CPU (px_my_cpu_time) для межузлового пути */
    char *data = NULL;       /* в межузловом пути — ОЗУ под MPI; в intra-node ветке ниже — память на GPU (cudaMalloc) */
    char *dataGPU = NULL;    /* указатель на выделенную cudaMalloc память на текущем device */
    /* По одному значению на каждый повтор измерения; потом calc_stats() сводит их в average/median/deviation/min */
    px_my_time_type *tmp_results=NULL;
    MPI_Status status;       /* сюда MPI_Recv кладёт источник, тег и т.д. (для тега 100 достаточно знать, что пришло) */
    int i;
    /* Приёмник после H2D шлёт отправителю MPI_Send(..., тег 100); отправитель делает Recv в tmp — синхронизация раундов */
    int tmp;
    cudaError_t cuda_error;
    /* stride = число GPU на рангах < source_proc; stride+source_gpu — глобальный индекс отправителя.
       Индекс в times: dest_gpu * total_gpu + stride + source_gpu */
    int stride = 0;
    for ( i = 0; i < source_proc; i++) 
        stride += gpu_count[i];

    if ( source_proc == dest_proc )
    {
        if ( source_gpu == dest_gpu )
        {
            /* Нет передачи между разными концами — ячейку матрицы обнуляем и выходим */
	        times[dest_gpu * total_gpu + stride + source_gpu].average = 0;
            times[dest_gpu * total_gpu + stride + source_gpu].deviation = 0;
            times[dest_gpu * total_gpu + stride + source_gpu].median = 0;
            return;
        }
        else
        {
            /* Два GPU на одном MPI-процессе: только CUDA P2P, без MPI */
            float timing; /* длительность одного копирования (мс), потом перевод в tmp_results */
            tmp_results = ( px_my_time_type* )malloc( num_repeats * sizeof( px_my_time_type ) ); /* время каждого повтора */
            cudaEvent_t start, stop; /* CUDA-события: засечь момент до/после копирования */
            cudaStream_t src_dst_stream; /* асинхронная очередь команд на GPU */
            cudaEventCreate( &start ); /* создать событие «старт» */
            cudaEventCreate( &stop ); /* создать событие «стоп» */
            cudaSetDevice( source_gpu ); /* дальше все cuda* относятся к GPU-отправителю */
            cudaMalloc( ( void** ) &data, mes_length ); /* выделить mes_length байт в VRAM источника */
            cudaDeviceEnablePeerAccess ( dest_gpu, 0 ); /* источнику разрешить доступ к памяти dest GPU */
            cudaStreamCreate ( &src_dst_stream ); /* создать поток для MemcpyPeerAsync */
            cudaSetDevice( dest_gpu ); /* переключиться на GPU-приёмник */
            cudaMalloc( ( void** ) &dataGPU, mes_length ); /* буфер назначения в VRAM приёмника */
            cudaSetDevice( source_gpu ); /* вернуться на источник: sync/события привязаны к нему */
            /* data/dataGPU — просто два куска VRAM размера mes_length; содержимое для замера не задают */
            for ( i = 0; i < num_repeats; i++ )
            {
                cudaEventRecord ( start, src_dst_stream );
                cudaMemcpyPeerAsync( dataGPU, dest_gpu, data, source_gpu, mes_length, src_dst_stream ); /* копия mes_length байт: VRAM→VRAM */
                cudaEventRecord ( stop, src_dst_stream );
                cuda_error = cudaDeviceSynchronize(); /* CPU ждёт конца всех задач на source_gpu, иначе время по events было бы неверным */
                if ( cuda_error )
                {
                    printf("Assync error%s\n", cudaGetErrorString( cuda_error ) );
                }
                cudaEventElapsedTime ( &timing, start, stop ); /* timing в миллисекундах (документация CUDA) */
                /* *0.0001 — не стандартные секунды (для с → с было бы *0.001); так принято в этом проекте для единообразия с другими тестами */
                tmp_results[i] = (double)timing * 0.0001;
            }
            cudaFree ( ( void** ) &data); /* VRAM источника */
            cudaDeviceReset(); /* сброс контекста на source_gpu */
            cudaSetDevice( dest_gpu );
            cudaFree ( ( void** ) &dataGPU); /* VRAM приёмника */
            cudaDeviceReset(); /* сброс контекста на dest_gpu */
            times[dest_gpu * total_gpu + stride + source_gpu] = calc_stats( tmp_results, num_repeats ); /* сводка повторов → ячейка матрицы */
            printf("Test between %d:%d and %d:%d finished with %.10lf med, %.10lf dev and %.10lf avg\n",
                    source_proc, source_gpu, dest_proc, dest_gpu, times[dest_gpu * total_gpu + stride + source_gpu].median,
                    times[dest_gpu * total_gpu + stride + source_gpu].deviation, times[dest_gpu * total_gpu + stride + source_gpu].average);
            fflush(stdout); /* сброс буфера stdout */
            free ( tmp_results ); /* хост: сырые времена повторов */
            return;
        }
    }

    /* Разные MPI-ранги: GPU→ОЗУ→MPI→ОЗУ→GPU; тег 0 — данные, тег 100 — «раунд готов» */
    tmp_results = ( px_my_time_type* )malloc( num_repeats * sizeof( px_my_time_type ) );

    if ( comm_rank == source_proc )
        cudaSetDevice( source_gpu ); /* буфер на своём GPU */
    if ( comm_rank == dest_proc )
        cudaSetDevice( dest_gpu );

    cuda_error = cudaMalloc( ( void** ) &dataGPU, mes_length );
    if ( cuda_error )
	printf("Malloc error%s\n", cudaGetErrorString( cuda_error ) );
    data = ( char* )malloc( sizeof( char ) * mes_length ); /* ОЗУ под байты для MPI на этом ранге */

    for ( i = 0; i < num_repeats; i++ )
    {
        if ( comm_rank == source_proc )
        {
            time_beg = px_my_cpu_time();
            cuda_error = cudaMemcpy( data, dataGPU, mes_length, cudaMemcpyDeviceToHost ); /* с GPU в RAM */
            if ( cuda_error )
                printf("Src copy error%s\n", cudaGetErrorString( cuda_error ) );
            MPI_Send( data, mes_length, MPI_BYTE, dest_proc, 0, MPI_COMM_WORLD); /* в сеть */
            time_end = px_my_cpu_time();
            tmp_results[i] = ( time_end - time_beg ); /* D2H + Send */
            MPI_Recv( &tmp, 1, MPI_INT, dest_proc, 100, MPI_COMM_WORLD, &status ); /* ждём приёмник */
        }
        if ( comm_rank == dest_proc )
        {
            time_beg = px_my_cpu_time();
            MPI_Recv( data, mes_length, MPI_BYTE, source_proc, 0, MPI_COMM_WORLD, &status);
            cuda_error = cudaMemcpy( dataGPU, data, mes_length, cudaMemcpyHostToDevice ); /* с RAM на GPU */
            if ( cuda_error )
                printf("dst copy error\n", cudaGetErrorString( cuda_error ) );
            time_end = px_my_cpu_time();
            tmp_results[i] = ( time_end - time_beg ); /* Recv + H2D */
            MPI_Send( &comm_rank, 1, MPI_INT, source_proc, 100, MPI_COMM_WORLD ); /* отпускаем отправителя */
        }
    }

    cudaFree ( (void**) &dataGPU );
    cudaDeviceReset ();
    free( data );
 
    if ( source_proc == comm_rank ) 
    {
        free( tmp_results ); /* в times пишет только приёмник, просто договоренность */
        return;
    }
    times[dest_gpu * total_gpu + stride + source_gpu] = calc_stats( tmp_results, num_repeats );
    printf("Test between %d:%d and %d:%d finished with %lf med, %lf dev and %lf avg\n",
                    source_proc, source_gpu, dest_proc, dest_gpu, times[dest_gpu * total_gpu + stride + source_gpu].median,
                    times[dest_gpu * total_gpu + stride + source_gpu].deviation, times[dest_gpu * total_gpu + stride + source_gpu].average);
    fflush(stdout);
}

/* Сводка по num_repeats замерам; qsort портит порядок all_times — avg/dev считают до него */
Test_time_result_type calc_stats( px_my_time_type* all_times, int num_repeats )
{
    int i;
    Test_time_result_type res_times;
    px_my_time_type sum = 0;
    for(i=0; i<num_repeats; i++)
    {
        sum+=all_times[i];
    }
	
    res_times.average=(sum/(double)num_repeats); /* среднее арифметическое */

    px_my_time_type st_deviation = 0;
    for(i=0; i<num_repeats; i++)
    {
        st_deviation+=(all_times[i]-res_times.average)*(all_times[i]-res_times.average);
    }
    st_deviation/=(double)(num_repeats);
    res_times.deviation=sqrt(st_deviation); /* sqrt среднего квадрата отклонения */

    qsort(all_times, num_repeats, sizeof(px_my_time_type), my_time_cmp );
    res_times.median=all_times[num_repeats/2]; /* центр после сортировки */

    res_times.min=all_times[0]; /* минимум = первый после qsort по возрастанию */

    return res_times;
}
