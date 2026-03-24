#include "my_time.h"
#include "my_malloc.h"
#include "tests_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <pthread.h>
#include <cuda_runtime.h>

extern int comm_rank; // network_test2.c
extern int comm_size; // network_test2.c
extern int* gpu_count; // network_test2.c
extern int total_gpu; // network_test2.c

/*
 * all_to_all_cuda_normal.c — одновременная «все-ко-всем» передача между всеми GPU кластера.
 *
 * Идея: у каждого MPI-процесса gpu_count[rank] локальных GPU. Глобально нумеруются GPU от 0 до total_gpu-1.
 * Для каждой пары (отправитель GPU j на этом узле → получатель GPU k в кластере) есть буферы send/recv
 * и при межузловой передаче цепочка: D2H (async) → MPI_Isend/Irecv → по приёму H2D (async).
 *
 * Локальные пары (k на том же MPI-ранге): cudaMemcpyPeerAsync + CUDA events — время без MPI.
 * Удалённые: время на приёмнике считается как (CPU time после завершения H2D) минус (CPU time в начале D2H
 * на том же повторе), см. tmp_results[...][i].
 *
 * Теги MPI кодируют глобальные индексы GPU в старших байтах (см. tag_r/tag_s), чтобы на Waitany понять,
 * какой слот результата обновлять.
 */

int all_to_all_cuda( Test_time_result_type * times, int mes_length, int num_repeats );

extern Test_time_result_type calc_stats( px_my_time_type* all_times, int num_repeats );

int all_to_all_cuda( Test_time_result_type * times, int mes_length, int num_repeats )
{
    /* Каждый ранг выполняет полный набор операций для своих GPU; матрица times заполняется локально,
     * затем (если нужно глобально) её обычно собирают снаружи через MPI — здесь только локальная часть. */

    px_my_time_type **tmp_results=NULL;
    px_my_time_type time_beg, time_end;
    
    MPI_Status status;
    int finished;
    px_my_time_type st_deviation;
    int i,j;
    int l_bound = 0, r_bound = 0;
    int flag=0;
    /* tag_s / tag_r: упаковка глобальных номеров GPU в MPI-тег (биты 24–31 и 16–23). */
    int tag_s = 0, tag_r = 0;
    int recv = 0; /* не везде используется — остаток от черновика */
    double sum;
    cudaError_t cuda_error;
    int *gpu_mpi_host_rank=NULL;
    int *gpu_global_rank=NULL;

    MPI_Request *send_request=NULL;
    MPI_Request *recv_request=NULL;

    char **send_data;
    char **recv_data;

    char **send_data_host = NULL;
    char **recv_data_host = NULL;

    cudaStream_t *send_streams;
    cudaStream_t *recv_streams;

    cudaEvent_t *start_events;
    cudaEvent_t *stop_events;

    /* gpu_mpi_host_rank[g] = MPI-ранг узла, на котором живёт глобальный GPU с индексом g. */
    gpu_mpi_host_rank = ( int* )malloc( sizeof( int ) * total_gpu );
    /* gpu_global_rank[i] — глобальный индекс i-го локального GPU на ЭТОМ ранге (в пределах [l_bound..r_bound]). */
    gpu_global_rank = ( int* )malloc( sizeof( int ) * gpu_count[comm_rank] );
    //time_beg =  ( px_my_time_type* )malloc( sizeof( int ) ) * gpu_count[comm_rank] );
    //time_end =  ( px_my_time_type* )malloc( sizeof( int ) ) * gpu_count[comm_rank] );
   
    /* Заполняем карту: глобальные GPU 0..total_gpu-1 идут блоками по узлам в порядке рангов MPI. */
    int k = 0;
    int stride = 0;
    for ( i = 0; i < comm_size; i++ ) 
    {
	stride += gpu_count[i];
        while ( k < stride )
        {
            gpu_mpi_host_rank[k] = i;
            k++;
        printf("K:%d, i:%d ", k, i);
        }
	
        printf("\n");
    }

    for ( i = 0; i < total_gpu; i++ )
        printf( "GPU-MPI:%d ", gpu_mpi_host_rank[i] );

    printf("\n");

    /* l_bound / r_bound — диапазон глобальных индексов GPU, принадлежащих текущему MPI-процессу. */
    for ( i = 0 ; i < comm_rank; i++)
    {
        l_bound += gpu_count[i];
        r_bound += gpu_count[i];
    }


    r_bound += gpu_count[comm_rank] - 1;
    for ( i = l_bound; i <= r_bound; i++ )
    {
        gpu_global_rank[i - l_bound] = i;
    }
	printf("r_bound:%d, l_bound:%d\n", r_bound, l_bound);
    printf("rank:%d ", comm_rank);
    for ( i = 0; i < gpu_count[comm_rank]; i++ )
	printf("SS:%d, ", gpu_global_rank[i]);
    printf("\n");

    /* Число «чужих» GPU с точки зрения этого узла × число локальных GPU ≈ столько пар требуют MPI. */
    int another_gpus = ( total_gpu - gpu_count[comm_rank] )* gpu_count[comm_rank] ;
    /* Индексация потоков/событий: локальный GPU j × total_gpu + глобальный индекс k. */
    send_streams = ( cudaStream_t* )malloc( sizeof( cudaStream_t ) * total_gpu * gpu_count[comm_rank] );
    recv_streams = ( cudaStream_t* )malloc( sizeof( cudaStream_t ) * total_gpu * gpu_count[comm_rank] );
    
    start_events = ( cudaEvent_t* )malloc( sizeof( cudaEvent_t ) * total_gpu * gpu_count[comm_rank] );
    stop_events = ( cudaEvent_t* )malloc( sizeof( cudaEvent_t ) * total_gpu * gpu_count[comm_rank] );

    /* Пары Isend/Irecv нумеруются отдельным смещением (см. offset ниже), массив длины another_gpus. */
    send_request = ( MPI_Request* )malloc( sizeof( MPI_Request ) *  another_gpus );
    recv_request = ( MPI_Request* )malloc( sizeof( MPI_Request ) *  another_gpus );
    cudaEvent_t start, stop;
    cudaStream_t src_dst_stream; /* создано, но в основном цикле не используется — задел/старый вариант */

    cudaEventCreate( &start );
    cudaEventCreate( &stop ); 


    /* send_data/recv_data: указатели на device-память; индекс j * total_gpu + k — «с j-го локального GPU к k-му глобальному». */
    send_data = ( char** ) malloc ( sizeof( char* ) * total_gpu * gpu_count[comm_rank] );
    recv_data = ( char** ) malloc ( sizeof( char* ) * total_gpu * gpu_count[comm_rank]  );
    /* Хостовые буферы под каждую пару (глобальный×глобальный) для стадии MPI. */
    send_data_host = ( char** ) malloc ( sizeof( char* ) * total_gpu * total_gpu );
    recv_data_host = ( char** ) malloc ( sizeof( char* ) * total_gpu * total_gpu );

    for ( i = 0; i < total_gpu * total_gpu; i++ )
    {
        send_data_host[i] = ( char* )malloc( sizeof ( char ) * mes_length );
        recv_data_host[i] = ( char* )malloc( sizeof ( char ) * mes_length );
    }

    /* tmp_results[локальная_строка * total_gpu + столбец][повтор] — время для пары (локальный GPU → глобальный k). */
    tmp_results = ( px_my_time_type** )malloc( sizeof( px_my_time_type* ) * total_gpu * gpu_count[comm_rank] );
    for ( i = 0; i < total_gpu * gpu_count[comm_rank]; i++ )
        tmp_results[i] = ( px_my_time_type* )malloc( sizeof( px_my_time_type ) * num_repeats );

    for ( i = 0; i < gpu_count[comm_rank] * total_gpu; i++ )
	for ( j = 0; j < num_repeats; j++ )
		tmp_results[i][j] = 0.0;
	printf("%d rank %d\n", gpu_count[comm_rank], comm_rank);
    /* Подготовка: на каждом локальном GPU выделяем буфер на каждый «столбец» k и создаём stream/event. */
    for ( i = 0; i < gpu_count[comm_rank]; i++ ) 
    {
        cudaSetDevice( i );
//	cuda_error = cudaMalloc ( ( void** ) &send_data[i], total_gpu * mes_length );
//	if ( cuda_error)
//		printf("mallocsend error\n");
//	cuda_error = cudaMalloc ( ( void** ) &recv_data[i], total_gpu * mes_length );
//	if ( cuda_error)
//		printf("mallocrecv error\n");
        for ( j = 0; j < total_gpu; j++ ) 
        {
	   printf("%d   %d\n", i, j);
	   cuda_error = cudaMalloc ( ( void** ) &send_data[i * total_gpu + j], mes_length );
	if (cuda_error)
		printf( "cudamalloc error%s\n", cudaGetErrorString( cuda_error ) );
	cuda_error = cudaMalloc ( ( void** ) &recv_data[i * total_gpu + j], mes_length );
	if (cuda_error)
		printf( "cudamalloc error%s\n", cudaGetErrorString( cuda_error ) );
            cuda_error = cudaStreamCreate ( &send_streams[i * total_gpu + j] );
            if (cuda_error)
                printf("send stream malloc error\n");
            cuda_error = cudaStreamCreate ( &recv_streams[i * total_gpu + j] );
            if (cuda_error)
                printf("recv stream malloc error\n");

            cuda_error = cudaEventCreate ( &start_events[i * total_gpu + j] );
            if (cuda_error)
                printf("start event malloc error\n");
           cuda_error = cudaEventCreate ( &stop_events[i * total_gpu + j] );
            if (cuda_error)
                printf("stop event malloc error\n");
            //cudaMalloc( ( void** ) &data, mes_length );
        }

        /* P2P между разными GPU на одной машине (индексы j — локальные device id). */
        for ( j = 0; j < gpu_count[comm_rank]; j++ )
        {
            if ( i == j )
                continue;
            cudaDeviceEnablePeerAccess ( j, 0 );
        }
    } /* конец подготовки устройств */

printf("BBBBBBBBBBBBBBBb%d\n", comm_rank);
   
    /* === Основной цикл измерений: i — номер повтора === */
    for ( i = 0; i < num_repeats; i++ )
    {
	printf( "it:%d\n", i);
        /* Фаза 1: для каждого локального GPU j и каждого глобального k — либо peer-копия, либо выгрузка на хост. */
        for ( j = 0; j < gpu_count[comm_rank]; j++ )
        {
            cuda_error = cudaSetDevice( j );
            for ( k = 0; k < total_gpu; k++ )
            {
                if ( comm_rank == gpu_mpi_host_rank[k] ) 
                {
                    /* GPU k физически на этом же узле: не MPI, а прямое копирование в recv-буфер «целевого» локального GPU. */
                    if ( k - l_bound == j )
                        continue;

                    printf("Processing transmission on single host\n");
                    fflush(stdout);
                    cudaEventRecord( start_events[j * total_gpu + k], send_streams[j * total_gpu + k] );
                    /* Источник: device j, приёмник: device (k - l_bound) на том же хосте; recv_data индексируется по паре. */
                    cuda_error = cudaMemcpyPeerAsync( recv_data[k - l_bound + gpu_global_rank[j]], k - l_bound, send_data[j * total_gpu + k], j, mes_length, send_streams[j * total_gpu + k] );
                    if (cuda_error)
                        printf("device peer memcpyasync error%s\n", cudaGetErrorString( cuda_error ) );
                    cudaEventRecord( stop_events[j * total_gpu + k], send_streams[j * total_gpu + k] );
                    continue;
                }
                else
                {
                    /* Удалённый GPU k: начало замера — отметка CPU времени до асинхронного D2H. */
                    printf("Processing transmission to CPU\n");
                    tmp_results[j * total_gpu + k][i] = px_my_cpu_time();
                    cuda_error = cudaMemcpyAsync( send_data_host[j * total_gpu + k], send_data[j * total_gpu + k], mes_length, cudaMemcpyDeviceToHost, send_streams[j * total_gpu + k] );
                    if (cuda_error)
                        printf("host memcpyasync error%s\n", cudaGetErrorString(cuda_error) );
                   //cudaDeviceSynchronize();
                   //MPI_Isend(send_data[j], gpu_mpi_rank[j], mes_length, MPI_BYTE, );
                   //MPI_Irecv();

                }
            }

        }
        /* Фаза 2: синхронизация потоков; для локальных k — запись времени по CUDA events; для удалённых — Isend/Irecv. */
        for ( j = 0; j < gpu_count[comm_rank]; j++ ) 
        {
            cudaSetDevice( j );
            for ( k = 0; k < total_gpu; k++ )
            {
		cuda_error = cudaStreamSynchronize( send_streams[j * total_gpu + k] );
		if ( cuda_error )
			printf("StreamSynchError%s\n", cudaGetErrorString( cuda_error ) );
                if ( comm_rank == gpu_mpi_host_rank[k] ) {
                    if ( gpu_global_rank[j] == k ) 
                    {
                        /* Сам себе — нет передачи. */
                        tmp_results[j * total_gpu + k][i] = 0.0;
                    }
                    else
                    {
                        /* Время peer-копии относится к получателю в глобальной матрице: строка (k-l_bound), столбец gpu_global_rank[j]. */
                        float tmp_time = 0.0;
                        cudaEventElapsedTime(&tmp_time, start_events[j * total_gpu + k], stop_events[j * total_gpu + k] ); 
                        tmp_results[(k - l_bound) * total_gpu + gpu_global_rank[j]][i] = (double)tmp_time * 0.0001;
                        printf("Local copy finished via %lf time\n", tmp_time * 0.0001);
                    }
                    continue;
                }
                /* Приёмник декодирует tag: кто получатель (старший байт) и кто отправитель (следующий). */
                tag_r = ( gpu_global_rank[j] << 24 ) | ( k << 16 );
                tag_s = ( k << 24 ) | ( gpu_global_rank[j] << 16 );
                printf( "%d:%d to %d:%d\n", comm_rank, gpu_global_rank[j], gpu_mpi_host_rank[k], k );
                printf( "1.Processing transmission to another host %d\n", gpu_mpi_host_rank[k] );
                //MPI_Isend( send_data_host[j * total_gpu + k], mes_length, MPI_BYTE, gpu_mpi_rank[k], 0, MPI_COMM_WORLD, &send_request[j * total_gpu + k] );
                //MPI_Irecv( recv_data_host[j * total_gpu + k], mes_length, MPI_BYTE, gpu_mpi_rank[k], 0, MPI_COMM_WORLD, &recv_request[j * total_gpu + k] );
                printf("lll:%d\n", gpu_global_rank[j] * total_gpu + k);
 //               MPI_Isend( send_data_host[j * total_gpu + k], mes_length, MPI_BYTE, gpu_mpi_host_rank[k], 0, MPI_COMM_WORLD, &send_request[gpu_mpi_host_rank[j]] );
  //              MPI_Irecv( recv_data_host[j * total_gpu + k], mes_length, MPI_BYTE, gpu_mpi_host_rank[k], 0, MPI_COMM_WORLD, &recv_request[gpu_mpi_host_rank[k]] );
                /* Смещение индекса в массивах request: унификация нумерации между рангами (избежать дубля пар). */
                int offset = 0;
                if (comm_rank < gpu_mpi_host_rank[k])
                offset = gpu_count[comm_rank]; 
                MPI_Isend( send_data_host[j * total_gpu + k], mes_length, MPI_BYTE, gpu_mpi_host_rank[k], tag_s, MPI_COMM_WORLD, &send_request[j * (total_gpu - gpu_count[comm_rank] ) + k - offset ] );
                MPI_Irecv( recv_data_host[j * total_gpu + k], mes_length, MPI_BYTE, gpu_mpi_host_rank[k], tag_r, MPI_COMM_WORLD, &recv_request[j * (total_gpu - gpu_count[comm_rank] ) + k - offset ] );
                printf("sss:%d\n", k * total_gpu + gpu_global_rank[j] );
                printf("2.Processing transmission to another host %d\n", gpu_mpi_host_rank[k]);
            }
        }
        
        //for ( j = 0; j < gpu_count[comm_rank]; j++ ) 
          //  for ( k = 0; k < total_gpu; k++ )
        /* Фаза 3: ждём завершения всех приёмов MPI для этого повтора; finished — индекс завершившегося request. */
        k = 0;
        while ( k < ( total_gpu - gpu_count[comm_rank] ) * gpu_count[comm_rank])
        {
            k++;
            MPI_Waitany( another_gpus, recv_request, &finished, &status );
            printf("Something was recieved from host: %d\n", finished);
            printf("KK::%d\n", k);
            /* Восстанавливаем глобальные индексы GPU из тега сообщения. */
            int gpu_recv = ( ( status.MPI_TAG >> 24 ) & 0x000000FF );
            int gpu_send = ( ( status.MPI_TAG >> 16 ) & 0x000000FF );
            int local_gpu_rank =  gpu_recv - l_bound;
            cuda_error = cudaSetDevice( local_gpu_rank );
            if ( cuda_error )
		    printf("ERR%s\n", cudaGetErrorString( cuda_error ) );
	    cuda_error = cudaMemcpyAsync( recv_data[local_gpu_rank * total_gpu + gpu_send], recv_data_host[local_gpu_rank * total_gpu + gpu_send], mes_length, cudaMemcpyHostToDevice, recv_streams[local_gpu_rank * total_gpu + gpu_send] );
            if ( cuda_error )
            printf("memcpyasync dst%s\n", cudaGetErrorString( cuda_error ) );
                    cuda_error = cudaStreamSynchronize( recv_streams[local_gpu_rank * total_gpu + gpu_send] );
            if ( cuda_error )
            printf("strsynch%s\n", cudaGetErrorString( cuda_error ));
            /* tmp_results[..][i] на этапе D2H хранило time_beg; здесь получаем длительность «от начала D2H до конца H2D на приёме». */
            double time_end = px_my_cpu_time();
                    tmp_results[local_gpu_rank * total_gpu + gpu_send][i] = time_end - tmp_results[local_gpu_rank * total_gpu + gpu_send][i]; 
            printf("local_gpu_rank:%d\n", local_gpu_rank);
                    printf("GPURECV:%d , GPUSEND:%d, Finished:%lf\n", gpu_recv, gpu_send, tmp_results[( gpu_recv - l_bound ) * total_gpu + gpu_send][i]);

        }
    
    }
    /* Агрегация: по каждой паре (локальный i → глобальный j) — среднее, медиана, разброс по повторам. */
    for ( i = 0; i < gpu_count[comm_rank]; i++ )
	    for ( j = 0; j < total_gpu; j++ ) 
        {
	
	     printf("@@@@TMPRES: 0:%lf, 1:%lf, 2:%lf", tmp_results[i*total_gpu + j][0], tmp_results[i * total_gpu + j][1], tmp_results[i * total_gpu + j][2] );
 	     times[i * total_gpu + j] = calc_stats( tmp_results[i * total_gpu + j], num_repeats); 
	    }

    printf("BBBB%d\n", comm_rank);
    /* Освобождение GPU-памяти и сброс контекста по каждому локальному устройству. */
    for ( i = 0 ; i < gpu_count[comm_rank]; i++ ) 
    {
	cudaSetDevice( i );

	for ( j = 0; j < total_gpu; j++ ) 
	{
		cudaFree( ( void** ) recv_data[i * total_gpu + j] );
		cudaFree( ( void** ) send_data[i * total_gpu + j] );
	}

	cudaDeviceReset();	
    }
    printf("DDDD%d\n", comm_rank);
    free (gpu_mpi_host_rank);
    free (gpu_global_rank);
    printf("FFFF%d\n", comm_rank);
    printf("QQQQ%d\n", comm_rank);
    printf("EEEE%d\n", comm_rank);
    free (send_data);
    free (recv_data);
    printf("GGGG%d\n", comm_rank);
    free (send_streams);
    free (recv_streams);
    printf("HHHH%d\n", comm_rank);
    free (start_events);
    free (stop_events);
    printf("JJJJ%d\n", comm_rank);
    for ( i = 0; i < total_gpu * total_gpu; i++ )
    {
       free( send_data_host[i] );
       free( recv_data_host[i] );
    }
    free( send_data_host );
    free( recv_data_host );
    for ( i = 0; i < total_gpu * gpu_count[comm_rank]; i++ )
        free( tmp_results[i] );
    free( tmp_results );
    printf("CCCC%d\n", comm_rank);
    return 0;
}

