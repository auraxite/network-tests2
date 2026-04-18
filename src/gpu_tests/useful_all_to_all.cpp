/*
 * gpu_all_to_all_timing_variants.cpp — справочный (не сборочный) файл.
 *
 * Назначение: разобрать, как именно можно мерить задержки в режиме all-to-all,
 * почему в этом режиме есть разница «измерять у отправителя / у получателя / по
 * двум часам», и показать три варианта реализации do_one() для одного и того же
 * протокола обмена сообщениями.
 *
 * Файл намеренно обёрнут в #if 0 ... #endif: он НЕ компилируется и НЕ участвует
 * в сборке через build.sh. Это документация уровня кода, чтобы можно было
 * сослаться на конкретный псевдокод в работе/защите и не путаться в общем
 * gpu_all_to_all.cpp.
 *
 * Общая исходная точка для всех трёх вариантов одна и та же:
 *
 *   1) Каждый ранг для каждого peer != self заранее ставит MPI_Irecv в свой
 *      буфер recv_bufs[peer]. После этой фазы у каждого ранга «висит» N-1
 *      ожидающих приёма запросов.
 *   2) Каждый ранг проходит по списку получателей dst и для каждого делает
 *      (если режим host) D2H в send_bufs[dst] и MPI_Send в dst.
 *   3) Для каждого источника src нужно завершить соответствующий MPI_Irecv
 *      и (если режим host) перенести данные H2D в d_recv.
 *
 * Различия между вариантами — ровно в том, КУДА поставлены t0/t1 и КТО
 * (отправитель или получатель) формирует ячейку матрицы (src, dst).
 *
 * Все три варианта приведены упрощённо, без debug_log, без обработки --timer,
 * без работы с raw-файлами и без сборки матрицы на ранге 0 — только то, что
 * относится к измерению одной итерации do_one().
 */

#if 0

#include <cuda_runtime.h>
#include <mpi.h>

#include <chrono>
#include <vector>

#include "gpu_all_to_all.hpp"
#include "gpu_common.hpp"

namespace gpu_benchmark {
namespace timing_variants {

/* =========================================================================
 * Вспомогательные функции и структуры — общие для всех трёх вариантов.
 * В реальном коде это уже есть в gpu_common.{hpp,cpp}; здесь воспроизведено
 * только ради компилируемости фрагментов как самостоятельных функций.
 * ========================================================================= */

static int alltoall_pair_tag(int src_rank, int dst_rank, int nproc) {
	return src_rank * nproc + dst_rank;
}

/* «Контекст итерации»: всё, что и так доступно в do_one() в gpu_all_to_all.cpp.
 * Передаётся в функции вариантов исключительно для читаемости подписей. */
struct IterCtx {
	int rank;
	int nproc;
	int count;             /* args.nbytes как int для MPI_BYTE */
	bool check_host;       /* env=host => идём через pinned-host буферы */
	char *d_send;
	char *d_recv;
	std::vector<char *> send_bufs;
	std::vector<char *> recv_bufs;
};

/* =========================================================================
 * ВАРИАНТ A: sender-side. «Как было до» — самый простой и стабильный.
 *
 * Семантика: для каждой исходящей пары (rank → dst) измеряется ВРЕМЯ ОТДАЧИ
 * сообщения с моей стороны: D2H (если host) + MPI_Send.
 *
 * - t0 ставится у меня (отправителя) непосредственно перед началом работы к
 *   данному dst.
 * - t1 ставится у меня сразу после возврата из MPI_Send.
 * - В сэмпл НЕ входят: дорога по сети до конца, приём у получателя, H2D у
 *   получателя.
 * - Сэмпл лежит в samples[dst]; матрицу (src, dst) собирает сам ранг src.
 * - Side в raw: Sender.
 *
 * Плюсы:
 *  + Самый простой код, всё на одном процессе и одних часах.
 *  + Числа стабильные и монотонные относительно размера сообщения.
 *  + Естественно для пропускной способности «темп моей отдачи»:
 *      bw = bytes / sample.
 *
 * Минусы:
 *  - Это НЕ e2e: не учитывается «когда у получателя данные оказались на GPU».
 *  - Для маленьких сообщений (eager MPI) MPI_Send может вернуться почти
 *    мгновенно, ещё до реального ухода пакета в сеть, — измеряется только
 *    локальное помещение в буфер MPI, а не сеть.
 *  - Для больших сообщений (rendezvous) MPI_Send уже включает почти весь
 *    путь — поэтому семантика метрики «плавает» по размеру.
 * ========================================================================= */

static void do_one_sender_side(const IterCtx &ctx, int iter_idx, bool measure,
								std::vector<std::vector<double>> *samples_us) {
	(void)iter_idx;

	std::vector<MPI_Request> recv_req(ctx.nproc, MPI_REQUEST_NULL);
	for (int src = 0; src < ctx.nproc; ++src) {
		if (src == ctx.rank)
			continue;
		MPI_Irecv(ctx.recv_bufs[src], ctx.count, MPI_BYTE, src,
				  alltoall_pair_tag(src, ctx.rank, ctx.nproc), MPI_COMM_WORLD,
				  &recv_req[src]);
	}

	for (int dst = 0; dst < ctx.nproc; ++dst) {
		if (dst == ctx.rank)
			continue;

		const double t0 = measure ? MPI_Wtime() : 0.0;

		if (ctx.check_host) {
			cudaMemcpy(ctx.send_bufs[dst], ctx.d_send, ctx.count,
					   cudaMemcpyDeviceToHost);
		}
		MPI_Send(ctx.send_bufs[dst], ctx.count, MPI_BYTE, dst,
				 alltoall_pair_tag(ctx.rank, dst, ctx.nproc), MPI_COMM_WORLD);

		if (measure) {
			const double t1 = MPI_Wtime();
			(*samples_us)[dst].push_back((t1 - t0) * 1e6);
		}
	}

	/* Приём дочитывается ОТДЕЛЬНО, ПОСЛЕ всех замеров — он не входит ни в один
	 * сэмпл sender-side. Это и есть ключевое отличие от других вариантов. */
	for (int src = 0; src < ctx.nproc; ++src) {
		if (src == ctx.rank)
			continue;
		MPI_Wait(&recv_req[src], MPI_STATUS_IGNORE);
		if (ctx.check_host) {
			cudaMemcpy(ctx.d_recv, ctx.recv_bufs[src], ctx.count,
					   cudaMemcpyHostToDevice);
		}
	}
}

/* =========================================================================
 * ВАРИАНТ B: receiver-side. Это «честный e2e на одних часах» — то, что
 * рекомендуется заявлять как сквозную задержку при отсутствии общей шкалы
 * времени между процессами.
 *
 * Семантика: для каждой ВХОДЯЩЕЙ пары (src → rank) у получателя измеряется
 * время от «раунд начался» до «данные от этого src оказались у меня на GPU».
 *
 * - t0 ставится у меня (получателя) ОДИН раз на итерацию — сразу после того,
 *   как все MPI_Irecv выставлены и я готов принимать. Это «начало раунда».
 * - Используется MPI_Waitany: порядок прихода сообщений в all-to-all
 *   непредсказуем, фиксировать его через for(src=0..N) — это искусственно
 *   замедлять замер на пирах с медленной сетью.
 * - Для каждого пришедшего src отдельно делается H2D и пишется t1 - t0 в
 *   samples[src] получателя.
 * - Матрицу (src, dst) собирает ранг dst (получатель). Side в raw: Receiver.
 *
 * Плюсы:
 *  + В сэмпл попадает то, что важно для пользователя приёма: ожидание сети,
 *    Recv, H2D — «когда у меня данные на GPU».
 *  + Все метки времени берутся на ОДНОМ процессе → числа достоверные, нет
 *    эффекта рассинхронизации часов.
 *  + Корректно работает с одним d_recv в host-режиме: сначала Wait под этот
 *    src гарантированно завершился, потом сразу H2D в d_recv.
 *
 * Минусы:
 *  - В сэмпл НЕ входит D2H и Send у отправителя — этих событий получатель
 *    просто не видит. Если хочется именно «от GPU до GPU», без общих часов
 *    добавить эти куски нечем.
 *  - Зависит от общего темпа раунда: когда сеть/очереди заняты, latency ребра
 *    растёт. Это ЧЕСТНОЕ свойство all-to-all-нагрузки, но единичную «голую»
 *    задержку одного ребра отсюда выудить нельзя.
 *  - Верхний слой меняется: ack-сообщения и raw-файлы формирует получатель,
 *    а не отправитель. В нашей кодовой базе это локальная правка в run_all_to_all.
 * ========================================================================= */

static void do_one_receiver_side(const IterCtx &ctx, int iter_idx, bool measure,
								  std::vector<std::vector<double>> *samples_us) {
	(void)iter_idx;

	std::vector<MPI_Request> recv_req(ctx.nproc, MPI_REQUEST_NULL);
	for (int src = 0; src < ctx.nproc; ++src) {
		if (src == ctx.rank)
			continue;
		MPI_Irecv(ctx.recv_bufs[src], ctx.count, MPI_BYTE, src,
				  alltoall_pair_tag(src, ctx.rank, ctx.nproc), MPI_COMM_WORLD,
				  &recv_req[src]);
	}

	/* Один общий t0 на всю итерацию — старт «раунда». */
	const double t0 = measure ? MPI_Wtime() : 0.0;

	/* Send-фаза. Сюда таймер уже не смотрит: отправки нужны просто чтобы у
	 * других ранков завершились их Irecv. Сами отправки в e2e получателя не
	 * входят и не интересны. */
	for (int dst = 0; dst < ctx.nproc; ++dst) {
		if (dst == ctx.rank)
			continue;
		if (ctx.check_host) {
			cudaMemcpy(ctx.send_bufs[dst], ctx.d_send, ctx.count,
					   cudaMemcpyDeviceToHost);
		}
		MPI_Send(ctx.send_bufs[dst], ctx.count, MPI_BYTE, dst,
				 alltoall_pair_tag(ctx.rank, dst, ctx.nproc), MPI_COMM_WORLD);
	}

	/* Recv-фаза. Конец интервала — индивидуальный для каждого src. */
	for (int k = 0; k < ctx.nproc - 1; ++k) {
		int idx = MPI_UNDEFINED;
		MPI_Waitany(ctx.nproc, recv_req.data(), &idx, MPI_STATUS_IGNORE);
		const int src = idx;
		if (ctx.check_host) {
			cudaMemcpy(ctx.d_recv, ctx.recv_bufs[src], ctx.count,
					   cudaMemcpyHostToDevice);
		}
		if (measure) {
			const double t1 = MPI_Wtime();
			(*samples_us)[src].push_back((t1 - t0) * 1e6);
		}
	}
}

/* =========================================================================
 * ВАРИАНТ C: «псевдо-e2e» по двум часам — как сделано в network_test
 *            (all_to_all_cuda_normal.c у Бегаева).
 *
 * Семантика: t0 ставит ОТПРАВИТЕЛЬ перед D2H, t1 ставит ПОЛУЧАТЕЛЬ после H2D,
 * результат хранится у получателя в ячейке (src, dst). Если дать оси времени
 * имена T_src и T_dst, то sample = T_dst(post H2D) − T_src(pre D2H).
 *
 * Плюсы:
 *  + Самая «правильная» по интерпретации семантика: один путь сообщения
 *    «GPU отправителя → GPU получателя».
 *  + Естественно ложится в матрицу (src, dst).
 *
 * Минусы (фундаментальные на нашем кластере):
 *  - Часы T_src и T_dst — это часы РАЗНЫХ процессов на РАЗНЫХ узлах. Между
 *    ними есть смещение offset_dst − offset_src, не нулевое и не известное.
 *    Реальная величина измерения:
 *        sample = (истинная e2e) + (offset_dst − offset_src).
 *    Смещение может превышать саму e2e и быть отрицательным, тогда выходят
 *    отрицательные / абсурдно маленькие задержки (это и наблюдается в выводе
 *    network_test).
 *  - Для корректности нужна либо PTP/NTP-синхронизация наносекундной точности
 *    (на узлах кластера её, как правило, нет), либо ping-pong-калибровка
 *    каждой пары перед замером, либо вычисление clock-skew по статистике —
 *    в обоих случаях это самостоятельная инфраструктура, влияющая на
 *    сам замер.
 *  - Технически нужно ПЕРЕДАВАТЬ t0 вместе с сообщением (отдельным MPI_Send
 *    или внутри пейлоада), чтобы получатель смог посчитать разность —
 *    лишний трафик и риск сбить сам замер на больших размерах.
 *
 * Ниже — упрощённый каркас. Реализация показывает, ЧТО именно нужно
 * передать, и почему практически это делать не стоит без синхронизации
 * часов.
 * ========================================================================= */

static void do_one_pseudo_e2e(const IterCtx &ctx, int iter_idx, bool measure,
							   std::vector<std::vector<double>> *samples_us) {
	(void)iter_idx;

	std::vector<MPI_Request> recv_req(ctx.nproc, MPI_REQUEST_NULL);
	std::vector<double> recv_t0(ctx.nproc, 0.0); /* сюда придут t0 от src */

	/* Принимаем сразу два сообщения от каждого src: данные и метку времени t0.
	 * В реальном коде t0 удобнее упаковывать в начало payload, чтобы избежать
	 * двух отдельных рукопожатий — но для иллюстрации показано раздельно. */
	for (int src = 0; src < ctx.nproc; ++src) {
		if (src == ctx.rank)
			continue;
		MPI_Irecv(ctx.recv_bufs[src], ctx.count, MPI_BYTE, src,
				  alltoall_pair_tag(src, ctx.rank, ctx.nproc), MPI_COMM_WORLD,
				  &recv_req[src]);
		MPI_Request t0_req = MPI_REQUEST_NULL;
		MPI_Irecv(&recv_t0[src], 1, MPI_DOUBLE, src,
				  alltoall_pair_tag(src, ctx.rank, ctx.nproc) + 100000,
				  MPI_COMM_WORLD, &t0_req);
		MPI_Request_free(&t0_req); /* условно: в реальной реализации Wait отдельно */
	}

	for (int dst = 0; dst < ctx.nproc; ++dst) {
		if (dst == ctx.rank)
			continue;

		/* Сторона ОТПРАВИТЕЛЯ берёт собственное t0 непосредственно перед D2H. */
		const double t0_src = MPI_Wtime();
		if (ctx.check_host) {
			cudaMemcpy(ctx.send_bufs[dst], ctx.d_send, ctx.count,
					   cudaMemcpyDeviceToHost);
		}
		/* Шлём данные и отдельным сообщением — собственное t0 для получателя. */
		MPI_Send(ctx.send_bufs[dst], ctx.count, MPI_BYTE, dst,
				 alltoall_pair_tag(ctx.rank, dst, ctx.nproc), MPI_COMM_WORLD);
		MPI_Send(&t0_src, 1, MPI_DOUBLE, dst,
				 alltoall_pair_tag(ctx.rank, dst, ctx.nproc) + 100000,
				 MPI_COMM_WORLD);
	}

	/* На стороне получателя: дождались данных, сделали H2D, вычислили
	 * sample = t1_dst − t0_src. Здесь и кроется проблема: t1_dst и t0_src —
	 * с разных часов, поэтому реальная величина смещена на offset_dst − offset_src. */
	for (int k = 0; k < ctx.nproc - 1; ++k) {
		int idx = MPI_UNDEFINED;
		MPI_Waitany(ctx.nproc, recv_req.data(), &idx, MPI_STATUS_IGNORE);
		const int src = idx;
		if (ctx.check_host) {
			cudaMemcpy(ctx.d_recv, ctx.recv_bufs[src], ctx.count,
					   cudaMemcpyHostToDevice);
		}
		if (measure) {
			const double t1_dst = MPI_Wtime();
			const double sample_us = (t1_dst - recv_t0[src]) * 1e6;
			/* ВНИМАНИЕ: это не «реальные» микросекунды; это смещённая величина.
			 * Без синхронизации часов её нельзя интерпретировать как e2e. */
			(*samples_us)[src].push_back(sample_us);
		}
	}
}

/* =========================================================================
 * Сводная таблица:
 *
 *   Вариант            | t0  | t1  | На каких часах | Что в сэмпле          | Куда матрица | Реализация
 *   -------------------+-----+-----+----------------+-----------------------+--------------+-----------
 *   A. sender-side     | src | src | одни (src)     | D2H + Send            | у src        | проще всего
 *   B. receiver-side   | dst | dst | одни (dst)     | wait + H2D            | у dst        | средне
 *   C. псевдо-e2e      | src | dst | разные         | весь путь (со смещ-м) | у dst        | сложнее,
 *                      |     |     |                |                       |              | результат
 *                      |     |     |                |                       |              | смещён
 *
 * Рекомендация для этой работы: вариант B (receiver-side). Это «честная»
 * сквозная задержка, которую можно получить БЕЗ синхронизации часов между
 * процессами, и она наиболее близка к смысловому ожиданию «когда данные у
 * меня на GPU». Вариант A удобен как референс пропускной способности.
 * Вариант C показан в работе как пример того, как делает network_test, и
 * почему его числа нельзя интерпретировать буквально на нашем стенде.
 * ========================================================================= */

} /* namespace timing_variants */
} /* namespace gpu_benchmark */

#endif /* справочный файл, не сборочный */
