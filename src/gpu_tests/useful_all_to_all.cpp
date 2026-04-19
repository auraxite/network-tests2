/*
 * gpu_all_to_all_timing_variants.cpp — справочный (не сборочный) файл.
 *
 * Назначение: разобрать, как именно можно мерить задержки в режиме all-to-all,
 * почему в этом режиме есть разница «измерять у отправителя / у получателя / по
 * двум часам», и показать пять вариантов реализации do_one() для одного и того
 * же протокола обмена сообщениями:
 *   A. sender-side                       — t0,t1 у отправителя
 *   B. receiver-side                     — t0,t1 у получателя (рекомендуемый e2e)
 *   C. псевдо-e2e (двое часов)           — t0 у src, t1 у dst (как network_test)
 *   D. send + ack от получателя          — t0,t1 у отправителя, в сэмпл входит обратный ack
 *   E. ping-pong (RTT/2)                 — t0,t1 у отправителя, в сэмпл — половина RTT
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
#include <cstring> /* std::memcpy для упаковки t0_src в начало payload в варианте C */
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
 *  - Нужно ПЕРЕДАВАТЬ t0 вместе с сообщением, чтобы получатель смог
 *    посчитать разность. В наивной реализации это делают отдельным
 *    MPI_Send — получается лишний обмен на каждую пару, удваивается
 *    число сообщений и добавляется «дрожание» на очередях MPI.
 *
 * Оптимизация относительно наивной версии (один MPI-вызов на пару вместо
 * двух): t0_src упаковывается в ПЕРВЫЕ sizeof(double) байт payload'а
 * (служебный «заголовок»), дальше идут обычные данные. Тогда на пару
 * нужен один MPI_Send/MPI_Irecv на count + sizeof(double) байт — никаких
 * отдельных рукопожатий под метку времени.
 *
 * Контракт буферов в этом варианте: send_bufs[peer] и recv_bufs[peer]
 * выделены размером count + sizeof(double), чтобы уместить заголовок.
 *
 * Плата за объединение:
 *  - Сам замер чуть завышается на время передачи 8 лишних байт (на типовых
 *    размерах сообщений — пренебрежимо).
 *  - Внутри узла/кластера с одинаковым ABI это безопасно; при реальной
 *    работе через гетерогенные архитектуры понадобился бы MPI_Type_create_struct
 *    с {MPI_DOUBLE, MPI_BYTE} вместо MPI_BYTE-«сырой» упаковки.
 *
 * Ниже — упрощённый каркас. Реализация показывает, ЧТО именно нужно
 * передать, и почему практически это делать не стоит без синхронизации
 * часов.
 * ========================================================================= */

static void do_one_pseudo_e2e(const IterCtx &ctx, int iter_idx, bool measure,
							   std::vector<std::vector<double>> *samples_us) {
	(void)iter_idx;

	/* Заголовок payload'а под t0_src: первые HDR байт буфера. Контракт с
	 * аллокатором — send_bufs/recv_bufs выделены на count + HDR байт. */
	constexpr int HDR = static_cast<int>(sizeof(double));

	std::vector<MPI_Request> recv_req(ctx.nproc, MPI_REQUEST_NULL);

	/* Одно Irecv на пару: принимаем заголовок (t0_src) и данные одним
	 * сообщением. Никаких отдельных meta-сообщений и, что важно, никаких
	 * «подвешенных» MPI_Request_free, которым пользовалась наивная версия. */
	for (int src = 0; src < ctx.nproc; ++src) {
		if (src == ctx.rank)
			continue;
		MPI_Irecv(ctx.recv_bufs[src], ctx.count + HDR, MPI_BYTE, src,
				  alltoall_pair_tag(src, ctx.rank, ctx.nproc), MPI_COMM_WORLD,
				  &recv_req[src]);
	}

	for (int dst = 0; dst < ctx.nproc; ++dst) {
		if (dst == ctx.rank)
			continue;

		/* Сторона ОТПРАВИТЕЛЯ берёт собственное t0 непосредственно перед D2H:
		 * важно, чтобы в интервал попало именно D2H + уход в сеть. */
		const double t0_src = MPI_Wtime();

		/* Кладём t0 в первые HDR байт send-буфера. Делаем это ДО D2H, чтобы
		 * cudaMemcpy не затёр заголовок: данные кладутся со смещением HDR. */
		std::memcpy(ctx.send_bufs[dst], &t0_src, HDR);

		if (ctx.check_host) {
			cudaMemcpy(ctx.send_bufs[dst] + HDR, ctx.d_send, ctx.count,
					   cudaMemcpyDeviceToHost);
		}
		/* Один MPI_Send на пару: заголовок и данные склеены. */
		MPI_Send(ctx.send_bufs[dst], ctx.count + HDR, MPI_BYTE, dst,
				 alltoall_pair_tag(ctx.rank, dst, ctx.nproc), MPI_COMM_WORLD);
	}

	/* На стороне получателя: дожидаемся единого сообщения, разбираем его на
	 * (t0_src | данные). sample = t1_dst − t0_src. Проблема «двух часов»
	 * никуда не делась: t1_dst и t0_src берутся на РАЗНЫХ процессах, просто
	 * теперь нет накладных расходов на два отдельных MPI-обмена. */
	for (int k = 0; k < ctx.nproc - 1; ++k) {
		int idx = MPI_UNDEFINED;
		MPI_Waitany(ctx.nproc, recv_req.data(), &idx, MPI_STATUS_IGNORE);
		const int src = idx;

		double t0_src = 0.0;
		std::memcpy(&t0_src, ctx.recv_bufs[src], HDR);

		if (ctx.check_host) {
			cudaMemcpy(ctx.d_recv, ctx.recv_bufs[src] + HDR, ctx.count,
					   cudaMemcpyHostToDevice);
		}
		if (measure) {
			const double t1_dst = MPI_Wtime();
			const double sample_us = (t1_dst - t0_src) * 1e6;
			/* ВНИМАНИЕ: это не «реальные» микросекунды; это смещённая величина.
			 * Без синхронизации часов её нельзя интерпретировать как e2e. */
			(*samples_us)[src].push_back(sample_us);
		}
	}
}

/* =========================================================================
 * ВАРИАНТ D: «send + ack от получателя». Аппроксимация e2e через короткий
 * подтверждающий ответ от получателя — оба замера у отправителя, на одних
 * часах.
 *
 * Идея: отправитель шлёт данные dst и сразу выставляет MPI_Recv маленького
 * ack от dst. Получатель, как только данные пришли (и при необходимости H2D
 * выполнен), посылает обратно ack нулевой длины. Сэмпл = «от старта Send до
 * прихода ack» у отправителя.
 *
 * Семантика: sample ≈ (отправка данных) + (сеть туда) + (Recv+H2D у dst) +
 *                     (отправка ack у dst) + (сеть обратно для маленького ack).
 *
 * Плюсы:
 *  + Без межузловой синхронизации часов — обе метки у отправителя.
 *  + В отличие от варианта A, в сэмпл реально входит «доставлено и обработано
 *    у получателя» (а не «локально отдано в MPI»).
 *  + Естественно ложится в матрицу (src, dst): её формирует сам отправитель.
 *
 * Минусы:
 *  - В сэмпл входит лишний путь обратного ack: для маленьких сообщений он может
 *    быть сопоставим с самим измеряемым временем.
 *  - Симметрия трафика «прямого» и «обратного» направления может различаться
 *    (особенно при асимметричной маршрутизации) — сэмпл системно завышен на
 *    время обратного пути.
 *  - В режиме all-to-all требует, чтобы получатель планомерно слал ack по
 *    всем входящим — это удваивает количество MPI-сообщений на итерацию.
 * ========================================================================= */

static void do_one_send_plus_ack(const IterCtx &ctx, int iter_idx, bool measure,
								  std::vector<std::vector<double>> *samples_us) {
	(void)iter_idx;

	/* Тег ack-сообщения должен не пересекаться с тегом данных. Здесь — простое
	 * смещение; в реальном коде брали бы alltoall_ack_tag(src,dst,nproc). */
	auto ack_tag = [&](int src, int dst) {
		return alltoall_pair_tag(src, dst, ctx.nproc) + 200000;
	};

	/* Я как получатель данных: предварительно ставлю Irecv по данным от каждого
	 * src и сразу планирую отправку ack — но ack отправляю только ПОСЛЕ того,
	 * как данные реально пришли (в recv-фазе ниже). */
	std::vector<MPI_Request> recv_data_req(ctx.nproc, MPI_REQUEST_NULL);
	for (int src = 0; src < ctx.nproc; ++src) {
		if (src == ctx.rank)
			continue;
		MPI_Irecv(ctx.recv_bufs[src], ctx.count, MPI_BYTE, src,
				  alltoall_pair_tag(src, ctx.rank, ctx.nproc), MPI_COMM_WORLD,
				  &recv_data_req[src]);
	}

	/* Я как отправитель: для каждого dst заранее ставлю Irecv маленького ack
	 * от dst — чтобы потом «закрыть» его в момент прихода ответа. */
	std::vector<MPI_Request> recv_ack_req(ctx.nproc, MPI_REQUEST_NULL);
	std::vector<char> ack_dummy(ctx.nproc, 0);
	for (int dst = 0; dst < ctx.nproc; ++dst) {
		if (dst == ctx.rank)
			continue;
		MPI_Irecv(&ack_dummy[dst], 0, MPI_BYTE, dst, ack_tag(dst, ctx.rank),
				  MPI_COMM_WORLD, &recv_ack_req[dst]);
	}

	/* Send-фаза: фиксируем t0 у отправителя ДО D2H, шлём данные. Ack ждём НЕ
	 * сразу — даём всем dst шанс параллельно начать обработку. */
	std::vector<double> t0_per_dst(ctx.nproc, 0.0);
	for (int dst = 0; dst < ctx.nproc; ++dst) {
		if (dst == ctx.rank)
			continue;
		if (measure)
			t0_per_dst[dst] = MPI_Wtime();
		if (ctx.check_host) {
			cudaMemcpy(ctx.send_bufs[dst], ctx.d_send, ctx.count,
					   cudaMemcpyDeviceToHost);
		}
		MPI_Send(ctx.send_bufs[dst], ctx.count, MPI_BYTE, dst,
				 alltoall_pair_tag(ctx.rank, dst, ctx.nproc), MPI_COMM_WORLD);
	}

	/* Recv-фаза «как получатель»: для каждого пришедшего src делаем H2D и
	 * сразу шлём ack обратно — это нужно, чтобы соответствующий отправитель
	 * смог закрыть свой замер. */
	for (int k = 0; k < ctx.nproc - 1; ++k) {
		int idx = MPI_UNDEFINED;
		MPI_Waitany(ctx.nproc, recv_data_req.data(), &idx, MPI_STATUS_IGNORE);
		const int src = idx;
		if (ctx.check_host) {
			cudaMemcpy(ctx.d_recv, ctx.recv_bufs[src], ctx.count,
					   cudaMemcpyHostToDevice);
		}
		MPI_Send(nullptr, 0, MPI_BYTE, src, ack_tag(ctx.rank, src),
				 MPI_COMM_WORLD);
	}

	/* Закрываем «как отправитель»: ловим ack от каждого dst и пишем сэмпл
	 * t1 - t0 в samples[dst]. */
	for (int k = 0; k < ctx.nproc - 1; ++k) {
		int idx = MPI_UNDEFINED;
		MPI_Waitany(ctx.nproc, recv_ack_req.data(), &idx, MPI_STATUS_IGNORE);
		const int dst = idx;
		if (measure) {
			const double t1 = MPI_Wtime();
			(*samples_us)[dst].push_back((t1 - t0_per_dst[dst]) * 1e6);
		}
	}
}

/* =========================================================================
 * ВАРИАНТ E: ping-pong (RTT/2). Прямая аппроксимация задержки одного
 * направления как половины «полного оборота» — оба замера у отправителя, на
 * одних часах.
 *
 * Идея: для каждой пары (rank → dst) отправитель шлёт сообщение тем же объёмом
 * данных, получатель echo-шлёт его обратно. Сэмпл = (t1 - t0) / 2.
 *
 * Семантика в all-to-all: каждый ранг по очереди (или параллельно для разных
 * dst) запускает свой ping-pong. Чтобы это оставалось «all-to-all», нужно либо
 * проводить N-1 параллельных ping-pong'ов одновременно (как ниже), либо
 * сделать N независимых раундов «один-к-одному», что уже не all-to-all.
 *
 * Плюсы:
 *  + Без межузловой синхронизации часов.
 *  + Хорошо работает для «голой» задержки маленького сообщения, когда обратный
 *    путь по времени почти равен прямому.
 *  + Стандартная схема в OSU/IMB — легко сравнивать с эталоном.
 *
 * Минусы:
 *  - Завышает результат, когда сеть/обработка асимметричны (D2H+H2D у одной
 *    стороны быстрее, у другой медленнее; обратный путь маршрутизируется иначе
 *    и т.п.) — половина RTT не равна односторонней задержке.
 *  - Удваивает трафик: каждое измеряемое сообщение фактически уходит дважды
 *    (туда и обратно), что меняет нагрузку сети по сравнению с настоящим
 *    one-way all-to-all.
 *  - В режиме all-to-all каждый ранг одновременно и инициирует ping-pong'и для
 *    своих dst, и обслуживает входящие ping'и от других src — код получается
 *    более громоздким, чем в вариантах A/B.
 * ========================================================================= */

static void do_one_ping_pong(const IterCtx &ctx, int iter_idx, bool measure,
							  std::vector<std::vector<double>> *samples_us) {
	(void)iter_idx;

	/* Echo-сообщение шлётся тем же размером, что и исходное, но с другим тегом,
	 * чтобы не путать «ping» и «pong» на стороне инициатора. */
	auto pong_tag = [&](int src, int dst) {
		return alltoall_pair_tag(src, dst, ctx.nproc) + 300000;
	};

	/* Я как «эхо»: для каждого src заранее ставлю Irecv на ping-данные. Когда
	 * ping придёт — сразу отправлю эти же байты обратно как pong. */
	std::vector<MPI_Request> recv_ping_req(ctx.nproc, MPI_REQUEST_NULL);
	for (int src = 0; src < ctx.nproc; ++src) {
		if (src == ctx.rank)
			continue;
		MPI_Irecv(ctx.recv_bufs[src], ctx.count, MPI_BYTE, src,
				  alltoall_pair_tag(src, ctx.rank, ctx.nproc), MPI_COMM_WORLD,
				  &recv_ping_req[src]);
	}

	/* Я как инициатор: для каждого dst заранее ставлю Irecv на возврат pong. */
	std::vector<MPI_Request> recv_pong_req(ctx.nproc, MPI_REQUEST_NULL);
	std::vector<char *> pong_bufs(ctx.nproc, nullptr); /* в реальном коде —
	                                                      отдельные буферы */
	for (int dst = 0; dst < ctx.nproc; ++dst) {
		if (dst == ctx.rank)
			continue;
		/* Для иллюстрации переиспользуем send_bufs[dst] под приём pong:
		 * в реальном коде нужны независимые буферы. */
		pong_bufs[dst] = ctx.send_bufs[dst];
		MPI_Irecv(pong_bufs[dst], ctx.count, MPI_BYTE, dst,
				  pong_tag(dst, ctx.rank), MPI_COMM_WORLD,
				  &recv_pong_req[dst]);
	}

	/* Send-фаза «ping»: t0 у отправителя ДО D2H, отправка ping. Ack/pong ждём
	 * отдельным проходом ниже. */
	std::vector<double> t0_per_dst(ctx.nproc, 0.0);
	for (int dst = 0; dst < ctx.nproc; ++dst) {
		if (dst == ctx.rank)
			continue;
		if (measure)
			t0_per_dst[dst] = MPI_Wtime();
		if (ctx.check_host) {
			cudaMemcpy(ctx.send_bufs[dst], ctx.d_send, ctx.count,
					   cudaMemcpyDeviceToHost);
		}
		MPI_Send(ctx.send_bufs[dst], ctx.count, MPI_BYTE, dst,
				 alltoall_pair_tag(ctx.rank, dst, ctx.nproc), MPI_COMM_WORLD);
	}

	/* Echo-фаза: для каждого пришедшего ping от src сразу делаем H2D (если
	 * нужно для семантики «GPU→GPU») и отправляем pong тех же байтов обратно.
	 * H2D у эха входит в RTT — для half-RTT приближения это корректно. */
	for (int k = 0; k < ctx.nproc - 1; ++k) {
		int idx = MPI_UNDEFINED;
		MPI_Waitany(ctx.nproc, recv_ping_req.data(), &idx, MPI_STATUS_IGNORE);
		const int src = idx;
		if (ctx.check_host) {
			cudaMemcpy(ctx.d_recv, ctx.recv_bufs[src], ctx.count,
					   cudaMemcpyHostToDevice);
			/* И обратно D2H в send-буфер для отправки pong. */
			cudaMemcpy(ctx.recv_bufs[src], ctx.d_recv, ctx.count,
					   cudaMemcpyDeviceToHost);
		}
		MPI_Send(ctx.recv_bufs[src], ctx.count, MPI_BYTE, src,
				 pong_tag(ctx.rank, src), MPI_COMM_WORLD);
	}

	/* Закрываем замеры «как инициатор»: дожидаемся pong от каждого dst,
	 * сэмпл = (t1 - t0) / 2 в samples[dst]. */
	for (int k = 0; k < ctx.nproc - 1; ++k) {
		int idx = MPI_UNDEFINED;
		MPI_Waitany(ctx.nproc, recv_pong_req.data(), &idx, MPI_STATUS_IGNORE);
		const int dst = idx;
		if (ctx.check_host) {
			cudaMemcpy(ctx.d_recv, pong_bufs[dst], ctx.count,
					   cudaMemcpyHostToDevice);
		}
		if (measure) {
			const double t1 = MPI_Wtime();
			const double half_rtt_us = (t1 - t0_per_dst[dst]) * 1e6 * 0.5;
			(*samples_us)[dst].push_back(half_rtt_us);
		}
	}
}

/* =========================================================================
 * Сводная таблица:
 *
 *   Вариант            | t0  | t1  | На каких часах | Что в сэмпле               | Куда матрица | Реализация
 *   -------------------+-----+-----+----------------+----------------------------+--------------+-----------
 *   A. sender-side     | src | src | одни (src)     | D2H + Send                 | у src        | проще всего
 *   B. receiver-side   | dst | dst | одни (dst)     | wait + H2D                 | у dst        | средне
 *   C. псевдо-e2e      | src | dst | разные         | весь путь (со смещ-м)      | у dst        | сложнее,
 *                      |     |     |                |                            |              | результат
 *                      |     |     |                |                            |              | смещён
 *   D. send + ack      | src | src | одни (src)     | путь туда + обработка у dst| у src        | средне,
 *                      |     |     |                | + короткий ack обратно     |              | завышено
 *                      |     |     |                |                            |              | на ack
 *   E. ping-pong RTT/2 | src | src | одни (src)     | (RTT туда+обратно)/2       | у src        | громоздко,
 *                      |     |     |                |                            |              | завышено
 *                      |     |     |                |                            |              | при асимм.
 *
 * Рекомендация для этой работы: вариант B (receiver-side). Это «честная»
 * сквозная задержка, которую можно получить БЕЗ синхронизации часов между
 * процессами, и она наиболее близка к смысловому ожиданию «когда данные у
 * меня на GPU». Вариант A удобен как референс пропускной способности.
 * Вариант C показан в работе как пример того, как делает network_test, и
 * почему его числа нельзя интерпретировать буквально на нашем стенде.
 * Варианты D и E — типовые приёмы из мира MPI-бенчмарков (OSU/IMB), оба
 * измеряют у отправителя и не требуют общих часов; D ближе по семантике к
 * «доставлено и обработано», E — к «голой» задержке одного направления.
 * ========================================================================= */

} /* namespace timing_variants */
} /* namespace gpu_benchmark */

#endif /* справочный файл, не сборочный */
