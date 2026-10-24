#!/bin/bash
# Запуск: bash submit.sh — обычное измерение (experiments/default.conf);
#         bash submit.sh task.sbatch <эксперимент> — experiments/<эксперимент>.conf
# Раздел берётся из config.sh, список экспериментов и их данные — experiments/README.md
#SBATCH --job-name=clustbench
#SBATCH --time=02:00:00
#SBATCH --output=clustbench-%j.out
#SBATCH --error=clustbench-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --gres-flags=enforce-binding

# Три флага ниже управляют РАЗНЫМИ вещами, их легко перепутать:
#   --gpu-bind=<тип>             что попадёт в CUDA_VISIBLE_DEVICES задачи
#   --cpu-bind=<тип>             маска ядер задачи
#   --gres-flags=enforce-binding какие ядра вообще выделить под задачу
#
# --cpus-per-task=8 вместо 1: одного ядра на ранг мало. UCX крутит прогресс
# (rc_mlx5/cuda_copy) в том же потоке, плюс CUDA-драйверу нужен CPU на
# обслуживание копирований — на одном ядре прогресс упирается в CPU и
# GPU-путь не выходит на полосу.
#
# --gres-flags=enforce-binding требует, чтобы выданные ранку ядра лежали
# на том же сокете, что и выданный ему GPU. Это единственный флаг, который
# реально связывает CPU с GPU (--gpu-bind этого НЕ делает, он лишь решает,
# какие GPU видны). Нужен он потому, что дистрибуция Slurm по умолчанию
# раздаёт ядра с NUMA 0, а на g8600-* все четыре GPU висят на NUMA 1 —
# без этого флага ранг садится на чужой сокет относительно своего GPU,
# и трафик CPU↔GPU идёт через UPI между сокетами.
# Флаг относится к стадии ВЫДЕЛЕНИЯ ресурсов, а allow-task-sharing на srun —
# к стадии шага, поэтому они не конфликтуют.
#
# ПРОВЕРЕНО на g8600-1 (22.09.2026), 4 ранга по 8 ядер:
#   NUMA 0 = ядра 0-31,64-95;  NUMA 1 = ядра 32-63,96-127
#   все 4 GPU: NUMA 1, CPU-аффинность 32-47,96-111
#   ранг 0 → 32-35,96-99    ранг 2 → 40-43,104-107
#   ранг 1 → 36-39,100-103  ранг 3 → 44-47,108-111
# То есть все ранги попали ровно в GPU-локальный диапазон, без перекрытий.
# Повторить проверку можно так (маски печатает --cpu-bind=verbose):
#   srun -p <партиция> --gres-flags=enforce-binding --nodes=1 --ntasks=4 \
#        --cpus-per-task=8 --gpus-per-node=4 --cpu-bind=verbose,cores \
#        bash -c 'echo "rank=$SLURM_PROCID $(numactl --show | grep ^physcpubind)"'
#
# --cpu-bind НЕ указывается здесь: в sbatch эта опция игнорируется,
# она задаётся ниже на srun (--cpu-bind=verbose,cores). Проверять результат
# надо по строкам "cpu-bind" в логе задания: маски должны попадать в ядра
# того NUMA-узла, где стоят GPU.
#
# --gpu-bind намеренно НЕ ставим. Программа сама выбирает GPU по локальному
# рангу (local_rank → cudaSetDevice), что даёт строгую биекцию ранг↔GPU;
# ей нужно видеть все GPU узла, отсюда --gres-flags=allow-task-sharing на srun.
#
# ОСОБЕННО не ставь --gpu-bind=closest. Он ищет ближайший GPU для каждой
# задачи независимо и не гарантирует уникальности. На узлах этого кластера
# все GPU имеют одинаковую CPU-аффинность (g5500: "0,32" у обоих, g8600:
# "32,96" у всех четырёх), поэтому "ближайший" — ничья, и несколько рангов
# получают один и тот же GPU. Это уже случалось: коммит 4e4c763 (27.05.2026)
# добавил в gpu_benchmark.cpp проверку на дубликаты по PCI-адресам именно
# из-за этого, а 6f59951 (10.08.2026) вернул --gpu-bind=none.
# Если нужен ровно один GPU на ранг — бери детерминированные варианты:
# --gpu-bind=map_gpu:0,1,2,3 или --gpu-bind=per_task:1.

# --- Что запускать: experiments/<имя>.conf (по умолчанию default) ---
# Описание задаёт:
#   PLAN          строки "блок повторов среда режим байт:итераций:прогрев [ключи программы]"
#   COMMON        ключи программы для всех запусков
#   SHUFFLE_SEED  перемешать запуски (пусто — по порядку PLAN)
#   SEED_BASE     зерно порядка пар у запуска = SEED_BASE + номер (для one_to_one_v2)
#   OUT_LAYOUT    flat — results/<узлы>_<среда>_<режим>_b.._w.._i...txt (обычное измерение)
#                 job  — results/hyp_<задание>/ c runs.csv и архивом .tgz (эксперименты)
#   REQUIRE       строка, которая должна быть в программе (иначе: пересобери)
#   LOG_FREQ      1 — фоновая запись частот GPU и CPU каждые 200 мс
#   LOG_TOPOLOGY  1 — lscpu и маски ядер рангов
#   COLLECT_ENV   1 — collect_env.sh: топология, ucx_info -d, адаптеры
#   UCX_PATHS     1 — журнал UCX по рангам и счётчики адаптеров/NVLink до и после запуска
#   PROTO_BLOCKS  блоки, где включаются UCX_PROTO_ENABLE=y UCX_PROTO_INFO=y
EXPERIMENT="${1:-default}"
GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONF="${GPU_TESTS}/experiments/${EXPERIMENT}.conf"
test -f "${CONF}" || { echo "error: нет ${CONF} (список: ls experiments)" >&2; exit 1; }

PLAN=(); COMMON=(); SHUFFLE_SEED=""; SEED_BASE=0; OUT_LAYOUT=job; REQUIRE=""
LOG_FREQ=0; LOG_TOPOLOGY=0; COLLECT_ENV=0; UCX_PATHS=0; PROTO_BLOCKS=()
source "${CONF}"

# Пути к Open MPI, UCX и CUDA — общие, из config.sh в каталоге отправки
source "${GPU_TESTS}/config.sh"

set -eEuo pipefail

BG_PIDS=()
stop_bg() { for p in ${BG_PIDS[@]+"${BG_PIDS[@]}"}; do kill "$p" 2>/dev/null || true; done; }
on_error() {
	local exit_code=$?
	echo "FATAL: шаг завершился с ошибкой (exit=${exit_code}), задание остановлено" >&2
	stop_bg
	if [[ -n "${SLURM_JOB_ID:-}" ]]; then scancel "${SLURM_JOB_ID}" >/dev/null 2>&1 || true; fi
	exit "${exit_code}"
}
trap on_error ERR

export PATH="$OMPI_PATH/bin:$PATH"
_LDPATH="$OMPI_PATH/lib:${CUDA_PATH}/lib64"
if [[ -d "$UCX_PATH/lib" ]]; then
	_LDPATH="$UCX_PATH/lib:$_LDPATH"
fi
export LD_LIBRARY_PATH="$_LDPATH:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export OMPI_MCA_pml=ucx
# В OpenMPI 5 компонент vader переименован в sm; старое имя
# btl_vader_single_copy_mechanism уже не действует. При pml=ucxshared-memory
# путь всё равно обслуживает UCX (cma/sm в UCX_TLS), так что btl тут не нужен.
export OMPI_MCA_btl_sm_single_copy_mechanism=cma

# ВАЖНО: снимаем возможные "залипшие" экспорты из ~/.bashrc или прошлых
# сессий. Программа выставляет свои значения с overwrite=0, то есть любой
# внешний export её перебивает. В прошлом прогоне именно так в auto-режим
# попали UCX_RNDV_SCHEME=put_zcopy и UCX_MEMTYPE_CACHE=n (видно в шапке
# results/*_auto_*.txt, хотя код их не ставит) — и убили GPU-путь.
unset UCX_RNDV_SCHEME
unset UCX_MEMTYPE_CACHE
unset UCX_RNDV_THRESH
unset UCX_TLS
unset UCX_PROTO_ENABLE UCX_PROTO_INFO UCX_LOG_LEVEL

# UCX_TLS, UCX_IB_GPU_DIRECT_RDMA, UCX_RNDV_SCHEME, UCX_MEMTYPE_CACHE
# выставляются программой до MPI_Init на основе --env:
#   auto → cuda_copy+cuda_ipc+IB+GDR (gdr_copy нет: в системе нет gdrcopy)
#   host → только IB+cma
# Чтобы переопределить — раскомментируй нужное:
# export UCX_TLS=rc_mlx5,ud_mlx5,dc_mlx5,cuda_copy,cuda_ipc,cma,sm,self
# export UCX_IB_GPU_DIRECT_RDMA=yes
# export UCX_RNDV_THRESH=0      # форсировать rendezvous на всех размерах.
#                                # НЕ рекомендуется: душит мелкие сообщения,
#                                # из-за этого auto проигрывал host на 1–8 КБ.

# Привязка к конкретному HCA. ВНИМАНИЕ: имя карты зависит от партиции,
# поэтому по умолчанию ничего не фиксируем — UCX выбирает сам.
#
#   intel-a100-pci5 (g5500-*): 4 карты, InfiniBand только mlx5_0
#       (ConnectX-6, 100 Гбит/с). Остальные Ethernet: mlx5_1 = 25 Гбит/с,
#       mlx5_2 = 10 Гбит/с, mlx5_3 = down. Фиксация mlx5_0:1 здесь
#       логична, но на замерах разницы не дала — проверено, полоса между
#       узлами осталась 9.2 ГБ/с.
#
#   intel-a100-sxm4 (g8600-*): 18 карт, и mlx5_0 для GPU НЕ оптимален —
#       он NODE от всех GPU. Ближние (PXB): mlx5_8..11 к GPU0/GPU1 и
#       mlx5_12..15 к GPU2/GPU3. Здесь фиксировать одну карту на все
#       ранги прямо вредно: правильный выбор зависит от того, какой GPU
#       достался рангу.
#
# Если захочешь зафиксировать — сверься с "nvidia-smi topo -m" и ibstat
# на нужной партиции, затем раскомментируй и подставь своё:
# export UCX_NET_DEVICES=mlx5_0:1

BIN="${GPU_TESTS}/gpu"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
JOB_NODES="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-$(hostname -s)}}"
SAFE_NODES="$(echo "${JOB_NODES}" | tr '[],' '_' | tr -s '_' | sed 's/^_//; s/_$//')"
if [[ "${OUT_LAYOUT}" == flat ]]; then
	OUT_DIR="${GPU_TESTS}/results"
else
	OUT_DIR="${GPU_TESTS}/results/hyp_${SLURM_JOB_ID:-local}"
fi
RUNS="${OUT_DIR}/runs.csv"

test -x "$BIN" || { echo "error: нет $BIN (bash build.sh)" >&2; exit 1; }
[[ -z "${REQUIRE}" ]] || grep -qa -- "${REQUIRE}" "$BIN" || {
	echo "error: $BIN собран без ${REQUIRE} — пересобери: bash build.sh" >&2; exit 1; }
mkdir -p "${OUT_DIR}"
echo "experiment: ${EXPERIMENT} (${CONF})"
echo "out: ${OUT_DIR}"

# --- Диагностика окружения ---
echo "=== clustbench env ==="
echo "host: $(hostname -s)"
echo "OMPI_PATH=$OMPI_PATH"
echo "UCX_PATH=$UCX_PATH"
command -v mpirun && mpirun --version | head -3
"$OMPI_PATH/bin/ompi_info" --parsable --all 2>/dev/null | grep -E 'opal_built_with_cuda_support|opal_cuda_support' || true
ldd "$BIN" | grep -E 'libmpi|libcudart|libucp|libuct' || true
if command -v ucx_info >/dev/null 2>&1; then
	echo "UCX: транспорты cma, cuda и gdr в ucx_info -d:"
	ucx_info -d 2>/dev/null | grep -E '^# *Transport:' | grep -iE 'cma|cuda|gdr' | sort -u | sed 's/^/  /' || \
		echo "  WARNING: не найдены — UCX собран без CMA или CUDA?"
fi
echo "yama.ptrace_scope на узлах задания:"
srun --ntasks-per-node=1 -n"${NNODES}" \
	bash -c 'echo "  $(hostname -s): $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo n/a)"' \
	2>/dev/null || true

# Без модуля peer-memory GPUDirect RDMA не работает, и auto упирается в ~0.8 ГБ/с:
# сетевой адаптер не может читать память GPU, всё идёт через ОЗУ.
# GPU и адаптер должны быть рядом: PIX/PXB — хорошо, SYS (через UPI между сокетами) — плохо.
echo "=== GPUDirect diagnostics ==="
srun --ntasks-per-node=1 -n"${NNODES}" bash -c '
echo "--- $(hostname -s) ---"
if lsmod | grep -qE "nvidia_peermem|nv_peer_mem"; then
	echo "  peermem: OK ($(lsmod | grep -oE "nvidia_peermem|nv_peer_mem" | head -1))"
else
	echo "  peermem: ОТСУТСТВУЕТ — GPUDirect RDMA не работает (modprobe nvidia_peermem, к админам)"
fi
if lsmod | grep -q gdrdrv; then echo "  gdrdrv:  OK"; else echo "  gdrdrv:  нет (gdr_copy пропускается)"; fi
echo "  топология GPU<->HCA:"
nvidia-smi topo -m 2>/dev/null | sed "s/^/    /" || echo "    nvidia-smi topo недоступен"
' 2>/dev/null || true
echo "======================"

if [[ "${COLLECT_ENV}" == 1 ]]; then
	srun --ntasks-per-node=1 -n"${NNODES}" --gpus-per-node=2 bash "${GPU_TESTS}/collect_env.sh" "${OUT_DIR}"
fi

if [[ "${LOG_TOPOLOGY}" == 1 ]]; then
	srun --ntasks-per-node=1 -n"${NNODES}" bash -c '
		f="$0/topology_$(hostname -s).txt"
		{ lscpu; echo; lscpu -e; echo; echo "# CPU: thread_siblings_list";
		  for c in /sys/devices/system/cpu/cpu[0-9]*; do
			echo "${c##*cpu} $(cat $c/topology/thread_siblings_list 2>/dev/null)"; done | sort -n
		  echo; echo "# scaling_governor:"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
		} > "$f"' "${OUT_DIR}"
	srun --gres-flags=allow-task-sharing --cpu-bind=cores bash -c \
		'echo "rank=$SLURM_PROCID local=$SLURM_LOCALID host=$(hostname -s) $(grep Cpus_allowed_list /proc/self/status)"' \
		| sort > "${OUT_DIR}/rank_masks.txt"
fi

# Фоновые записи частот GPU и CPU каждые 200 мс; --overlap не занимает ресурсы замеров
if [[ "${LOG_FREQ}" == 1 ]]; then
	srun --overlap --ntasks-per-node=1 -n"${NNODES}" --gpus-per-node=2 bash -c '
		nvidia-smi --query-gpu=timestamp,index,pci.bus_id,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu \
			--format=csv,nounits -lms 200 > "$0/gpu_clocks_$(hostname -s).csv"' "${OUT_DIR}" \
		2>"${OUT_DIR}/gpu_clocks.err" &
	BG_PIDS+=($!)
	srun --overlap --ntasks-per-node=1 -n"${NNODES}" bash -c '
		f="$0/cpu_freq_$(hostname -s).csv"
		while true; do
			echo "$(date +%s.%N),$(awk "/^cpu MHz/ {printf \"%s;\", \$4}" /proc/cpuinfo)" >> "$f"
			sleep 0.2
		done' "${OUT_DIR}" 2>"${OUT_DIR}/cpu_freq.err" &
	BG_PIDS+=($!)
fi

# Счётчики: байты, отправленные каждым сетевым адаптером (port_xmit_data в 4-байтовых словах),
# и трафик NVLink. Разбор — parse_paths.py.
snap() {
	srun --overlap --ntasks-per-node=1 -n"${NNODES}" --gpus-per-node=2 bash -c '
		f="$0/counters/$(hostname -s)_$1.txt"
		{ for d in /sys/class/infiniband/*; do
			for p in "$d"/ports/*; do
				echo "nic $(basename "$d") $(basename "$p") xmit_words $(cat "$p"/counters/port_xmit_data 2>/dev/null || echo NA)"
			done
		  done
		  echo "# nvidia-smi nvlink -gt d"
		  nvidia-smi nvlink -gt d 2>&1 || echo "NA"
		} > "$f"' "${OUT_DIR}" "$1"
}
if [[ "${UCX_PATHS}" == 1 ]]; then
	mkdir -p "${OUT_DIR}/ucx" "${OUT_DIR}/counters"
	# Трафик каждого GPU по PCIe раз в секунду (МБ/с приёма и передачи), строка с временем эпохи:
	# путь через ОЗУ или адаптер проходит по PCIe GPU, путь по NVLink — нет
	srun --overlap --ntasks-per-node=1 -n"${NNODES}" --gpus-per-node=2 bash -c '
		stdbuf -oL nvidia-smi dmon -s t -d 1 | while IFS= read -r l; do echo "$(date +%s.%N) $l"; done \
			> "$0/counters/pcie_$(hostname -s).txt"' "${OUT_DIR}" 2>"${OUT_DIR}/counters/pcie.err" &
	BG_PIDS+=($!)
fi

# --- Список запусков; перемешивание — Фишер — Йетс на bash (shuf с <(yes) рвал задание SIGPIPE) ---
LIST=()
for LINE in "${PLAN[@]}"; do
	read -r BLOCK REPS ENV MODE CASE ARGS <<<"${LINE}"
	for REP in $(seq 1 "${REPS}"); do
		LIST+=("${BLOCK} ${REP} ${ENV} ${MODE} ${CASE} ${ARGS}")
	done
done
if [[ -n "${SHUFFLE_SEED}" ]]; then
	RANDOM=${SHUFFLE_SEED}
	for ((K = ${#LIST[@]} - 1; K > 0; K--)); do
		J=$((RANDOM % (K + 1)))
		TMP=${LIST[K]}; LIST[K]=${LIST[J]}; LIST[J]=${TMP}
	done
fi
echo "запусков: ${#LIST[@]}, перемешивание: ${SHUFFLE_SEED:-нет}"
if [[ "${OUT_LAYOUT}" == job ]]; then echo "index,block,rep,seed,env,mode,bytes,iters,warmup,args,start,end" > "${RUNS}"; fi

for IDX in "${!LIST[@]}"; do
	read -r BLOCK REP ENV MODE CASE ARGS <<<"${LIST[$IDX]}"
	IFS=":" read -r B I W <<<"${CASE}"
	SEED=$((SEED_BASE + IDX))
	EXTRA=()
	if [[ -n "${ARGS}" ]]; then read -r -a EXTRA <<<"${ARGS}"; fi
	if [[ "${MODE}" == one_to_one_v2 ]]; then EXTRA+=(--seed "${SEED}"); fi
	if [[ "${OUT_LAYOUT}" == flat ]]; then
		NAME="${SAFE_NODES}_${ENV}_${MODE}_b${B}_w${W}_i${I}"
	else
		NAME="${SAFE_NODES}_${BLOCK}_rep${REP}_s${SEED}_${ENV}_${MODE}_b${B}_w${W}_i${I}"
	fi
	UCX_ENV=()
	if [[ "${UCX_PATHS}" == 1 ]]; then UCX_ENV=(UCX_LOG_LEVEL=info); fi
	for P in ${PROTO_BLOCKS[@]+"${PROTO_BLOCKS[@]}"}; do
		if [[ "${BLOCK}" == "${P}" ]]; then UCX_ENV+=(UCX_PROTO_ENABLE=y UCX_PROTO_INFO=y); fi
	done
	# Вывод программы: по файлу на ранг (журнал UCX), в журнал задания (обычное измерение) или никуда
	if [[ "${UCX_PATHS}" == 1 ]]; then
		STDIO=(--output="${OUT_DIR}/ucx/${NAME}.r%t.out" --error="${OUT_DIR}/ucx/${NAME}.r%t.err")
	elif [[ "${OUT_LAYOUT}" == flat ]]; then
		STDIO=()
	else
		STDIO=(--output=/dev/null)
	fi
	echo "RUN ${IDX}: ${BLOCK} rep=${REP} env=${ENV} mode=${MODE} bytes=${B} warmup=${W} iters=${I} ${ARGS}"
	if [[ "${UCX_PATHS}" == 1 ]]; then snap "${IDX}_before"; fi
	T0=$(date +%s.%N)
	# --cpu-bind=verbose,cores: ранг жёстко на своих ядрах, маска печатается в журнал задания
	env ${UCX_ENV[@]+"${UCX_ENV[@]}"} srun --mpi=pmix \
		--kill-on-bad-exit=1 \
		--gres-flags=allow-task-sharing \
		--cpu-bind=verbose,cores \
		${STDIO[@]+"${STDIO[@]}"} \
		"${BIN}" \
		--bytes "${B}" --warmup "${W}" --iters "${I}" --env "${ENV}" --mode "${MODE}" \
		${COMMON[@]+"${COMMON[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} --out "${OUT_DIR}/${NAME}.txt"
	T1=$(date +%s.%N)
	if [[ "${UCX_PATHS}" == 1 ]]; then snap "${IDX}_after"; fi
	if [[ "${OUT_LAYOUT}" == job ]]; then
		echo "${IDX},${BLOCK},${REP},${SEED},${ENV},${MODE},${B},${I},${W},${ARGS},${T0},${T1}" >> "${RUNS}"
	fi
done

stop_bg
wait 2>/dev/null || true
if [[ "${OUT_LAYOUT}" == job ]]; then
	cp "${GPU_TESTS}/clustbench-${SLURM_JOB_ID:-local}.err" "${OUT_DIR}/job.err" 2>/dev/null || true
	# Один архив: при скачивании множества мелких файлов часть терялась (hyp_11065, hyp_11078)
	tar czf "${OUT_DIR}.tgz" -C "$(dirname "${OUT_DIR}")" "$(basename "${OUT_DIR}")"
	echo "архив: ${OUT_DIR}.tgz"
fi
echo "DONE: ${OUT_DIR}"
