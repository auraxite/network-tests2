#!/bin/bash
#SBATCH --partition=intel-a100-pci5
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

ENVS=( # закомментировать лишнее
	auto
	host
)
MODES=( # закомментировать лишнее
	one_to_one
	all_to_all
)

ONE_TO_ONE_CASES=( # "bytes:iters:warmup"
	"1000:2000:5"
	"2000:2000:5"
	"4000:2000:5"
	"8000:2000:5"
	"16000:2000:5"
	"32000:2000:5"
	"64000:2000:5"
	"128000:2000:5"
	"256000:2000:5"
	"512000:2000:5"
	"1024000:1000:5"
	"2048000:1000:5"
	"4096000:1000:5"
	"8192000:100:5"
	"16384000:100:5"
)

ALL_TO_ALL_CASES=( # "bytes:iters:warmup"
	"512000:1000:5"
	"4096000:500:5"
	"16384000:200:5"
)

OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10-cuda-clean}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

set -eEuo pipefail

on_error() {
	local exit_code=$?
	echo "FATAL: benchmark step failed (exit=${exit_code}). Stopping job immediately." >&2
	if [[ -n "${SLURM_JOB_ID:-}" ]]; then
		scancel "${SLURM_JOB_ID}" >/dev/null 2>&1 || true
	fi
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


GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/results"
mkdir -p "${OUT_DIR}"
NTASKS="${SLURM_NTASKS:-4}"
BIN="${GPU_TESTS}/gpu"
JOB_NODES="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-$(hostname -s)}}"
SAFE_NODES="$(echo "${JOB_NODES}" | tr '[],' '_' | tr -s '_' | sed 's/^_//; s/_$//')"

echo "=== clustbench env ==="
echo "host: $(hostname -s)"
echo "OMPI_PATH=$OMPI_PATH"
echo "UCX_PATH=$UCX_PATH"
if [[ -d "$UCX_PATH/lib" ]]; then
	echo "UCX lib: OK ($UCX_PATH/lib)"
else
	echo "UCX lib: (нет каталога — используется системный UCX из libmpi, если есть)"
fi

command -v mpirun && mpirun --version | head -3
"$OMPI_PATH/bin/ompi_info" --parsable --all 2>/dev/null | grep -E 'opal_built_with_cuda_support|opal_cuda_support' || true
echo "BIN=$BIN"
test -x "$BIN" || { echo "error: missing or not executable: $BIN"; exit 1; }
ldd "$BIN" | grep -E 'libmpi|libcudart|libucp|libuct' || true
ldd "$OMPI_PATH/lib/libmpi.so" 2>/dev/null | grep -E 'libucp|libuct' || true
if command -v ucx_info >/dev/null 2>&1; then
	echo "UCX transports with cma:"
	ucx_info -d 2>/dev/null | grep -E '^# *Transport:' | grep -i cma || \
		echo "  WARNING: cma не найден в ucx_info -d (UCX собран без CMA?)"
else
	echo "ucx_info: не в PATH (диагностику CMA пропускаем)"
fi

echo "yama.ptrace_scope на узлах задания:"
srun --ntasks-per-node=1 -n"${SLURM_JOB_NUM_NODES:-1}" \
	bash -c 'echo "  $(hostname -s): $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo n/a)"' \
	2>/dev/null || true

# --- Диагностика GPUDirect RDMA ---------------------------------------
# Без этих трёх пунктов auto-режим будет упираться в ~0.8 ГБ/с независимо
# от любых UCX-настроек, потому что NIC просто не сможет читать память GPU
# напрямую и всё пойдёт через staging в хостовой памяти.
echo "=== GPUDirect diagnostics ==="
srun --ntasks-per-node=1 -n"${SLURM_JOB_NUM_NODES:-1}" bash -c '
echo "--- $(hostname -s) ---"

# 1. Модуль peer-memory. Без него GPUDirect RDMA физически не работает.
if lsmod | grep -qE "nvidia_peermem|nv_peer_mem"; then
	echo "  peermem: OK ($(lsmod | grep -oE "nvidia_peermem|nv_peer_mem" | head -1))"
else
	echo "  peermem: ОТСУТСТВУЕТ — GPUDirect RDMA не работает!"
	echo "           нужен modprobe nvidia_peermem (права root, к админам кластера)"
fi

# 2. gdrdrv — ускоряет мелкие сообщения через gdr_copy.
if lsmod | grep -q gdrdrv; then
	echo "  gdrdrv:  OK"
else
	echo "  gdrdrv:  нет (gdr_copy будет пропущен, крупные размеры не пострадают)"
fi

# 3. Топология: GPU и HCA должны быть на одном PCIe-свитче/NUMA-узле.
#    Связь PIX/PXB — хорошо, SYS (через UPI между сокетами) — те самые 0.8 ГБ/с.
echo "  топология GPU<->HCA:"
nvidia-smi topo -m 2>/dev/null | sed "s/^/    /" || echo "    nvidia-smi topo недоступен"
' 2>/dev/null || true

# Какие CUDA/GDR-транспорты реально собрались в UCX
if command -v ucx_info >/dev/null 2>&1; then
	echo "CUDA/GDR транспорты в ucx_info -d:"
	ucx_info -d 2>/dev/null | grep -E '^# *Transport:' | grep -iE 'cuda|gdr' | sort -u | sed 's/^/  /' || \
		echo "  WARNING: cuda_copy/cuda_ipc не найдены — UCX собран без CUDA?"
fi
echo "======================"

for ENV in "${ENVS[@]}"; do
	for MODE in "${MODES[@]}"; do
		case "${MODE}" in
		one_to_one) CASES=("${ONE_TO_ONE_CASES[@]}") ;;
		all_to_all) CASES=("${ALL_TO_ALL_CASES[@]}") ;;
		*) echo "unknown mode: ${MODE}" >&2; exit 1 ;;
		esac

		for CASE in "${CASES[@]}"; do
			IFS=":" read -r BYTES ITERS WARMUP <<<"${CASE}"

			OUT_TXT="${OUT_DIR}/${SAFE_NODES}_${ENV}_${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
			echo "RUN env=${ENV} mode=${MODE} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"
			echo "  -> ${OUT_TXT}"

			echo "  launch: rank-per-gpu mode (${NTASKS} MPI rank(s), 2 visible GPU/node)"
			# --cpu-bind=verbose,cores — жёсткая привязка ранга к ядрам
			# (verbose печатает фактическую маску, чтобы было видно, что
			# ранг сел рядом со своим GPU, а не на чужой сокет).
			srun --mpi=pmix \
				--kill-on-bad-exit=1 \
				--gres-flags=allow-task-sharing \
				--cpu-bind=verbose,cores \
				"${BIN}" \
				--bytes "${BYTES}" \
				--warmup "${WARMUP}" \
				--iters "${ITERS}" \
				--env "${ENV}" \
				--mode "${MODE}" \
				--out "${OUT_TXT}"

			echo "OK  env=${ENV} mode=${MODE} bytes=${BYTES}"
		done
	done
done