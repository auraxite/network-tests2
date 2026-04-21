#!/bin/bash
#SBATCH --job-name=clustbench
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --output=clustbench-%j.out
#SBATCH --error=clustbench-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
# Запрашиваем ВСЕ GPU узла как общий ресурс ноды, а не как «по одному на task».
# При --gpus-per-task=1 Slurm создавал на каждый процесс отдельный cgroup, в
# котором виден ровно один GPU. cuda_ipc между процессами одного узла из разных
# cgroup-изолированных «обзоров» не работает (UCX не может разделить
# IPC-handle), поэтому intra-node путь в env=auto уезжал на cuda_copy через
# host staging — отсюда и плоские ~745 мкс на heatmap. С --gres=gpu:4 все 4
# GPU видны всем процессам узла, а раздачу «кому какой device» делает уже сам
# код (gpu_benchmark.cpp: cudaSetDevice(local_rank % gpu_count)).
#SBATCH --gres=gpu:4

REPS=(1) # задать число
ENVS=( # закомментировать лишнее
	auto
	host
)
MODES=( # закомментировать лишнее
	one_to_one
	# one_to_one_node
	all_to_all
)
CASES=( # "bytes:iters:warmup", закомментировать лишнее
	"500000:1500:5"
	"4000000:500:5"
	"16000000:200:5"
)

OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10-cuda}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

set -euo pipefail

export PATH="$OMPI_PATH/bin:$PATH"
_LDPATH="$OMPI_PATH/lib:${CUDA_PATH}/lib64"
if [[ -d "$UCX_PATH/lib" ]]; then
	_LDPATH="$UCX_PATH/lib:$_LDPATH"
fi
export LD_LIBRARY_PATH="$_LDPATH:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
# export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx

# === Intra-node single-copy: убираем двойной копир через /dev/shm ===
# По умолчанию intra-node обмен host-буферов идёт через /dev/shm-сегмент:
#   sender->buf  -> [/dev/shm staging] -> receiver->buf   (две копии через DRAM)
# CMA (Cross Memory Attach, ядро >=3.2) даёт одну копию напрямую между
# адресными пространствами процессов через process_vm_readv/writev:
#   sender->buf  -> receiver->buf                          (одна копия)
# Главный канал у нас pml=ucx, поэтому ключевая переменная — UCX_TLS,
# где явно перечислены допустимые транспорты с cma в списке. btl_vader_*
# подстрахует на случай, если какие-то компоненты пойдут мимо UCX.
# Требования: yama.ptrace_scope <= 1 (типично на HPC-узлах) и UCX,
# собранный с поддержкой CMA (проверяется в diagnostics ниже).
export UCX_TLS=cma,cuda_copy,cuda_ipc,rc_verbs,rc_mlx5,dc_mlx5,ud_verbs,ud_mlx5,sm,self
export OMPI_MCA_btl_vader_single_copy_mechanism=cma

# Диагностика выбранных транспортов UCX (раскомментировать при отладке;
# даёт много вывода — по строке на каждое создание endpoint'а).
# export UCX_LOG_LEVEL=info
# export OMPI_MCA_btl_base_verbose=10

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/fixed"
mkdir -p "${OUT_DIR}"
NTASKS="${SLURM_NTASKS:-8}"
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

# --- Диагностика intra-node single-copy (CMA) ---
# 1) CMA должен быть в списке UCX-транспортов (если нет — UCX собран без CMA).
if command -v ucx_info >/dev/null 2>&1; then
	echo "UCX transports with cma:"
	ucx_info -d 2>/dev/null | grep -E '^# *Transport:' | grep -i cma || \
		echo "  WARNING: cma не найден в ucx_info -d (UCX собран без CMA?)"
else
	echo "ucx_info: не в PATH (диагностику CMA пропускаем)"
fi
# 2) На каждом из выделенных узлов проверяем yama.ptrace_scope (нужно <= 1).
echo "yama.ptrace_scope на узлах задания:"
srun --ntasks-per-node=1 -n"${SLURM_JOB_NUM_NODES:-1}" \
	bash -c 'echo "  $(hostname -s): $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo n/a)"' \
	2>/dev/null || true
echo "======================"

for REP in "${REPS[@]}"; do
	for ENV in "${ENVS[@]}"; do
		for MODE in "${MODES[@]}"; do
			for CASE in "${CASES[@]}"; do
				IFS=":" read -r BYTES ITERS WARMUP <<<"${CASE}"

				OUT_TXT="${OUT_DIR}/${SAFE_NODES}_rep${REP}_${ENV}_${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
				echo "RUN rep=${REP} env=${ENV} mode=${MODE} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"
				echo "  -> ${OUT_TXT}"

				if [[ "${MODE}" == "one_to_one_node" ]]; then
					# Новый режим: 1 MPI-процесс на узел, внутри процесса доступны
					# все GPU узла. Локальные пары идут без MPI через cudaMemcpyPeer
					# (auto) или D2H->H2D (host).
					RUN_NTASKS="${SLURM_JOB_NUM_NODES:-2}"
					RUN_TPN=1
					echo "  launch: node-mode (${RUN_NTASKS} MPI rank(s), 1 per node)"
					srun --mpi=pmix \
						--cpu-bind=cores \
						--ntasks="${RUN_NTASKS}" \
						--ntasks-per-node="${RUN_TPN}" \
						"${BIN}" \
						--bytes "${BYTES}" \
						--warmup "${WARMUP}" \
						--iters "${ITERS}" \
						--env "${ENV}" \
						--mode "${MODE}" \
						--out "${OUT_TXT}"
				else
					# Старые режимы: 1 MPI-процесс на 1 GPU.
					# --cpu-bind=cores привязывает каждый ранг к своему ядру.
					# --gpu-bind=closest закрепляет ранг за ближайшим GPU.
					echo "  launch: rank-per-gpu mode (${NTASKS} MPI rank(s))"
					srun --mpi=pmix \
						--cpu-bind=cores \
						--gpu-bind=closest \
						-n"${NTASKS}" "${BIN}" \
						--bytes "${BYTES}" \
						--warmup "${WARMUP}" \
						--iters "${ITERS}" \
						--env "${ENV}" \
						--mode "${MODE}" \
						--out "${OUT_TXT}"
				fi

				echo "OK  rep=${REP} env=${ENV} mode=${MODE} bytes=${BYTES}"
			done
		done
	done
done
