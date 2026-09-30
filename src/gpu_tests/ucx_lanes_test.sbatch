#!/bin/bash
#SBATCH --partition=intel-a100-pci5
#SBATCH --job-name=ucx-lanes
#SBATCH --time=00:30:00
#SBATCH --output=ucx-lanes-%j.out
#SBATCH --error=ucx-lanes-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --gres-flags=enforce-binding

# Проверка гипотезы: на g5500 UCX делит крупное rendezvous-сообщение между
# двумя путями (UCX_MAX_RNDV_LANES, по умолчанию 2) — часть идёт между GPU
# узла через cuda_ipc, часть через сетевой адаптер mlx5_0. Тогда в all_to_all
# внутриузловая пара стоит в очереди к адаптеру вместе с межузловыми и
# получается почти такой же медленной.
#
# Шаги (все при UCX_RNDV_THRESH=0, 16 МБ, среда auto):
#   1. one_to_one с UCX_PROTO_INFO=y — UCX печатает выбранный протокол и
#      доли сообщения по путям (в ucx-lanes-<job>.out/.err).
#   2. all_to_all как в основном прогоне (UCX_MAX_RNDV_LANES не задан).
#   3. all_to_all и one_to_one с UCX_MAX_RNDV_LANES=1.
# Если в шаге 3 внутриузловые пары all_to_all станут заметно быстрее
# межузловых — гипотеза подтверждена.
#
# Запуск: cd ~/network-tests2/src/gpu_tests && sbatch ucx_lanes_test.sbatch

OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10-cuda-clean}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

set -eEuo pipefail

export PATH="$UCX_PATH/bin:$OMPI_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$UCX_PATH/lib:$OMPI_PATH/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export OMPI_MCA_pml=ucx

unset UCX_RNDV_SCHEME UCX_MEMTYPE_CACHE UCX_TLS UCX_MAX_RNDV_LANES UCX_PROTO_INFO
export UCX_RNDV_THRESH=0

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
BIN="${GPU_TESTS}/gpu"
OUT_DIR="${GPU_TESTS}/results/results_lanes"
mkdir -p "${OUT_DIR}"
test -x "$BIN" || { echo "error: missing or not executable: $BIN"; exit 1; }

echo "=== UCX: параметры, связанные с делением сообщения ==="
ucx_info -c 2>/dev/null | grep -iE 'RNDV_LANES|MAX_RNDV|PROTO_ENABLE' || echo "ucx_info недоступен"

run() {  # run <метка> <mode> <iters> — остальное берётся из окружения
	local tag="$1" mode="$2" iters="$3"
	local out="${OUT_DIR}/${tag}_auto_${mode}_b16384000_w5_i${iters}.txt"
	echo "=== RUN ${tag} mode=${mode} UCX_MAX_RNDV_LANES=${UCX_MAX_RNDV_LANES:-<not set>} UCX_PROTO_INFO=${UCX_PROTO_INFO:-<not set>}"
	srun --mpi=pmix --kill-on-bad-exit=1 \
		--gres-flags=allow-task-sharing --cpu-bind=cores \
		"${BIN}" --bytes 16384000 --warmup 5 --iters "${iters}" \
		--env auto --mode "${mode}" --out "${out}"
	grep '^pair' "${out}" | awk '$2!=$4' || true
}

# 1. Как UCX делит сообщение
UCX_PROTO_INFO=y run proto_info one_to_one 20

# 2. Как в основном прогоне
run lanes_default all_to_all 200
run lanes_default one_to_one 100

# 3. Без деления сообщения
export UCX_MAX_RNDV_LANES=1
run lanes_1 all_to_all 200
run lanes_1 one_to_one 100

echo "=== DONE ==="
