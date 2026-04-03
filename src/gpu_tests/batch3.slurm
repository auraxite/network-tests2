#!/bin/bash
#SBATCH --job-name=gpu1to1-cpu
#SBATCH --partition=batch
# Отдельный прогон: только cpus-per-task × 3 размера (без полной сетки из batch2).
# 3 CPU × 3 размера = 9 шагов srun; при добавлении MODE ниже — умножается.
#SBATCH --time=08:00:00
#SBATCH --output=slurm-cpu-%j.out
#SBATCH --error=slurm-cpu-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${ROOT}/Makefile" && -f "${ROOT}/gpu_one_to_one.cpp" ]]; then
	GPU_TESTS="${ROOT}"
elif [[ -f "${ROOT}/src/gpu_tests/Makefile" && -f "${ROOT}/src/gpu_tests/gpu_one_to_one.cpp" ]]; then
	GPU_TESTS="${ROOT}/src/gpu_tests"
else
	echo "cannot find gpu_tests (Makefile + gpu_one_to_one.cpp); SLURM_SUBMIT_DIR=${ROOT}" >&2
	exit 1
fi

BIN="${GPU_TESTS}/gpu_one_to_one"
OUT_DIR="${GPU_TESTS}/grid_out_cpu"
mkdir -p "${OUT_DIR}"

NTASKS="${SLURM_NTASKS:-8}"

if [[ ! -x "${BIN}" ]]; then
	echo "build first: (cd ${GPU_TESTS} && make)" >&2
	exit 1
fi

# Только влияние cpus-per-task: один режим (для host — второй запуск или поменяй MODE).
MODE="${MODE:-auto}"
CPUS_LIST=(1 2 4)

# 64 KB, 4 MB, 32 MB (десятичные KB/MB)
SIZES=(
	$((64 * 1000))
	$((4 * 1000 * 1000))
	$((32 * 1000 * 1000))
)

for CPUS in "${CPUS_LIST[@]}"; do
	for BYTES in "${SIZES[@]}"; do
		if ((BYTES <= 512 * 1000)); then
			WARMUP=24
			ITERS=400
		elif ((BYTES <= 8 * 1000 * 1000)); then
			WARMUP=16
			ITERS=200
		else
			WARMUP=8
			ITERS=50
		fi

		OUT_TXT="${OUT_DIR}/bench_m${MODE}_c${CPUS}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
		echo "RUN mode=${MODE} cpus_per_task=${CPUS} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"

		srun --mpi=pmix -n"${NTASKS}" --cpus-per-task="${CPUS}" "${BIN}" \
			--bytes "${BYTES}" \
			--warmup "${WARMUP}" \
			--iters "${ITERS}" \
			--mode "${MODE}" \
			--out "${OUT_TXT}"
	done
done
