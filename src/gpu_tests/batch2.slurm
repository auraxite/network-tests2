#!/bin/bash
#SBATCH --job-name=gpu1to1-grid
#SBATCH --partition=batch
# Лимит «настенного» времени: после этого Slurm убьёт job (формат ЧЧ:ММ:СС).
# Сетка: 10 размеров × 3 CPU × 2 режима = 60 прогонов — при нехватке увеличь.
#SBATCH --time=24:00:00
#SBATCH --output=slurm-grid-%j.out
#SBATCH --error=slurm-grid-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
# Максимум из перебора ниже; srun шагами запрашивает 1, 2 или 4 CPU на задачу.
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
OUT_DIR="${GPU_TESTS}/grid_out"
mkdir -p "${OUT_DIR}"

NTASKS="${SLURM_NTASKS:-8}"

if [[ ! -x "${BIN}" ]]; then
	echo "build or copy binary first: ${BIN} (cd ${GPU_TESTS} && make)" >&2
	exit 1
fi

SIZES=(
	$((64 * 1000))
	$((256 * 1000))
	$((512 * 1000))
	$((1 * 1000 * 1000))
	$((2 * 1000 * 1000))
	$((4 * 1000 * 1000))
	$((8 * 1000 * 1000))
	$((16 * 1000 * 1000))
	$((24 * 1000 * 1000))
	$((32 * 1000 * 1000))
)

CPUS_LIST=(1 2 4)
MODES=(auto host)

for CPUS in "${CPUS_LIST[@]}"; do
	for MODE in "${MODES[@]}"; do
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

			OUT_TXT="${OUT_DIR}/bench_c${CPUS}_m${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
			echo "RUN cpus=${CPUS} mode=${MODE} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"

			srun --mpi=pmix -n"${NTASKS}" --cpus-per-task="${CPUS}" "${BIN}" \
				--bytes "${BYTES}" \
				--warmup "${WARMUP}" \
				--iters "${ITERS}" \
				--mode "${MODE}" \
				--out "${OUT_TXT}"
		done
	done
done
