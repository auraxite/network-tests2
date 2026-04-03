#!/bin/bash
#SBATCH --job-name=gpu1to1-avggrid
#SBATCH --partition=batch
# 3 повтора × 2 режима × 4 CPU = 24 прогона; при необходимости увеличь.
#SBATCH --time=24:00:00
#SBATCH --output=slurm-batch4-%j.out
#SBATCH --error=slurm-batch4-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32

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
OUT_DIR="${GPU_TESTS}/grid_avg_repeat"
mkdir -p "${OUT_DIR}"

NTASKS="${SLURM_NTASKS:-8}"

if [[ ! -x "${BIN}" ]]; then
	echo "build first: (cd ${GPU_TESTS} && make)" >&2
	exit 1
fi

# 64 MB (десятичные байты), 100 измеряемых итераций; прогрев фиксированный.
BYTES=$((64 * 1000 * 1000))
WARMUP=16
ITERS=100

MODES=(auto host)
CPUS_LIST=(1 2 8 32)

for REP in 1 2 3; do
	for MODE in "${MODES[@]}"; do
		for CPT in "${CPUS_LIST[@]}"; do
			OUT_TXT="${OUT_DIR}/rep${REP}_m${MODE}_c${CPT}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
			echo "RUN rep=${REP} mode=${MODE} cpus_per_task=${CPT} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"

			srun --mpi=pmix -n"${NTASKS}" --cpus-per-task="${CPT}" "${BIN}" \
				--bytes "${BYTES}" \
				--warmup "${WARMUP}" \
				--iters "${ITERS}" \
				--mode "${MODE}" \
				--out "${OUT_TXT}"
		done
	done
done
