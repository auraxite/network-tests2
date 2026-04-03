#!/bin/bash
#SBATCH --job-name=gpu1to1-small-demo
#SBATCH --partition=batch
#SBATCH --time=00:20:00
#SBATCH --output=slurm-small-%j.out
#SBATCH --error=slurm-small-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2

set -euo pipefail

HOSTNAME="${SLURMD_NODENAME:-$(hostname -s)}"
GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/small_demo_raw"
PNG_DIR="${GPU_TESTS}/small_demo_png"
mkdir -p "${OUT_DIR}" "${PNG_DIR}"
NTASKS="${SLURM_NTASKS:-8}"
BIN="${GPU_TESTS}/gpu_one_to_one"
RENDER="${GPU_TESTS}/gpu_render.py"

# Один набор параметров для демо.
BYTES=$((24 * 1000 * 1000))
WARMUP=16
ITERS=100
MODE="host"
CPUS="${SLURM_CPUS_PER_TASK:-2}"

for REP in 1 2 3; do
	OUT_TXT="${OUT_DIR}/${HOSTNAME}_rep${REP}_c${CPUS}_m${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
	echo "RUN rep=${REP} mode=${MODE} cpus=${CPUS} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"
	srun --mpi=pmix -n"${NTASKS}" --cpus-per-task="${CPUS}" "${BIN}" \
		--bytes "${BYTES}" \
		--warmup "${WARMUP}" \
		--iters "${ITERS}" \
		--mode "${MODE}" \
		--stat all \
		--out "${OUT_TXT}"
done

echo "Render PNG for all 3 logs..."
python3 "${RENDER}" "${OUT_DIR}" --out-dir "${PNG_DIR}"

# Для демо оставляем только avg/var (median/min/max убираем).
find "${PNG_DIR}" -type f \( -name "*_median.png" -o -name "*_min.png" -o -name "*_max.png" \) -delete

echo "Done."
echo "Raw logs: ${OUT_DIR}"
echo "PNG (avg/var): ${PNG_DIR}"
