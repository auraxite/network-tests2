#!/bin/bash
#SBATCH --job-name=clustbench
#SBATCH --partition=batch
#SBATCH --time=00:20:00
#SBATCH --output=slurm-grid-%j.out
#SBATCH --error=slurm-grid-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1

set -euo pipefail

HOSTNAME="${SLURMD_NODENAME:-$(hostname -s)}"
GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/grid_out"
mkdir -p "${OUT_DIR}"
NTASKS="${SLURM_NTASKS:-8}"
BIN="${GPU_TESTS}/gpu_one_to_one"

CASES=(
	"500000:1600:80"
	# "2000000:800:40"
	# "8000000:400:20"
	# "32000000:200:10"
)

REPS=(1)
MODES=(
	auto
	# host
)

for REP in "${REPS[@]}"; do
	for MODE in "${MODES[@]}"; do
		for CASE in "${CASES[@]}"; do
			IFS=":" read -r BYTES ITERS WARMUP <<<"${CASE}"

			OUT_TXT="${OUT_DIR}/${HOSTNAME}_rep${REP}_m${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
			echo "RUN rep=${REP} mode=${MODE} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"

			srun --mpi=pmix -n"${NTASKS}" --cpus-per-task=1 "${BIN}" \
				--bytes "${BYTES}" \
				--warmup "${WARMUP}" \
				--iters "${ITERS}" \
				--mode "${MODE}" \
				--out "${OUT_TXT}"
		done
	done
done
