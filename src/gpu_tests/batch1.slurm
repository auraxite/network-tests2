#!/bin/bash
#SBATCH --job-name=gpu1to1-grid
#SBATCH --partition=batch
#SBATCH --time=06:00:00
#SBATCH --output=slurm-grid-%j.out
#SBATCH --error=slurm-grid-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2

set -euo pipefail

HOSTNAME="${SLURMD_NODENAME:-$(hostname -s)}"
GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/grid_out"
mkdir -p "${OUT_DIR}"
NTASKS="${SLURM_NTASKS:-8}"
BIN="${GPU_TESTS}/gpu_one_to_one"

SIZES=(
	$((64 * 1000))
	$((256 * 1000))
	$((1 * 1000 * 1000))
	$((4 * 1000 * 1000))
	$((16 * 1000 * 1000))
	$((32 * 1000 * 1000))
)

MODES=(auto host)
CPUS_LIST=(1 2)

for REP in 1 2; do
	for CPUS in "${CPUS_LIST[@]}"; do
		for MODE in "${MODES[@]}"; do
			for BYTES in "${SIZES[@]}"; do
				if ((BYTES <= 256 * 1000)); then
					WARMUP=90
					ITERS=900
				elif ((BYTES <= 4 * 1000 * 1000)); then
					WARMUP=60
					ITERS=600
				else
					WARMUP=30
					ITERS=300
				fi

				OUT_TXT="${OUT_DIR}/${HOSTNAME}_rep${REP}_c${CPUS}_m${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
				echo "RUN rep=${REP} cpus=${CPUS} mode=${MODE} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"

				srun --mpi=pmix -n"${NTASKS}" --cpus-per-task="${CPUS}" "${BIN}" \
					--bytes "${BYTES}" \
					--warmup "${WARMUP}" \
					--iters "${ITERS}" \
					--mode "${MODE}" \
					--out "${OUT_TXT}"
			done
		done
	done
done
