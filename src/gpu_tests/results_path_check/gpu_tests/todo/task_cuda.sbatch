#!/bin/bash
# CUDA IPC/P2P payload benchmark: one rank per GPU, all ranks on one node.
# MPI is used only for process control, CUDA IPC handles, and result gathering.
#SBATCH --job-name=clustbench-cuda-ipc
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --output=clustbench-cuda-ipc-%j.out
#SBATCH --error=clustbench-cuda-ipc-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=none

set -eEuo pipefail

ONE_TO_ONE_CASES=(
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
)
ALL_TO_ALL_CASES=(
	"500000:1500:5"
	"4000000:500:5"
	"16000000:200:5"
)

OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10-cuda}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/results_cuda_ipc"
BIN="${GPU_TESTS}/gpu"
mkdir -p "${OUT_DIR}"

export PATH="$OMPI_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$OMPI_PATH/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"

test -x "${BIN}" || { echo "missing executable: ${BIN}" >&2; exit 1; }

run_mode() {
	local mode="$1"
	shift
	for case_spec in "$@"; do
		IFS=":" read -r bytes iters warmup <<<"${case_spec}"
		local out="${OUT_DIR}/$(hostname -s)_${mode}_b${bytes}_w${warmup}_i${iters}.txt"
		echo "RUN mode=${mode} bytes=${bytes} iters=${iters} warmup=${warmup}"
		srun --mpi=pmix --kill-on-bad-exit=1 --cpu-bind=cores --gpu-bind=none \
			"${BIN}" --mode "${mode}" --env auto --bytes "${bytes}" \
			--warmup "${warmup}" --iters "${iters}" --out "${out}"
	done
}

run_mode cuda_one_to_one "${ONE_TO_ONE_CASES[@]}"
run_mode cuda_all_to_all "${ALL_TO_ALL_CASES[@]}"
