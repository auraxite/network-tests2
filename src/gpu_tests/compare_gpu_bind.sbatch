#!/bin/bash
#SBATCH --job-name=gpu-bind-cmp
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=gpu-bind-cmp-%j.out
#SBATCH --error=gpu-bind-cmp-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1

# Один job Slurm → одна allocation (любые 2 узла) → оба srun на тех же узлах.
# Сравнение --gpu-bind=closest vs none, auto one_to_one, 16 МБ.

OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10-cuda}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

set -eEuo pipefail

export PATH="$OMPI_PATH/bin:$PATH"
_LDPATH="$OMPI_PATH/lib:${CUDA_PATH}/lib64"
if [[ -d "$UCX_PATH/lib" ]]; then
	_LDPATH="$UCX_PATH/lib:$_LDPATH"
fi
export LD_LIBRARY_PATH="$_LDPATH:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl_vader_single_copy_mechanism=cma
export UCX_RNDV_THRESH_LEAVE_DEFAULT=1

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
BIN="${GPU_TESTS}/gpu"
OUT_DIR="${GPU_TESTS}/results_gpu_bind_cmp"
mkdir -p "${OUT_DIR}"

JOB_NODES="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-$(hostname -s)}}"
SAFE_NODES="$(echo "${JOB_NODES}" | tr '[],' '_' | tr -s '_' | sed 's/^_//; s/_$//')"

BYTES=16000000
WARMUP=5
ITERS=200
NTASKS="${SLURM_NTASKS:-8}"

test -x "${BIN}" || { echo "error: missing or not executable: ${BIN}"; exit 1; }

echo "=== gpu-bind compare ==="
echo "nodes: ${JOB_NODES}"
echo "BIN=${BIN}"
echo "OUT_DIR=${OUT_DIR}"

run_one() {
	local bind="$1"
	local out_txt="${OUT_DIR}/${SAFE_NODES}_auto_one_to_one_b${BYTES}_gpu_bind_${bind}.txt"

	echo ""
	echo ">>> srun --gpu-bind=${bind}"
	echo "    -> ${out_txt}"

	srun --mpi=pmix \
		--kill-on-bad-exit=1 \
		--cpu-bind=cores \
		--gpu-bind="${bind}" \
		-n"${NTASKS}" "${BIN}" \
		--bytes "${BYTES}" \
		--warmup "${WARMUP}" \
		--iters "${ITERS}" \
		--env auto \
		--mode one_to_one \
		--out "${out_txt}"

	echo "OK  gpu-bind=${bind}"
}

run_one closest
run_one none

echo ""
echo "=== done: results in ${OUT_DIR} ==="
