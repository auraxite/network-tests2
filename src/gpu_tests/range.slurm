#!/bin/bash
#SBATCH --job-name=clustbench
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --output=clustbench-%j.out
#SBATCH --error=clustbench-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1

REPS=(1)
ENVS=( # auto (CUDA-aware при поддержке MPI) или host (через host-staging).
	auto
	host
)
MODES=(
	one_to_one
	all_to_all
)
RANGES=( # "bytes_begin:bytes_end:bytes_step:iters:warmup"
	"4000:64000:8000:4000:80"
)

OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10-cuda}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"
NETCDF_PATH="${NETCDF_PATH:-$HOME/opt/netcdf-c-4.9.2}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

set -euo pipefail

export PATH="$OMPI_PATH/bin:$NETCDF_PATH/bin:$PATH"
_LDPATH="$OMPI_PATH/lib:$NETCDF_PATH/lib:${CUDA_PATH}/lib64"
if [[ -d "$UCX_PATH/lib" ]]; then
	_LDPATH="$UCX_PATH/lib:$_LDPATH"
fi
export LD_LIBRARY_PATH="$_LDPATH:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export UCX_MEMTYPE_CACHE=N
export OMPI_MCA_pml=UCX

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/range_out"
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
ldd "$BIN" | grep -E 'libmpi|libcudart|libnetcdf|libucp|libuct' || true
ldd "$OMPI_PATH/lib/libmpi.so" 2>/dev/null | grep -E 'libucp|libuct' || true
echo "======================"

for REP in "${REPS[@]}"; do
	for ENV in "${ENVS[@]}"; do
		for MODE in "${MODES[@]}"; do
			for CASE in "${RANGES[@]}"; do
				IFS=":" read -r BYTES_BEGIN BYTES_END BYTES_STEP ITERS WARMUP <<<"${CASE}"

				OUT_PREFIX="${OUT_DIR}/${SAFE_NODES}_rep${REP}_${ENV}_${MODE}_bb${BYTES_BEGIN}_be${BYTES_END}_bs${BYTES_STEP}_w${WARMUP}_i${ITERS}"
				echo "RUN rep=${REP} env=${ENV} mode=${MODE} bytes_begin=${BYTES_BEGIN} bytes_end=${BYTES_END} bytes_step=${BYTES_STEP} warmup=${WARMUP} iters=${ITERS}"
				echo "  -> ${OUT_PREFIX}_{avg,med,min,max,var,std}.nc"

				srun --mpi=pmix -n"${NTASKS}" "${BIN}" \
					--bytes-begin "${BYTES_BEGIN}" \
					--bytes-end "${BYTES_END}" \
					--bytes-step "${BYTES_STEP}" \
					--warmup "${WARMUP}" \
					--iters "${ITERS}" \
					--env "${ENV}" \
					--mode "${MODE}" \
					--out "${OUT_PREFIX}" \
					-d

				echo "OK  rep=${REP} env=${ENV} mode=${MODE} bytes_begin=${BYTES_BEGIN} bytes_end=${BYTES_END} bytes_step=${BYTES_STEP}"
			done
		done
	done
done
