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

set -euo pipefail

OMPI_PREFIX="${OMPI_PREFIX:-$HOME/opt/openmpi-5.0.10-cuda}"
UCX_PREFIX="${UCX_PREFIX:-$HOME/opt/ucx-1.15.0}"
NETCDF_HOME="${NETCDF_HOME:-$HOME/opt/netcdf-c-4.9.2}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"

export PATH="$OMPI_PREFIX/bin:$NETCDF_HOME/bin:$PATH"
_LDPATH="$OMPI_PREFIX/lib:$NETCDF_HOME/lib:${CUDA_ROOT}/lib64"
if [[ -d "$UCX_PREFIX/lib" ]]; then
	_LDPATH="$UCX_PREFIX/lib:$_LDPATH"
fi
export LD_LIBRARY_PATH="$_LDPATH:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/sweep_out"
mkdir -p "${OUT_DIR}"
NTASKS="${SLURM_NTASKS:-8}"
BIN="${GPU_TESTS}/gpu"
JOB_NODES="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-$(hostname -s)}}"
SAFE_NODES="$(echo "${JOB_NODES}" | tr '[],' '_' | tr -s '_' | sed 's/^_//; s/_$//')"

echo "=== clustbench env ==="
echo "host: $(hostname -s)"
echo "OMPI_PREFIX=$OMPI_PREFIX"
echo "UCX_PREFIX=$UCX_PREFIX"
if [[ -d "$UCX_PREFIX/lib" ]]; then
	echo "UCX lib: OK ($UCX_PREFIX/lib)"
else
	echo "UCX lib: (нет каталога — используется системный UCX из libmpi, если есть)"
fi
command -v mpirun && mpirun --version | head -3
"$OMPI_PREFIX/bin/ompi_info" --parsable --all 2>/dev/null | grep -E 'opal_built_with_cuda_support|opal_cuda_support' || true
echo "BIN=$BIN"
test -x "$BIN" || { echo "error: missing or not executable: $BIN"; exit 1; }
ldd "$BIN" | grep -E 'libmpi|libcudart|libnetcdf|libucp|libuct' || true
ldd "$OMPI_PREFIX/lib/libmpi.so" 2>/dev/null | grep -E 'libucp|libuct' || true
echo "======================"

SWEEPS=(
	"8000:64000:8000:8000:80"
)

REPS=(1)
MODES=(
	# auto
	host
)
SCHEMES=(
	one_to_one
	all_to_all
)

for REP in "${REPS[@]}"; do
	for MODE in "${MODES[@]}"; do
		for SCHEME in "${SCHEMES[@]}"; do
			for CASE in "${SWEEPS[@]}"; do
				IFS=":" read -r BYTES_BEGIN BYTES_END BYTES_STEP ITERS WARMUP <<<"${CASE}"

				OUT_PREFIX="${OUT_DIR}/${SAFE_NODES}_rep${REP}_${MODE}_${SCHEME}_bb${BYTES_BEGIN}_be${BYTES_END}_bs${BYTES_STEP}_w${WARMUP}_i${ITERS}"
				echo "RUN rep=${REP} mode=${MODE} scheme=${SCHEME} bytes_begin=${BYTES_BEGIN} bytes_end=${BYTES_END} bytes_step=${BYTES_STEP} warmup=${WARMUP} iters=${ITERS}"
				echo "  -> ${OUT_PREFIX}_{avg,med,min,max,var,std}.nc"

				srun --mpi=pmix -n"${NTASKS}" "${BIN}" \
					--bytes-begin "${BYTES_BEGIN}" \
					--bytes-end "${BYTES_END}" \
					--bytes-step "${BYTES_STEP}" \
					--warmup "${WARMUP}" \
					--iters "${ITERS}" \
					--mode "${MODE}" \
					--scheme "${SCHEME}" \
					--out "${OUT_PREFIX}" \
					-d

				echo "OK  rep=${REP} mode=${MODE} scheme=${SCHEME} bytes_begin=${BYTES_BEGIN} bytes_end=${BYTES_END} bytes_step=${BYTES_STEP}"
			done
		done
	done
done
