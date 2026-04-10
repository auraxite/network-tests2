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

# --- Open MPI (CUDA-aware), UCX (пользовательский префикс), NetCDF, CUDA ---
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
# Bus error / SIGBUS в libuct_cuda + ucp_worker_progress + pml_ucx_recv: часто кеш
# регистрации CUDA-памяти в UCX; см. clustbench-141060.err
export UCX_MEMTYPE_CACHE=n
# При ошибках CMA (process_vm_readv / Bad address) можно раскомментировать:
# export UCX_TLS=^cma
# Если всё ещё падает на IB+GPU — попробуй (может ухудшить производительность):
# export UCX_IB_GPU_DIRECT=no

# Если на кластере без этого ломается выбор интерфейсов — раскомментируй блок ниже.
# unset OMPI_MCA_btl_tcp_if_include OMPI_MCA_oob_tcp_if_include
# unset OMPI_MCA_btl_tcp_if_exclude OMPI_MCA_oob_tcp_if_exclude
# export OMPI_MCA_btl=tcp,self,vader
# export OMPI_MCA_btl_tcp_if_exclude=lo,docker0,virbr0
# export OMPI_MCA_oob_tcp_if_exclude=lo,docker0,virbr0

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/grid_out"
mkdir -p "${OUT_DIR}"
NTASKS="${SLURM_NTASKS:-8}"
BIN="${GPU_TESTS}/gpu_one_to_one"
JOB_NODES="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-$(hostname -s)}}"
SAFE_NODES="$(echo "${JOB_NODES}" | tr '[],' '___')"

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

CASES=(
	"500000:1600:80"
	# "2000000:800:40"
	# "8000000:400:20"
	# "32000000:200:10"
)

REPS=(1)
MODES=(
	auto
	host
)

for REP in "${REPS[@]}"; do
	for MODE in "${MODES[@]}"; do
		for CASE in "${CASES[@]}"; do
			IFS=":" read -r BYTES ITERS WARMUP <<<"${CASE}"

			OUT_TXT="${OUT_DIR}/${SAFE_NODES}_rep${REP}_m${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
			echo "RUN rep=${REP} mode=${MODE} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"
			echo "  -> ${OUT_TXT}"

			srun --mpi=pmix -n"${NTASKS}" "${BIN}" \
				--bytes "${BYTES}" \
				--warmup "${WARMUP}" \
				--iters "${ITERS}" \
				--mode "${MODE}" \
				--out "${OUT_TXT}"

			echo "OK  rep=${REP} mode=${MODE} bytes=${BYTES}"
		done
	done
done
