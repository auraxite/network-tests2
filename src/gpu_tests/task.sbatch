#!/bin/bash
#SBATCH --job-name=clustbench
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --output=clustbench-%j.out
#SBATCH --error=clustbench-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

REPS=(1) # задать число
ENVS=( # закомментировать лишнее
	auto
	host
)
MODES=( # закомментировать лишнее
	one_to_one
	# all_to_all
)
CASES=( # "bytes:iters:warmup", закомментировать лишнее
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
	
	# "500000:1500:5"
	# "4000000:500:5"
	# "16000000:200:5"
)

OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10-cuda}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

set -eEuo pipefail

on_error() {
	local exit_code=$?
	echo "FATAL: benchmark step failed (exit=${exit_code}). Stopping job immediately." >&2
	if [[ -n "${SLURM_JOB_ID:-}" ]]; then
		scancel "${SLURM_JOB_ID}" >/dev/null 2>&1 || true
	fi
	exit "${exit_code}"
}
trap on_error ERR

export PATH="$OMPI_PATH/bin:$PATH"
_LDPATH="$OMPI_PATH/lib:${CUDA_PATH}/lib64"
if [[ -d "$UCX_PATH/lib" ]]; then
	_LDPATH="$UCX_PATH/lib:$_LDPATH"
fi
export LD_LIBRARY_PATH="$_LDPATH:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl_vader_single_copy_mechanism=cma
# UCX_TLS, UCX_IB_GPU_DIRECT_RDMA, UCX_RNDV_SCHEME выставляются программой
# до MPI_Init на основе --env: auto → cuda_copy+cuda_ipc+IB+GDR, host → только IB+cma.
# Чтобы переопределить — раскомментируй нужное:
# export UCX_TLS=rc_mlx5,ud_mlx5,dc_mlx5,cuda_copy,cuda_ipc,cma,sm,self
# export UCX_IB_GPU_DIRECT_RDMA=yes
# export UCX_RNDV_THRESH=8192   # default UCX threshold (~8–16 КБ); не трогай если
#                                # хочешь видеть естественный переход eager→rendezvous


GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/results"
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
ldd "$BIN" | grep -E 'libmpi|libcudart|libucp|libuct' || true
ldd "$OMPI_PATH/lib/libmpi.so" 2>/dev/null | grep -E 'libucp|libuct' || true
if command -v ucx_info >/dev/null 2>&1; then
	echo "UCX transports with cma:"
	ucx_info -d 2>/dev/null | grep -E '^# *Transport:' | grep -i cma || \
		echo "  WARNING: cma не найден в ucx_info -d (UCX собран без CMA?)"
else
	echo "ucx_info: не в PATH (диагностику CMA пропускаем)"
fi

echo "yama.ptrace_scope на узлах задания:"
srun --ntasks-per-node=1 -n"${SLURM_JOB_NUM_NODES:-1}" \
	bash -c 'echo "  $(hostname -s): $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo n/a)"' \
	2>/dev/null || true
echo "======================"

for REP in "${REPS[@]}"; do
	for ENV in "${ENVS[@]}"; do
		for MODE in "${MODES[@]}"; do
				for CASE in "${CASES[@]}"; do
				IFS=":" read -r BYTES ITERS WARMUP <<<"${CASE}"

				OUT_TXT="${OUT_DIR}/${SAFE_NODES}_rep${REP}_${ENV}_${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
				echo "RUN rep=${REP} env=${ENV} mode=${MODE} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"
				echo "  -> ${OUT_TXT}"

				echo "  launch: rank-per-gpu mode (${NTASKS} MPI rank(s))"
				srun --mpi=pmix \
					--kill-on-bad-exit=1 \
					--cpu-bind=cores \
					--gpu-bind=none \
					-n"${NTASKS}" "${BIN}" \
					--bytes "${BYTES}" \
					--warmup "${WARMUP}" \
					--iters "${ITERS}" \
					--env "${ENV}" \
					--mode "${MODE}" \
					--out "${OUT_TXT}"

				echo "OK  rep=${REP} env=${ENV} mode=${MODE} bytes=${BYTES}"
			done
		done
	done
done