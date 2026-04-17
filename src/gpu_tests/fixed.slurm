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
ENVS=(
	auto
	host
)
MODES=(
	one_to_one
	# all_to_all
)
CASES=( # "bytes:iters:warmup"
	"500000:6400:64"
	# "4000000:1600:16"
	# "32000000:400:4"
)
TIMERS=(
	all
	mpi
	cpu
	cuda
)
SIDES=(
	sender
	receiver
)

OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10-cuda}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"
NETCDF_PATH="${NETCDF_PATH:-$HOME/opt/netcdf-c-4.9.2}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

set -euo pipefail

# Запуск в одной Slurm-задаче:
# - по всем таймерам/сторонам (по умолчанию),
# - либо только по одному, если передать TIMER и/или SIDE через env.
if [[ -n "${TIMER:-}" ]]; then
	TIMERS=("${TIMER}")
fi
if [[ -n "${SIDE:-}" ]]; then
	case "${SIDE}" in
		both)
			SIDES=(sender receiver)
			;;
		sender|receiver)
			SIDES=("${SIDE}")
			;;
		*)
			echo "error: SIDE must be sender|receiver|both, got '${SIDE}'" >&2
			exit 1
			;;
	esac
fi

export PATH="$OMPI_PATH/bin:$NETCDF_PATH/bin:$PATH"
_LDPATH="$OMPI_PATH/lib:$NETCDF_PATH/lib:${CUDA_PATH}/lib64"
if [[ -d "$UCX_PATH/lib" ]]; then
	_LDPATH="$UCX_PATH/lib:$_LDPATH"
fi
export LD_LIBRARY_PATH="$_LDPATH:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/fixed"
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
			for TIMER in "${TIMERS[@]}"; do
				for SIDE in "${SIDES[@]}"; do
					for CASE in "${CASES[@]}"; do
						IFS=":" read -r BYTES ITERS WARMUP <<<"${CASE}"

						OUT_TXT="${OUT_DIR}/${SAFE_NODES}_rep${REP}_${ENV}_${MODE}_t${TIMER}_size${SIDE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
						echo "RUN rep=${REP} env=${ENV} mode=${MODE} timer=${TIMER} size=${SIDE} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"
						echo "  -> ${OUT_TXT}"

						srun --mpi=pmix -n"${NTASKS}" "${BIN}" \
							--bytes "${BYTES}" \
							--warmup "${WARMUP}" \
							--iters "${ITERS}" \
							--env "${ENV}" \
							--mode "${MODE}" \
							--timer "${TIMER}" \
							--side "${SIDE}" \
							--out "${OUT_TXT}" \
							-d

						echo "OK  rep=${REP} env=${ENV} mode=${MODE} timer=${TIMER} size=${SIDE} bytes=${BYTES}"
					done
				done
			done
		done
	done
done
