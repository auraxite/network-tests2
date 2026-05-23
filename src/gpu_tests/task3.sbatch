#!/bin/bash
#SBATCH --job-name=clustbench3
#SBATCH --partition=batch
#SBATCH --time=08:00:00
#SBATCH --output=clustbench3-%j.out
#SBATCH --error=clustbench3-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

ENVS=(
	auto
	host
)
MODES=(
	one_to_one
	all_to_all
)

BIG_CASES=(
	"500000:1500:5"
	"4000000:500:5"
	"16000000:200:5"
)

SMALL_CASES=(
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

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
NTASKS="${SLURM_NTASKS:-8}"
BIN="${GPU_TESTS}/gpu"
JOB_NODES="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-$(hostname -s)}}"
SAFE_NODES="$(echo "${JOB_NODES}" | tr '[],' '_' | tr -s '_' | sed 's/^_//; s/_$//')"

echo "=== clustbench3 env ==="
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
echo "======================="

# $1 — каталог результатов (относительно GPU_TESTS)
# $2 — метка порога: "program" | "0" | "default"
# $3+ — CASES ("bytes:iters:warmup" ...)
run_suite() {
	local out_rel="$1"
	local thresh_mode="$2"
	shift 2
	local -a cases=("$@")

	local out_dir="${GPU_TESTS}/${out_rel}"
	mkdir -p "${out_dir}"

	unset UCX_RNDV_THRESH_LEAVE_DEFAULT
	case "${thresh_mode}" in
	0)
		export UCX_RNDV_THRESH=0
		;;
	default)
		unset UCX_RNDV_THRESH
		export UCX_RNDV_THRESH_LEAVE_DEFAULT=1
		;;
	program|*)
		unset UCX_RNDV_THRESH
		;;
	esac

	echo ""
	echo ">>> suite: ${out_rel}  UCX_RNDV_THRESH_mode=${thresh_mode}"
	echo "    UCX_RNDV_THRESH=${UCX_RNDV_THRESH:-<unset>}"
	echo "    UCX_RNDV_THRESH_LEAVE_DEFAULT=${UCX_RNDV_THRESH_LEAVE_DEFAULT:-<unset>}"

	for ENV in "${ENVS[@]}"; do
		for MODE in "${MODES[@]}"; do
			for CASE in "${cases[@]}"; do
				IFS=":" read -r BYTES ITERS WARMUP <<<"${CASE}"

				OUT_TXT="${out_dir}/${SAFE_NODES}_${ENV}_${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
				echo "RUN env=${ENV} mode=${MODE} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"
				echo "  -> ${OUT_TXT}"

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

				echo "OK  env=${ENV} mode=${MODE} bytes=${BYTES}"
			done
		done
	done
}

# 1) Большие сообщения — порог из программы (по умолчанию 0, если не задан в sbatch)
run_suite results_big program "${BIG_CASES[@]}"

# 2) Малые — явно UCX_RNDV_THRESH=0 для auto и host
run_suite results_small_threshold_0 0 "${SMALL_CASES[@]}"

# 3) Малые — внутренний порог UCX (переменная не задаётся)
run_suite results_small_threshold_default default "${SMALL_CASES[@]}"

echo ""
echo "=== clustbench3 done ==="
