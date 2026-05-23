#!/bin/bash
# task5.sbatch — проверка корректности тайминга после добавления cudaDeviceSynchronize()
# в one_to_one/auto на стороне получателя.
#
# Цель: убедиться, что после синхронизации внутриузловая задержка при 16 МБ
# вырастает (ожидаем ~60–100 мкс вместо ~28 мкс) и эффективная ПС
# оказывается ниже однонаправленного предела NVLink3 (~300 ГБ/с).
#
# Запуск: sbatch task5.sbatch
# Результаты: results_sync_check/

#SBATCH --job-name=clustbench-synccheck
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=clustbench-synccheck-%j.out
#SBATCH --error=clustbench-synccheck-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=1

# Только auto: именно в этой среде cudaDeviceSynchronize() влияет на таймер
ENVS=(
	auto
)
MODES=(
	one_to_one
)
# Несколько крупных размеров, чтобы оценить кривую ПС = size/latency
CASES=(
	"1000000:500:5"
	"2000000:500:5"
	"4000000:300:5"
	"8000000:200:5"
	"16000000:200:5"
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
OUT_DIR="${GPU_TESTS}/results_sync_check"
mkdir -p "${OUT_DIR}"
NTASKS="${SLURM_NTASKS:-8}"
BIN="${GPU_TESTS}/gpu"
JOB_NODES="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-$(hostname -s)}}"
SAFE_NODES="$(echo "${JOB_NODES}" | tr '[],' '_' | tr -s '_' | sed 's/^_//; s/_$//')"

echo "=== clustbench-synccheck env ==="
echo "host: $(hostname -s)"
echo "OMPI_PATH=$OMPI_PATH"
echo "BIN=$BIN"
test -x "$BIN" || { echo "error: missing or not executable: $BIN"; exit 1; }
ldd "$BIN" | grep -E 'libmpi|libcudart|libucp|libuct' || true
echo "==============================="

for ENV in "${ENVS[@]}"; do
	for MODE in "${MODES[@]}"; do
		for CASE in "${CASES[@]}"; do
			IFS=":" read -r BYTES ITERS WARMUP <<<"${CASE}"

			OUT_TXT="${OUT_DIR}/${SAFE_NODES}_${ENV}_${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
			echo "RUN env=${ENV} mode=${MODE} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"
			echo "  -> ${OUT_TXT}"

			srun --mpi=pmix \
				--kill-on-bad-exit=1 \
				--cpu-bind=cores \
				--gpu-bind=closest \
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

echo ""
echo "=== Быстрая проверка медиан ==="
for f in "${OUT_DIR}"/*.txt; do
	echo "--- ${f##*/} ---"
	grep "^pair" "${f}" | awk -F'\t' '
		$1=="pair" && $2 != $3 {
			# колонка 5 = медиана (мкс), $2=src, $3=dst
			printf "  %s -> %s  med=%.1f мкс  BW=%.1f ГБ/с\n",
				$2, $3, $5, ('"${BYTES}"' / ($5 * 1e-6)) / 1e9
		}
	' 2>/dev/null || grep "pair" "${f}" | head -5
done
