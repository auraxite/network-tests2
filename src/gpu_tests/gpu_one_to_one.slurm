#!/bin/bash
#SBATCH --job-name=gpu1to1
#SBATCH --partition=batch
#SBATCH --time=00:02:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2

set -euo pipefail

init_modules() {
	if type module >/dev/null 2>&1; then
		return 0
	fi
	local f
	for f in /etc/profile.d/modules.sh \
		/usr/share/modules/init/bash \
		/usr/share/lmod/lmod/init/bash \
		/opt/ohpc/pub/libs/lmod/lmod/init/bash; do
		if [[ -f "$f" ]]; then
			# shellcheck source=/dev/null
			source "$f"
			break
		fi
	done
}

ensure_cuda_home() {
	if [[ -n "${CUDA_HOME:-}" && -f "${CUDA_HOME}/include/cuda_runtime.h" ]]; then
		export CUDA_HOME
		return 0
	fi
	if [[ -n "${CUDA_PATH:-}" && -f "${CUDA_PATH}/include/cuda_runtime.h" ]]; then
		export CUDA_HOME="${CUDA_PATH}"
		return 0
	fi
	init_modules
	if type module >/dev/null 2>&1; then
		local m
		for m in cuda default_cuda cuda/toolkit nvhpc nvhpc-hpcx \
			cuda/12.4 cuda/12.3 cuda/12.2 cuda/12.1 cuda/12.0 cuda/11.8; do
			module load "$m" 2>/dev/null || continue
			if [[ -n "${CUDA_HOME:-}" && -f "${CUDA_HOME}/include/cuda_runtime.h" ]]; then
				return 0
			fi
			if [[ -n "${CUDA_PATH:-}" && -f "${CUDA_PATH}/include/cuda_runtime.h" ]]; then
				export CUDA_HOME="${CUDA_PATH}"
				return 0
			fi
		done
	fi
	local d
	for d in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11 /opt/cuda; do
		if [[ -f "$d/include/cuda_runtime.h" ]]; then
			export CUDA_HOME="$d"
			return 0
		fi
	done
	return 1
}

if ! ensure_cuda_home; then
	echo "CUDA toolkit not found (cuda_runtime.h). Tried: module + common paths." >&2
	type module >/dev/null 2>&1 && module list >&2 || echo "module: not available" >&2
	echo "Fix: sbatch --export=ALL,CUDA_HOME=/path/to/cuda  OR  module load <cuda> on login, then echo \\\$CUDA_HOME" >&2
	exit 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${ROOT}/Makefile" && -f "${ROOT}/gpu_one_to_one.cpp" ]]; then
	GPU_TESTS="${ROOT}"
elif [[ -f "${ROOT}/src/gpu_tests/Makefile" && -f "${ROOT}/src/gpu_tests/gpu_one_to_one.cpp" ]]; then
	GPU_TESTS="${ROOT}/src/gpu_tests"
else
	echo "cannot find gpu_tests (Makefile + gpu_one_to_one.cpp); SLURM_SUBMIT_DIR=${ROOT}" >&2
	exit 1
fi
BIN="${GPU_TESTS}/gpu_one_to_one"
cd "${GPU_TESTS}"
if [[ ! -x "${BIN}" ]]; then
	make clean 2>/dev/null || true
	make CUDA_HOME="${CUDA_HOME}"
fi
if [[ ! -x "${BIN}" ]]; then
	echo "build failed: ${BIN} after make CUDA_HOME=${CUDA_HOME}" >&2
	exit 1
fi
OUT_TXT="${ROOT}/bench-${SLURM_JOB_ID}.txt"
srun --mpi=pmix -n8 "${BIN}" \
	--bytes $((4 * 1024 * 1024)) \
	--warmup 10 \
	--iters 50 \
	--mode auto \
	--out "${OUT_TXT}"
