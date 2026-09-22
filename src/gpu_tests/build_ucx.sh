#!/bin/bash

set -euo pipefail

# ==== Настройки — правь тут при необходимости ====
PARTITION="${PARTITION:-intel-a100-pci5}"
BUILD_JOBS="${BUILD_JOBS:-4}"
UCX_VERSION="${UCX_VERSION:-1.15.0}"
UCX_PREFIX="${UCX_PREFIX:-$HOME/opt/ucx-${UCX_VERSION}}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
WORK="${BUILD_DIR:-$HOME/build_ucx_tmp}"
# ===================================================

TARBALL="ucx-${UCX_VERSION}.tar.gz"
URL="https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/${TARBALL}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -e "${UCX_PREFIX}" ]; then
	echo "FATAL: ${UCX_PREFIX} уже существует (старая/рабочая сборка)."
	echo "Удали или переименуй вручную и запусти скрипт снова, например:"
	echo "  mv '${UCX_PREFIX}' '${UCX_PREFIX}.bak.\$(date +%s)'"
	exit 1
fi

echo "Запускаю сборку UCX ${UCX_VERSION} одной джобой (partition=${PARTITION}, gpu=1, cpus=${BUILD_JOBS})"

export UCX_VERSION UCX_PREFIX CUDA_PATH WORK TARBALL URL BUILD_JOBS

srun --partition="${PARTITION}" --nodes=1 --ntasks=1 \
	--cpus-per-task="${BUILD_JOBS}" --gres=gpu:1 --time=00:45:00 \
	--job-name=ucx-build bash "${SCRIPT_DIR}/_ucx_build_job.sh"
