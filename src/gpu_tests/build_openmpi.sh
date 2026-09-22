#!/bin/bash

set -euo pipefail

# ==== Настройки — правь тут при необходимости ====
PARTITION="${PARTITION:-intel-a100-pci5}"
BUILD_JOBS="${BUILD_JOBS:-4}"
OMPI_VERSION="${OMPI_VERSION:-5.0.10}"
OMPI_PREFIX="${OMPI_PREFIX:-$HOME/opt/openmpi-${OMPI_VERSION}}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
SRC_DIR="${SRC_DIR:-$HOME/src}"
# ===================================================

# ВАЖНО: OpenMPI нужно скачивать именно с официального сайта
#   https://www.open-mpi.org/software/ompi/
# а НЕ с GitHub. В архивах с GitHub (Source code (tar.gz)) нет
# сгенерированных configure/Makefile.in — там только "сырой" git-репозиторий,
# и для сборки потребуется GNU Autotools + ./autogen.pl, чего обычно нет
# на вычислительных узлах. Официальный release-тарбол уже содержит
# готовый configure и собирается "из коробки".
#
# ВАЖНО: порядок сборки — сначала UCX (build_ucx.sh), и только потом
# OpenMPI, т.к. ниже OpenMPI собирается с --with-ucx="$UCX_PATH" и
# ему нужен уже готовый (собранный и установленный) UCX по этому пути.

OMPI_URL="https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OMPI_VERSION}.tar.gz"
TARBALL="openmpi-${OMPI_VERSION}.tar.gz"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# === Скачивание и распаковка исходников (на логин-узле, GPU не нужен) ===
mkdir -p "${SRC_DIR}"
cd "${SRC_DIR}"

if [ -d "openmpi-${OMPI_VERSION}" ]; then
	echo "Исходники уже распакованы: ${SRC_DIR}/openmpi-${OMPI_VERSION}"
else
	if [ -f "$HOME/${TARBALL}" ]; then
		echo "Использую кэш $HOME/${TARBALL}"
		cp "$HOME/${TARBALL}" .
	else
		echo "Скачиваю (официальный сайт): ${OMPI_URL}"
		wget -q "${OMPI_URL}" -O "${TARBALL}"
		cp "${TARBALL}" "$HOME/${TARBALL}"
	fi
	echo "Распаковываю ${TARBALL} в ${SRC_DIR}"
	tar xzf "${TARBALL}"
fi

# === Сборка — одна srun-джоба на узле с GPU ===
if [ ! -x "${UCX_PATH}/bin/ucx_info" ]; then
	echo "FATAL: не найден собранный UCX в ${UCX_PATH} (сначала запусти build_ucx.sh)"
	exit 1
fi

echo "Запускаю сборку OpenMPI ${OMPI_VERSION} одной джобой (partition=${PARTITION}, gpu=1)"

export OMPI_VERSION OMPI_PREFIX UCX_PATH CUDA_PATH SRC_DIR BUILD_JOBS

srun --partition="${PARTITION}" --nodes=1 --ntasks=1 \
	--cpus-per-task="${BUILD_JOBS}" --gres=gpu:1 --time=01:00:00 \
	--job-name=openmpi-build bash "${SCRIPT_DIR}/_openmpi_build_job.sh"
