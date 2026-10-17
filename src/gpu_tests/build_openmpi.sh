#!/bin/bash
# Сборка Open MPI одной srun-джобой на узле с GPU (тело — _openmpi_build_job.sh).
# Сначала нужен UCX (build_ucx.sh). Исходники — официальный архив
# https://www.open-mpi.org/software/ompi/ (не с GitHub: там нет готового
# configure), распакованный в $SRC_DIR.

set -euo pipefail

PARTITION="${PARTITION:-intel-a100-pci5}"
BUILD_JOBS="${BUILD_JOBS:-16}"
OMPI_VERSION="${OMPI_VERSION:-5.0.10}"
OMPI_PREFIX="${OMPI_PREFIX:-$HOME/opt/openmpi-${OMPI_VERSION}}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
SRC_DIR="${SRC_DIR:-$HOME/src}"

[ -x "$UCX_PATH/bin/ucx_info" ] || { echo "FATAL: нет UCX в $UCX_PATH (сначала build_ucx.sh)"; exit 1; }
[ -d "$SRC_DIR/openmpi-$OMPI_VERSION" ] || { echo "FATAL: нет исходников $SRC_DIR/openmpi-$OMPI_VERSION"; exit 1; }

echo "Сборка Open MPI $OMPI_VERSION (partition=$PARTITION, cpus=$BUILD_JOBS)"
export OMPI_VERSION OMPI_PREFIX UCX_PATH CUDA_PATH SRC_DIR BUILD_JOBS
srun --partition="$PARTITION" --nodes=1 --ntasks=1 \
	--cpus-per-task="$BUILD_JOBS" --gres=gpu:1 --time=02:00:00 \
	--job-name=openmpi-build bash "$(dirname "$0")/_openmpi_build_job.sh"
