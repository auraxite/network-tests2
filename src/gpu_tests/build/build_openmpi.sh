#!/bin/bash
# Сборка Open MPI одной srun-джобой на узле с GPU (тело — _openmpi_build_job.sh).
# Сначала нужен UCX (build_ucx.sh). Исходники — официальный архив
# https://www.open-mpi.org/software/ompi/ (не с GitHub: там нет готового
# configure), распакованный в $SRC_DIR.

set -euo pipefail

source "$(dirname "$0")/../config.sh"   # раздел, версии и пути — общие

# --- Настройки этого скрипта ---
BUILD_JOBS="${BUILD_JOBS:-16}"
SRC_DIR="${SRC_DIR:-$HOME/src}"

[ -x "$UCX_PATH/bin/ucx_info" ] || { echo "FATAL: нет UCX в $UCX_PATH (сначала build_ucx.sh)"; exit 1; }
[ -d "$SRC_DIR/openmpi-$OMPI_VERSION" ] || { echo "FATAL: нет исходников $SRC_DIR/openmpi-$OMPI_VERSION"; exit 1; }
# Не ставить поверх готовой сборки: она собрана с другим UCX или уже работает
if [ -e "$OMPI_PATH/bin/mpicc" ]; then
	echo "FATAL: $OMPI_PATH уже существует (Open MPI для UCX $UCX_VERSION). Убери старую сборку, например:"
	echo "  mv '$OMPI_PATH' '$OMPI_PATH.bak.\$(date +%s)'"
	exit 1
fi

echo "Сборка Open MPI $OMPI_VERSION с UCX $UCX_VERSION в $OMPI_PATH (partition=$PARTITION, cpus=$BUILD_JOBS)"
export OMPI_VERSION OMPI_PREFIX="$OMPI_PATH" UCX_PATH CUDA_PATH SRC_DIR BUILD_JOBS
srun --partition="$PARTITION" --nodes=1 --ntasks=1 \
	--cpus-per-task="$BUILD_JOBS" --gres=gpu:1 --time=02:00:00 \
	--job-name=openmpi-build bash "$(dirname "$0")/_openmpi_build_job.sh"
