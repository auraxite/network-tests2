#!/bin/bash
# Сборка Open MPI на узле с GPU. Запускается из build_openmpi.sh

# Перед сборкой вносится одна правка в pml ucx: публиковать один полный адрес
# UCX для всех процессов. Без неё в задании srun на нескольких узлах пары GPU
# одного узла передают данные через сетевой адаптер.

set -euo pipefail

cd "$SRC_DIR/openmpi-$OMPI_VERSION"
echo "host=$(hostname -s) OMPI_PREFIX=$OMPI_PREFIX UCX_PATH=$UCX_PATH CUDA_PATH=$CUDA_PATH"
make distclean || true

# --- Правка pml_ucx.c: заменить одну строку (или вернуть исходную) ----------
PML_UCX="ompi/mca/pml/ucx/pml_ucx.c"
ORIG_LINE='#if !HAVE_UCP_WORKER_ADDRESS_FLAGS'
PATCH_LINE='#if 1 /* clustbench: полный адрес UCX для всех процессов */'

if [ "${OMPI_PATCH_FULLADDR:-1}" = 1 ]; then
	from=$ORIG_LINE; to=$PATCH_LINE
else
	from=$PATCH_LINE; to=$ORIG_LINE
fi

if grep -qF "$to" "$PML_UCX"; then
	echo "pml_ucx.c: уже в нужном состоянии"
elif [ "$(grep -cF "$from" "$PML_UCX")" = 1 ]; then
	line=$(grep -nF "$from" "$PML_UCX" | cut -d: -f1)
	sed -i "${line}s|.*|$to|" "$PML_UCX"
	echo "pml_ucx.c, строка $line: $to"
else
	echo "FATAL: в $PML_UCX не найдена ровно одна строка '$from'"
	exit 1
fi

./configure \
	--prefix="$OMPI_PREFIX" \
	--with-ucx="$UCX_PATH" \
	--with-cuda="$CUDA_PATH" \
	--with-slurm \
	--disable-mpi-fortran \
	--without-hcoll \
	--enable-mca-no-build=coll-hcoll

make -j"$BUILD_JOBS"
make install
echo "Готово. Open MPI установлен в: $OMPI_PREFIX"
