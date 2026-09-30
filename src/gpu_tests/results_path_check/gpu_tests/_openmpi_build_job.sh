#!/bin/bash
# Тело сборки OpenMPI. Выполняется внутри srun-джобы на узле с GPU,
# запускается из build_openmpi.sh (там же экспортируются переменные ниже).
# Отдельным файлом — чтобы не терять подсветку синтаксиса в редакторе.

set -euo pipefail

cd "$SRC_DIR/openmpi-$OMPI_VERSION"

echo "host=$(hostname -s) OMPI_PREFIX=$OMPI_PREFIX UCX_PATH=$UCX_PATH CUDA_PATH=$CUDA_PATH"

make distclean || true

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

echo "Готово. OpenMPI установлен в: $OMPI_PREFIX"
