#!/bin/bash
# Тело сборки OpenMPI. Выполняется внутри srun-джобы на узле с GPU,
# запускается из build_openmpi.sh (там же экспортируются переменные ниже).
# Отдельным файлом — чтобы не терять подсветку синтаксиса в редакторе.

set -euo pipefail

cd "$SRC_DIR/openmpi-$OMPI_VERSION"

echo "host=$(hostname -s) OMPI_PREFIX=$OMPI_PREFIX UCX_PATH=$UCX_PATH CUDA_PATH=$CUDA_PATH"

make distclean || true

# --- Правка pml ucx: полный адрес UCX для всех процессов -----------------
# Open MPI 5 публикует для процессов других узлов адрес UCX только с сетевыми
# транспортами (UCP_WORKER_ADDRESS_FLAG_NET_ONLY), а полный (с cuda_ipc, sm) —
# только для процессов своего узла. При запуске через srun --mpi=pmix в
# задании на нескольких узлах соседи по узлу получают сокращённый адрес, не
# видят cuda_ipc, и UCX передаёт данные между GPU узла через сетевой адаптер:
# на g5500 16 МБ за ~830 мкс вместо ~72 мкс в задании на одном узле.
# Правка включает ветку, которой Open MPI пользуется со старыми UCX: один
# полный адрес для всех. Отключить: OMPI_PATCH_FULLADDR=0 bash build_openmpi.sh
PML_UCX="ompi/mca/pml/ucx/pml_ucx.c"
ORIG_LINE='#if !HAVE_UCP_WORKER_ADDRESS_FLAGS'
PATCH_LINE='#if 1 /* clustbench: полный адрес UCX для всех процессов */'
if [ "${OMPI_PATCH_FULLADDR:-1}" = 1 ]; then
	if grep -qF "$PATCH_LINE" "$PML_UCX"; then
		echo "pml_ucx.c: правка уже внесена"
	elif [ "$(grep -cF "$ORIG_LINE" "$PML_UCX")" = 1 ]; then
		sed -i "s|^$ORIG_LINE.*|$PATCH_LINE|" "$PML_UCX"
		echo "pml_ucx.c: правка внесена"
	else
		echo "FATAL: в $PML_UCX не найдена ровно одна строка '$ORIG_LINE'"
		exit 1
	fi
	echo "--- $PML_UCX вокруг правки:"
	line=$(grep -nF "$PATCH_LINE" "$PML_UCX" | cut -d: -f1)
	sed -n "$((line - 3)),$((line + 25))p" "$PML_UCX"
	echo "---"
elif grep -qF "$PATCH_LINE" "$PML_UCX"; then
	line=$(grep -nF "$PATCH_LINE" "$PML_UCX" | cut -d: -f1)
	sed -i "${line}s|.*|$ORIG_LINE|" "$PML_UCX"
	echo "pml_ucx.c: правка откатана"
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

echo "Готово. OpenMPI установлен в: $OMPI_PREFIX"
