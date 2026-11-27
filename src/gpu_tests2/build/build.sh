#!/bin/bash
# Сборка программы gpu2 (C): bash build/build.sh (сам отправляет сборку через srun на узел с GPU).
# Open MPI и UCX — те же, что у gpu_tests (собираются ../gpu_tests/build/build_ucx.sh и build_openmpi.sh).

set -euo pipefail

source "$(dirname "$0")/../config.sh"   # раздел, пути и имя программы

SRUN_TIME="${SRUN_TIME:-00:10:00}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # корень gpu_tests2: исходники в src/, программа — здесь

case "${1:-srun}" in
srun)
	exec srun --partition="$PARTITION" -N1 --gres=gpu:1 --time="$SRUN_TIME" \
		bash "$ROOT/build/build.sh" build ;;
build) ;;
*)
	echo "Usage: bash build/build.sh [build]" >&2
	exit 1 ;;
esac

MPICC="$OMPI_PATH/bin/mpicc"
[ -x "$MPICC" ] || { echo "build.sh: нет $MPICC" >&2; exit 1; }
[ -f "$CUDA_PATH/include/cuda_runtime.h" ] || { echo "build.sh: нет $CUDA_PATH/include/cuda_runtime.h" >&2; exit 1; }

UCX_LIB=""
[ -d "$UCX_PATH/lib" ] && UCX_LIB="$UCX_PATH/lib"
export PATH="$OMPI_PATH/bin:$PATH"
export LD_LIBRARY_PATH="${UCX_LIB:+$UCX_LIB:}$OMPI_PATH/lib:${LD_LIBRARY_PATH:-}"
RPATH="${UCX_LIB:+-Wl,-rpath,$UCX_LIB }-Wl,-rpath,$OMPI_PATH/lib -Wl,-rpath,$CUDA_PATH/lib64"

# Имя переменной UCX для типа памяти промежуточных буферов передачи частями (pipeline). В разных
# версиях UCX оно разное (UCX_RNDV_FRAG_MEM_TYPE или UCX_RNDV_FRAG_MEM_TYPES), поэтому берётся из
# списка параметров той сборки UCX, с которой собирается программа. Нужна среде host.
FRAG_VAR=""
if [ -x "$UCX_PATH/bin/ucx_info" ]; then
	FRAG_VAR=$("$UCX_PATH/bin/ucx_info" -c 2>/dev/null | grep -oE '^UCX_RNDV_FRAG_MEM_TYPES?=' | head -1 | tr -d '=' || true)
fi
if [ -n "$FRAG_VAR" ]; then
	echo "UCX $UCX_VERSION: тип памяти буферов pipeline — $FRAG_VAR ($("$UCX_PATH/bin/ucx_info" -c | grep "^$FRAG_VAR="))"
else
	echo "WARNING: в ucx_info -c нет UCX_RNDV_FRAG_MEM_TYPE(S): среда host не сможет запретить буферы pipeline в VRAM" >&2
fi

"$MPICC" --showme:version 2>/dev/null || true
"$MPICC" -O2 -std=gnu11 -Wall -Wextra -I"$CUDA_PATH/include" \
	-DUCX_FRAG_MEM_VAR="\"$FRAG_VAR\"" \
	"$ROOT/src/gpu_benchmark.c" "$ROOT/src/gpu_common.c" \
	"$ROOT/src/gpu_one_to_one.c" "$ROOT/src/gpu_simultaneous.c" \
	-o "$ROOT/$GPU_BIN" \
	$RPATH \
	-Wl,--no-as-needed -Wl,-rpath-link,"$OMPI_PATH/lib" \
	-L"$OMPI_PATH/lib" -lmpi -lopen-pal \
	-Wl,--as-needed -L"$CUDA_PATH/lib64" -lcudart -lm

echo "OK: $ROOT/$GPU_BIN (C, Open MPI $OMPI_VERSION, UCX $UCX_VERSION)"
ldd "$ROOT/$GPU_BIN" | grep -E 'libmpi|open-pal|libcudart|libucp|libuct' || true
