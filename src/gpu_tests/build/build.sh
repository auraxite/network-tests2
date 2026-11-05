#!/bin/bash
# Сборка программы gpu: bash build/build.sh (сам отправляет сборку через srun на узел с GPU)

set -euo pipefail

source "$(dirname "$0")/../config.sh"   # раздел и пути — общие

# --- Настройки этого скрипта ---
SRUN_TIME="${SRUN_TIME:-00:10:00}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # корень gpu_tests: исходники в src/, программа gpu — здесь

case "${1:-srun}" in
srun)
	exec srun --partition="$PARTITION" -N1 --gres=gpu:1 --time="$SRUN_TIME" \
		bash "$ROOT/build/build.sh" build ;;
build) ;;
*)
	echo "Usage: bash build/build.sh [build]" >&2
	exit 1 ;;
esac

MPICXX="$OMPI_PATH/bin/mpicxx"
[ -x "$MPICXX" ] || { echo "build.sh: нет $MPICXX" >&2; exit 1; }
[ -f "$CUDA_PATH/include/cuda_runtime.h" ] || { echo "build.sh: нет $CUDA_PATH/include/cuda_runtime.h" >&2; exit 1; }

UCX_LIB=""
[ -d "$UCX_PATH/lib" ] && UCX_LIB="$UCX_PATH/lib"
export PATH="$OMPI_PATH/bin:$PATH"
export LD_LIBRARY_PATH="${UCX_LIB:+$UCX_LIB:}$OMPI_PATH/lib:${LD_LIBRARY_PATH:-}"
RPATH="${UCX_LIB:+-Wl,-rpath,$UCX_LIB }-Wl,-rpath,$OMPI_PATH/lib -Wl,-rpath,$CUDA_PATH/lib64"

"$MPICXX" --showme:version 2>/dev/null || true
"$MPICXX" -O2 -std=c++17 -I"$CUDA_PATH/include" \
	"$ROOT/src/gpu_benchmark.cpp" "$ROOT/src/gpu_common.cpp" \
	"$ROOT/src/gpu_one_to_one.cpp" "$ROOT/src/gpu_simultaneous.cpp" \
	-o "$ROOT/$GPU_BIN" \
	$RPATH \
	-Wl,--no-as-needed -Wl,-rpath-link,"$OMPI_PATH/lib" \
	-L"$OMPI_PATH/lib" -lmpi -lopen-pal \
	-Wl,--as-needed -L"$CUDA_PATH/lib64" -lcudart

echo "OK: $ROOT/$GPU_BIN (Open MPI $OMPI_VERSION, UCX $UCX_VERSION)"
ldd "$ROOT/$GPU_BIN" | grep -E 'libmpi|open-pal|libcudart|libucp|libuct' || true
