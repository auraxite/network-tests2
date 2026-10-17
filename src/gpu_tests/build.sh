#!/bin/sh

set -eu

# ==== Настройки — правь тут при необходимости ====
PARTITION="${PARTITION:-intel-a100-pci5}"
SRUN_NODES="${SRUN_NODES:-1}"
SRUN_TIME="${SRUN_TIME:-00:10:00}"
OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
# ===================================================

SRC="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if [ "${1:-srun}" = srun ]; then
  export OMPI_PATH UCX_PATH CUDA_PATH

  exec srun --partition="$PARTITION" -N"$SRUN_NODES" --gres=gpu:1 \
    --time="$SRUN_TIME" sh "$SRC/build.sh" build

elif [ "${1:-}" != build ]; then
  echo "Usage: sh build.sh [build]" >&2
  echo "  без аргументов: запустить сборку через srun на узле с GPU" >&2
  echo "  build:          собрать прямо здесь" >&2
  exit 1
fi

# === Сборка ===

UCX_LIB=""
[ -d "$UCX_PATH/lib" ] && UCX_LIB="$UCX_PATH/lib"

export PATH="$OMPI_PATH/bin:$PATH"
export LD_LIBRARY_PATH="${UCX_LIB:+$UCX_LIB:}$OMPI_PATH/lib:${LD_LIBRARY_PATH:-}"

MPICXX="$OMPI_PATH/bin/mpicxx"

test -x "$MPICXX" || {
  echo "build.sh: not executable: $MPICXX" >&2
  exit 1
}

if ! test -f "$CUDA_PATH/include/cuda_runtime.h"; then
  echo "build.sh: missing $CUDA_PATH/include/cuda_runtime.h" >&2
  exit 1
fi

echo "$MPICXX"
"$MPICXX" --showme:version 2>/dev/null || true

CXXFLAGS="-O2 -std=c++17 -I$CUDA_PATH/include"

RPATH="${UCX_LIB:+-Wl,-rpath,$UCX_LIB }-Wl,-rpath,$OMPI_PATH/lib -Wl,-rpath,$CUDA_PATH/lib64"

# === Компиляция ===

"$MPICXX" $CXXFLAGS \
  "$SRC/gpu_benchmark.cpp" \
  "$SRC/gpu_common.cpp" \
  "$SRC/gpu_one_to_one.cpp" \
  "$SRC/gpu_all_to_all.cpp" \
  -o "$SRC/gpu" \
  $RPATH \
  -Wl,--no-as-needed \
  -Wl,-rpath-link,"$OMPI_PATH/lib" \
  -L"$OMPI_PATH/lib" -lmpi -lopen-pal \
  -Wl,--as-needed \
  -L"$CUDA_PATH/lib64" -lcudart

echo "OK: $SRC/gpu"

ldd "$SRC/gpu" | grep -E 'libmpi|libmpi_cxx|open-pal|libcudart|libucp|libuct' || true