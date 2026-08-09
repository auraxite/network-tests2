#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_GPU_TESTS="$SCRIPT_DIR"
SELF="$SCRIPT_DIR/build.sh"

OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10-cuda-clean}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"

case "${1:-srun}" in
srun)
	SRUN_NODES="${SRUN_NODES:-1}"
	SRUN_GPUS="${SRUN_GPUS:-1}"
	SRUN_TIME="${SRUN_TIME:-00:01:00}"
	SRUN_PART=""
	if [ -n "${SRUN_PARTITION:-}" ]; then
		SRUN_PART="-p $SRUN_PARTITION"
	fi
	exec srun $SRUN_PART -N"${SRUN_NODES}" --gpus="${SRUN_GPUS}" --time="${SRUN_TIME}" bash -lc \
		"export OMPI_PATH=\"$OMPI_PATH\"; export CUDA_PATH=\"$CUDA_PATH\"; export UCX_PATH=\"$UCX_PATH\"; cd \"$REPO_GPU_TESTS\" && exec sh \"$SELF\" build"
	;;
build)
	shift
	;;
*)
	echo "Usage: sh build.sh [build]" >&2
	echo "  no args: submit build via srun" >&2
	exit 1
	;;
esac

export PATH="$OMPI_PATH/bin:$PATH"
if [ -d "$UCX_PATH/lib" ]; then
	export LD_LIBRARY_PATH="$UCX_PATH/lib:$OMPI_PATH/lib:${LD_LIBRARY_PATH:-}"
else
	export LD_LIBRARY_PATH="$OMPI_PATH/lib:${LD_LIBRARY_PATH:-}"
fi

MPICXX="$OMPI_PATH/bin/mpicxx"
test -x "$MPICXX" || { echo "build.sh: not executable: $MPICXX" >&2; exit 1; }
NVCC="$CUDA_PATH/bin/nvcc"
test -x "$NVCC" || { echo "build.sh: not executable: $NVCC" >&2; exit 1; }

if ! test -f "$CUDA_PATH/include/cuda_runtime.h"; then
	echo "build.sh: missing $CUDA_PATH/include/cuda_runtime.h" >&2
	echo "  На login-узле часто нет CUDA headers — запусти: sh build.sh" >&2
	exit 1
fi

echo "$MPICXX"
"$MPICXX" --showme:version 2>/dev/null || true

CXXFLAGS="-O2 -std=c++17 -I$CUDA_PATH/include"
RPATH="-Wl,-rpath,$OMPI_PATH/lib -Wl,-rpath,$CUDA_PATH/lib64"
if [ -d "$UCX_PATH/lib" ]; then
	RPATH="-Wl,-rpath,$UCX_PATH/lib $RPATH"
fi

CUDA_OBJS="$REPO_GPU_TESTS/gpu_cuda_one_to_one.o $REPO_GPU_TESTS/gpu_cuda_all_to_all.o"
trap 'rm -f $CUDA_OBJS' EXIT
"$NVCC" $CXXFLAGS -ccbin "$MPICXX" -I"$OMPI_PATH/include" \
	-c "$REPO_GPU_TESTS/gpu_cuda_one_to_one.cu" \
	-o "$REPO_GPU_TESTS/gpu_cuda_one_to_one.o"
"$NVCC" $CXXFLAGS -ccbin "$MPICXX" -I"$OMPI_PATH/include" \
	-c "$REPO_GPU_TESTS/gpu_cuda_all_to_all.cu" \
	-o "$REPO_GPU_TESTS/gpu_cuda_all_to_all.o"

"$MPICXX" $CXXFLAGS \
	"$REPO_GPU_TESTS/gpu_benchmark.cpp" \
	"$REPO_GPU_TESTS/gpu_common.cpp" \
	"$REPO_GPU_TESTS/gpu_one_to_one.cpp" \
	"$REPO_GPU_TESTS/gpu_all_to_all.cpp" \
	$CUDA_OBJS \
	-o "$REPO_GPU_TESTS/gpu" \
	$RPATH \
	-Wl,--no-as-needed \
	-Wl,-rpath-link,"$OMPI_PATH/lib" \
	-L"$OMPI_PATH/lib" -lmpi -lopen-pal \
	-Wl,--as-needed \
	-L"$CUDA_PATH/lib64" -lcudart

echo "OK: $REPO_GPU_TESTS/gpu"
ldd "$REPO_GPU_TESTS/gpu" | grep -E 'libmpi|libmpi_cxx|open-pal|libcudart|libucp|libuct' || true
