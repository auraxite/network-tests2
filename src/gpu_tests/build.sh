#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_GPU_TESTS="$SCRIPT_DIR"
SELF="$SCRIPT_DIR/build.sh"

OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10-cuda}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"

case "${1:-}" in
srun|remote)
	SRUN_NODES="${SRUN_NODES:-1}"
	SRUN_GPUS="${SRUN_GPUS:-1}"
	SRUN_TIME="${SRUN_TIME:-00:05:00}"
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
esac

export PATH="$OMPI_PATH/bin:$PATH"
if [ -d "$UCX_PATH/lib" ]; then
	export LD_LIBRARY_PATH="$UCX_PATH/lib:$OMPI_PATH/lib:${LD_LIBRARY_PATH:-}"
else
	export LD_LIBRARY_PATH="$OMPI_PATH/lib:${LD_LIBRARY_PATH:-}"
fi

MPICXX="$OMPI_PATH/bin/mpicxx"
test -x "$MPICXX" || { echo "build.sh: not executable: $MPICXX" >&2; exit 1; }

if ! test -f "$CUDA_PATH/include/cuda_runtime.h"; then
	echo "build.sh: missing $CUDA_PATH/include/cuda_runtime.h" >&2
	echo "  На login-узле часто нет CUDA headers — запусти: sh build.sh srun" >&2
	exit 1
fi

echo "$MPICXX"
"$MPICXX" --showme:version 2>/dev/null || true

CXXFLAGS="-O2 -I$CUDA_PATH/include"
RPATH="-Wl,-rpath,$OMPI_PATH/lib -Wl,-rpath,$CUDA_PATH/lib64"
if [ -d "$UCX_PATH/lib" ]; then
	RPATH="-Wl,-rpath,$UCX_PATH/lib $RPATH"
fi

"$MPICXX" $CXXFLAGS \
	"$REPO_GPU_TESTS/gpu_benchmark.cpp" \
	"$REPO_GPU_TESTS/gpu_common.cpp" \
	"$REPO_GPU_TESTS/gpu_one_to_one.cpp" \
	"$REPO_GPU_TESTS/gpu_one_to_one_node.cpp" \
	"$REPO_GPU_TESTS/gpu_all_to_all.cpp" \
	-o "$REPO_GPU_TESTS/gpu" \
	$RPATH \
	-Wl,--no-as-needed \
	-L"$OMPI_PATH/lib" -lmpi -lopen-pal \
	-Wl,--as-needed \
	-L"$CUDA_PATH/lib64" -lcudart

echo "OK: $REPO_GPU_TESTS/gpu"
ldd "$REPO_GPU_TESTS/gpu" | grep -E 'libmpi|libmpi_cxx|open-pal|libcudart|libucp|libuct' || true
