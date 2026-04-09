#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

docker run --rm -v "$REPO_ROOT":/network-tests2 -w /network-tests2/src/gpu_tests msu270 bash -lc " \
  mpicc -O2 -fPIC -c ../core/data_write_operations.c -o ../core/data_write_operations.o && \
  mpicc -O2 -fPIC -c ../core/string_id_converters.c -o ../core/string_id_converters.o && \
  mpicxx -O2 gpu_one_to_one.cpp ../core/data_write_operations.o ../core/string_id_converters.o \
    -o gpu_one_to_one -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lnetcdf \
"