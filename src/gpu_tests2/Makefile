#
# This file is a part of the network-tests2 project.
# GPU-to-GPU latency benchmark, C version (CUDA + MPI).
# На кластере удобнее build/build.sh: он сам подставляет пути Open MPI, UCX и CUDA и находит
# имя переменной UCX для среды host.
#

include ../../config

# GPU_MPI_CC — mpicc CUDA-aware сборки MPI.
#   make GPU_MPI_CC=$(HOME)/opt/openmpi-5.0.10-ucx-1.22.0/bin/mpicc UCX_FRAG_MEM_VAR=UCX_RNDV_FRAG_MEM_TYPES
GPU_MPI_CC ?= $(MPI_CC)
CUDA_INSTALL_PATH ?= /usr/local/cuda
UCX_FRAG_MEM_VAR ?=

EXECS = gpu2

vpath %.c src

FILES_GPU = \
	gpu_benchmark.o \
	gpu_common.o \
	gpu_one_to_one.o \
	gpu_simultaneous.o

all: $(EXECS)

$(EXECS): $(FILES_GPU)
	$(GPU_MPI_CC) $(MPI_CCLINKER_FLAGS) $(MPI_LIB_PATH) \
		-o $(EXECS) $(FILES_GPU) \
		-L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lm

clean:
	rm -f $(FILES_GPU) $(EXECS)

install: all
	cp -rf $(EXECS) $(INSTALL_DIR)/bin/

%.o: %.c
	$(GPU_MPI_CC) $(MPI_CC_FLAGS) $(MPI_CC_INCLUDE) -std=gnu11 \
		-I$(CUDA_INSTALL_PATH)/include -DUCX_FRAG_MEM_VAR='"$(UCX_FRAG_MEM_VAR)"' \
		-c $< -o $@
