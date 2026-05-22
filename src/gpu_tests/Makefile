#
# This file is a part of the network-tests2 project.
# GPU-to-GPU latency benchmark (CUDA + MPI).
#

include ../../config

# GPU_MPI_CXX should point to a CUDA-aware MPI build.
# Override on the command line or in config if the system mpicxx is not CUDA-aware:
#   make GPU_MPI_CXX=$(HOME)/opt/openmpi-5.0.10-cuda/bin/mpicxx
GPU_MPI_CXX ?= $(MPI_CXX)

CUDA_INSTALL_PATH ?= /usr/local/cuda

EXECS = gpu

FILES_GPU = \
	gpu_benchmark.o \
	gpu_common.o \
	gpu_one_to_one.o \
	gpu_all_to_all.o

all: $(EXECS)

$(EXECS): $(FILES_GPU)
	$(GPU_MPI_CXX) $(MPI_CXXLINKER_FLAGS) $(MPI_LIB_PATH) \
		-o $(EXECS) $(FILES_GPU) \
		-L$(CUDA_INSTALL_PATH)/lib64 -lcudart

clean:
	rm -f $(FILES_GPU) $(EXECS)

install: all
	cp -rf $(EXECS) $(INSTALL_DIR)/bin/

%.o: %.cpp
	$(GPU_MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) \
		-I$(CUDA_INSTALL_PATH)/include \
		-c $< -o $@
