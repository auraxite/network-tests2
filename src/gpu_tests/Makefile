include ../../config

# Бинарь: блокирующий one-to-one GPU↔GPU через MPI (см. gpu_one_to_one.cpp).
EXECS = gpu_one_to_one

SRCS = gpu_one_to_one.cpp
OBJS = $(SRCS:.cpp=.o)

# Линковка с CUDA runtime (cudaMalloc, cudaMemcpy, …).
CUDA_LIBS = -lcudart

all: $(EXECS)

gpu_one_to_one: $(OBJS)
	$(MPI_CXXLINKER) $(MPI_CXXLINKER_FLAGS) -o $@ $^ $(CUDA_LIBS)

%.o: %.cpp
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) -I ../core -I ../.. -c $< -o $@

clean:
	rm -f $(OBJS) $(EXECS)

install: all
	cp -rf $(EXECS) $(INSTALL_DIR)/bin/

.PHONY: all clean install
