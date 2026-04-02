MPI_CXX ?= mpicxx
MPI_CXXFLAGS ?= -O2 -g

CUDA_ROOT := $(if $(CUDA_HOME),$(CUDA_HOME),$(CUDA_PATH))
CUDA_RPATH ?= 0
ifneq ($(strip $(CUDA_ROOT)),)
CUDA_CPPFLAGS := -I$(CUDA_ROOT)/include
ifneq ($(wildcard $(CUDA_ROOT)/lib64),)
ifneq ($(strip $(CUDA_RPATH)),0)
CUDA_LDFLAGS := -L$(CUDA_ROOT)/lib64 -Wl,-rpath,$(CUDA_ROOT)/lib64
else
CUDA_LDFLAGS := -L$(CUDA_ROOT)/lib64
endif
else
ifneq ($(strip $(CUDA_RPATH)),0)
CUDA_LDFLAGS := -L$(CUDA_ROOT)/lib -Wl,-rpath,$(CUDA_ROOT)/lib
else
CUDA_LDFLAGS := -L$(CUDA_ROOT)/lib
endif
endif
CUDA_LIBS := $(CUDA_LDFLAGS) -lcudart
else
CUDA_CPPFLAGS :=
CUDA_LIBS := -lcudart
endif

MPI_CXXFLAGS += $(CUDA_CPPFLAGS)

EXECS = gpu_one_to_one
SRCS = gpu_one_to_one.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(EXECS)

gpu_one_to_one: $(OBJS)
	$(MPI_CXX) $(MPI_CXXFLAGS) -o $@ $^ $(CUDA_LIBS)

%.o: %.cpp
	$(MPI_CXX) $(MPI_CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXECS)

.PHONY: all clean
