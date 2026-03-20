include ../../config

EXECS = gpu_load

SRCS_NET = gpu_load.cpp

OBJS_NET = $(SRCS_NET:.cpp=.o)

all: $(EXECS)

gpu_load: $(OBJS_NET)
	$(MPI_CXXLINKER) $(MPI_CXXLINKER_FLAGS) -o $@ $^

%.o: %.cpp
	$(MPI_CXX) $(MPI_CXX_FLAGS) $(MPI_CXX_INCLUDE) -I ../core -I ../.. -c $< -o $@

clean:
	rm -f $(OBJS_NET) $(EXECS)

install: all
	cp -rf $(EXECS) $(INSTALL_DIR)/bin/

.PHONY: all clean install
