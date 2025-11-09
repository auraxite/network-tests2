FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    build-essential automake autoconf cmake git wget unzip which \
    libx11-dev libxext-dev libxrender-dev mesa-common-dev \
    libglu1-mesa-dev freeglut3-dev mesa-utils \
    libssl-dev libfftw3-dev libhdf5-dev libnetcdf-dev \
    qt4-qmake libqt4-dev libqwt-qt4-dev \
    openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run && \
    sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit --override && \
    rm cuda_11.8.0_520.61.05_linux.run

ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN wget -q https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz && \
    tar -xzf openmpi-4.1.6.tar.gz && cd openmpi-4.1.6 && \
    ./configure --prefix=/usr/local && make -j$(nproc) && make install && \
    cd .. && rm -rf openmpi-4.1.6*

WORKDIR /root
RUN git clone -b tests_as_plugins https://github.com/auraxite/network-tests2.git

WORKDIR /root/network-tests2
RUN ./make_configure.sh && \
    ./configure --enable-all --prefix=/usr/local || true
    # оставляем make на ручной запуск, чтобы при ошибке не сбрасывался кэш
ENV PATH=/usr/local/qt4/bin:/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/qt4/lib:/usr/local/lib:$LD_LIBRARY_PATH

RUN apt update && apt install -y xauth x11-apps sudo && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["/bin/bash"]