FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    apt update && \
    apt install -y \
        build-essential \
        openmpi-bin \
        libopenmpi-dev \
        libnetcdf-dev \
        libnuma-dev \
        tzdata && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*