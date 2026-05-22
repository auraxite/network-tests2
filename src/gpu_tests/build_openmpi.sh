srun -N1 --gpus=1 --time=01:00:00 bash -lc '
set -eu
cd ~/src/ompi-5.0.10

export OMPI_NEW=$HOME/opt/openmpi-5.0.10-cuda
export UCX_PATH=$HOME/opt/ucx-1.15.0
export CUDA_PATH=/usr/local/cuda

make distclean || true

./configure \
  --prefix="$OMPI_NEW" \
  --with-ucx="$UCX_PATH" \
  --with-cuda="$CUDA_PATH" \
  --with-slurm \
  --disable-mpi-fortran \
  --without-hcoll \
  --enable-mca-no-build=coll-hcoll

make -j"$(nproc)"
make install
'