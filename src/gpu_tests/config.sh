# Общие настройки кластера для всех скриптов gpu_tests (подключается через source).
# Для другого кластера правится только этот файл.
# Любую настройку можно переопределить при запуске: PARTITION=другой bash build.sh

PARTITION="${PARTITION:-intel-a100-pci5}"   # раздел Slurm для сборки и измерений

UCX_VERSION="${UCX_VERSION:-1.15.0}"
OMPI_VERSION="${OMPI_VERSION:-5.0.10}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-$UCX_VERSION}"          # куда ставит build_ucx.sh
OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-$OMPI_VERSION}"   # куда ставит build_openmpi.sh
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
