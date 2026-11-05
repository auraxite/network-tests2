# Общие настройки кластера для всех скриптов gpu_tests (подключается через source).
# Для другого кластера правится только этот файл.
# Любую настройку можно переопределить при запуске: PARTITION=другой bash build.sh

PARTITION="${PARTITION:-intel-a100-pci5}"   # раздел Slurm для сборки и измерений

# Версии. Сборки разных версий UCX живут рядом и не мешают друг другу:
# у каждой свой UCX, свой Open MPI (собранный именно с ним) и своя программа.
# Вернуться к прежней: UCX_VERSION=1.15.0 bash submit.sh ...
#   1.15.0 — все данные до ноября 2026 (results/hyp_*, paths_*);
#   1.22.0 — с ноября 2026. С UCX 1.16 по умолчанию новый механизм выбора протоколов.
UCX_VERSION="${UCX_VERSION:-1.22.0}"
OMPI_VERSION="${OMPI_VERSION:-5.0.10}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-$UCX_VERSION}"          # куда ставит build_ucx.sh
# Open MPI собирается под конкретный UCX. Сборка с 1.15.0 исторически лежит без суффикса.
if [ "$UCX_VERSION" = 1.15.0 ]; then
	OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-$OMPI_VERSION}"
else
	OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-$OMPI_VERSION-ucx-$UCX_VERSION}"
fi
# Программа измерений в каталоге gpu_tests: своя для каждой версии UCX
GPU_BIN="${GPU_BIN:-gpu-ucx-$UCX_VERSION}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
