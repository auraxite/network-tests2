#!/bin/bash
# Запуск gpu_one_to_one: 3 узла × 2 GPU = 6 MPI-рангов (у каждого ровно один видимый GPU).
#
# Перед первым запуском:
#   1) Подставьте свою partition / account (и при необходимости --nodelist).
#   2) module load ... для CUDA + OpenMPI (или тот стек, что на МГУ-270).
#   3) Соберите бинарь: из корня network-tests2 с настроенным config — make в src/gpu_tests.
#
# Запуск:
#   sbatch src/gpu_tests/gpu_one_to_one_batch.slurm
#   или из каталога src/gpu_tests:
#   sbatch gpu_one_to_one_batch.slurm
#
# Логи: slurm-%j.out ; дублирующая копия таблицы пар — bench-%j.txt (--out).

#SBATCH --job-name=gpu1to1
#SBATCH --nodes=3
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

## Раскомментируйте и поправьте под раздел:
#SBATCH --partition=batch

## Опционально: зафиксировать узлы (две «полки» + одна):
#SBATCH --nodelist=node01,node02,node25

## Если на кластере нет --gpus-per-task, попробуйте вариант с GRES (закомментируйте блок выше и раскомментируйте ниже):
## #SBATCH --nodes=3
## #SBATCH --ntasks=6
## #SBATCH --ntasks-per-node=2
## #SBATCH --gres=gpu:2
## Тогда убедитесь, что на каждом узле два таска получают разные GPU (документация Slurm / тест nvidia-smi).

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
GPU_TESTS="${ROOT}/src/gpu_tests"
BIN="${GPU_TESTS}/gpu_one_to_one"

if [[ ! -x "${BIN}" ]]; then
	echo "Нет исполняемого ${BIN}. Соберите: cd src/gpu_tests && make" >&2
	exit 1
fi

# Раскомментируйте при необходимости:
# module load cuda
# module load openmpi

cd "${GPU_TESTS}"

OUT_TXT="${ROOT}/bench-${SLURM_JOB_ID}.txt"

# srun раздаёт процессы по узлам согласно SBATCH; с OpenMPI часто нужен один из вариантов:
#   --mpi=pmix | --mpi=pmi2 | без флага (зависит от сборки Slurm)
srun --mpi=pmix "${BIN}" \
	--bytes $((4 * 1024 * 1024)) \
	--warmup 10 \
	--iters 50 \
	--mode auto \
	--out "${OUT_TXT}"

echo "Текст с парами: ${OUT_TXT}"
echo "Heatmap (на login-узле или в том же job, если есть matplotlib):"
echo "  python3 ${GPU_TESTS}/gpu_render.py ${OUT_TXT} -o ${ROOT}/gpu_plots --prefix job${SLURM_JOB_ID}"
