#!/bin/bash
#SBATCH --job-name=clustbench
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:05
#SBATCH --output=slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

cd "${PROJECT_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

gpu_dir="${PROJECT_ROOT}/src/gpu_tests"
input_dir="${gpu_dir}/input"
out_dir="${PROJECT_ROOT}/output"
mkdir -p "${input_dir}" "${out_dir}"

python3 "${gpu_dir}/gpu_info_collect.py" --out-dir "${out_dir}"