#!/bin/bash
#SBATCH --job-name=clustbench
# MSU270: default GPU partition is "batch" (see: sinfo -s -o "%P %a %G")
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Example override:
#   sbatch --gpus-per-node=8 msu270.slurm
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# GPU topology: gpu_info_collect.py (stdlib) + gpu_info_render.py (matplotlib).
# Optional: PROJECT_ROOT=/path/to/network-tests2

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

# Render intentionally disabled in batch job.
# Run locally or in a separate job if PNGs are needed:
#   python3 "${gpu_dir}/gpu_info_render.py" "${out_dir}/gpu_snapshot.json" --out-dir "${out_dir}"
