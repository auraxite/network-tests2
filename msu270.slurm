#!/bin/bash
#SBATCH --job-name=clustbench
# MSU270: default GPU partition is "batch" (see: sinfo -s -o "%P %a %G")
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Example override:
#   sbatch --gpus-per-node=8 --export=ALL,APP_MODE=gpu_info_intra msu270.slurm
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# GPU topology: gpu_info_collect.py (stdlib) + gpu_info_render_snapshot.py (matplotlib).
# APP_MODE:
#   gpu_info_intra   — one node: snapshot + optional PNGs (default)
#   gpu_info         — multi-node: set NET_MAP, srun collects live_node*.in per node
#
# Optional: PROJECT_ROOT=/path/to/network-tests2
#           NET_MAP=src/gpu_tests/input/1_8.in   (required only for gpu_info)

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
APP_MODE="${APP_MODE:-gpu_info_intra}"
NET_MAP="${NET_MAP:-}"

cd "${PROJECT_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

run_gpu_info() {
  local gpu_dir="${PROJECT_ROOT}/src/gpu_tests"
  local input_dir="${gpu_dir}/input"
  local out_dir="${PROJECT_ROOT}/output"
  local nnodes="${SLURM_JOB_NUM_NODES:-1}"

  if [[ -z "${NET_MAP}" ]]; then
    echo "ERROR: NET_MAP is required for APP_MODE=gpu_info"
    echo "Example: sbatch --export=ALL,APP_MODE=gpu_info,NET_MAP=src/gpu_tests/input/1_8.in msu270.slurm"
    exit 2
  fi

  mkdir -p "${input_dir}"
  srun --ntasks="${nnodes}" --ntasks-per-node=1 bash -lc '
    idx=${SLURM_NODEID:-0}
    out="'"${input_dir}"'/live_node${idx}.in"
    nvidia-smi topo -m > "${out}"
    echo "saved ${out} on $(hostname)"
  '

  python3 "${gpu_dir}/gpu_info_collect.py" \
    --input-dir "${input_dir}" \
    --net-map "${NET_MAP}" \
    --out-dir "${out_dir}"

  if ! python3 "${gpu_dir}/gpu_info_render_snapshot.py" "${out_dir}/gpu_snapshot.json" --out-dir "${out_dir}"; then
    echo "Note: PNG render skipped (matplotlib/numpy missing or render error). Run locally:" >&2
    echo "  python3 ${gpu_dir}/gpu_info_render_snapshot.py ${out_dir}/gpu_snapshot.json --out-dir ${out_dir}" >&2
  fi
}

run_gpu_info_intra() {
  local gpu_dir="${PROJECT_ROOT}/src/gpu_tests"
  local input_dir="${gpu_dir}/input"
  local out_dir="${PROJECT_ROOT}/output"
  mkdir -p "${input_dir}" "${out_dir}"
  rm -f "${input_dir}"/live_node*.in

  python3 "${gpu_dir}/gpu_info_collect.py" --out-dir "${out_dir}"

  if ! python3 "${gpu_dir}/gpu_info_render_snapshot.py" "${out_dir}/gpu_snapshot.json" --out-dir "${out_dir}"; then
    echo "Note: PNG render skipped (matplotlib/numpy missing or render error). Run locally:" >&2
    echo "  python3 ${gpu_dir}/gpu_info_render_snapshot.py ${out_dir}/gpu_snapshot.json --out-dir ${out_dir}" >&2
  fi
}

case "${APP_MODE}" in
  gpu_info_intra) run_gpu_info_intra ;;
  gpu_info) run_gpu_info ;;
  *)
    echo "ERROR: unknown APP_MODE='${APP_MODE}' (expected gpu_info_intra or gpu_info)"
    exit 2
    ;;
esac
