#!/bin/bash
#SBATCH --job-name=clustbench
#SBATCH --partition=batch
#SBATCH --time=00:05:00
#SBATCH --output=clustbench-%j.out
#SBATCH --error=clustbench-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1

set -euo pipefail

# Очистка настроек при компиляции в Докере
unset OMPI_MCA_btl_tcp_if_include OMPI_MCA_oob_tcp_if_include
unset OMPI_MCA_btl_tcp_if_exclude OMPI_MCA_oob_tcp_if_exclude
export OMPI_MCA_btl=tcp,self,vader
export OMPI_MCA_btl_tcp_if_exclude=lo,docker0,virbr0
export OMPI_MCA_oob_tcp_if_exclude=lo,docker0,virbr0

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export NETCDF_HOME="$HOME/opt/netcdf-c-4.9.2"
export PATH="$NETCDF_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NETCDF_HOME/lib:$LD_LIBRARY_PATH"

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/grid_out"
mkdir -p "${OUT_DIR}"
NTASKS="${SLURM_NTASKS:-8}"
BIN="${GPU_TESTS}/gpu_one_to_one"
JOB_NODES="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-$(hostname -s)}}"
SAFE_NODES="$(echo "${JOB_NODES}" | tr '[],' '___')"


CASES=(
	"500000:1600:80"
	# "2000000:800:40"
	# "8000000:400:20"
	"32000000:200:10"
)

REPS=(1)
MODES=(
	auto
	host
)

for REP in "${REPS[@]}"; do
	for MODE in "${MODES[@]}"; do
		for CASE in "${CASES[@]}"; do
			IFS=":" read -r BYTES ITERS WARMUP <<<"${CASE}"

			OUT_TXT="${OUT_DIR}/${SAFE_NODES}_rep${REP}_m${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
			echo "RUN rep=${REP} mode=${MODE} bytes=${BYTES} warmup=${WARMUP} iters=${ITERS}"

			srun --mpi=pmix -n"${NTASKS}" "${BIN}" \
				--bytes "${BYTES}" \
				--warmup "${WARMUP}" \
				--iters "${ITERS}" \
				--mode "${MODE}" \
				--out "${OUT_TXT}"
		done
	done
done
