#!/bin/bash
#SBATCH --partition=intel-a100-pci5
#SBATCH --job-name=launcher-check
#SBATCH --time=00:20:00
#SBATCH --output=launcher-check-%j.out
#SBATCH --error=launcher-check-%j.err
#SBATCH --nodelist=g5500-[2,4]
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --gres-flags=enforce-binding

# В задании на двух узлах UCX передаёт данные между GPU одного узла через
# сетевой адаптер (rc_mlx5), а в задании на одном узле — через cuda_ipc.
# Гипотеза: Open MPI (pml_ucx.c) публикует для удалённых процессов адрес UCX
# только с сетевыми транспортами (UCP_WORKER_ADDRESS_FLAG_NET_ONLY), а полный
# адрес — только для процессов своего узла. При запуске через srun --mpi=pmix
# соседи по узлу, видимо, получают сокращённый адрес и не видят cuda_ipc.
#
# Проверка: одно и то же задание, два способа запуска:
#   1. srun --mpi=pmix (как во всех измерениях);
#   2. mpirun из Open MPI (сам раздаёт процессам сведения о соседях).
# Для каждого — one_to_one 16 МБ и таблица UCX_PROTO_INFO.
# Если через mpirun пары внутри узла дают ~70 мкс, а через srun ~830 мкс —
# причина в способе запуска.
#
# Запуск: cd ~/network-tests2/src/gpu_tests && sbatch launcher_check.sbatch
# Результаты: results_launcher_check/ и launcher-check-<job>.out/.err

OMPI_PATH="$HOME/opt/openmpi-5.0.10"
UCX_PATH="$HOME/opt/ucx-1.15.0"
CUDA_PATH="/usr/local/cuda"

set -eEuo pipefail

export PATH="$UCX_PATH/bin:$OMPI_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$UCX_PATH/lib:$OMPI_PATH/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export OMPI_MCA_pml=ucx
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset UCX_RNDV_SCHEME UCX_MEMTYPE_CACHE UCX_TLS UCX_NET_DEVICES \
      UCX_MAX_RNDV_RAILS UCX_PROTO_ENABLE UCX_PROTO_INFO
export UCX_RNDV_THRESH=0

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
BIN="${GPU_TESTS}/gpu"
OUT_DIR="${GPU_TESTS}/results_launcher_check"
rm -rf "${OUT_DIR}"          # сырые замеры дописываются в конец файла
mkdir -p "${OUT_DIR}"
test -x "$BIN" || { echo "нет $BIN"; exit 1; }

echo "=== launcher check: nodes=${SLURM_JOB_NODELIST}"
echo "mpirun: $(command -v mpirun)"
mpirun --version | head -2 || true
echo "--- параметры pml ucx"
ompi_info --param pml ucx --level 9 2>/dev/null | grep -iE "addr|net|local" || true

show() {
	grep 'pci=' "$1" | sed 's/^/   /'
	grep '^pair' "$1" | awk '$2!=$4 {print "   ", $2, "->", $4, $6}' || true
}

run_srun() {  # run_srun <метка> <iters>
	local out="${OUT_DIR}/srun_$1_auto_one_to_one_b16384000_w5_i$2.txt"
	echo "=== RUN srun $1 UCX_PROTO_ENABLE=${UCX_PROTO_ENABLE:-<not set>}"
	srun --mpi=pmix --kill-on-bad-exit=1 \
		--gres-flags=allow-task-sharing --cpu-bind=cores \
		"${BIN}" --bytes 16384000 --warmup 5 --iters "$2" \
		--env auto --mode one_to_one --out "${out}"
	show "${out}"
}

run_mpirun() {  # run_mpirun <метка> <iters>
	local out="${OUT_DIR}/mpirun_$1_auto_one_to_one_b16384000_w5_i$2.txt"
	echo "=== RUN mpirun $1 UCX_PROTO_ENABLE=${UCX_PROTO_ENABLE:-<not set>}"
	# Номер процесса внутри узла программа берёт из OMPI_COMM_WORLD_LOCAL_RANK.
	# В прошлом запуске под mpirun MPIX_Query_cuda_support() вернул 0 и
	# программа ушла на путь через оперативную память. Явно требуем модуль
	# CUDA и просим Open MPI объяснить выбор (в .err).
	mpirun -np 4 --map-by ppr:2:node --bind-to core \
		--mca accelerator cuda --mca accelerator_base_verbose 10 \
		-x PATH -x LD_LIBRARY_PATH -x CUDA_DEVICE_ORDER \
		-x OMPI_MCA_opal_cuda_support -x OMPI_MCA_pml \
		-x UCX_RNDV_THRESH ${UCX_PROTO_ENABLE:+-x UCX_PROTO_ENABLE} \
		${UCX_PROTO_INFO:+-x UCX_PROTO_INFO} \
		"${BIN}" --bytes 16384000 --warmup 5 --iters "$2" \
		--env auto --mode one_to_one --out "${out}"
	show "${out}"
}

run_mpirun plain 100
grep -m1 '^CUDA-aware' "${OUT_DIR}/mpirun_plain_auto_one_to_one_b16384000_w5_i100.txt" || true

export UCX_PROTO_ENABLE=y UCX_PROTO_INFO=y
run_mpirun protoinfo 20
unset UCX_PROTO_ENABLE UCX_PROTO_INFO

# Для сравнения — то же через srun (в прошлом запуске: ~830 мкс внутри узла)
run_srun plain 100

echo "=== DONE"
