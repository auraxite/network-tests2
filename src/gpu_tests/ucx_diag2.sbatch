#!/bin/bash
#SBATCH --partition=intel-a100-pci5
#SBATCH --job-name=ucx-diag2
#SBATCH --time=00:15:00
#SBATCH --output=ucx-diag2-%j.out
#SBATCH --error=ucx-diag2-%j.err
#SBATCH --nodelist=g5500-[2,4]
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --gres-flags=enforce-binding

# ucx_diag.sbatch показал: в шаге на двух узлах даже соединение процесса с
# самим собой идёт через rc_mlx5 — значит, адреса UCX, которые Open MPI
# раздаёт процессам, содержат только сетевые транспорты. Правка pml_ucx.c
# должна была это исправить. Проверяем:
#   1. пересобран ли pml_ucx после правки (время pml_ucx.c, объектного файла
#      и установленной libmpi);
#   2. какой код отправки адреса работает на самом деле: сообщения pml ucx
#      (OMPI_MCA_pml_ucx_verbose) в шагах на одном и на двух узлах.
#
# Запуск: cd ~/network-tests2/src/gpu_tests && sbatch ucx_diag2.sbatch

OMPI_PATH="$HOME/opt/openmpi-5.0.10"
UCX_PATH="$HOME/opt/ucx-1.15.0"
CUDA_PATH="/usr/local/cuda"
SRC="$HOME/src/openmpi-5.0.10"

set -uo pipefail

export PATH="$UCX_PATH/bin:$OMPI_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$UCX_PATH/lib:$OMPI_PATH/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export OMPI_MCA_pml=ucx
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset UCX_RNDV_SCHEME UCX_MEMTYPE_CACHE UCX_TLS UCX_NET_DEVICES \
      UCX_MAX_RNDV_RAILS UCX_PROTO_ENABLE UCX_PROTO_INFO UCX_LOG_LEVEL
export UCX_RNDV_THRESH=0

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
BIN="${GPU_TESTS}/gpu"
OUT_DIR="${GPU_TESTS}/results_ucx_diag2"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo "=== 1. Пересобран ли pml ucx после правки"
ls -l --time-style=full-iso "$SRC/ompi/mca/pml/ucx/pml_ucx.c" 2>&1
find "$SRC/ompi/mca/pml/ucx" -name "*pml_ucx.*o" -exec ls -l --time-style=full-iso {} + 2>&1
ls -l --time-style=full-iso "$OMPI_PATH"/lib/libmpi.so.40.* 2>&1
ls "$OMPI_PATH"/lib/openmpi 2>/dev/null | grep -i pml || echo "(pml встроен в libmpi)"
grep -n "clustbench" "$SRC/ompi/mca/pml/ucx/pml_ucx.c" || echo "правки в исходнике нет"
echo "--- с какой libmpi запускается программа:"
ldd "$BIN" | grep -E 'libmpi|libucp'

SRUN="srun --mpi=pmix --gres-flags=allow-task-sharing --cpu-bind=cores"
export OMPI_MCA_pml_ucx_verbose=100

run() {  # run <метка> <параметры srun>
	local tag="$1"; shift
	local out="${OUT_DIR}/${tag}_auto_one_to_one_b16384000_w5_i20.txt"
	echo "=== RUN ${tag}: srun $*"
	$SRUN "$@" "$BIN" --bytes 16384000 --warmup 5 --iters 20 \
		--env auto --mode one_to_one --out "$out" > "${OUT_DIR}/${tag}.log" 2>&1
	grep '^pair' "$out" | awk '$2!=$4 {print "   ", $2, "->", $4, $6}'
	echo "--- сообщения pml ucx об адресах:"
	grep -iE 'address|worker|ucp_ep|add_proc|connect' "${OUT_DIR}/${tag}.log" \
		| sed 's/^\[[^]]*\] *//' | sort | uniq -c | head -40
}

echo "=== 2. Сообщения pml ucx"
run step1node -N1 -n2 --ntasks-per-node=2
run step2node -N2 -n4

echo "=== DONE"
