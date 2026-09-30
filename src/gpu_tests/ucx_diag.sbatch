#!/bin/bash
#SBATCH --partition=intel-a100-pci5
#SBATCH --job-name=ucx-diag
#SBATCH --time=00:20:00
#SBATCH --output=ucx-diag-%j.out
#SBATCH --error=ucx-diag-%j.err
#SBATCH --nodelist=g5500-[2,4]
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --gres-flags=enforce-binding

# Почему в задании на двух узлах UCX не видит транспорты внутри узла.
#
# Факты: в задании на одном узле для пары GPU узла UCX выбирает cuda_ipc
# (~72 мкс на 16 МБ), для памяти хоста — sysv. В задании на двух узлах для
# тех же пар нет ни cuda_ipc, ни sysv, ни cma — только rc_mlx5 через сетевой
# адаптер (~830 мкс). Полный адрес UCX до соседа доходит (правка Open MPI),
# а картина та же.
#
# Проверки в одном выделении из двух узлов:
#   1. окружение (UCX_*, OMPI_*, PMIX_*, SLURM_PMIX*) у шага на 1 узле и на 2
#      узлах — Slurm-плагин PMIx может сам выставлять UCX_TLS/UCX_NET_DEVICES;
#   2. шаг на ОДНОМ узле внутри этого выделения (srun -N1 -n2): если быстро,
#      то дело в шаге, охватывающем несколько узлов, а не в выделении;
#   3. шаг на двух узлах с отладочным журналом UCX (UCX_LOG_LEVEL=debug):
#      какие транспорты UCX выбрал для каждой пары процессов.
#
# Запуск: cd ~/network-tests2/src/gpu_tests && sbatch ucx_diag.sbatch
# Результат: ucx-diag-<job>.out и results_ucx_diag/

OMPI_PATH="$HOME/opt/openmpi-5.0.10"
UCX_PATH="$HOME/opt/ucx-1.15.0"
CUDA_PATH="/usr/local/cuda"

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
OUT_DIR="${GPU_TESTS}/results_ucx_diag"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}/log"

SRUN="srun --mpi=pmix --gres-flags=allow-task-sharing --cpu-bind=cores"
ENVRE='^(UCX|OMPI|PMIX|SLURM_PMIX|SLURM_MPI|OMPI_MCA)'

echo "=== 1. Окружение процессов"
echo "--- шаг на 1 узле (srun -N1 -n2):"
$SRUN -N1 -n2 --ntasks-per-node=2 bash -c "env | grep -E '$ENVRE' | sort" \
	| sort | uniq -c | sort -k2 > "${OUT_DIR}/env_1node.txt"
cat "${OUT_DIR}/env_1node.txt"
echo "--- шаг на 2 узлах (srun -N2 -n4):"
$SRUN -N2 -n4 bash -c "env | grep -E '$ENVRE' | sort" \
	| sort | uniq -c | sort -k2 > "${OUT_DIR}/env_2node.txt"
cat "${OUT_DIR}/env_2node.txt"
echo "--- различия (имена переменных):"
diff <(awk '{print $2}' "${OUT_DIR}/env_1node.txt" | cut -d= -f1 | sort -u) \
     <(awk '{print $2}' "${OUT_DIR}/env_2node.txt" | cut -d= -f1 | sort -u) || true
echo "--- Slurm: параметры плагина PMIx"
scontrol show config 2>/dev/null | grep -iE 'mpi|pmix' || true

show() {
	grep 'pci=' "$1" | sed 's/^/   /'
	grep '^pair' "$1" | awk '$2!=$4 {print "   ", $2, "->", $4, $6}' || true
}

echo "=== 2. Шаг на ОДНОМ узле внутри выделения из двух узлов"
out="${OUT_DIR}/step1node_auto_one_to_one_b16384000_w5_i50.txt"
UCX_PROTO_ENABLE=y UCX_PROTO_INFO=y \
$SRUN -N1 -n2 --ntasks-per-node=2 "$BIN" --bytes 16384000 --warmup 5 --iters 50 \
	--env auto --mode one_to_one --out "$out" > "${OUT_DIR}/step1node_proto.log" 2>&1
show "$out"
echo "   транспорты в таблице протоколов:"
grep -oE '[a-z_0-9]+/[A-Za-z_0-9:]+ *\|$' "${OUT_DIR}/step1node_proto.log" | sort | uniq -c | sed 's/^/     /'

echo "=== 3. Шаг на двух узлах с отладочным журналом UCX"
out="${OUT_DIR}/step2node_auto_one_to_one_b16384000_w5_i5.txt"
UCX_LOG_LEVEL=debug UCX_LOG_FILE="${OUT_DIR}/log/ucx.%h.%p.log" \
$SRUN -N2 -n4 "$BIN" --bytes 16384000 --warmup 5 --iters 5 \
	--env auto --mode one_to_one --out "$out" > /dev/null 2>&1
show "$out"
echo "--- выбранные пути (lane) по процессам:"
for f in "${OUT_DIR}"/log/ucx.*.log; do
	echo "## $(basename "$f")"
	grep -E 'lane\[[0-9]+\]' "$f" | sed 's/^\[[^]]*\] *//' | sort -u | head -40
done
echo "--- упоминания транспортов внутри узла и недоступности:"
grep -hiE 'cuda_ipc|sysv|posix|cma|unreachable|not reachable|machine|system_id|guid' \
	"${OUT_DIR}"/log/ucx.*.log | sed 's/^\[[^]]*\] *//' | sort | uniq -c | sort -rn | head -60
gzip -q "${OUT_DIR}"/log/*.log 2>/dev/null || true

echo "=== DONE"
