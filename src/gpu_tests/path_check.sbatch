#!/bin/bash
#SBATCH --job-name=path-check
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres-flags=enforce-binding
#SBATCH --output=path-check-%x-%j.out
#SBATCH --error=path-check-%x-%j.err

# Каким путём UCX передаёт данные между двумя GPU одного узла.
#
# На G5500 UCX_PROTO_INFO показал, что передача GPU -> GPU внутри узла идёт
# через сетевой адаптер (rc_mlx5/mlx5_0, чтение по RDMA), а cuda_ipc не
# используется. Проверки:
#   G5500: 1) что разрешает система: nvidia-smi topo -p2p r/w;
#          2) обычный режим auto;
#          3) auto без транспортов InfiniBand (UCX_TLS без rc/ud/dc): UCX не
#             может уйти через адаптер. Быстро — cuda_ipc работает, но UCX
#             его не выбирает. Медленно или ошибка — прямого пути нет;
#          4) таблица UCX_PROTO_INFO для 2) и 3).
#   G8600: таблица UCX_PROTO_INFO в обычном режиме (ожидается cuda_ipc).
# Все прогоны: один узел, два процесса, UCX_RNDV_THRESH=0, one_to_one.
#
# Запуск:
#   cd ~/network-tests2/src/gpu_tests
#   bash path_check.sbatch          # БЕЗ sbatch: скрипт сам поставит 2 задания
# Результаты: results_path_check/{pci5,sxm}/ и логи path-check-*.out/.err

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
BIN="${GPU_TESTS}/gpu"

# ---------- Запуск без Slurm: поставить два задания ----------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
	test -x "$BIN" || { echo "нет $BIN — сначала sh build.sh"; exit 1; }
	cd "$GPU_TESTS"
	# G5500: все 4 GPU узла, чтобы при CUDA_DEVICE_ORDER=PCI_BUS_ID процессам
	# достались 56:00.0 и 57:00.0, как в основных прогонах.
	sbatch --job-name=pc-pci5 --partition=intel-a100-pci5 --gpus-per-node=4 \
		--export=ALL,GROUP=pci5 "$0"
	sbatch --job-name=pc-sxm --partition=intel-a100-sxm4 --gpus-per-node=2 \
		--export=ALL,GROUP=sxm "$0"
	echo "Поставлено 2 задания. Следить: squeue -u \$USER"
	exit 0
fi

# ---------- Внутри задания ----------------------------------------------
OMPI_PATH="${OMPI_PATH:-$HOME/opt/openmpi-5.0.10-cuda-clean}"
UCX_PATH="${UCX_PATH:-$HOME/opt/ucx-1.15.0}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

set -eEuo pipefail
trap 'echo "FATAL: шаг завершился с ошибкой (exit=$?)" >&2' ERR

export PATH="$UCX_PATH/bin:$OMPI_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$UCX_PATH/lib:$OMPI_PATH/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export OMPI_MCA_pml=ucx
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset UCX_RNDV_SCHEME UCX_MEMTYPE_CACHE UCX_TLS UCX_NET_DEVICES \
      UCX_MAX_RNDV_RAILS UCX_PROTO_ENABLE UCX_PROTO_INFO
export UCX_RNDV_THRESH=0

: "${GROUP:?GROUP не задан — запускай через bash path_check.sbatch}"
OUT_DIR="${GPU_TESTS}/results_path_check/${GROUP}"
rm -rf "${OUT_DIR}"          # сырые замеры дописываются в конец файла
mkdir -p "${OUT_DIR}"
SAFE_NODES="$(echo "${SLURM_JOB_NODELIST}" | tr '[],' '_' | tr -s '_' | sed 's/^_//; s/_$//')"

echo "=== path check: GROUP=${GROUP} node=${SLURM_JOB_NODELIST}"
echo "--- nvidia-smi topo -m"
nvidia-smi topo -m || true
echo "--- nvidia-smi topo -p2p r (разрешено ли чтение памяти соседнего GPU)"
nvidia-smi topo -p2p r || true
echo "--- nvidia-smi topo -p2p w"
nvidia-smi topo -p2p w || true
echo "--- nvidia-smi nvlink --status -i 0"
nvidia-smi nvlink --status -i 0 || true

run() {  # run <метка> <bytes> <iters> — остальное из окружения
	local tag="$1" bytes="$2" iters="$3"
	local out="${OUT_DIR}/${SAFE_NODES}_${tag}_auto_one_to_one_b${bytes}_w5_i${iters}.txt"
	echo "=== RUN ${tag} bytes=${bytes} UCX_TLS=${UCX_TLS:-<module>} UCX_PROTO_ENABLE=${UCX_PROTO_ENABLE:-<not set>}"
	srun --mpi=pmix --kill-on-bad-exit=1 \
		--gres-flags=allow-task-sharing --cpu-bind=cores \
		"${BIN}" --bytes "${bytes}" --warmup 5 --iters "${iters}" \
		--env auto --mode one_to_one --out "${out}"
	grep 'pci=' "${out}" | sed 's/^/   /'
	grep '^pair' "${out}" | awk '$2!=$4 {print "   ", $2, "->", $4, $6}' || true
}

# Обычный режим, как в основных измерениях
run default 1024000 1000
run default 16384000 100
# Таблица выбора протоколов для обычного режима
UCX_PROTO_ENABLE=y UCX_PROTO_INFO=y run default_protoinfo 16384000 20

if [[ "${GROUP}" == "pci5" ]]; then
	# Без транспортов InfiniBand: путь через адаптер недоступен
	export UCX_TLS=cuda_copy,cuda_ipc,sm,cma,self
	export UCX_NET_DEVICES=all
	run noib 1024000 1000
	run noib 16384000 100
	UCX_PROTO_ENABLE=y UCX_PROTO_INFO=y run noib_protoinfo 16384000 20
fi

echo "=== DONE GROUP=${GROUP}"
