#!/bin/bash
#SBATCH --job-name=a2a-rerun
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres-flags=enforce-binding
#SBATCH --output=a2a-rerun-%x-%j.out
#SBATCH --error=a2a-rerun-%x-%j.err

# Повторные измерения all_to_all после добавления MPI_Barrier в каждую
# итерацию (gpu_all_to_all.cpp). Те же узлы, пороги, размеры и имена файлов,
# что в основном прогоне 30.09.2026, чтобы результаты можно было положить
# поверх старых all_to_all в src/gpu_tests/results.
#
# Порядок:
#   cd ~/network-tests2/src/gpu_tests
#   sh build.sh                      # пересобрать ./gpu (обязательно)
#   bash all_to_all_rerun.sbatch     # БЕЗ sbatch: скрипт сам поставит 2 задания
#
# Результат: results/a2a_rerun/results_{pci5,sxm}_{default,0,32}/ — внутри
# *.txt и raw/*.raw с теми же именами, что в старых папках. Скопировать
# содержимое каждой папки поверх одноимённой в results/.

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
BIN="${GPU_TESTS}/gpu"
RERUN_DIR="${GPU_TESTS}/results/a2a_rerun"

# ---------- Запуск без Slurm: поставить два задания ----------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
	test -x "$BIN" || { echo "нет $BIN — сначала sh build.sh"; exit 1; }
	# Бинарник должен содержать барьер в каждой итерации.
	if ! grep -q "MPI_Barrier(iteration)" "$BIN"; then
		echo "В $BIN нет барьера в каждой итерации all_to_all."
		echo "Обнови gpu_all_to_all.cpp и пересобери: sh build.sh"
		exit 1
	fi
	cd "$GPU_TESTS"
	sbatch --job-name=a2a-pci5 --partition=intel-a100-pci5 --nodelist='g5500-[1-2]' \
		--nodes=2 --ntasks=4 --ntasks-per-node=2 --gpus-per-node=2 \
		--export=ALL,GROUP=pci5 "$0"
	sbatch --job-name=a2a-sxm --partition=intel-a100-sxm4 --nodelist=g8600-1 \
		--nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 \
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

export PATH="$OMPI_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$UCX_PATH/lib:$OMPI_PATH/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export OMPI_MCA_pml=ucx
unset UCX_RNDV_SCHEME UCX_MEMTYPE_CACHE UCX_RNDV_THRESH UCX_TLS \
      UCX_MAX_RNDV_LANES UCX_PROTO_INFO UCX_NET_DEVICES

: "${GROUP:?GROUP не задан — запускай через bash all_to_all_rerun.sbatch}"
SAFE_NODES="$(echo "${SLURM_JOB_NODELIST}" | tr '[],' '_' | tr -s '_' | sed 's/^_//; s/_$//')"

echo "=== a2a rerun: GROUP=${GROUP} nodes=${SLURM_JOB_NODELIST} prefix=${SAFE_NODES}"
echo "git: $(git -C "$GPU_TESTS" rev-parse --short HEAD 2>/dev/null || echo '?')"
grep -q "MPI_Barrier(iteration)" "$BIN" && echo "барьер в итерации: есть"

CASES=( # "bytes:iters:warmup" — как в task.sbatch
	"512000:1000:5"
	"4096000:500:5"
	"16384000:200:5"
)

for THRESH in default 0 32M; do
	case "$THRESH" in
		default) unset UCX_RNDV_THRESH; SUFFIX=default ;;
		0)       export UCX_RNDV_THRESH=0; SUFFIX=0 ;;
		32M)     export UCX_RNDV_THRESH=32M; SUFFIX=32 ;;
	esac
	OUT_DIR="${RERUN_DIR}/results_${GROUP}_${SUFFIX}"
	# Сырые замеры программа дописывает в конец файла, поэтому папка
	# должна быть пустой.
	rm -rf "${OUT_DIR}"
	mkdir -p "${OUT_DIR}"

	for ENV in auto host; do
		for CASE in "${CASES[@]}"; do
			IFS=":" read -r BYTES ITERS WARMUP <<<"${CASE}"
			OUT_TXT="${OUT_DIR}/${SAFE_NODES}_${ENV}_all_to_all_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
			echo "RUN thresh=${THRESH} env=${ENV} bytes=${BYTES} -> ${OUT_TXT}"
			srun --mpi=pmix --kill-on-bad-exit=1 \
				--gres-flags=allow-task-sharing --cpu-bind=cores \
				"${BIN}" --bytes "${BYTES}" --warmup "${WARMUP}" \
				--iters "${ITERS}" --env "${ENV}" --mode all_to_all \
				--out "${OUT_TXT}"
			grep '^pair' "${OUT_TXT}" | awk '$2!=$4 {print "   ", $2, "->", $4, $6}'
		done
	done
done

echo "=== DONE GROUP=${GROUP}"
