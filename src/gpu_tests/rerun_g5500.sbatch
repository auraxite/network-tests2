#!/bin/bash
#SBATCH --partition=intel-a100-pci5
#SBATCH --job-name=rerun-g5500
#SBATCH --time=01:30:00
#SBATCH --output=rerun-g5500-%j.out
#SBATCH --error=rerun-g5500-%j.err
#SBATCH --nodelist=g5500-[1-2]
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --gres-flags=enforce-binding

# Повтор прогонов G5500 от 30.09.2026 (логи: logs_2026-09-30/results_pci5_*.out)
# с исправленным Open MPI: pml ucx публикует полный адрес UCX для всех
# процессов (см. _openmpi_build_job.sh). До исправления в задании на двух
# узлах пары GPU одного узла передавали данные через сетевой адаптер.
#
# Те же 36 запусков, что в task.sbatch (one_to_one 15 размеров и all_to_all
# 3 размера, среды auto и host), те же узлы g5500-1/2, для трёх порогов
# UCX_RNDV_THRESH (по умолчанию, 0, 32M) в одном задании.
# GPU закреплены за 56:00.0 и 57:00.0, как в прежних прогонах: узлу
# выделяются все 4 GPU, процессов два, порядок устройств — по шине PCI.
#
# Запуск: cd ~/network-tests2/src/gpu_tests && sbatch rerun_g5500.sbatch
#   если g5500-1/2 заняты: sbatch --nodelist=g5500-[2,4] rerun_g5500.sbatch
# Результаты: results_fixed/results_pci5_{default,0,32}/ (имена как раньше),
#   лог — rerun-g5500-<job>.out

ENVS=(auto host)
MODES=(one_to_one all_to_all)
ONE_TO_ONE_CASES=( # "bytes:iters:warmup" — как в task.sbatch
	"1000:2000:5" "2000:2000:5" "4000:2000:5" "8000:2000:5" "16000:2000:5"
	"32000:2000:5" "64000:2000:5" "128000:2000:5" "256000:2000:5" "512000:2000:5"
	"1024000:1000:5" "2048000:1000:5" "4096000:1000:5" "8192000:100:5" "16384000:100:5"
)
ALL_TO_ALL_CASES=("512000:1000:5" "4096000:500:5" "16384000:200:5")

OMPI_PATH="$HOME/opt/openmpi-5.0.10"
UCX_PATH="$HOME/opt/ucx-1.15.0"
CUDA_PATH="/usr/local/cuda"

set -eEuo pipefail
trap 'echo "FATAL: шаг завершился с ошибкой (exit=$?)" >&2' ERR

export PATH="$UCX_PATH/bin:$OMPI_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$UCX_PATH/lib:$OMPI_PATH/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export OMPI_MCA_pml=ucx
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset UCX_RNDV_SCHEME UCX_MEMTYPE_CACHE UCX_RNDV_THRESH UCX_TLS UCX_NET_DEVICES \
      UCX_MAX_RNDV_RAILS UCX_PROTO_ENABLE UCX_PROTO_INFO UCX_LOG_LEVEL

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
BIN="${GPU_TESTS}/gpu"
RESULTS="${GPU_TESTS}/results_fixed"
test -x "$BIN" || { echo "нет $BIN — сначала sh build.sh"; exit 1; }
SAFE_NODES="$(echo "${SLURM_JOB_NODELIST}" | tr '[],' '_' | tr -s '_' | sed 's/^_//; s/_$//')"

echo "=== rerun g5500: nodes=${SLURM_JOB_NODELIST} prefix=${SAFE_NODES}"
echo "git: $(git -C "$GPU_TESTS" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "правка Open MPI в исходниках: $(grep -c 'clustbench: полный адрес' "$HOME/src/openmpi-5.0.10/ompi/mca/pml/ucx/pml_ucx.c" 2>/dev/null || echo '?')"
ls -l --time-style=long-iso "$OMPI_PATH"/lib/libmpi.so.40.* || true
ldd "$BIN" | grep -E 'libmpi|libucp'

for THRESH in default 0 32M; do
	case "$THRESH" in
		default) unset UCX_RNDV_THRESH; SUFFIX=default ;;
		0)       export UCX_RNDV_THRESH=0; SUFFIX=0 ;;
		32M)     export UCX_RNDV_THRESH=32M; SUFFIX=32 ;;
	esac
	OUT_DIR="${RESULTS}/results_pci5_${SUFFIX}"
	rm -rf "${OUT_DIR}"          # сырые замеры дописываются в конец файла
	mkdir -p "${OUT_DIR}"
	echo "######## THRESH=${THRESH} -> ${OUT_DIR}"

	for ENV in "${ENVS[@]}"; do
		for MODE in "${MODES[@]}"; do
			case "$MODE" in
				one_to_one) CASES=("${ONE_TO_ONE_CASES[@]}") ;;
				all_to_all) CASES=("${ALL_TO_ALL_CASES[@]}") ;;
			esac
			for CASE in "${CASES[@]}"; do
				IFS=":" read -r BYTES ITERS WARMUP <<<"${CASE}"
				OUT_TXT="${OUT_DIR}/${SAFE_NODES}_${ENV}_${MODE}_b${BYTES}_w${WARMUP}_i${ITERS}.txt"
				echo "RUN thresh=${THRESH} env=${ENV} mode=${MODE} bytes=${BYTES} iters=${ITERS}"
				srun --mpi=pmix --kill-on-bad-exit=1 \
					--gres-flags=allow-task-sharing --cpu-bind=cores \
					"${BIN}" --bytes "${BYTES}" --warmup "${WARMUP}" \
					--iters "${ITERS}" --env "${ENV}" --mode "${MODE}" \
					--out "${OUT_TXT}" > /dev/null
				grep '^pair' "${OUT_TXT}" | awk '$2!=$4 {printf " %s>%s:%s", $2, $4, substr($6,8)} END{print ""}'
			done
		done
	done
done

echo "=== DONE"
