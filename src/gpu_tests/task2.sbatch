#!/bin/bash
# Проверка гипотез о прогреве, повторы, новые режимы, длинные серии.
# Запуск: bash submit.sh task2.sbatch
# Окружение и привязка как в task.sbatch (пояснения там и в NOTES.md).
#SBATCH --job-name=clustbench-hyp
#SBATCH --time=01:30:00
#SBATCH --output=clustbench-hyp-%j.out
#SBATCH --error=clustbench-hyp-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --gres-flags=enforce-binding

# --- Блоки: закомментировать лишнее ---
BLOCKS=(
	warm    # прогрев: w5 / w500 / w2000, 3 повтора (они же независимые повторы)
	modes   # one_to_all, all_to_one и all_to_all для сравнения
	long    # длинные серии one_to_one
)
ENVS=( auto host )
REPS=3   # повторов в блоке warm

WARM_CASES=( # "bytes:iters:warmup"
	"16000:2000:5"   "16000:2000:500"   "16000:2000:2000"
	"4096000:2000:5" "4096000:2000:500" "4096000:2000:2000"
)
MODES_LIST=( one_to_all all_to_one all_to_all )
MODES_CASES=( # "bytes:iters:warmup"
	"512000:1000:5"
	"4096000:500:5"
	"16384000:200:5"
)
LONG_CASES=( # "bytes:iters:warmup"
	"16000:20000:5"
	"4096000:10000:5"
)

source "${SLURM_SUBMIT_DIR:-.}/config.sh"

set -eEuo pipefail

SMI_PID=""
on_error() {
	local exit_code=$?
	echo "FATAL: шаг завершился с ошибкой (exit=${exit_code})" >&2
	[[ -n "$SMI_PID" ]] && kill "$SMI_PID" 2>/dev/null || true
	[[ -n "${SLURM_JOB_ID:-}" ]] && scancel "${SLURM_JOB_ID}" >/dev/null 2>&1 || true
	exit "${exit_code}"
}
trap on_error ERR

export PATH="$OMPI_PATH/bin:$PATH"
_LDPATH="$OMPI_PATH/lib:${CUDA_PATH}/lib64"
[[ -d "$UCX_PATH/lib" ]] && _LDPATH="$UCX_PATH/lib:$_LDPATH"
export LD_LIBRARY_PATH="$_LDPATH:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl_sm_single_copy_mechanism=cma
# Внешние export перебили бы значения программы (см. task.sbatch)
unset UCX_RNDV_SCHEME UCX_MEMTYPE_CACHE UCX_RNDV_THRESH UCX_TLS

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
BIN="${GPU_TESTS}/gpu"
OUT_DIR="${GPU_TESTS}/results/hyp_${SLURM_JOB_ID:-local}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
JOB_NODES="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-$(hostname -s)}}"
SAFE_NODES="$(echo "${JOB_NODES}" | tr '[],' '_' | tr -s '_' | sed 's/^_//; s/_$//')"
TIMELINE="${OUT_DIR}/timeline.csv"

test -x "$BIN" || { echo "error: нет $BIN (bash build.sh)" >&2; exit 1; }
grep -qa one_to_all "$BIN" || {
	echo "error: $BIN собран без one_to_all — пересобери: bash build.sh" >&2; exit 1; }
mkdir -p "${OUT_DIR}"
echo "out: ${OUT_DIR}"
echo "block,rep,env,mode,bytes,iters,warmup,start,end" > "${TIMELINE}"

# Частоты, P-state и температура GPU каждые 200 мс на всех узлах — для
# гипотезы о прогреве после запуска. --overlap: шаг не занимает ресурсы
# у замеров. Если не запустится, замеры всё равно идут.
srun --overlap --ntasks-per-node=1 -n"${NNODES}" --gpus-per-node=2 bash -c '
	nvidia-smi --query-gpu=timestamp,index,pci.bus_id,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu \
		--format=csv,nounits -lms 200 > "$0/gpu_clocks_$(hostname -s).csv"' "${OUT_DIR}" \
	2>"${OUT_DIR}/gpu_clocks.err" &
SMI_PID=$!

# run BLOCK REP ENV MODE BYTES ITERS WARMUP
run() {
	local block=$1 rep=$2 env=$3 mode=$4 bytes=$5 iters=$6 warmup=$7
	local out="${OUT_DIR}/${SAFE_NODES}_${block}_rep${rep}_${env}_${mode}_b${bytes}_w${warmup}_i${iters}.txt"
	echo "RUN ${block} rep=${rep} env=${env} mode=${mode} bytes=${bytes} warmup=${warmup} iters=${iters}"
	local t0; t0=$(date +%s.%N)
	srun --mpi=pmix \
		--kill-on-bad-exit=1 \
		--gres-flags=allow-task-sharing \
		--cpu-bind=cores \
		"${BIN}" \
		--bytes "${bytes}" --warmup "${warmup}" --iters "${iters}" \
		--env "${env}" --mode "${mode}" --out "${out}" > /dev/null
	echo "${block},${rep},${env},${mode},${bytes},${iters},${warmup},${t0},$(date +%s.%N)" >> "${TIMELINE}"
}

for BLOCK in "${BLOCKS[@]}"; do
	case "${BLOCK}" in
	warm)
		for REP in $(seq 1 "${REPS}"); do
			for ENV in "${ENVS[@]}"; do
				for CASE in "${WARM_CASES[@]}"; do
					IFS=":" read -r B I W <<<"${CASE}"
					run warm "${REP}" "${ENV}" one_to_one "${B}" "${I}" "${W}"
				done
			done
		done ;;
	modes)
		for ENV in "${ENVS[@]}"; do
			for MODE in "${MODES_LIST[@]}"; do
				for CASE in "${MODES_CASES[@]}"; do
					IFS=":" read -r B I W <<<"${CASE}"
					run modes 1 "${ENV}" "${MODE}" "${B}" "${I}" "${W}"
				done
			done
		done ;;
	long)
		for ENV in "${ENVS[@]}"; do
			for CASE in "${LONG_CASES[@]}"; do
				IFS=":" read -r B I W <<<"${CASE}"
				run long 1 "${ENV}" one_to_one "${B}" "${I}" "${W}"
			done
		done ;;
	*) echo "unknown block: ${BLOCK}" >&2; exit 1 ;;
	esac
done

kill "${SMI_PID}" 2>/dev/null || true
wait "${SMI_PID}" 2>/dev/null || true
echo "DONE: ${OUT_DIR}"
