#!/bin/bash
# «Быстрое» состояние узла: тесты частоты ядра и L3 (uncore) до и после каждой пары.
# Запуск: bash submit.sh task5.sbatch
# Окружение и привязка как в task.sbatch (пояснения там и в NOTES.md).
#SBATCH --job-name=clustbench-hyp5
#SBATCH --time=01:00:00
#SBATCH --output=clustbench-hyp5-%j.out
#SBATCH --error=clustbench-hyp5-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --gres-flags=enforce-binding

SHUFFLE_SEED=5   # порядок запусков; записывается в runs.csv

# Запуски: "блок повторов среда привязка байт:итераций:прогрев"
#   probe — без привязки; до и после каждой пары оба участника меряют
#   цепочку умножений (частота ядра) и переходы по 8 МБ (задержка L3, uncore).
# Везде: one_to_one_v2, случайный порядок пар, номер CPU на каждой итерации,
# 1 с нагрузки перед первой парой.
PLAN=(
	"probe 12 auto -1 16000:2000:500"
	"probe 12 host -1 16000:2000:500"
)
COMMON=(--mode one_to_one_v2 --order random --cpu-log --prespin-sec 1 --probe)

source "${SLURM_SUBMIT_DIR:-.}/config.sh"

set -eEuo pipefail

BG_PIDS=()
stop_bg() { for p in "${BG_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
on_error() {
	local exit_code=$?
	echo "FATAL: шаг завершился с ошибкой (exit=${exit_code})" >&2
	stop_bg
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
RUNS="${OUT_DIR}/runs.csv"

test -x "$BIN" || { echo "error: нет $BIN (bash build.sh)" >&2; exit 1; }
grep -qa -- "--probe" "$BIN" || {
	echo "error: $BIN собран без --probe — пересобери: bash build.sh" >&2; exit 1; }
mkdir -p "${OUT_DIR}"
echo "out: ${OUT_DIR}"

# --- Топология CPU узлов и маски рангов ---
srun --ntasks-per-node=1 -n"${NNODES}" bash -c '
	f="$0/topology_$(hostname -s).txt"
	{ lscpu; echo; lscpu -e; echo; echo "# CPU: thread_siblings_list";
	  for c in /sys/devices/system/cpu/cpu[0-9]*; do
		echo "${c##*cpu} $(cat $c/topology/thread_siblings_list 2>/dev/null)"; done | sort -n
	  echo; echo "# scaling_governor:"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
	} > "$f"' "${OUT_DIR}"
srun --gres-flags=allow-task-sharing --cpu-bind=cores bash -c \
	'echo "rank=$SLURM_PROCID local=$SLURM_LOCALID host=$(hostname -s) $(grep Cpus_allowed_list /proc/self/status)"' \
	| sort > "${OUT_DIR}/rank_masks.txt"

# --- Фоновые логи частот GPU и CPU каждые 200 мс ---
srun --overlap --ntasks-per-node=1 -n"${NNODES}" --gpus-per-node=2 bash -c '
	nvidia-smi --query-gpu=timestamp,index,pci.bus_id,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu \
		--format=csv,nounits -lms 200 > "$0/gpu_clocks_$(hostname -s).csv"' "${OUT_DIR}" \
	2>"${OUT_DIR}/gpu_clocks.err" &
BG_PIDS+=($!)
srun --overlap --ntasks-per-node=1 -n"${NNODES}" bash -c '
	f="$0/cpu_freq_$(hostname -s).csv"
	while true; do
		echo "$(date +%s.%N),$(awk "/^cpu MHz/ {printf \"%s;\", \$4}" /proc/cpuinfo)" >> "$f"
		sleep 0.2
	done' "${OUT_DIR}" 2>"${OUT_DIR}/cpu_freq.err" &
BG_PIDS+=($!)

# --- Список запусков в случайном порядке (Фишер — Йетс на bash, см. task3) ---
LIST=()
for LINE in "${PLAN[@]}"; do
	read -r BLOCK REPS ENV PIN CASE <<<"${LINE}"
	for REP in $(seq 1 "${REPS}"); do
		LIST+=("${BLOCK} ${REP} ${ENV} ${PIN} ${CASE}")
	done
done
RANDOM=${SHUFFLE_SEED}
for ((K = ${#LIST[@]} - 1; K > 0; K--)); do
	J=$((RANDOM % (K + 1)))
	TMP=${LIST[K]}; LIST[K]=${LIST[J]}; LIST[J]=${TMP}
done
echo "запусков: ${#LIST[@]}, порядок перемешан (SHUFFLE_SEED=${SHUFFLE_SEED})"
echo "index,block,rep,seed,env,pin,bytes,iters,warmup,start,end" > "${RUNS}"

for IDX in "${!LIST[@]}"; do
	read -r BLOCK REP ENV PIN CASE <<<"${LIST[$IDX]}"
	IFS=":" read -r B I W <<<"${CASE}"
	SEED=$((3000 + IDX))
	OUT="${OUT_DIR}/${SAFE_NODES}_${BLOCK}_rep${REP}_s${SEED}_${ENV}_one_to_one_v2_b${B}_w${W}_i${I}.txt"
	PIN_ARGS=()
	[[ "${PIN}" != "-1" ]] && PIN_ARGS=(--pin-main "${PIN}")
	echo "RUN ${IDX}: ${BLOCK} rep=${REP} env=${ENV} pin=${PIN} bytes=${B}"
	T0=$(date +%s.%N)
	srun --mpi=pmix \
		--kill-on-bad-exit=1 \
		--gres-flags=allow-task-sharing \
		--cpu-bind=verbose,cores \
		"${BIN}" \
		--bytes "${B}" --warmup "${W}" --iters "${I}" --env "${ENV}" \
		"${COMMON[@]}" --seed "${SEED}" ${PIN_ARGS[@]+"${PIN_ARGS[@]}"} --out "${OUT}" > /dev/null
	echo "${IDX},${BLOCK},${REP},${SEED},${ENV},${PIN},${B},${I},${W},${T0},$(date +%s.%N)" >> "${RUNS}"
done

stop_bg
wait 2>/dev/null || true
cp "${GPU_TESTS}/clustbench-hyp5-${SLURM_JOB_ID:-local}.err" "${OUT_DIR}/job.err" 2>/dev/null || true
# Один архив: при скачивании множества мелких файлов часть терялась (hyp_11065, hyp_11078)
tar czf "${OUT_DIR}.tgz" -C "$(dirname "${OUT_DIR}")" "$(basename "${OUT_DIR}")"
echo "архив: ${OUT_DIR}.tgz"
echo "DONE: ${OUT_DIR}"
