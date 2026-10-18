#!/bin/bash
# Состояния запусков и прогрев: случайный порядок запусков и пар (one_to_one_v2).
# Запуск: bash submit.sh task3.sbatch
# Окружение и привязка как в task.sbatch (пояснения там и в NOTES.md).
#SBATCH --job-name=clustbench-hyp3
#SBATCH --time=01:30:00
#SBATCH --output=clustbench-hyp3-%j.out
#SBATCH --error=clustbench-hyp3-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --gres-flags=enforce-binding

SHUFFLE_SEED=3   # порядок запусков; записывается в runs.csv

# Запуски: "блок повторов среда режим порядок_пар байт:итераций:прогрев"
#   stateR — случайный порядок пар, много повторов одной конфигурации
#   stateS — тот же замер в обычном порядке пар (сравнение с прошлыми данными)
#   stateV — старый режим one_to_one (проверка, что v2 меряет то же)
#   warm   — прогрев w5 / w500 / w2000 при случайном порядке пар
#   long   — длинные серии
#   rooted — host one_to_all и all_to_one (разница внутри узла)
PLAN=(
	"stateR 6 auto one_to_one_v2 random 16000:2000:500"
	"stateR 6 host one_to_one_v2 random 16000:2000:500"
	"stateS 3 auto one_to_one_v2 seq    16000:2000:500"
	"stateS 3 host one_to_one_v2 seq    16000:2000:500"
	"stateV 3 auto one_to_one    seq    16000:2000:500"
	"stateV 3 host one_to_one    seq    16000:2000:500"
	"warm   2 auto one_to_one_v2 random 16000:2000:5"
	"warm   2 auto one_to_one_v2 random 16000:2000:2000"
	"warm   2 auto one_to_one_v2 random 4096000:2000:5"
	"warm   2 auto one_to_one_v2 random 4096000:2000:500"
	"warm   2 auto one_to_one_v2 random 4096000:2000:2000"
	"warm   2 host one_to_one_v2 random 16000:2000:5"
	"warm   2 host one_to_one_v2 random 16000:2000:2000"
	"warm   2 host one_to_one_v2 random 4096000:2000:5"
	"warm   2 host one_to_one_v2 random 4096000:2000:500"
	"warm   2 host one_to_one_v2 random 4096000:2000:2000"
	"long   2 auto one_to_one_v2 random 16000:20000:5"
	"long   2 host one_to_one_v2 random 16000:20000:5"
	"rooted 2 host one_to_all    seq    4096000:500:5"
	"rooted 2 host all_to_one    seq    4096000:500:5"
)

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
grep -qa one_to_one_v2 "$BIN" || {
	echo "error: $BIN собран без one_to_one_v2 — пересобери: bash build.sh" >&2; exit 1; }
mkdir -p "${OUT_DIR}"
echo "out: ${OUT_DIR}"

# --- Фоновые логи на всех узлах, шаги --overlap не занимают ресурсы замеров ---
# Частоты GPU каждые 200 мс.
srun --overlap --ntasks-per-node=1 -n"${NNODES}" --gpus-per-node=2 bash -c '
	nvidia-smi --query-gpu=timestamp,index,pci.bus_id,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu \
		--format=csv,nounits -lms 200 > "$0/gpu_clocks_$(hostname -s).csv"' "${OUT_DIR}" \
	2>"${OUT_DIR}/gpu_clocks.err" &
BG_PIDS+=($!)
# Частоты всех ядер CPU каждые 200 мс: время эпохи, затем МГц по номерам ядер.
srun --overlap --ntasks-per-node=1 -n"${NNODES}" bash -c '
	f="$0/cpu_freq_$(hostname -s).csv"
	while true; do
		echo "$(date +%s.%N),$(awk "/^cpu MHz/ {printf \"%s;\", \$4}" /proc/cpuinfo)" >> "$f"
		sleep 0.2
	done' "${OUT_DIR}" 2>"${OUT_DIR}/cpu_freq.err" &
BG_PIDS+=($!)

# --- Список запусков: каждый повтор отдельно, затем случайный порядок ---
LIST=()
for LINE in "${PLAN[@]}"; do
	read -r BLOCK REPS ENV MODE ORDER CASE <<<"${LINE}"
	for REP in $(seq 1 "${REPS}"); do
		LIST+=("${BLOCK} ${REP} ${ENV} ${MODE} ${ORDER} ${CASE}")
	done
done
# Перемешивание Фишера — Йетса на bash: shuf с <(yes) даёт SIGPIPE, а с
# set -E это сработало бы как ошибка и сняло задание. Фактический порядок — в runs.csv.
RANDOM=${SHUFFLE_SEED}
for ((K = ${#LIST[@]} - 1; K > 0; K--)); do
	J=$((RANDOM % (K + 1)))
	TMP=${LIST[K]}; LIST[K]=${LIST[J]}; LIST[J]=${TMP}
done
echo "запусков: ${#LIST[@]}, порядок перемешан (SHUFFLE_SEED=${SHUFFLE_SEED})"
echo "index,block,rep,seed,env,mode,order,bytes,iters,warmup,start,end" > "${RUNS}"

for IDX in "${!LIST[@]}"; do
	read -r BLOCK REP ENV MODE ORDER CASE <<<"${LIST[$IDX]}"
	IFS=":" read -r B I W <<<"${CASE}"
	SEED=$((1000 + IDX))   # зерно порядка пар, у каждого запуска своё
	OUT="${OUT_DIR}/${SAFE_NODES}_${BLOCK}_rep${REP}_s${SEED}_${ENV}_${MODE}_b${B}_w${W}_i${I}.txt"
	EXTRA=()
	[[ "${MODE}" == one_to_one_v2 ]] && EXTRA=(--order "${ORDER}" --seed "${SEED}")
	echo "RUN ${IDX}: ${BLOCK} rep=${REP} env=${ENV} mode=${MODE} order=${ORDER} bytes=${B} warmup=${W} iters=${I}"
	T0=$(date +%s.%N)
	srun --mpi=pmix \
		--kill-on-bad-exit=1 \
		--gres-flags=allow-task-sharing \
		--cpu-bind=verbose,cores \
		"${BIN}" \
		--bytes "${B}" --warmup "${W}" --iters "${I}" \
		--env "${ENV}" --mode "${MODE}" ${EXTRA[@]+"${EXTRA[@]}"} --out "${OUT}" > /dev/null
	echo "${IDX},${BLOCK},${REP},${SEED},${ENV},${MODE},${ORDER},${B},${I},${W},${T0},$(date +%s.%N)" >> "${RUNS}"
done

stop_bg
wait 2>/dev/null || true
echo "DONE: ${OUT_DIR}"
