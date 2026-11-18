#!/bin/bash
# Поставить N независимых заданий одного эксперимента друг за другом на одни и те же узлы (шаг 5 методики:
# оценка — по нескольким независимым запускам).
#   bash experiments/submit_series.sh modes_method 4
#   bash experiments/submit_series.sh p2pcheck 4
#   NODES=g5500-2,g5500-3 bash experiments/submit_series.sh modes_method 4
# Число процессов и время берутся по эксперименту (ниже); их можно переопределить: NTASKS=, PER_NODE=, TIME=.
# Каждое следующее задание стартует после окончания предыдущего (afterany: даже если оно упало).
# Независимость — между заданиями: свой порядок запусков, пар и раундов (зерно — номер задания).

set -euo pipefail
cd "$(dirname "$0")/.."

CONF="${1:?имя эксперимента (experiments/<имя>.conf)}"
N="${2:-4}"
[[ -f "experiments/${CONF}.conf" ]] || { echo "нет experiments/${CONF}.conf" >&2; exit 1; }

case "${CONF}" in
	modes|modes_method) DEF_NTASKS=4; DEF_PER_NODE=2; DEF_TIME=00:20:00 ;;
	autothresh_method)  DEF_NTASKS=8; DEF_PER_NODE=4; DEF_TIME=00:50:00 ;;
	warmup_method)      DEF_NTASKS=4; DEF_PER_NODE=2; DEF_TIME=00:06:00 ;;
	autotransport_method) DEF_NTASKS=8; DEF_PER_NODE=4; DEF_TIME=00:40:00 ;;
	*)                  DEF_NTASKS=8; DEF_PER_NODE=4; DEF_TIME=00:25:00 ;;
esac
NTASKS="${NTASKS:-$DEF_NTASKS}"
PER_NODE="${PER_NODE:-$DEF_PER_NODE}"
TIME="${TIME:-$DEF_TIME}"
NODES="${NODES:-g5500-4,g5500-5}"

prev=""
for ((i = 1; i <= N; ++i)); do
	jid=$(bash submit.sh --parsable --time="$TIME" --nodes=2 --ntasks="$NTASKS" --ntasks-per-node="$PER_NODE" \
		--gpus-per-node=4 --nodelist="$NODES" ${prev:+--dependency=afterany:$prev} task.sbatch "$CONF")
	echo "${CONF}: задание $i из $N: $jid"
	prev="$jid"
done
