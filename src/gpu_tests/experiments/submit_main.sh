#!/bin/bash
# Поставить N заданий основного эксперимента (main.conf) друг за другом на одни и те же узлы.
#   bash experiments/submit_main.sh 8          — 8 заданий подряд
#   NODES=g5500-2,g5500-3 bash experiments/submit_main.sh 4
# Каждое следующее стартует после окончания предыдущего (afterany: даже если оно упало).
# Независимость — между заданиями: свой порядок запусков и пар (зерно — номер задания).
# Лучше ставить не все сразу, а частями в разное время дня: bash ... 4 утром, bash ... 4 вечером.

set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-8}"
NODES="${NODES:-g5500-4,g5500-5}"
TIME="${TIME:-00:25:00}"   # оценка ~14 мин на задание
prev=""
for ((i = 1; i <= N; ++i)); do
	jid=$(bash submit.sh --parsable --time="$TIME" --nodes=2 --ntasks=8 --ntasks-per-node=4 \
		--gpus-per-node=4 --nodelist="$NODES" ${prev:+--dependency=afterany:$prev} task.sbatch main)
	echo "задание $i из $N: $jid"
	prev="$jid"
done
