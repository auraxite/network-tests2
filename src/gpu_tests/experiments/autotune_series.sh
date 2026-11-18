#!/bin/bash
# Совместный подбор транспорта и порога (autotune_method) на 4 GPU узла: по заданию на набор транспортов,
# задания друг за другом на одни и те же узлы. Первым — all (эталон и его пороги), дальше остальные; результаты
# можно обрабатывать по мере готовности: python3 analysis/autotune.py results/hyp_<готовые задания>.
#   bash experiments/autotune_series.sh
#   SETS="all nogdr" ITERS=100 NODES=g5500-2,g5500-3 bash experiments/autotune_series.sh
# Повтор для проверки разброса между заданиями: REPEAT=2 (каждый набор — два задания).
# Время задания: all — 30 запусков, остальные — 20; при 56 парах ~15–25 с на запуск.

set -euo pipefail
cd "$(dirname "$0")/.."

SETS="${SETS:-all nogdr noipc nic}"
ITERS="${ITERS:-100}"
REPEAT="${REPEAT:-1}"
NODES="${NODES:-g5500-4,g5500-5}"
TIME="${TIME:-00:15:00}"

prev=""
for ((r = 1; r <= REPEAT; ++r)); do
	for S in ${SETS}; do
		jid=$(AUTOTUNE_SETS="${S}" AUTOTUNE_GPUS=all AUTOTUNE_ITERS="${ITERS}" \
			bash submit.sh --parsable --time="${TIME}" --nodes=2 --ntasks=8 --ntasks-per-node=4 \
			--gpus-per-node=4 --nodelist="${NODES}" ${prev:+--dependency=afterany:$prev} task.sbatch autotune_method)
		echo "autotune_method: набор ${S}, повтор ${r}: ${jid}"
		prev="${jid}"
	done
done
