#!/bin/bash
# Отправка задания с разделом из config.sh (в строках #SBATCH переменные не работают).
#   bash submit.sh                          — task.sbatch, обычное измерение
#   bash submit.sh task.sbatch probe        — эксперимент experiments/probe.conf
#   bash submit.sh --time=00:30:00 my.sbatch — свои опции sbatch и скрипт

set -euo pipefail

cd "$(dirname "$0")"   # задания ищут config.sh и gpu в каталоге отправки
source ./config.sh

[ $# -gt 0 ] || set -- task.sbatch
sbatch --partition="$PARTITION" "$@"
