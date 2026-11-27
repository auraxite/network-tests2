#!/bin/bash
# Отправка задания для программы gpu2 (C). Скрипт задания — общий, ../gpu_tests/task.sbatch:
# он берёт config.sh, experiments/, программу и results/ из каталога отправки, то есть отсюда.
#   bash submit.sh                           — task.sbatch, обычное измерение (experiments/default.conf)
#   bash submit.sh task.sbatch host_compare  — эксперимент experiments/host_compare.conf
#   bash submit.sh --time=00:30:00 task.sbatch main
# Описания экспериментов, которых здесь нет, копируются из ../gpu_tests/experiments при первом запуске.

set -euo pipefail

cd "$(dirname "$0")"
source ./config.sh

TASK="$(cd ../gpu_tests && pwd)/task.sbatch"
[ $# -gt 0 ] || set -- task.sbatch

ARGS=()
SEEN_TASK=0
for a in "$@"; do
	if [[ "$SEEN_TASK" == 0 && ( "$a" == task.sbatch || "$a" == */task.sbatch ) ]]; then
		ARGS+=("$TASK")
		SEEN_TASK=1
	elif [[ "$SEEN_TASK" == 1 ]]; then
		ARGS+=("$a")
		SEEN_TASK=2   # первый аргумент после скрипта — имя эксперимента
		if [[ ! -f "experiments/$a.conf" && -f "../gpu_tests/experiments/$a.conf" ]]; then
			cp "../gpu_tests/experiments/$a.conf" "experiments/$a.conf"
			echo "submit.sh: experiments/$a.conf скопирован из gpu_tests" >&2   # stdout занят номером задания (--parsable)
		fi
	else
		ARGS+=("$a")
	fi
done
if [[ "$SEEN_TASK" == 1 && ! -f experiments/default.conf ]]; then
	cp ../gpu_tests/experiments/default.conf experiments/default.conf
fi

[ -x "./$GPU_BIN" ] || { echo "submit.sh: нет ./$GPU_BIN — сначала bash build/build.sh" >&2; exit 1; }
sbatch --partition="$PARTITION" "${ARGS[@]}"
