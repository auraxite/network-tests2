#!/bin/bash
# task.sbatch вызывает paths/collect_env.sh из каталога отправки; сбор окружения общий с gpu_tests.
exec bash "$(cd "$(dirname "$0")/../../gpu_tests/paths" && pwd)/collect_env.sh" "$@"
