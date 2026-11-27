# Настройки gpu_tests2: те же раздел, версии и пути, что у gpu_tests (config.sh рядом),
# своя только программа. Подключается через source из build/build.sh, submit.sh и task.sbatch.
_GPU2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_GPU2_DIR}/../gpu_tests/config.sh"

# Программа на C (gpu_tests2), своя для каждой версии UCX. Имя не совпадает с программой
# gpu_tests (gpu-ucx-*), чтобы их нельзя было перепутать в журналах заданий.
GPU_BIN="${GPU2_BIN:-gpu2-ucx-$UCX_VERSION}"
