#!/bin/bash
# Снимок топологии узла и возможностей UCX: из чего складываются возможные пути передачи.
# Запуск на каждом узле задания:
#   srun --ntasks-per-node=1 -n<число узлов> bash collect_env.sh <каталог>
# или на одном узле вручную: bash collect_env.sh <каталог>
# Пишет <каталог>/env_<узел>.txt. Недоступные команды пропускаются.

set -u
OUT_DIR="${1:-.}"
source "$(dirname "$0")/config.sh"
mkdir -p "${OUT_DIR}"
F="${OUT_DIR}/env_$(hostname -s).txt"

UCX_INFO=ucx_info
[[ -x "${UCX_PATH}/bin/ucx_info" ]] && UCX_INFO="${UCX_PATH}/bin/ucx_info"

section() {
	echo "===== $* ====="
	"$@" 2>&1 || echo "(недоступно: $1)"
	echo
}

{
	echo "# $(hostname -f) $(date -Is)"
	echo
	# соединения GPU, NIC и CPU: NV#, PIX, PXB, PHB, NODE, SYS
	section nvidia-smi topo -m
	# разрешён ли прямой доступ GPU → GPU (чтение и запись)
	section nvidia-smi topo -p2p r
	section nvidia-smi topo -p2p w
	section nvidia-smi nvlink -s
	section nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version --format=csv
	# сетевые адаптеры: тип, скорость, состояние порта
	section ibv_devinfo
	section ibstat
	# дерево PCIe: какие устройства под общим коммутатором
	section lspci -tv
	section lscpu
	# какие транспорты есть в собранной UCX и с какими параметрами
	section "${UCX_INFO}" -v
	section "${UCX_INFO}" -d
	# загружен ли модуль для GPUDirect RDMA
	section bash -c 'lsmod | grep -E "nvidia_peermem|nv_peer_mem|gdrdrv" || echo "модули не найдены"'
} > "${F}"
echo "env: ${F}"
