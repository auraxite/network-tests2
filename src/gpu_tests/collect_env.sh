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
	# где в дереве PCIe каждый адаптер и GPU и к какому процессору (NUMA) он подключён
	section bash -c 'for d in /sys/class/infiniband/*; do
		p=$(readlink -f "$d/device"); n=$(basename "$d")
		echo "nic $n pci=$(basename "$p") numa=$(cat "$p/numa_node" 2>/dev/null)" \
			"netdev=$(ls "$p/net" 2>/dev/null | tr "\n" " ")path=$p"
	done
	for g in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null); do
		b=$(echo "0000:${g#*:}" | tr "A-F" "a-f")   # 00000000:56:00.0 → 0000:56:00.0
		p=$(readlink -f "/sys/bus/pci/devices/$b")
		echo "gpu $g numa=$(cat "$p/numa_node" 2>/dev/null) path=$p"
	done'
	# таблица GID RoCE: номер, тип, интерфейс (у VLAN свои GID), адрес
	section bash -c 'for d in /sys/class/infiniband/*; do for a in "$d"/ports/1/gid_attrs/ndevs/*; do
		i=${a##*/}; nd=$(cat "$a" 2>/dev/null) || continue
		echo "gid $(basename "$d") $i $(cat "$d/ports/1/gid_attrs/types/$i" 2>/dev/null | tr " " _) $nd $(cat "$d/ports/1/gids/$i")"
	done; done'
	# процессоры, их память и расстояния между NUMA-узлами
	section numactl -H
	# адреса сетевых интерфейсов: по ним проверяется, связаны ли Ethernet-адаптеры узлов
	section ip -br -4 addr
	section lscpu
	# какие транспорты есть в собранной UCX и с какими параметрами
	section "${UCX_INFO}" -v
	section "${UCX_INFO}" -d
	# загружен ли модуль для GPUDirect RDMA
	section bash -c 'lsmod | grep -E "nvidia_peermem|nv_peer_mem|gdrdrv" || echo "модули не найдены"'
} > "${F}"
echo "env: ${F}"
