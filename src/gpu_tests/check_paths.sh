#!/bin/bash
# Проверка на практике, какие пути передачи из перечня enum_paths.py работают.
# Каждый путь включается принудительно настройками UCX и проверяется ucx_perftest
# (тест задержки ping-pong, 4 МБ из памяти GPU): прошла ли передача, задержка,
# журнал UCX и счётчики NVLink и адаптеров до и после.
# Запуск: bash submit.sh check_paths.sh ; результат: results/paths_<задание>.tgz
# 10–20 минут (около 35 тестов). На узле берутся все GPU, чтобы снимок
# показал полную топологию. За 90с до лимита времени архив собирается из того, что успело.
#SBATCH --job-name=clustbench-paths
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=clustbench-paths-%j.out
#SBATCH --error=clustbench-paths-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4

source "${SLURM_SUBMIT_DIR:-.}/config.sh"
set -u   # без -e: неработающий путь — тоже результат

export PATH="$OMPI_PATH/bin:$PATH"
_LDPATH="$OMPI_PATH/lib:${CUDA_PATH}/lib64"
[[ -d "$UCX_PATH/lib" ]] && _LDPATH="$UCX_PATH/lib:$_LDPATH"
export LD_LIBRARY_PATH="$_LDPATH:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # номера GPU как в nvidia-smi
unset UCX_TLS UCX_NET_DEVICES UCX_IB_GPU_DIRECT_RDMA UCX_RNDV_THRESH UCX_RNDV_SCHEME UCX_PROTO_ENABLE UCX_PROTO_INFO

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
PERF="${UCX_PATH}/bin/ucx_perftest"
OUT="${GPU_TESTS}/results/paths_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT}/perf" "${OUT}/counters"
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
A=${NODES[0]}; B=${NODES[1]}
SIZE=4194304; ITERS=100
GPN="${SLURM_GPUS_PER_NODE:-4}"; GPN=${GPN##*:}   # GPU на узел в выделении (вид «4» или «a100:4»)
test -x "${PERF}" || { echo "error: нет ${PERF}" >&2; exit 1; }
echo "out: ${OUT}, узлы ${A} ${B}"

# --- Упаковка результатов: в конце или по сигналу перед лимитом времени ---
pack() {
	cp "${GPU_TESTS}/clustbench-paths-${SLURM_JOB_ID:-local}.out" "${OUT}/job.out" 2>/dev/null || true
	cp "${GPU_TESTS}/clustbench-paths-${SLURM_JOB_ID:-local}.err" "${OUT}/job.err" 2>/dev/null || true
	tar czf "${OUT}.tgz" -C "$(dirname "${OUT}")" "$(basename "${OUT}")"
	echo "архив: ${OUT}.tgz"
}
trap 'echo "лимит времени близко, упаковка"; pack; exit 1' USR1

# --- Снимок топологии обоих узлов ---
srun --ntasks-per-node=1 -N2 --gpus-per-node="${GPN}" bash "${GPU_TESTS}/collect_env.sh" "${OUT}"

# --- Связность сети: ping с первого узла до каждого адреса второго ---
srun -N1 -n1 -w "${B}" ip -o -4 addr show | awk '{print $2, $4}' > "${OUT}/ip_${B}.txt"
echo "iface,ip,ping_ok" > "${OUT}/ping.csv"
while read -r ifc cidr; do
	[[ "${ifc}" == lo ]] && continue
	ip=${cidr%/*}
	# < /dev/null: иначе srun съедает остальные строки списка
	if srun -N1 -n1 -w "${A}" ping -c 2 -W 1 "${ip}" < /dev/null > "${OUT}/ping_${ifc}.txt" 2>&1; then ok=1; else ok=0; fi
	echo "${ifc},${ip},${ok}" >> "${OUT}/ping.csv"
done < "${OUT}/ip_${B}.txt"

# --- Счётчики: байты от адаптеров (port_xmit_data в 4-байтовых словах, для Ethernet ещё
# tx_bytes интерфейса) и трафик NVLink ---
snap() {
	srun --overlap --ntasks-per-node=1 -N2 --gpus-per-node="${GPN}" bash -c '
		f="$0/counters/$(hostname -s)_$1.txt"
		{ for d in /sys/class/infiniband/*; do for p in "$d"/ports/*; do
			echo "nic $(basename "$d") $(basename "$p") xmit_words $(cat "$p"/counters/port_xmit_data 2>/dev/null || echo NA)"
		  done
		  for n in "$d"/device/net/*; do   # Ethernet: байты сетевого интерфейса
			[ -e "$n" ] || continue
			echo "netdev $(basename "$d") $(basename "$n") tx_bytes $(cat "$n"/statistics/tx_bytes)"
			ethtool -S "$(basename "$n")" 2>/dev/null | grep -i "rdma.*bytes" | sed "s/^ */ethtool $(basename "$n") /"
		  done; done
		  echo "# nvidia-smi nvlink -gt d"; nvidia-smi nvlink -gt d 2>&1 || echo NA
		} > "$f"' "${OUT}" "$1"
}

# --- Один прогон ucx_perftest: путь включается переменными UCX ---
# run ИМЯ УЗЕЛ_СЕРВЕРА GPU_СЕРВЕРА УЗЕЛ_КЛИЕНТА GPU_КЛИЕНТА ПАМЯТЬ [NUMA=с:к] ПЕРЕМЕННЫЕ...
# NUMA=с:к — память процессов сервера и клиента на NUMA-узлах с и к (буферы в ОЗУ).
# UCX нужен транспорт служебных сообщений: cuda_ipc и cma их не передают (нужен posix),
# rc_mlx5 соединяется через ud_mlx5.
echo "name,server,server_gpu,client,client_gpu,mem,numa,env,exit,sec" > "${OUT}/perf.csv"
PORT=20000
run() {
	local name=$1 sn=$2 sg=$3 cn=$4 cg=$5 mem=$6; shift 6
	local numa="" nsrv="" ncli=""
	if [[ "${1:-}" == NUMA=* ]]; then
		numa=${1#NUMA=}; shift
		nsrv="numactl --membind=${numa%:*}"; ncli="numactl --membind=${numa#*:}"
	fi
	PORT=$((PORT + 1))
	local envs=(UCX_LOG_LEVEL=info "$@")
	echo "RUN ${name}: ${sn}:GPU${sg} <-> ${cn}:GPU${cg} mem=${mem} ${envs[*]}"
	local t0=${SECONDS}
	snap "${name}_before"
	env "${envs[@]}" srun --overlap -N1 -n1 -w "${sn}" --gpus-per-node="${GPN}" --cpu-bind=none bash -c \
		"CUDA_VISIBLE_DEVICES=${sg} exec timeout 40 ${nsrv} ${PERF} -p ${PORT} -t tag_lat -m ${mem} -s ${SIZE} -n ${ITERS}" \
		> "${OUT}/perf/${name}.server" 2>&1 &
	local spid=$!
	sleep 3
	env "${envs[@]}" srun --overlap -N1 -n1 -w "${cn}" --gpus-per-node="${GPN}" --cpu-bind=none bash -c \
		"CUDA_VISIBLE_DEVICES=${cg} exec timeout 30 ${ncli} ${PERF} ${sn} -p ${PORT} -t tag_lat -m ${mem} -s ${SIZE} -n ${ITERS}" \
		> "${OUT}/perf/${name}.client" 2>&1
	local rc=$?
	wait "${spid}" 2>/dev/null
	snap "${name}_after"
	local envstr="$*"
	echo "${name},${sn},${sg},${cn},${cg},${mem},${numa},\"${envstr// /;}\",${rc},$((SECONDS - t0))" >> "${OUT}/perf.csv"
	echo "  exit=${rc} за $((SECONDS - t0))с"
}

# работающие порты адаптеров (из снимка первого узла)
NICS=($(awk '/^hca_id:/{d=$2} /PORT_ACTIVE/{print d}' "${OUT}/env_${A}.txt" | sort -u))
NGPU=$(srun -N1 -n1 -w "${A}" --gpus-per-node="${GPN}" nvidia-smi -L 2>/dev/null | wc -l)
echo "адаптеры: ${NICS[*]}, GPU на узле: ${NGPU}"

# Внутри узла: GPU0–GPU1 (NVLink) и, если GPU больше двух, GPU0–GPU2 (PCIe через оба процессора)
PAIRS=("0 1")
[[ "${NGPU}" -ge 3 ]] && PAIRS+=("0 2")
RC="rc_mlx5,ud_mlx5"
for PAIR in "${PAIRS[@]}"; do
	read -r g1 g2 <<<"${PAIR}"
	tag="intra_g${g1}g${g2}"
	run "${tag}_direct"  "${A}" "${g1}" "${A}" "${g2}" cuda UCX_TLS=cuda_ipc,cuda_copy,posix,self
	# через ОЗУ: буфер в памяти процессора 0 или 1, у сервера и клиента одинаково или по-разному
	for nm in 0:0 1:1 0:1; do
		run "${tag}_ram_posix_n${nm/:/}" "${A}" "${g1}" "${A}" "${g2}" cuda NUMA=${nm} UCX_TLS=cuda_copy,posix,self
	done
	run "${tag}_ram_cma"  "${A}" "${g1}" "${A}" "${g2}" cuda UCX_TLS=cuda_copy,posix,cma,self
	for d in "${NICS[@]}"; do
		run "${tag}_nic_${d}_gdr" "${A}" "${g1}" "${A}" "${g2}" cuda 			UCX_TLS=${RC},cuda_copy,self UCX_NET_DEVICES="${d}:1" UCX_IB_GPU_DIRECT_RDMA=y
		run "${tag}_nic_${d}_ram" "${A}" "${g1}" "${A}" "${g2}" cuda 			UCX_TLS=${RC},cuda_copy,self UCX_NET_DEVICES="${d}:1" UCX_IB_GPU_DIRECT_RDMA=n
	done
done

# Между узлами: GPU0 ↔ GPU0 через каждый адаптер
for d in "${NICS[@]}"; do
	run "inter_${d}_gdr"  "${B}" 0 "${A}" 0 cuda UCX_TLS=${RC},cuda_copy,self UCX_NET_DEVICES="${d}:1" UCX_IB_GPU_DIRECT_RDMA=y
	run "inter_${d}_ram"  "${B}" 0 "${A}" 0 cuda UCX_TLS=${RC},cuda_copy,self UCX_NET_DEVICES="${d}:1" UCX_IB_GPU_DIRECT_RDMA=n
	run "inter_${d}_host" "${B}" 0 "${A}" 0 host UCX_TLS=${RC},self UCX_NET_DEVICES="${d}:1"
done
# GPU2 ↔ GPU2: GPU у второго процессора, все адаптеры у первого
if [[ "${NGPU}" -ge 3 && ${#NICS[@]} -gt 0 ]]; then
	d=${NICS[0]}
	run "inter_g2_${d}_gdr" "${B}" 2 "${A}" 2 cuda UCX_TLS=${RC},cuda_copy,self UCX_NET_DEVICES="${d}:1" UCX_IB_GPU_DIRECT_RDMA=y
	run "inter_g2_${d}_ram" "${B}" 2 "${A}" 2 cuda UCX_TLS=${RC},cuda_copy,self UCX_NET_DEVICES="${d}:1" UCX_IB_GPU_DIRECT_RDMA=n
fi

pack
