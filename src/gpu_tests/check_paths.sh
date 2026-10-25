#!/bin/bash
# Проверка на практике, какие пути передачи из перечня enum_paths.py работают.
# Каждый путь включается принудительно настройками UCX и проверяется ucx_perftest
# (тест задержки ping-pong, 4 МБ из памяти GPU): прошла ли передача, задержка,
# журнал UCX и счётчики NVLink и адаптеров до и после.
# Запуск: bash submit.sh check_paths.sh ; результат: results/paths_<задание>.tgz
# Около 3–5 минут. На узле берутся все GPU, чтобы снимок показал полную топологию.
#SBATCH --job-name=clustbench-paths
#SBATCH --time=00:10:00
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

# --- Снимок топологии обоих узлов ---
srun --ntasks-per-node=1 -N2 --gpus-per-node="${GPN}" bash "${GPU_TESTS}/collect_env.sh" "${OUT}"

# --- Связность сети: ping с первого узла до каждого адреса второго ---
srun -N1 -n1 -w "${B}" ip -o -4 addr show | awk '{print $2, $4}' > "${OUT}/ip_${B}.txt"
echo "iface,ip,ping_ok" > "${OUT}/ping.csv"
while read -r ifc cidr; do
	[[ "${ifc}" == lo ]] && continue
	ip=${cidr%/*}
	if srun -N1 -n1 -w "${A}" ping -c 2 -W 1 "${ip}" > "${OUT}/ping_${ifc}.txt" 2>&1; then ok=1; else ok=0; fi
	echo "${ifc},${ip},${ok}" >> "${OUT}/ping.csv"
done < "${OUT}/ip_${B}.txt"

# --- Счётчики: байты от адаптеров (port_xmit_data в 4-байтовых словах) и трафик NVLink ---
snap() {
	srun --overlap --ntasks-per-node=1 -N2 --gpus-per-node="${GPN}" bash -c '
		f="$0/counters/$(hostname -s)_$1.txt"
		{ for d in /sys/class/infiniband/*; do for p in "$d"/ports/*; do
			echo "nic $(basename "$d") $(basename "$p") xmit_words $(cat "$p"/counters/port_xmit_data 2>/dev/null || echo NA)"
		  done; done
		  echo "# nvidia-smi nvlink -gt d"; nvidia-smi nvlink -gt d 2>&1 || echo NA
		} > "$f"' "${OUT}" "$1"
}

# --- Один прогон ucx_perftest: путь включается переменными UCX ---
# run ИМЯ УЗЕЛ_СЕРВЕРА GPU_СЕРВЕРА УЗЕЛ_КЛИЕНТА GPU_КЛИЕНТА ПАМЯТЬ ПЕРЕМЕННЫЕ...
echo "name,server,server_gpu,client,client_gpu,mem,env,exit" > "${OUT}/perf.csv"
PORT=20000
run() {
	local name=$1 sn=$2 sg=$3 cn=$4 cg=$5 mem=$6; shift 6
	PORT=$((PORT + 1))
	local envs=(UCX_LOG_LEVEL=info "$@")
	echo "RUN ${name}: ${sn}:GPU${sg} <-> ${cn}:GPU${cg} mem=${mem} ${envs[*]}"
	snap "${name}_before"
	env "${envs[@]}" srun --overlap -N1 -n1 -w "${sn}" --gpus-per-node="${GPN}" --cpu-bind=none bash -c \
		"CUDA_VISIBLE_DEVICES=${sg} exec timeout 60 ${PERF} -p ${PORT} -t tag_lat -m ${mem} -s ${SIZE} -n ${ITERS}" \
		> "${OUT}/perf/${name}.server" 2>&1 &
	local spid=$!
	sleep 3
	env "${envs[@]}" srun --overlap -N1 -n1 -w "${cn}" --gpus-per-node="${GPN}" --cpu-bind=none bash -c \
		"CUDA_VISIBLE_DEVICES=${cg} exec timeout 50 ${PERF} ${sn} -p ${PORT} -t tag_lat -m ${mem} -s ${SIZE} -n ${ITERS}" \
		> "${OUT}/perf/${name}.client" 2>&1
	local rc=$?
	wait "${spid}" 2>/dev/null
	snap "${name}_after"
	local envstr="$*"
	echo "${name},${sn},${sg},${cn},${cg},${mem},\"${envstr// /;}\",${rc}" >> "${OUT}/perf.csv"
}

# работающие порты адаптеров (из снимка первого узла)
NICS=($(awk '/^hca_id:/{d=$2} /PORT_ACTIVE/{print d}' "${OUT}/env_${A}.txt" | sort -u))
NGPU=$(srun -N1 -n1 -w "${A}" --gpus-per-node="${GPN}" nvidia-smi -L 2>/dev/null | wc -l)
echo "адаптеры: ${NICS[*]}, GPU на узле: ${NGPU}"

# Внутри узла: пары GPU0–GPU1 и, если GPU больше двух, GPU0–GPU2 (другой коммутатор PCIe)
PAIRS=("0 1")
[[ "${NGPU}" -ge 3 ]] && PAIRS+=("0 2")
for PAIR in "${PAIRS[@]}"; do
	read -r g1 g2 <<<"${PAIR}"
	tag="intra_g${g1}g${g2}"
	run "${tag}_direct"  "${A}" "${g1}" "${A}" "${g2}" cuda UCX_TLS=cuda_ipc,cuda_copy,self
	run "${tag}_ram_sm"  "${A}" "${g1}" "${A}" "${g2}" cuda UCX_TLS=cuda_copy,sm,self
	run "${tag}_ram_cma" "${A}" "${g1}" "${A}" "${g2}" cuda UCX_TLS=cuda_copy,cma,self
	for d in "${NICS[@]}"; do
		run "${tag}_nic_${d}_gdr" "${A}" "${g1}" "${A}" "${g2}" cuda \
			UCX_TLS=rc_mlx5,cuda_copy,self UCX_NET_DEVICES="${d}:1" UCX_IB_GPU_DIRECT_RDMA=y
		run "${tag}_nic_${d}_ram" "${A}" "${g1}" "${A}" "${g2}" cuda \
			UCX_TLS=rc_mlx5,cuda_copy,self UCX_NET_DEVICES="${d}:1" UCX_IB_GPU_DIRECT_RDMA=n
	done
done

# Между узлами: GPU0 первого узла ↔ GPU0 второго через каждый адаптер
for d in "${NICS[@]}"; do
	run "inter_${d}_gdr"  "${B}" 0 "${A}" 0 cuda UCX_TLS=rc_mlx5,cuda_copy,self UCX_NET_DEVICES="${d}:1" UCX_IB_GPU_DIRECT_RDMA=y
	run "inter_${d}_ram"  "${B}" 0 "${A}" 0 cuda UCX_TLS=rc_mlx5,cuda_copy,self UCX_NET_DEVICES="${d}:1" UCX_IB_GPU_DIRECT_RDMA=n
	run "inter_${d}_host" "${B}" 0 "${A}" 0 host UCX_TLS=rc_mlx5,self UCX_NET_DEVICES="${d}:1"
done

cp "${GPU_TESTS}/clustbench-paths-${SLURM_JOB_ID:-local}.out" "${OUT}/job.out" 2>/dev/null || true
cp "${GPU_TESTS}/clustbench-paths-${SLURM_JOB_ID:-local}.err" "${OUT}/job.err" 2>/dev/null || true
tar czf "${OUT}.tgz" -C "$(dirname "${OUT}")" "$(basename "${OUT}")"
echo "архив: ${OUT}.tgz"
