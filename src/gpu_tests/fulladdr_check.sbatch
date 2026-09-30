#!/bin/bash
#SBATCH --partition=intel-a100-pci5
#SBATCH --job-name=fulladdr-check
#SBATCH --time=00:20:00
#SBATCH --output=fulladdr-check-%j.out
#SBATCH --error=fulladdr-check-%j.err
#SBATCH --nodelist=g5500-[2,4]
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --gres-flags=enforce-binding

# Проверка Open MPI, пересобранного с правкой pml_ucx.c (build_openmpi.sh,
# см. _openmpi_build_job.sh). Задание на двух узлах, запуск через srun, как во
# всех измерениях, программа ./gpu (её пересобирать не нужно: библиотека
# подхватывается из ~/opt/openmpi-5.0.10).
# Ожидание: пары внутри узла ~70 мкс при 16 МБ, в UCX_PROTO_INFO — cuda_ipc.
# До правки было ~830 мкс и rc_mlx5 (через сетевой адаптер).
#
# Запуск: cd ~/network-tests2/src/gpu_tests && sbatch fulladdr_check.sbatch
# Результаты: results_fulladdr_check/ и fulladdr-check-<job>.out/.err

UCX_PATH="$HOME/opt/ucx-1.15.0"
CUDA_PATH="/usr/local/cuda"

set -eEuo pipefail

export OMPI_MCA_opal_cuda_support=1
export OMPI_MCA_pml=ucx
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset UCX_RNDV_SCHEME UCX_MEMTYPE_CACHE UCX_TLS UCX_NET_DEVICES \
      UCX_MAX_RNDV_RAILS UCX_PROTO_ENABLE UCX_PROTO_INFO
export UCX_RNDV_THRESH=0

GPU_TESTS="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT_DIR="${GPU_TESTS}/results_fulladdr_check"
rm -rf "${OUT_DIR}"          # сырые замеры дописываются в конец файла
mkdir -p "${OUT_DIR}"

run() {  # run <бинарник> <каталог Open MPI> <метка> <mode> <bytes> <iters>
	local bin="$1" ompi="$2" tag="$3" mode="$4" bytes="$5" iters="$6"
	local out="${OUT_DIR}/${tag}_auto_${mode}_b${bytes}_w5_i${iters}.txt"
	test -x "${GPU_TESTS}/${bin}" || { echo "нет ${bin}"; return 0; }
	echo "=== RUN ${tag} bin=${bin} ompi=${ompi} mode=${mode} bytes=${bytes}"
	PATH="$UCX_PATH/bin:$ompi/bin:$PATH" \
	LD_LIBRARY_PATH="$UCX_PATH/lib:$ompi/lib:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
	srun --mpi=pmix --kill-on-bad-exit=1 \
		--gres-flags=allow-task-sharing --cpu-bind=cores \
		"${GPU_TESTS}/${bin}" --bytes "${bytes}" --warmup 5 --iters "${iters}" \
		--env auto --mode "${mode}" --out "${out}"
	grep -E 'pci=|^CUDA-aware' "${out}" | sed 's/^/   /'
	grep '^pair' "${out}" | awk '$2!=$4 {print "   ", $2, "->", $4, $6}' || true
}

OMPI="$HOME/opt/openmpi-5.0.10"
echo "правка в исходниках Open MPI: $(grep -c 'clustbench: полный адрес' "$HOME/src/openmpi-5.0.10/ompi/mca/pml/ucx/pml_ucx.c" 2>/dev/null || echo '?')"

run gpu "$OMPI" fulladdr one_to_one 16384000 100
run gpu "$OMPI" fulladdr one_to_one 1024000 1000
run gpu "$OMPI" fulladdr all_to_all 16384000 200

# Таблица выбора протоколов (ожидается cuda_ipc внутри узла)
UCX_PROTO_ENABLE=y UCX_PROTO_INFO=y \
	run gpu "$OMPI" fulladdr_protoinfo one_to_one 16384000 20

echo "=== DONE"
