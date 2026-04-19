#!/bin/bash
# Сборка UCX 1.15.0 с поддержкой CMA + CUDA + IB verbs.
#
# Зачем:
#   Существующая сборка UCX без CMA приводит к двойному копиру в intra-node
#   обмене host-буферов через /dev/shm (UCX выбирает транспорт `sm`).
#   С CMA UCX делает одну копию через process_vm_readv/writev в kernel mode.
#
# Использование:
#   bash build_ucx.sh                — запускает srun автоматически
#   srun ... bash build_ucx.sh       — если уже внутри srun
#
# Настраиваемые переменные окружения:
#   UCX_VERSION  (1.15.0)
#   UCX_PREFIX   ($HOME/opt/ucx-${UCX_VERSION})
#   CUDA_PATH    (/usr/local/cuda)
#   PARTITION    (batch)
#   BUILD_JOBS   (4)

set -euo pipefail

UCX_VERSION="${UCX_VERSION:-1.15.0}"
UCX_PREFIX="${UCX_PREFIX:-$HOME/opt/ucx-${UCX_VERSION}}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
PARTITION="${PARTITION:-batch}"
BUILD_JOBS="${BUILD_JOBS:-4}"

# === Самозапуск через srun, если не в Slurm-job ===
if [ -z "${SLURM_JOB_ID:-}" ]; then
	echo "Не в Slurm-job, перезапускаю через srun на узле с GPU..."
	echo "  partition=${PARTITION}, cpus=${BUILD_JOBS}, gpus=1, time=00:45:00"
	exec srun \
		--partition="${PARTITION}" \
		--nodes=1 \
		--ntasks=1 \
		--cpus-per-task="${BUILD_JOBS}" \
		--gpus=1 \
		--time=00:45:00 \
		--job-name=ucx-build \
		bash "$0" "$@"
fi

# === Дальше — мы на compute-узле ===

echo "=== Среда сборки ==="
echo "host:        $(hostname -s)"
echo "UCX_VERSION: ${UCX_VERSION}"
echo "UCX_PREFIX:  ${UCX_PREFIX}"
echo "CUDA_PATH:   ${CUDA_PATH}"
echo "BUILD_JOBS:  ${BUILD_JOBS}"

if [ -f "${CUDA_PATH}/include/cuda_runtime.h" ]; then
	echo "cuda_runtime.h: OK"
else
	echo "cuda_runtime.h: MISSING — поправь CUDA_PATH или возьми другой partition"
	exit 1
fi
nvidia-smi --query-gpu=index,name --format=csv,noheader || true
echo "===================="

# === 1. Удаляем устаревший бэкап (если остался от предыдущих попыток) ===
if [ -d "${UCX_PREFIX}.no-cma.bak" ]; then
	echo "Удаляю старый бэкап ${UCX_PREFIX}.no-cma.bak"
	rm -rf "${UCX_PREFIX}.no-cma.bak"
fi

# === 2. Защита от случайного перезаписывания работающей сборки ===
if [ -e "${UCX_PREFIX}" ]; then
	cat <<EOF
В целевом каталоге уже что-то есть:
  ${UCX_PREFIX}
Это либо сломанная сборка от прошлой попытки, либо рабочая.
Удали или переименуй вручную, например:
  mv "${UCX_PREFIX}" "${UCX_PREFIX}.bak.\$(date +%s)"
И запусти build_ucx.sh снова.
EOF
	exit 1
fi

# === 3. Скачивание исходников ===
WORK="${BUILD_DIR:-$HOME/build_ucx_tmp}"
rm -rf "${WORK}"
mkdir -p "${WORK}"
cd "${WORK}"
echo "Workdir: ${WORK}"

TARBALL="ucx-${UCX_VERSION}.tar.gz"
URL="https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/${TARBALL}"

if [ -f "$HOME/${TARBALL}" ]; then
	echo "Использую кэш $HOME/${TARBALL}"
	cp "$HOME/${TARBALL}" .
else
	echo "Скачиваю: ${URL}"
	wget -q "${URL}"
	# Кэшируем в $HOME, чтобы повторные запуски не лезли в интернет.
	cp "${TARBALL}" "$HOME/${TARBALL}"
fi
tar xzf "${TARBALL}"
cd "ucx-${UCX_VERSION}"

# === 4. Configure ===
CONFIG_LOG="/tmp/ucx_configure_$$.log"
./configure \
	--prefix="${UCX_PREFIX}" \
	--enable-cma \
	--with-cuda="${CUDA_PATH}" \
	--with-verbs \
	--enable-mt \
	--disable-logging \
	--disable-debug \
	--disable-assertions \
	--disable-params-check \
	--disable-go \
	--without-java \
	--disable-doxygen-doc 2>&1 | tee "${CONFIG_LOG}"

echo ""
echo "=== Опции, которые нашёл configure ==="
# В UCX summary имеет формат "UCT modules: < cma knem ... >", "Perf modules: < cuda >",
# "Compiling with verbs support from ...". Печатаем сводку.
grep -iE 'uct modules|cuda modules|perf modules|verbs support|gdr support' \
	"${CONFIG_LOG}" | tail -20 || true

# Жёсткая проверка: CMA обязателен, иначе вся сборка бесполезна.
# Самый надёжный сигнал — наличие HAVE_CMA в сгенерированном config.h.
if [ -f "config.h" ] && grep -qE '^#define[[:space:]]+HAVE_CMA[[:space:]]+1' config.h; then
	echo "OK: HAVE_CMA найден в config.h"
elif grep -qE 'UCT modules:.*\bcma\b' "${CONFIG_LOG}"; then
	echo "OK: cma в списке UCT modules"
else
	echo ""
	echo "FATAL: CMA не подхватился configure'ом."
	echo "Возможно, не хватает glibc заголовков или ядро < 3.2."
	echo "Полный лог: ${CONFIG_LOG}"
	echo ""
	echo "Полезные строки из лога для диагностики:"
	grep -iE 'cma|process_vm' "${CONFIG_LOG}" | tail -20 || true
	exit 1
fi

# Нейтрализуем проблемные bindings, которые UCX 1.15 release tarball пытается
# собрать, даже если configure-опции для их отключения "сработали" (они
# молча игнорируются). Заменяем их Makefile на пустышки — make войдёт туда,
# увидит валидный Makefile с правилами all/install/clean, которые ничего не
# делают, и пойдёт дальше.
for STUB_DIR in bindings/go bindings/java; do
	if [ -d "${STUB_DIR}" ]; then
		cat > "${STUB_DIR}/Makefile" <<'STUB'
# Stub Makefile: bindings отключён билд-скриптом, чтобы не ломать make.
all:
	@echo "skip: bindings отключены"
install:
	@echo "skip: bindings отключены"
install-data:
	@true
install-exec:
	@true
uninstall:
	@true
clean:
	@true
distclean:
	@rm -f Makefile
check:
	@true
mostlyclean:
	@true
maintainer-clean:
	@true
.PHONY: all install install-data install-exec uninstall clean distclean check mostlyclean maintainer-clean
STUB
		echo "Нейтрализован ${STUB_DIR}/Makefile (stub)"
	fi
done

# === 5. Сборка и установка ===
echo ""
echo "=== make -j${BUILD_JOBS} ==="
make -j"${BUILD_JOBS}"
make install

# === 6. Финальные проверки ===
echo ""
echo "=== Версия установленного UCX ==="
"${UCX_PREFIX}/bin/ucx_info" -v

echo ""
echo "=== CMA транспорт в ucx_info -d ==="
if "${UCX_PREFIX}/bin/ucx_info" -d 2>&1 | grep -qE 'Transport: cma'; then
	"${UCX_PREFIX}/bin/ucx_info" -d 2>&1 | grep -B1 -A2 'Transport: cma' | head -10
	echo "OK: CMA доступен"
else
	echo "WARN: CMA транспорта нет в ucx_info -d (что-то странное при сборке)"
fi

echo ""
echo "=== Build defines (HAVE_CMA / HAVE_CUDA) ==="
"${UCX_PREFIX}/bin/ucx_info" -b | grep -iE 'have_cma|have_cuda' | head -10

# === 7. Уборка ===
echo ""
echo "Удаляю workdir ${WORK}"
rm -rf "${WORK}"
rm -f "${CONFIG_LOG}"

cat <<EOF

=================================================================
Готово. UCX установлен в:
  ${UCX_PREFIX}

Следующий шаг — sbatch fixed.slurm (там уже UCX_TLS=cma,... настроен).
В diagnostic-блоке должно появиться:
  UCX transports with cma:
  #      Transport: cma
=================================================================
EOF
