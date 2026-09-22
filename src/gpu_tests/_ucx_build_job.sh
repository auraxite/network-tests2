#!/bin/bash
# Тело сборки UCX. Выполняется внутри srun-джобы на узле с GPU,
# запускается из build_ucx.sh (там же экспортируются переменные ниже).
# Отдельным файлом — чтобы не терять подсветку синтаксиса в редакторе.

set -euo pipefail

echo "host=$(hostname -s) UCX_VERSION=$UCX_VERSION UCX_PREFIX=$UCX_PREFIX CUDA_PATH=$CUDA_PATH"
[ -f "$CUDA_PATH/include/cuda_runtime.h" ] || { echo "FATAL: $CUDA_PATH/include/cuda_runtime.h не найден"; exit 1; }
nvidia-smi --query-gpu=index,name --format=csv,noheader || true

# Скачивание (с кэшем в $HOME)
rm -rf "$WORK"
mkdir -p "$WORK"
cd "$WORK"

if [ -f "$HOME/$TARBALL" ]; then
	echo "Использую кэш $HOME/$TARBALL"
	cp "$HOME/$TARBALL" .
else
	echo "Скачиваю: $URL"
	wget -q "$URL"
	cp "$TARBALL" "$HOME/$TARBALL"
fi
tar xzf "$TARBALL"
cd "ucx-$UCX_VERSION"

# Configure
CONFIG_LOG="/tmp/ucx_configure_$$.log"
./configure \
	--prefix="$UCX_PREFIX" \
	--enable-cma \
	--with-cuda="$CUDA_PATH" \
	--with-gdrcopy \
	--with-verbs \
	--enable-mt \
	--disable-debug \
	--disable-assertions \
	--disable-params-check \
	--disable-go \
	--without-java \
	--disable-doxygen-doc 2>&1 | tee "$CONFIG_LOG"

if ! grep -qE '^#define[[:space:]]+HAVE_CMA[[:space:]]+1' config.h 2>/dev/null \
	&& ! grep -qE 'UCT modules:.*\bcma\b' "$CONFIG_LOG"; then
	echo "FATAL: CMA не подхватился configure (см. $CONFIG_LOG)"
	exit 1
fi
echo "OK: CMA включён"

# Заглушки для go/java bindings, чтобы не мешали make
for d in bindings/go bindings/java; do
	[ -d "$d" ] || continue
	printf 'all install install-data install-exec uninstall clean distclean check mostlyclean maintainer-clean:\n\t@true\n' > "$d/Makefile"
done

make -j"$BUILD_JOBS"
make install

echo "=== Проверка установки ==="
"$UCX_PREFIX/bin/ucx_info" -v
if "$UCX_PREFIX/bin/ucx_info" -d 2>&1 | grep -qE 'Transport: cma'; then
	echo "OK: CMA транспорт доступен"
else
	echo "WARN: CMA транспорта нет в ucx_info -d"
fi
"$UCX_PREFIX/bin/ucx_info" -b | grep -iE 'have_cma|have_cuda'

rm -rf "$WORK" "$CONFIG_LOG"
echo "Готово. UCX установлен в: $UCX_PREFIX"
