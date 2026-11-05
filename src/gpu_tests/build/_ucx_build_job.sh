#!/bin/bash
# Сборка UCX на узле с GPU. Запускается из build_ucx.sh

set -euo pipefail

echo "host=$(hostname -s) UCX_VERSION=$UCX_VERSION UCX_PREFIX=$UCX_PREFIX CUDA_PATH=$CUDA_PATH"
[ -f "$CUDA_PATH/include/cuda_runtime.h" ] || { echo "FATAL: нет $CUDA_PATH/include/cuda_runtime.h"; exit 1; }
nvidia-smi --query-gpu=index,name --format=csv,noheader || true

# Исходники (архив кэшируется в $HOME)
rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK"
if [ ! -f "$HOME/$TARBALL" ]; then
	wget -q "$URL" -O "$TARBALL"   # в кэш — только целиком скачанный архив
	cp "$TARBALL" "$HOME/$TARBALL"
fi
tar xzf "$HOME/$TARBALL"
cd "ucx-$UCX_VERSION"

# gdrcopy — только если найден gdrapi.h (иначе configure падает).
# На крупные сообщения не влияет: там работает GPUDirect RDMA.
GDR_FLAG="--without-gdrcopy"
for p in "${GDRCOPY_PATH:-}" /usr /usr/local "$HOME/opt/gdrcopy"; do
	[ -f "$p/include/gdrapi.h" ] && { GDR_FLAG="--with-gdrcopy=$p"; break; }
done
echo "gdrcopy: ${GDR_FLAG#--}"

CONFIG_LOG="/tmp/ucx_configure_$$.log"
./configure \
	--prefix="$UCX_PREFIX" \
	--enable-cma \
	--with-cuda="$CUDA_PATH" \
	"$GDR_FLAG" \
	--with-verbs \
	--enable-mt \
	--disable-debug \
	--disable-assertions \
	--disable-params-check \
	--disable-go \
	--without-java \
	--disable-doxygen-doc 2>&1 | tee "$CONFIG_LOG"

grep -qE '^#define[[:space:]]+HAVE_CMA[[:space:]]+1' config.h 2>/dev/null \
	|| grep -qE 'UCT modules:.*\bcma\b' "$CONFIG_LOG" \
	|| { echo "FATAL: configure не включил CMA (см. $CONFIG_LOG)"; exit 1; }

# go/java не собираем, но make всё равно заходит в эти каталоги
for d in bindings/go bindings/java; do
	[ -d "$d" ] && printf 'all install install-data install-exec uninstall clean distclean check mostlyclean maintainer-clean:\n\t@true\n' > "$d/Makefile"
done

make -j"$BUILD_JOBS"
make install

"$UCX_PREFIX/bin/ucx_info" -v
"$UCX_PREFIX/bin/ucx_info" -d | grep -E 'Transport: (cma|cuda_copy|cuda_ipc)' \
	|| echo "WARN: в ucx_info -d нет cma или cuda-транспортов"

rm -rf "$WORK" "$CONFIG_LOG"
echo "Готово. UCX установлен в: $UCX_PREFIX"
