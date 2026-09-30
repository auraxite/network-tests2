#!/bin/sh
# Сборка статьи: pdflatex дважды (второй проход проставляет номера ссылок
# на литературу), удаление вспомогательных файлов и упаковка в submit/
# того, что отправляется в сборник.
set -e
cd "$(dirname "$0")"

# MiKTeX без --enable-installer открывает диалог установки пакетов,
# в TeX Live такого флага нет.
if pdflatex --version | grep -q MiKTeX; then
    OPTS="--enable-installer"
else
    OPTS=""
fi

for i in 1 2; do
    if ! pdflatex $OPTS -interaction=nonstopmode -halt-on-error main_preview.tex >/dev/null; then
        echo "Ошибка сборки, подробности в main_preview.log"
        exit 1
    fi
done

rm -f main_preview.aux main_preview.log main_preview.out main_preview.toc build.log

mkdir -p submit
cp main_preview.pdf submit/kibizov_421.pdf
cp kibizov_421_annotation.txt submit/
rm -f submit/kibizov_421.zip
# zip есть не везде (например, в Git Bash), поэтому архив собирает Python.
# В Windows python3 может оказаться заглушкой Microsoft Store, поэтому
# берётся первый интерпретатор, который действительно запускается.
for PY in python3 python; do
    "$PY" -c "" 2>/dev/null && break
done
"$PY" - <<'EOF'
import glob, zipfile
files = ["kibizov_421.tex"] + sorted(glob.glob("kibizov_421_*.pdf") + glob.glob("kibizov_421_*.eps"))
with zipfile.ZipFile("submit/kibizov_421.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in files:
        z.write(f)
EOF

echo "Готово: main_preview.pdf и submit/"
