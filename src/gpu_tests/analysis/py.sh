#!/bin/bash
# Один раз: виртуальное окружение .venv с зависимостями обработки (requirements.txt).

set -euo pipefail

cd "$(dirname "$0")"

python3 -m pip install --user virtualenv || python3 -m pip install --user --break-system-packages virtualenv
python3 -m virtualenv .venv
.venv/bin/python3 -m pip install -r requirements.txt
