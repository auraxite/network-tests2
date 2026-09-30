#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

python3 -m pip install --user virtualenv || python3 -m pip install --user --break-system-packages virtualenv

python3 -m virtualenv .venv
set +u
. .venv/bin/activate
set -u
python3 -m pip install -r requirements.txt
