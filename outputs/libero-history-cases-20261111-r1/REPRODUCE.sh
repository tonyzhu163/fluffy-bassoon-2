#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Refuse to overwrite an existing case bank.
case_root=${1:?Usage: bash REPRODUCE.sh /path/to/new-output-directory}
mkdir "$case_root"
uv run --no-project --python 3.11 --with numpy==2.0.2 \
  exploration/libero-scenes/scenes.py --preset positions --n 256 --seed 110000 \
  --out "$case_root/scenes"
uv run --no-project --python 3.11 --with numpy==2.0.2 \
  exploration/libero-scenes/histories.py --scenes "$case_root/scenes" \
  --out "$case_root/cases" --train 1024 --validation 256 --test 256 --attempts 12 --seed 0
