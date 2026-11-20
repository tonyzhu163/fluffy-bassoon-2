#!/usr/bin/env bash
set -euo pipefail

COMMIT="${1:?commit required}"
RUN_TAG="${2:-gpu01exp}"

BASE="/mowenjie/fluffy-bassoon-2"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$BASE/outputs/maniskill3-xemb-sysid-20261120-r1/$RUN_TAG"
PY="$BASE/setups/maniskill3-xemb/.venv/bin/python"
PROBE_PY="$BASE/setups/robocasa365-native-20261101-r1/env/bin/python"
SCRIPT="$REPO/outputs/maniskill3-xemb-sysid-20261120-r1/ms3_xemb_exp.py"

mkdir -p "$OUT"
ACTUAL_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
test "$ACTUAL_COMMIT" = "$COMMIT"

export PYTHONUNBUFFERED=1
export MS_ASSET_DIR="$BASE/setups/maniskill3-xemb/assets"
LIBS="$BASE/setups/libero90-full-action-20261119-r1/desktop-libs02/prefix/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$LIBS:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

START_TS="$(date +%s)"
echo "$COMMIT $(hostname) $$ ms3-xemb-exp" > "$OUT/launch.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT/gpu-start.txt"

PYTHONPATH="$REPO/src" "$PROBE_PY" -Bu "$REPO/scripts/[train]gpu_probe.py" \
  --expected-count 1 --purpose training --baige-container --baige-exclusive-training \
  --output "$OUT/gpu.json" > "$OUT/probe.log" 2>&1

"$PY" -Bu "$SCRIPT" --out-dir "$OUT" --steps 2500 > "$OUT/run.log" 2>&1
echo "0 $(date +%s) $START_TS" > "$OUT/exit.txt"
