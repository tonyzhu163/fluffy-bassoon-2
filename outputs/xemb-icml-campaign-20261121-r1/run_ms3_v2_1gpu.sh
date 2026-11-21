#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
run="$root/outputs/$(basename "$here")"
py="$root/setups/maniskill3-xemb/.venv/bin/python"
probe_py="$root/setups/robocasa365-native-20261101-r1/env/bin/python"

pin=$1
attempt=$2
host=$(hostname)
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ "$host" == job-* ]] || { echo "Require Baige Distributed Training container" >&2; exit 2; }
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]

out="$run/$attempt"
mkdir -p "$out"
df -h "$out" > "$out/storage.txt"
printf '%s %s %s stage1-ms3-v2\n' "$pin" "$host" "$$" > "$out/launch.txt"
start=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT

export PYTHONUNBUFFERED=1
export MS_ASSET_DIR="$root/setups/maniskill3-xemb/assets"
libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02/prefix/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$code/src:$code/outputs/maniskill3-xemb-sysid-20261120-r1:$here"
export CUDA_VISIBLE_DEVICES=0

date +%s > "$out/gpu-start.txt"
PYTHONPATH="$code/src" "$probe_py" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose training --baige-container --baige-exclusive-training \
    --output "$out/gpu.json" > "$out/probe.log" 2>&1

"$py" -Bu "$here/ms3_xemb_v2.py" --out-dir "$out/ms3_v2" --steps 3000 > "$out/run.log" 2>&1
date +%s > "$out/gpu-end.txt"
