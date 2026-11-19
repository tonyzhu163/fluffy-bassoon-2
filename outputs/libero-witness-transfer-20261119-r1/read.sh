#!/usr/bin/env bash
set -euo pipefail
pin=$1
arm=$2
attempt=$3
root=/mowenjie/fluffy-bassoon-2
run=libero-witness-transfer-20261119-r1
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
[[ $(cut -d' ' -f1 "$root/outputs/$run/cpu01/exit.txt") == 0 ]]
[[ "$arm" == robot || "$arm" == vlm || "$arm" == rgb || "$arm" == motion ]]
out="$root/outputs/$run/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
printf '%s %s %s\n' "$pin" "$(hostname)" "$$" > "$out/launch.txt"
timeout --kill-after=15s 30m "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/outputs/$run/transfer.py" --arm "$arm" --output "$out/$arm.json" > "$out/run.log" 2>&1
