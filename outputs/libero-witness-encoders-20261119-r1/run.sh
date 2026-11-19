#!/usr/bin/env bash
set -euo pipefail
pin=$1
arm=$2
attempt=$3
root=/mowenjie/fluffy-bassoon-2
run=libero-witness-encoders-20261119-r1
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
[[ "$arm" == robot || "$arm" == vlm || "$arm" == rgb ]]
py="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
out="$root/outputs/$run/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
printf '%s %s %s\n' "$pin" "$(hostname)" "$$" > "$out/launch.txt"
arg=()
if [[ $arm == vlm ]]; then
    arg=(--pt-weights "$root/inputs/libero-witness-paligemma-pt-20261119-r1/vision.safetensors")
fi
timeout --kill-after=15s 30m "$py" -Bu "$code/outputs/$run/probe.py" --arm "$arm" "${arg[@]}" --output "$out/$arm.json" > "$out/run.log" 2>&1
