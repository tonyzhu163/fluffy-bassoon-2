#!/usr/bin/env bash
# Prepare a registered, doubled extension under an immutable previous physical enrollment.
set -euo pipefail
[[ $# == 5 ]]
export EVAL_PIN=$1
source "$(dirname "$0")/environment.sh"
[[ $2 =~ ^prepare[0-9]+$ && $3 =~ ^prepare[0-9]+$ && $4 =~ ^[a-f0-9]{64}$ && $5 =~ ^[0-9]+$ ]]
mkdir "$shared/$2"
out=$shared/$2
exec > >(tee "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
manifest="$shared/$3/panel/manifest.json"
[[ $(sha256sum "$manifest" | cut -d' ' -f1) == "$4" ]]
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
taskset -c 40,41 "$sim/env/bin/python" -Bu "$here/plan.py" "$out/panel" --prefix "$manifest" --count "$5"
