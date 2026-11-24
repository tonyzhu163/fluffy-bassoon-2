#!/usr/bin/env bash
# In a Baige Distributed Training allocation with one RTX 4090. Usage: run.sh <pin> <attempt>
# Three panel runs through the shared harness launcher, each with its own pin and GPU check:
#   headroom on the six failures (8 streams), on the five known successes (2 streams), then the critic panel.
set -euo pipefail
[[ $# -eq 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[A-Za-z0-9_-]+$ ]]
here=$(cd "$(dirname "$0")" && pwd -P)
panel="$here/../frozen-vla-proposal-shift-20261124-r1/run.sh"
[[ -s "$here/critic.json" ]]
bash "$panel" "$1" "$2-sampling-failures" --starts 14,15,17,36,37,38 \
    --adapters noise-0,noise-1,noise-2,noise-3,noise-4,noise-5,noise-6,noise-7
bash "$panel" "$1" "$2-sampling-successes" --starts 16,31,34,35,39 --adapters noise-0,noise-1
bash "$panel" "$1" "$2-sampling-critic" --starts development --adapters critic-k4,random-k4,single-k1
source "$here/../frozen-vla-native-transfer-20261123-r1/env.sh"
base="$root/outputs/frozen-vla-proposal-shift-20261124-r1"
"$python" -B "$here/headroom.py" "$base/$2-sampling-failures" "$base/$2-sampling-successes" \
    > "$base/$2-sampling-headroom.json"
