#!/usr/bin/env bash
# In a Baige Distributed Training allocation with one RTX 4090.
# Usage: run.sh <pin> <attempt> [panel.py options]; default is the development starts.
set -euo pipefail
[[ $# -ge 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[A-Za-z0-9_-]+$ ]]
here=$(cd "$(dirname "$0")" && pwd -P)
harness="$here/../frozen-vla-proposal-shift-20261124-r1"
out="/mowenjie/fluffy-bassoon-2/outputs/$(basename "$here")/$2"
mkdir -p "$out"
# Per-replan render readbacks (geometry, pixel counts, validity) from the privileged twin.
export VIEW_TRACE="$out/views.jsonl"
pin=$1 attempt=$2
shift 2
[[ $# -gt 0 ]] || set -- --starts development
exec bash "$harness/run.sh" "$pin" "$attempt" --adapters view_kinova,view_panda_external,view_panda_both "$@"
