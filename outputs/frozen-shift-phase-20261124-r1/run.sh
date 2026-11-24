#!/usr/bin/env bash
# Phase-prompt lane on the shared panel. In a Baige Distributed Training allocation with one RTX 4090.
# Usage: run.sh <pin> <attempt> [panel.py options; a later --starts or --adapters replaces the default]
set -euo pipefail
[[ $# -ge 2 ]]
here=$(cd "$(dirname "$0")" && pwd -P)
shared="$here/../frozen-vla-proposal-shift-20261124-r1"
code=$(git -C "$here" rev-parse --show-toplevel)
root=$(code=$code; source "$here/../frozen-vla-native-transfer-20261123-r1/env.sh"; echo "$root")
# Per-replan drawer-front pixels and open score from the public camera, next to the panel records.
export PHASE_TRACE_DIR="$root/outputs/$(basename "$shared")/$2/phase-trace"
exec "$shared/run.sh" "$1" "$2" --starts development \
    --adapters phase-reader,phase-instant,phase-clock,phase-open,phase-place "${@:3}"
