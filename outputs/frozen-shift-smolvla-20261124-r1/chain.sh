#!/usr/bin/env bash
# Scheduling only: wait for one running harness process, then run queued cells in order.
# Usage: bash chain.sh SCRATCH "WAIT_ARM WAIT_TAG WAIT_LAYOUTS" "TAG ARM OFFSET LAYOUTS..." ...
here=$(cd "$(dirname "$0")" && pwd)
scratch=$1; read -r arm tag layouts <<<"$2"; shift 2
while pgrep -f "[-]-arm $arm .*evidence/$tag --layout $layouts" >/dev/null; do sleep 3; done
for cell in "$@"; do
    read -r tag arm offset layouts <<<"$cell"
    HOOKS=adapters:LiberoPrompt NOISE_OFFSET=$offset bash "$here/run.sh" "$scratch" "$tag" "$arm" $layouts \
        > "$here/logs/$tag-$arm-${layouts// /_}.log" 2>&1
done
