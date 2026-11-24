#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
pin=$1
[[ $pin =~ ^[0-9a-f]{40}$ && $(git -C "$here" rev-parse HEAD) == "$pin" ]]
[[ -z $(git --no-optional-locks -C "$here" status --porcelain --untracked-files=no) ]]
root=/mowenjie/fluffy-bassoon-2
study="$root/outputs/$(basename "$here")"
out="$study/cpu01"
mkdir -p "$study"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
df -h "$out" > "$out/storage.txt"
test -w "$out"
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" MPLCONFIGDIR="$out/cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$out/cases"
runtime="$here/../frozen-vla-reachability-20261123-r1/cpu.sh"
for cpu in 30 31; do
    taskset -c "$cpu" true
    mkdir -p "$out/cache/$cpu"
done
run_cpu() {
    local cpu=$1
    shift
    REACH_CPUS="$cpu" REACH_CACHE="$out/cache/$cpu" bash "$runtime" "$here/replay.py" "$out/cases" "$pin" "$cpu" "$@" > "$out/cpu-$cpu.log" 2>&1
}
run_cpu 30 34 36 &
p30=$!
run_cpu 31 35 37 &
p31=$!
rc=0
wait "$p30" || rc=$?
wait "$p31" || rc=$?
(( rc == 0 ))
[[ $(find "$out/cases" -name '*-from-*.json' | wc -l) == 32 ]]
