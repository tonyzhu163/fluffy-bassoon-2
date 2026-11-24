#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
pin=$1
attempt=$2
[[ $attempt =~ ^cpu[0-9]+$ && $(git -C "$here" rev-parse HEAD) == "$pin" ]]
base=/mowenjie/fluffy-bassoon-2/outputs
out=$base/$(basename "$here")/$attempt
[[ $(cut -d' ' -f1 "$base/symmeta-query-identifiability-20261124-r1/cpu02/exit.txt") == 0 ]]
mkdir -p "$(dirname "$out")"
mkdir "$out"
date -Is > "$out/launch.txt"
printf '%s %s\n' "$pin" "$(hostname)" >> "$out/launch.txt"
trap 'rc=$?; echo "$rc $(date -Is)" > "$out/exit.txt"' EXIT
df -h "$out" > "$out/storage.txt"
df -i "$out" >> "$out/storage.txt"
for core in 38 39; do taskset -c "$core" true; done
awk '/^cpu(38|39) /' /proc/stat >> "$out/core-idle.txt"
sleep 2
awk '/^cpu(38|39) /' /proc/stat >> "$out/core-idle.txt"
awk 'NR<=2 {idle[$1]=$5; total[$1]=$2+$3+$4+$5+$6+$7+$8+$9; next} {n=$2+$3+$4+$5+$6+$7+$8+$9-total[$1]; if (n<=0 || ($5-idle[$1])/n<0.90) exit 1}' "$out/core-idle.txt"
ps -eLo pid,psr,stat,comm | awk '($2==38 || $2==39) && $3 ~ /^R/' >> "$out/core-idle.txt"
export TMPDIR="$out/tmp" MPLCONFIGDIR="$out/cache/matplotlib" XDG_CACHE_HOME="$out/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
runtime="$here/../frozen-vla-reachability-20261123-r1/cpu.sh"
collect() (
    ulimit -t "${PROBE_CPU_LIMIT:-540}"
    export REACH_CPUS="$1" REACH_CACHE="$out/cache/$1"
    /usr/bin/time -f 'wall=%e user=%U sys=%S' -o "$out/time-$1.txt" bash "$runtime" "$here/collect.py" "$out" "$1" "$2" > "$out/$1.log" 2>&1
)
collect 38 UR5e &
p38=$!
collect 39 IIWA &
p39=$!
rc=0
wait "$p38" || rc=$?
wait "$p39" || rc=$?
((rc == 0))
ulimit -t 45
export REACH_CPUS=38 REACH_CACHE="$out/cache/38"
/usr/bin/time -f 'wall=%e user=%U sys=%S' -o "$out/time-score.txt" bash "$runtime" "$here/score.py" "$out" > "$out/score.log" 2>&1
