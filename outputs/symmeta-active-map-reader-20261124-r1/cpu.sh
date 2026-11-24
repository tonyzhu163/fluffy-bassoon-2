#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
pin=${1:?pin}
attempt=${2:?new directory}
[[ $attempt =~ ^cpu[0-9]+$ && $(git -C "$here" rev-parse HEAD) == "$pin" ]]
root=/mowenjie/fluffy-bassoon-2/outputs
[[ $(cut -d' ' -f1 "$root/symmeta-query-direction-probe-20261124-r1/cpu03/exit.txt") == 0 ]]
out="$root/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
df -h "$out" > "$out/storage.txt"
df -i "$out" >> "$out/storage.txt"
date -Is > "$out/launch.txt"
echo "$pin $(hostname)" >> "$out/launch.txt"
trap 'rc=$?; echo "$rc $(date -Is)" > "$out/exit.txt"' EXIT
cores=(38 39 35 36)
arms=(Panda Sawyer UR5e IIWA)
for cpu in "${cores[@]}"; do taskset -c "$cpu" true; done
awk '/^cpu(35|36|38|39) /' /proc/stat > "$out/idle.txt"
sleep 2
awk '/^cpu(35|36|38|39) /' /proc/stat >> "$out/idle.txt"
awk 'NR<=4 {i[$1]=$5; t[$1]=$2+$3+$4+$5+$6+$7+$8+$9; next} {n=$2+$3+$4+$5+$6+$7+$8+$9-t[$1]; if(n<=0 || ($5-i[$1])/n<0.90) exit 1}' "$out/idle.txt"
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" MPLCONFIGDIR="$out/cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
collect() (
    cpu=$1
    arm=$2
    export REACH_CPUS="$cpu" REACH_CACHE="$out/cache/$cpu"
    TIMEFORMAT='wall=%R user=%U sys=%S'
    { time timeout 2300 bash "$here/../frozen-vla-reachability-20261123-r1/cpu.sh" "$here/fixture.py" "$out" "$cpu" "$arm" > "$out/$arm.log" 2>&1; } 2>"$out/time-$arm.txt"
)
pids=()
for i in "${!cores[@]}"; do collect "${cores[$i]}" "${arms[$i]}" & pids+=($!); done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
exit "$rc"
