#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
pin=$1
[[ $(git -C "$here" rev-parse HEAD) == "$pin" ]]
root=/mowenjie/fluffy-bassoon-2/outputs/$(basename "$here")
out=$root/cpu03
mkdir "$out"
echo "$pin $(hostname) $(date -Is)" > "$out/launch.txt"
trap 'rc=$?; echo "$rc $(date -Is)" > "$out/exit.txt"' EXIT
df -h "$out" > "$out/storage.txt"
awk '/^cpu39 /' /proc/stat > "$out/core-idle.txt"
sleep 2
awk '/^cpu39 /' /proc/stat >> "$out/core-idle.txt"
awk 'NR==1 {idle=$5; total=$2+$3+$4+$5+$6+$7+$8+$9; next} {n=$2+$3+$4+$5+$6+$7+$8+$9-total; if (n<=0 || ($5-idle)/n<0.90) exit 1}' "$out/core-idle.txt"
export TMPDIR="$out/tmp" MPLCONFIGDIR="$out/cache/matplotlib" XDG_CACHE_HOME="$out/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
export REACH_CPUS=39 REACH_CACHE="$root/cpu02/cache/39"
# cpu02 consumed 1025.96 CPU seconds. Leave room under the original 1200 s ceiling.
ulimit -t 142
TIMEFORMAT='wall=%R user=%U sys=%S'
{ time bash "$here/../frozen-vla-reachability-20261123-r1/cpu.sh" "$here/finish.py" "$root/cpu02" "$out" > "$out/run.log" 2>&1; } 2>"$out/time.txt"
