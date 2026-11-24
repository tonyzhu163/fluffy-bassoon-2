#!/usr/bin/env bash
# Baige Distributed Training job command. One arm, one fresh attempt, one GPU-hour.
set -euo pipefail
started=$(date +%s)
source "$(dirname "$0")/runtime.sh"
[[ $# == 5 && $1 == --fresh && $2 =~ ^[0-9a-f]{40}$ && ( $3 == IIWA || $3 == Kinova3 ) ]]
[[ $4 =~ ^gpu[A-Za-z0-9_-]+$ && $5 =~ ^cpu[A-Za-z0-9_-]+$ && $(hostname) == job-*-master-0 ]]
pin=$2
arm=$3
out="$shared/$4-$arm"
qual="$shared/$5"
[[ -f "$qual/$arm/qualified.json" && $(cat "$qual/exit.txt") == 0 && $(cat "$qual/pin.txt") == "$pin" ]]
mkdir "$out"
trap 'rc=$?; end=$(date +%s); echo "$rc $started $end" > "$out/exit.txt"' EXIT
printf '%s\n' "pin=$pin host=$(hostname) arm=$arm qualifier=$qual start=$started" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
remaining=$((3540 - $(date +%s) + started))
(( remaining > 0 ))
timeout --kill-after=5s "${remaining}s" bash "$here/job.sh" "$pin" "$arm" "$out" "$qual" > "$out/run.log" 2>&1
