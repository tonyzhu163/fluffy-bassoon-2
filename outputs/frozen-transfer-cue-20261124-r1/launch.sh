#!/usr/bin/env bash
# One parent-dispatched Baige Distributed Training allocation, total runtime under one hour.
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
[[ $# == 4 && $1 == --fresh && $2 =~ ^[0-9a-f]{40}$ && $3 =~ ^gpu[A-Za-z0-9_-]+$ && $4 =~ ^cpu[A-Za-z0-9_-]+$ ]]
[[ $(hostname) == job-*-master-0 ]]
pin=$2
out="$shared/$3"
qual="$shared/$4"
pinned_tree "$repo" "$pin"
[[ $(cat "$qual/exit.txt") == 0 && -s "$qual/qualified.json" ]]
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
printf 'pin=%s host=%s cpu=%s start=%s\n' "$pin" "$(hostname)" "$qual" "$start" > "$out/launch.txt"
timeout --kill-after=5s 3540s bash "$here/job.sh" "$pin" "$out" "$qual" > "$out/run.log" 2>&1
