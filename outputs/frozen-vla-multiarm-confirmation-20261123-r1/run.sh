#!/usr/bin/env bash
# Distributed Training entry point. One fresh directory per arm enforces the single allocation contract.
set -euo pipefail
started=$(date +%s)
[[ $# == 2 && $1 =~ ^[0-9a-f]{40}$ && ( $2 == IIWA || $2 == UR5e ) ]]
[[ $(uname -s) == Linux && $(hostname) == job-*-master-0 ]]
here=$(cd "$(dirname "$0")" && pwd -P)
pin=$1
arm=$2
[[ $(git -C "$here" rev-parse HEAD) == "$pin" ]]
[[ -z $(GIT_OPTIONAL_LOCKS=0 git -C "$here" status --porcelain --untracked-files=no) ]]
out=/mowenjie/fluffy-bassoon-2/outputs/$(basename "$here")/gpu01-$arm
mkdir "$out"
exec > >(tee "$out/console.log") 2>&1
trap 'rc=$?; end=$(date +%s); echo "$rc $started $end" > "$out/exit.txt"; echo "$((end-started))" > "$out/one_card_launcher_seconds.txt"' EXIT
echo "host=$(hostname) pin=$pin arm=$arm start_unix=$started" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
# Bound the entire runtime, including GPU checks, load/compile, evaluation and summary.
remaining=$((3540 - $(date +%s) + started))
(( remaining > 0 ))
timeout --kill-after=5s "${remaining}s" bash "$here/job.sh" "$arm" "$out"
