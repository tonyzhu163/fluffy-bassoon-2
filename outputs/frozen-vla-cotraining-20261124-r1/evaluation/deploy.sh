#!/usr/bin/env bash
# Invoked by the root operator in an allocated Baige Distributed Training job.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
run=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cotraining-20261124-r1
[[ $# == 3 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^gpu[A-Za-z0-9_-]+$ && $3 =~ ^gpu[A-Za-z0-9_-]+$ ]]
pin=$1
fit="$run/model/$2"
manifest="$run/data/dataset-v1/manifest.json"
read -r rc _ < "$fit/exit.txt"
[[ $rc == 0 ]]
# Bind to the digest emitted by the completed training run, independent of the current file bytes.
mapfile -t locked < <(awk -v filename="$fit/fit.json" '$1 == "Ready" && $2 == filename && NF == 3 {print $3}' "$fit/console.log")
[[ ${#locked[@]} == 1 && ${locked[0]} =~ ^[0-9a-f]{64}$ ]]
[[ $(sha256sum "$manifest" | cut -d' ' -f1) == 05209732b5e3f04308c8acbef2f6c8c8a2f173673840fdb055e1b094ff7d6f40 ]]
exec bash "$here/launch.sh" gpu "$pin" "$3" "$fit" "${locked[0]}" "$manifest"
