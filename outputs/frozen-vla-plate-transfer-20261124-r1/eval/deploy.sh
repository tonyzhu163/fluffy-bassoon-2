#!/usr/bin/env bash
# Bind the exact successful trainer's own seal receipt, including after a named training recovery.
set -euo pipefail
[[ $# == 7 || $# == 9 ]]
export EVAL_PIN=$1
source "$(dirname "$0")/environment.sh"
fit_attempt=$3
[[ $fit_attempt =~ ^gpu[0-9]+$ ]]
fit="$root/outputs/frozen-vla-plate-transfer-20261124-r1/fit/$fit_attempt"
read -r rc _ < "$fit/exit.txt"; [[ $rc == 0 ]]
mapfile -t sealed < <(awk -v file="$fit/fit.json" '$1=="Sealed" && $2==file && NF==3 {print $3}' "$fit/console.log")
[[ ${#sealed[@]} == 1 && ${sealed[0]} =~ ^[0-9a-f]{64}$ ]]
fit_pin=$(sed -n '2p' "$fit/launch.txt")
[[ $fit_pin =~ ^[0-9a-f]{40}$ ]]
prepared=${8:-prepare01}
manifest=${9:-b7996a9e110c9ceed9babb081a18476c1dc0b9a8d7e623703f5912031e44bc3f}
exec bash "$here/launch.sh" gpu "$EVAL_PIN" "$2" "$fit_pin" "$fit_attempt" "${sealed[0]}" "$prepared" "$manifest" "$4" "$5" "$6" "$7"
