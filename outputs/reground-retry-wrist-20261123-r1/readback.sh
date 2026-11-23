#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && $2 =~ ^(full|goals)$ ]]
export RETRY_PIN=$1
here=$(cd "$(dirname "$0")" && pwd -P)
store=/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1/scored
attempt="$store/readback-$2"
mkdir -p "$attempt"
source "$here/env.sh"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
exec "$py" -Bu "$here/driver.py" summarize "$store" "$2"
