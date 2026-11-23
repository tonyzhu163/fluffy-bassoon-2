#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && -d $1 ]]
attempt=$(cd "$1" && pwd -P)
here=$(cd "$(dirname "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES='' GIT_OPTIONAL_LOCKS=0
export SYMMETA_CODE_REV=$(git -C "$here" rev-parse HEAD)
source "$here/env.sh"
export JAX_PLATFORMS=cpu
libs=$project/setups/libero90-full-action-20261119-r1/desktop-libs02/prefix
export LD_LIBRARY_PATH="$libs/usr/lib/x86_64-linux-gnu:$libs/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
exec "$root/env/bin/python" -Bu "$here/fixture.py" "$attempt/test"
