#!/usr/bin/env bash
# Run from the local clone; Baige receives the script over stdin and writes no files.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ssh -o BatchMode=yes baige-dev '
    set -eu
    root=/mowenjie/fluffy-bassoon-2
    legacy=/mowenjie/fluffy-bassoon
    bddl="$root/sources/libero-pro2-e5500c69/assets/bddl/libero-pro2.0-1680-v1/libero_goal/00_base/BASE__push_the_plate_to_the_front_of_the_stove.bddl"
    export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    timeout 30m taskset -c 29 "$legacy/envs/openpi-215abfb/bin/python" -B -         --base "$root/outputs/frozen-vla-plate-transfer-20261124-r1"         --checkpoint "$legacy/inputs/pi05-libero-openpi-20261027-r1"         --openpi "$legacy/third_party/openpi-215abfb"         --bddl "$bddl"
' < "$here/audit.py"
