#!/usr/bin/env bash
# Run on Baige development CPU as: taskset -c 44 bash native_readback.sh
set -euo pipefail
pin=d70e206a00c6c5d59f190ec4a9b0bc6df2ec82c4
study=frozen-transfer-progress-20261124-r1
checkout=/mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-${pin:0:12}
d="$checkout/outputs/$study"
source "$d/runtime.sh"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
[[ $(awk '/Cpus_allowed_list/{print $2}' /proc/self/status) == 44 ]]
attempt=2
for kernel in Haswell SkylakeX Cooperlake Sandybridge; do
    out=$(printf '%s/cpu-pair%02d' "$shared" "$attempt")
    mkdir "$out"
    runtime
    export OPENBLAS_CORETYPE=$kernel
    {
        date -u -Is
        hostname
        lscpu | sed -n '/^Model name:/p'
        taskset -pc $$
        sha256sum "$d/readback.py" "$d/contract.py"
        echo "pin=$pin OPENBLAS_CORETYPE=$kernel OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS JAX_PLATFORMS=$JAX_PLATFORMS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        "$python" -B -c 'import sys, numpy as np; print(sys.executable, sys.version, np.__version__); np.show_runtime()'
    } > "$out/runtime.log" 2>&1
    start=$(date +%s)
    rc=0
    "$python" -Bu "$d/readback.py" --iiwa "$shared/gpu01-IIWA" --kinova3 "$shared/gpu01-Kinova3" \
        --out "$out/paired-layout31.json" > "$out/reader.log" 2>&1 || rc=$?
    echo "$rc $start $(date +%s)" > "$out/exit.txt"
    echo "$out kernel=$kernel exit=$rc"
    [[ $rc != 0 ]] || exit 0
    tail -3 "$out/reader.log"
    attempt=$((attempt + 1))
done
exit 1
