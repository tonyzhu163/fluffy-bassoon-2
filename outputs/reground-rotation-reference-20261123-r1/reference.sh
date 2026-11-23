#!/usr/bin/env bash
# Baige Distributed Training, one exclusive node. See REFERENCE.md for deployment.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
root=${REGROUND_RUNTIME_ROOT:-/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1}
export REGROUND_RUNTIME_ROOT=$root
store="$root/reference-native"
py="$root/env/bin/python"
pinup=16a98486f560676b9036d1a6d067b9dd51a5917a

payload() {
    local attempt=$1 gpus=$2 phase=$3
    shift 3
    source "$root/runtime/activate.sh"
    export PYTHONPATH="$PYTHONPATH:$code/src"
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONDONTWRITEBYTECODE=1
    export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export NCCL_P2P_DISABLE=1 REGROUND_NEXEC=10
    unset MUJOCO_EGL_DEVICE_ID LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER
    local libs=/mowenjie/fluffy-bassoon-2/setups/libero90-full-action-20261119-r1/desktop-libs02
    if [[ -f "$libs/nvidia-vendor.json" ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
        export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    fi
    # Only the scheduler container defines the CUDA device namespace. If Baige leaves
    # CUDA_VISIBLE_DEVICES unset, number all its assigned devices in their native order.
    if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 "$((gpus - 1))")
        export CUDA_VISIBLE_DEVICES
    fi
    "$py" -Bu "$code/scripts/[train]gpu_probe.py" --baige-container --purpose training \
        --baige-exclusive-training --expected-count "$gpus" --output "$attempt/gpu.json"
    local stages=("$phase")
    if [[ $phase == all ]]; then stages=(babble collect fit base reground); fi
    for stage in "${stages[@]}"; do
        "$py" -Bu "$here/reference.py" dispatch --phase "$stage" --gpus "$gpus" \
            --attempt "$attempt" --seeds "$@"
    done
    if [[ $phase == all && $# == 5 ]]; then
        "$py" -Bu "$here/reference.py" summarize
    fi
}

if [[ ${1:-} == _payload ]]; then
    shift
    payload "$@"
    exit
fi
[[ $# -ge 5 ]] || { echo "usage: reference.sh CODE_SHA FRESH_ATTEMPT GPUS WALL_SECONDS PHASE [SEEDS...]" >&2; exit 2; }
pin=$1 id=$2 gpus=$3 wall=$4 phase=$5
shift 5
if (( $# == 0 )); then set -- 0 1 2 3 4; fi
[[ $pin =~ ^[a-f0-9]{40}$ && $id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ && $gpus =~ ^(2|4|5)$ ]]
[[ $wall =~ ^[1-9][0-9]{1,5}$ && $wall -gt 20 && $phase =~ ^(all|babble|collect|fit|base|reground)$ ]]
[[ $(hostname) == job-*-master-0 && -n ${KUBERNETES_SERVICE_HOST:-} && ${WORLD_SIZE:-1} == 1 ]] || {
    echo "Expected one Baige Distributed Training master container with all requested GPUs" >&2; exit 2; }
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && $(git -C "$root/upstream" rev-parse HEAD) == "$pinup" ]]
git -C "$code" diff --quiet HEAD -- "${here#"$code"/}" 'scripts/[train]gpu_probe.py' src/fluffy_bassoon/baige_gpu.py
[[ -x $py && -r $root/runtime/activate.sh && -s $root/libero-config/config.yaml ]]
export REFERENCE_STARTED=$(date +%s) REFERENCE_PHASE=$phase REFERENCE_CODE_PIN=$pin
attempt="$store/attempts/$id"
mkdir -p "$store/attempts"
mkdir "$attempt"
"$py" -B "$here/reference.py" reserve --attempt "$attempt" --gpus "$gpus" --wall "$wall" \
    --cap "${REFERENCE_CAP_GPUH:-50}" --seeds "$@"
finish() {
    local rc=$? end diff
    end=$(date +%s)
    diff=$(( (end - REFERENCE_STARTED) * gpus ))
    printf '{"exit_code":%s,"end_utc":%s,"allocated_gpu_seconds":%s}\n' "$rc" "$end" "$diff" > "$attempt/exit.tmp"
    mv "$attempt/exit.tmp" "$attempt/exit.json"
}
trap finish EXIT
remaining=$((REFERENCE_STARTED + wall - 15 - $(date +%s)))
(( remaining > 0 ))
timeout --kill-after=10s "${remaining}s" bash "$here/reference.sh" _payload "$attempt" "$gpus" "$phase" "$@" \
    > "$attempt/launch.log" 2>&1
