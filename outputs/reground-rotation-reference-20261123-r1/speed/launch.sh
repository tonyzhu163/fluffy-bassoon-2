#!/usr/bin/env bash
# One exclusive RTX 4090 on Baige Distributed Training. Parent supplies the remaining reference cap.
set -euo pipefail
[[ $(hostname) == job-*-master-0 && -n ${KUBERNETES_SERVICE_HOST:-} && ${WORLD_SIZE:-1} == 1 ]] || {
    echo "Scored babbling requires one Baige Distributed Training master" >&2; exit 2; }
export GIT_OPTIONAL_LOCKS=0
here=$(cd "$(dirname "$0")" && pwd -P)
root=/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1

if [[ ${1:-} == _payload ]]; then
    shift
    attempt=$1
    # CUDA is visible to the allocation probe only. The native generator has cameras=False.
    source "$here/environment.sh"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    "$py" -Bu "$reference_tree/scripts/[train]gpu_probe.py" --baige-container --purpose training \
        --baige-exclusive-training --expected-count 1 --output "$attempt/gpu.json"
    export CUDA_VISIBLE_DEVICES=""
    unset MUJOCO_EGL_DEVICE_ID
    export NUMBA_CACHE_DIR="$attempt/numba-cache"
    "$py" -B "$here/check_runtime.py" > "$attempt/runtime.json"
    exec "$py" -Bu "$here/babble.py" run "$@"
fi

[[ $# == 4 ]] || { echo "usage: launch.sh TRANSPORT_SHA FRESH_ATTEMPT WORKERS WALL_SECONDS" >&2; exit 2; }
pin=$1 id=$2 workers=$3 wall=$4
[[ $pin =~ ^[a-f0-9]{40}$ && $id =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]
[[ $workers =~ ^(8|9|1[0-9]|2[0-4])$ && $wall =~ ^[1-9][0-9]{1,5}$ && $wall -gt 20 ]]
: "${REFERENCE_CAP_GPUH:?parent must specify the cumulative reference ceiling}"
export REFERENCE_STARTED=$(date +%s) REFERENCE_PHASE=babble REFERENCE_CODE_PIN=$pin
[[ $(git -C "$here" rev-parse HEAD) == "$pin" ]]
if git -C "$here" symbolic-ref -q HEAD; then exit 2; fi
git -C "$here" diff --quiet HEAD -- .
[[ -z $(git -C "$here" ls-files --others --exclude-standard -- '*.py' '*.sh') ]]
(cd "$root" && sha256sum --strict --check --quiet "$here/runtime.sha256")
source "$here/environment.sh"
[[ $(git -C "$reference_tree" rev-parse HEAD) == "$reference_pin" ]]
if git -C "$reference_tree" symbolic-ref -q HEAD; then exit 2; fi
git -C "$reference_tree" diff --quiet HEAD -- "$run" ':(literal)scripts/[train]gpu_probe.py' src/fluffy_bassoon/baige_gpu.py
attempt="$root/reference-native/attempts/$id"
mkdir "$attempt"
args=("$attempt" --workers "$workers" --wall "$wall" --cap "$REFERENCE_CAP_GPUH")
"$py" -B "$here/babble.py" reserve "${args[@]}"
finish() {
    local rc=$? end
    end=$(date +%s)
    printf '{"exit_code":%s,"end_utc":%s,"allocated_gpu_seconds":%s}\n' \
        "$rc" "$end" "$((end - REFERENCE_STARTED))" >"$attempt/exit.tmp"
    mv "$attempt/exit.tmp" "$attempt/exit.json"
}
trap finish EXIT
remaining=$((REFERENCE_STARTED + wall - 15 - $(date +%s)))
(( remaining > 0 ))
# Default GNU timeout owns the process group, including all disposable CPU children.
timeout --kill-after=10s "${remaining}s" bash "$here/launch.sh" _payload "${args[@]}" >"$attempt/launch.log" 2>&1
