#!/usr/bin/env bash
# Parent invokes this only inside an exclusive Baige Distributed Training container.
set -euo pipefail
if [[ $# != 5 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^[a-zA-Z0-9._-]+$ ||
      ! $3 =~ ^[1-9][0-9]*$ || ! $4 =~ ^[1-9][0-9]*$ || ! $5 =~ ^(goals|full)$ ]]; then
    echo 'Usage: launch.sh FULL_SHA ATTEMPT_ID GPU_COUNT WALL_SECONDS goals|full' >&2
    exit 2
fi
export RETRY_PIN=$1
id=$2 n=$3 wall=$4 phase=$5
(( n * wall <= 7 * 3600 )) || { echo 'Invocation exceeds 7 GPU-hours' >&2; exit 2; }
here="/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1/src-$RETRY_PIN"
store=/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1/scored
mkdir -p "$store/attempts"
exec 9>>"$store/active.lock"
flock -n 9 || { echo 'Another retry launcher is active' >&2; exit 1; }
attempt="$store/attempts/$id"
mkdir "$attempt"
start=$(date +%s)
finish() {
    code=$?
    end=$(date +%s)
    echo "$code $start $end $n $((n*(end-start)))" > "$attempt/exit.txt"
}
trap finish EXIT
export here attempt n wall phase
sha256sum "$0" > "$attempt/launcher.sha256"
echo "$RETRY_PIN $HOSTNAME $phase $n $start $wall" > "$attempt/launch.txt"
payload() {
    set -euo pipefail
    source "$here/env.sh"
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER MUJOCO_EGL_DEVICE_ID
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
    if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
        IFS=, read -ra allocation <<< "${NVIDIA_VISIBLE_DEVICES:?missing scheduler UUIDs}"
        [[ ${#allocation[@]} == "$n" ]]
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 "$((n-1))")
        export CUDA_VISIBLE_DEVICES
    fi
    "$py" -Bu "$science/scripts/[train]gpu_probe.py" --baige-container --purpose training \
        --baige-exclusive-training --expected-count "$n" --output "$attempt/gpu.json"
    export py
    bash "$control/$name/transport/with-cache.sh" bash -c '
        set -eu
        pids=()
        for ((rank=0; rank<n; rank++)); do
            "$py" -Bu "$1" lane "$attempt" "$rank" "$phase" > "$attempt/rank$rank.log" 2>&1 &
            pids+=("$!")
        done
        status=0
        for pid in "${pids[@]}"; do
            if wait "$pid"; then
                continue
            else
                rc=$?
                if (( status == 0 )); then status=$rc; fi
            fi
        done
        exit "$status"
    ' retry-workers "$here/driver.py"
}
export -f payload
timeout --kill-after=10s "${wall}s" bash -c payload > "$attempt/launch.log" 2>&1
