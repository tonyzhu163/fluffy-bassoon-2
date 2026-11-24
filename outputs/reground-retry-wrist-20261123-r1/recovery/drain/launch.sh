#!/usr/bin/env bash
# Parent invokes this only inside an exclusive Baige Distributed Training container.
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^retry[0-9]+$ ]]; then
    echo 'Usage: launch.sh DISPATCH_FULL_SHA ATTEMPT_ID' >&2
    exit 2
fi
export WRIST_DISPATCH_PIN=$1
export RETRY_PIN=ac28987f221e336cef2ea64dc16dfffc9f0e74c9
export WRIST_RECOVERY_PIN=d6de8c725cfaf2dbb01a6a9c4895197441b8a6af
export GIT_OPTIONAL_LOCKS=0
id=$2 n=1 wall=5250 phase=full
run=/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1
export WRIST_DRAIN_DIR
WRIST_DRAIN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
[[ $(git -C "$WRIST_DRAIN_DIR" rev-parse HEAD) == "$WRIST_DISPATCH_PIN" ]]
[[ $(git -C "$WRIST_DRAIN_DIR" rev-parse HEAD^) == "$WRIST_RECOVERY_PIN" ]]
git -C "$WRIST_DRAIN_DIR" diff --quiet HEAD -- .
[[ -z $(git -C "$WRIST_DRAIN_DIR" ls-files --others --exclude-standard) ]]
legacy=$run/recovery-src-$WRIST_RECOVERY_PIN
[[ $(git -C "$legacy" rev-parse HEAD) == "$WRIST_RECOVERY_PIN" ]]
echo "c46a5996c99f28b93c965873ed1fd7e40f58e782a972f9afaea1127bfaeb4041  $legacy/recovery/launch.sh" | sha256sum --quiet -c -
here=$run/src-$RETRY_PIN
store=$run/scored
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
    # Baige binds the generated job ID to the master hostname used by native readback.
    local scheduler_host
    scheduler_host=$(hostname)
    [[ $scheduler_host == "$HOSTNAME" && $scheduler_host =~ ^(job-[a-z0-9]+)-master-0$ ]]
    export py WRIST_BAIGE_JOB_ID="${BASH_REMATCH[1]}"
    bash "$control/$name/transport/with-cache.sh" bash -c '
        set -eu
        pids=()
        for ((rank=0; rank<n; rank++)); do
            "$py" -Bu "$1" "$attempt" "$rank" > "$attempt/rank$rank.log" 2>&1 &
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
    ' retry-workers "$WRIST_DRAIN_DIR/dispatch.py"
}
export -f payload
timeout --kill-after=10s "${wall}s" bash -c payload > "$attempt/launch.log" 2>&1
