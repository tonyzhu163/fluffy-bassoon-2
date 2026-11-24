#!/usr/bin/env bash
# Parent-owned launch inside the original one-4090 Baige Distributed Training allocation.
set -euo pipefail
if [[ $# != 5 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^retry[0-9]+$ ||
      ! $4 =~ ^[1-9][0-9]*$ || ! $5 =~ ^[1-9][0-9]*$ ]]; then
    echo 'Usage: launch.sh OPERATIONAL_SHA ATTEMPT LOCKED_SELECTION_JSON ALLOCATION_SECONDS PAYLOAD_SECONDS' >&2
    exit 2
fi
pin=$1 id=$2 selection=$3 allocation=$4 payload=$5
export RETRY_PIN=ac28987f221e336cef2ea64dc16dfffc9f0e74c9
export GIT_OPTIONAL_LOCKS=0
run=/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1
variant=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
[[ $(git -C "$variant" rev-parse HEAD) == "$pin" ]]
git -C "$variant" diff --quiet HEAD -- .
[[ -z $(git -C "$variant" ls-files --others --exclude-standard -- .) ]]
here=$run/src-$RETRY_PIN
store=$run/scored
exec 9>>"$store/active.lock"
flock -n 9 || { echo 'Another retry launcher is active' >&2; exit 1; }
attempt=$store/attempts/$id
reservation_rc=0
python3 -IB "$variant/admission.py" "$store" "$run/recovery/paired-reset/admission" \
    "$selection" "$id" "$allocation" "$payload" "$pin" || reservation_rc=$?
if (( reservation_rc == 10 )); then
    echo 'All explicitly assigned batches are already sealed'
    exit 0
fi
(( reservation_rc == 0 )) || exit "$reservation_rc"
start=$(date +%s)
finish() {
    local code=$? end
    end=$(date +%s)
    echo "$code $start $end 1 $((end-start))" > "$attempt/exit.txt"
}
trap finish EXIT
sha256sum "$variant/launch.sh" "$variant/admission.py" "$variant/dispatch.py" "$variant/pairing.py" \
    > "$attempt/paired-reset-code.sha256"
echo "$RETRY_PIN $HOSTNAME full 1 $start $payload" > "$attempt/launch.txt"
export here attempt variant
run_payload() {
    set -euo pipefail
    source "$here/env.sh"
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER MUJOCO_EGL_DEVICE_ID
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
    if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
        [[ -n ${NVIDIA_VISIBLE_DEVICES:-} && $NVIDIA_VISIBLE_DEVICES != *,* ]]
        export CUDA_VISIBLE_DEVICES=0
    fi
    "$py" -Bu "$science/scripts/[train]gpu_probe.py" --baige-container --purpose training \
        --baige-exclusive-training --expected-count 1 --output "$attempt/gpu.json"
    exec bash "$control/$name/transport/with-cache.sh" \
        "$py" -Bu "$variant/dispatch.py" "$attempt" > "$attempt/rank0.log" 2>&1
}
export -f run_payload
timeout --kill-after=10s "${payload}s" bash -c run_payload > "$attempt/launch.log" 2>&1
