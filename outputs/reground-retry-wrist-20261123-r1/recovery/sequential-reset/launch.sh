#!/usr/bin/env bash
# Parent-owned one-GPU Baige launch. No aggregate or outcomes in launcher output.
set -euo pipefail
[[ $# == 3 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^retry[0-9]+$ && $3 =~ ^[1-9][0-9]*$ ]]
export SEQ_PIN=$1 SEQ_PAYLOAD_SECONDS=$3
export RETRY_PIN=ac28987f221e336cef2ea64dc16dfffc9f0e74c9
project=/mowenjie/fluffy-bassoon-2
wrist_root=$project/outputs/reground-retry-wrist-20261123-r1
store=$wrist_root/scored
checkout=$wrist_root/sequential-src-$SEQ_PIN
seq_dir=$checkout/recovery/sequential-reset
[[ $(git -C "$checkout" rev-parse HEAD) == "$SEQ_PIN" ]]
git -C "$checkout" diff --quiet HEAD -- recovery/sequential-reset
[[ $(realpath "$0") == "$seq_dir/launch.sh" ]]
exec 9>>"$store/active.lock"
flock -n 9 || { echo 'ReGround scoring lock is held' >&2; exit 1; }
[[ ! -e $store/batches/batch054 ]]
attempt=$store/attempts/$2
mkdir "$attempt"
export SEQ_STARTED_EPOCH
SEQ_STARTED_EPOCH=$(date +%s)
finish() {
    rc=$?
    end=$(date +%s)
    echo "$rc $SEQ_STARTED_EPOCH $end 1 $((end-SEQ_STARTED_EPOCH))" > "$attempt/exit.txt"
}
trap finish EXIT
sha256sum "$seq_dir/launch.sh" > "$attempt/launcher.sha256"
sha256sum "$seq_dir/sequential.py" > "$attempt/implementation.sha256"
echo "$RETRY_PIN $HOSTNAME full 1 $SEQ_STARTED_EPOCH $SEQ_PAYLOAD_SECONDS" > "$attempt/launch.txt"
here=$wrist_root/src-$RETRY_PIN
source "$here/env.sh"
unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER MUJOCO_EGL_DEVICE_ID
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    [[ -n ${NVIDIA_VISIBLE_DEVICES:-} && $NVIDIA_VISIBLE_DEVICES != *,* ]]
    export CUDA_VISIBLE_DEVICES=0
fi
remaining=$((SEQ_STARTED_EPOCH+SEQ_PAYLOAD_SECONDS-$(date +%s)))
if (( remaining <= 0 )); then
    exit 3
fi
export py science control name
# Include GPU admission, receipt checks, model load, candidate resets and terminal scoring in one timeout.
set +e
timeout --kill-after=10s "${remaining}s" bash -c '
    set -euo pipefail
    "$py" -Bu "$science/scripts/[train]gpu_probe.py" --baige-container --purpose training \
        --baige-exclusive-training --expected-count 1 --output "$1/gpu.json"
    exec bash "$control/$name/transport/with-cache.sh" "$py" -Bu "$2/sequential.py" "$3" "$1"
' sequential "$attempt" "$seq_dir" "$store" > "$attempt/rank0.log" 2>&1
rc=$?
set -e
if [[ $rc == 124 && ! -e $attempt/sequential-first-wrist-control.json ]]; then
    printf '%s\n' '{"state":"budget_exhausted_before_wrist_control","reason":"hard launcher wall cap"}' > "$attempt/sequential-status.json"
    rc=3
fi
exit "$rc"
