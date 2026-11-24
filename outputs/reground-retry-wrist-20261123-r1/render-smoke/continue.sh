#!/usr/bin/env bash
# Parent-owned Distributed Training only. A completed exact smoke is mandatory.
set -euo pipefail
[[ $# == 5 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[0-9a-f]{40}$ && $3 =~ ^[a-zA-Z0-9._-]+$ &&
    $4 =~ ^[a-zA-Z0-9._-]+$ && $5 =~ ^[1-9][0-9]*$ ]]
export CONTINUATION_PIN=$1
smoke_pin=$2
smoke_id=$3
resume_id=$4
wall=$5
wrist_root=/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1
checkout=$wrist_root/render-src-$CONTINUATION_PIN
run_path=outputs/reground-retry-wrist-20261123-r1/render-smoke
[[ $(git -C "$checkout" rev-parse HEAD) == "$CONTINUATION_PIN" ]]
git -C "$checkout" diff --quiet HEAD -- "$run_path"
smoke=$wrist_root/render-smoke/attempts/$smoke_id
[[ -s $smoke/paired-report.json ]]
export RETRY_PIN=ac28987f221e336cef2ea64dc16dfffc9f0e74c9
here=$wrist_root/src-$RETRY_PIN
store=$wrist_root/scored
exec 9>>"$store/active.lock"
flock -n 9 || { echo 'Another retry launcher is active' >&2; exit 1; }
attempt=$store/attempts/$resume_id
mkdir "$attempt"
start=$(date +%s)
finish() {
    code=$?
    end=$(date +%s)
    echo "$code $start $end 1 $((end-start))" > "$attempt/exit.txt"
}
trap finish EXIT
sha256sum "$0" "$checkout/$run_path/continue_pair.py" "$checkout/$run_path/smoke.py" \
    "$smoke/paired-report.json" > "$attempt/bridge.sha256"
source "$here/env.sh"
unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER MUJOCO_EGL_DEVICE_ID
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    [[ -n ${NVIDIA_VISIBLE_DEVICES:-} && $NVIDIA_VISIBLE_DEVICES != *,* ]]
    export CUDA_VISIBLE_DEVICES=0
fi
"$py" -Bu "$science/scripts/[train]gpu_probe.py" --baige-container --purpose training \
    --baige-exclusive-training --expected-count 1 --output "$attempt/gpu.json"
timeout --kill-after=10s "${wall}s" bash "$control/$name/transport/with-cache.sh" \
    "$py" -Bu "$checkout/$run_path/continue_pair.py" "$attempt" "$smoke" "$smoke_pin" \
    > "$attempt/rank0.log" 2>&1
