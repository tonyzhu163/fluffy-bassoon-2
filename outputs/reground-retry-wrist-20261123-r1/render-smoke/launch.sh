#!/usr/bin/env bash
# Run inside an allocated one-4090 Distributed Training container, or on the
# existing CPU development endpoint for the archive/preprocessor check only.
set -euo pipefail
[[ $# == 3 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[a-zA-Z0-9._-]+$ && $3 =~ ^(cuda|cuda-forkserver|archive)$ ]]
render_pin=$1
mode=$3
project=/mowenjie/fluffy-bassoon-2
wrist_root=$project/outputs/reground-retry-wrist-20261123-r1
checkout=$wrist_root/render-src-$render_pin
run_path=outputs/reground-retry-wrist-20261123-r1/render-smoke
[[ $(git -C "$checkout" rev-parse HEAD) == "$render_pin" ]]
git -C "$checkout" diff --quiet HEAD -- "$run_path"
script=$checkout/$run_path/smoke.py
render_script=$0
export RETRY_PIN=ac28987f221e336cef2ea64dc16dfffc9f0e74c9
export RENDER_PIN=$render_pin
here=$wrist_root/src-$RETRY_PIN
mkdir -p "$wrist_root/render-smoke/attempts"
attempt=$wrist_root/render-smoke/attempts/$2
mkdir "$attempt"
sha256sum "$render_script" "$script" > "$attempt/code.sha256"
source "$here/env.sh"
if [[ $mode != archive ]]; then
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER MUJOCO_EGL_DEVICE_ID
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
    if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
        [[ -n ${NVIDIA_VISIBLE_DEVICES:-} && $NVIDIA_VISIBLE_DEVICES != *,* ]]
        export CUDA_VISIBLE_DEVICES=0
    fi
    "$py" -Bu "$science/scripts/[train]gpu_probe.py" --baige-container --purpose training \
        --baige-exclusive-training --expected-count 1 --output "$attempt/gpu.json"
else
    export CUDA_VISIBLE_DEVICES=""
fi
# Positional arguments remain absolute across the transport wrapper.
seconds=1080
if [[ $mode == cuda-forkserver ]]; then
    seconds=480
fi
timeout --kill-after=15s "${seconds}s" bash "$control/$name/transport/with-cache.sh" \
    "$py" -Bu "$script" "$attempt" "$wrist_root/scored/attempts" "$mode" > "$attempt/launch.log" 2>&1
