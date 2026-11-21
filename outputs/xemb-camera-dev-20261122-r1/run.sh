#!/usr/bin/env bash
set -euo pipefail
start=$(date +%s)
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
campaign=xemb-icml-campaign-20261121-r1
run=xemb-camera-dev-20261122-r1
pin=${1:?pass the full parent-validated deployment commit}
attempt=${2:?pass a fresh attempt name}
[[ $# == 2 && $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[a-zA-Z0-9_-]+$ ]]
[[ $(hostname) == job-* && "$here" == "$code/outputs/$run" ]]
out="$root/outputs/$run/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT
export CAMERA_PIN="$pin"
payload() {
    local code=$1 out=$2 root=$3 legacy=$4 campaign=$5 here=$6
    local release="$root/sources/libero-pro2-e5500c69"
    local setup="$root/setups/libero-meta-learning-20261117-r1"
    local release_pin=e5500c690c1498eca1b7050350654964190cdf70
    local overlay="$root/setups/$campaign/cv2-headless-overlay"
    local libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
    local probe_python="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
    local python="$legacy/envs/openpi-215abfb/bin/python"
    local archive="$root/outputs/libero-xemb-ur5e-scale-20261120-r1/gpu01/archive/archive.json"
    local checkpoint="$root/outputs/$campaign/gpu04homog/stage1/checkpoints/equiv_flow_homog_nodz-active_probe-seed41.msgpack"
    local rel path driver_paths="" directory
    local -a paths
    [[ $(git -C "$code" rev-parse HEAD) == "$CAMERA_PIN" ]]
    [[ -z $(git --no-optional-locks -C "$code" status --porcelain) ]]
    for path in "$here"/{PLAN.md,evaluate.py,run.sh} "$code/outputs/libero-xemb-ur5e-sysid-20261120-r1/ur5e_sim_worker.py"; do
        rel=${path#"$code/"}
        [[ $(git -C "$code" rev-parse "$CAMERA_PIN:$rel") == "$(git hash-object "$path")" ]]
    done
    [[ $(git -C "$release" rev-parse HEAD) == "$release_pin" ]]
    export CROSSARM_RELEASE="$release" LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
    export OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export PYTHONPATH="$code/src"
    for directory in libero-crossarm-20261120-r1 libero-xemb-ur5e-sysid-20261120-r1 libero-xemb-ur5e-scale-20261120-r1 "$campaign"; do
        PYTHONPATH+=":$code/outputs/$directory"
    done
    PYTHONPATH+=":$here:$overlay:$release:$LIBERO_PRO_ROOT"
    export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige XLA_PYTHON_CLIENT_PREALLOCATE=false NCCL_P2P_DISABLE=1
    IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
    for path in "${paths[@]}"; do
        if [[ -n $path && ! -f $path/libtorch_python.so ]]; then driver_paths+=":$path"; fi
    done
    export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$driver_paths"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
    printf '%s\n' "commit=$CAMERA_PIN" "host=$(hostname)" "checkout=$code" "PYTHONPATH=$PYTHONPATH" \
        "release=$release_pin" "policy=$python" "archive=$archive" "checkpoint=$checkpoint" > "$out/config.txt"
    df -h "$out" > "$out/storage.txt"
    cd "$here"
    "$probe_python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training \
        --baige-container --baige-exclusive-training --output "$out/gpu.json" > "$out/probe.log" 2>&1
    "$python" -B - "$out/gpu.json" "$overlay" <<'PY'
import json
from pathlib import Path
import sys
import cv2

probe = json.loads(Path(sys.argv[1]).read_text())
device, = probe["devices"]
name = device["cuda_name"].upper()
if probe["status"] != "passed" or ("RTX PRO 6000" not in name and name != "NVIDIA RPBZZZ6"):
    raise RuntimeError("Expected one scheduler RTX PRO 6000-series GPU")
if (Path(sys.argv[2]) / "cv2").resolve() != Path(cv2.__file__).resolve().parent:
    raise RuntimeError(f"OpenCV came from {cv2.__file__}")
PY
    exec "$python" -Bu "$here/evaluate.py" "$out" "$archive" "$checkpoint"
}
export -f payload
# Leave a five-second termination grace within the single 1,500-second allocation.
remaining=$((start + 1494 - $(date +%s)))
((remaining > 0))
timeout --kill-after=5s "${remaining}s" bash -euo pipefail -c 'payload "$@"' _ \
    "$code" "$out" "$root" "$legacy" "$campaign" "$here" > "$out/run.log" 2>&1
