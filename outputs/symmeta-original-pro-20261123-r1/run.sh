#!/usr/bin/env bash
# Entry point inside a single-GPU Baige Distributed Training allocation.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
source "$here/prepare_runtime.sh"
original_pro_paths

paired_payload() {
    local pin=$1 out=$2 libs path drivers=""
    local -a paths
    verify_checkout "$code" "$pin"
    if git -C "$code" symbolic-ref -q HEAD >/dev/null; then
        echo "Use the detached deployment checkout for this run" >&2
        return 1
    fi
    original_pro_verify
    [[ -r "$here/CASES.json" && -r "$here/eval.py" ]]
    printf '%s  %s\n' ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170 \
        "$PI05_CHECKPOINT/READY.json" c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4 \
        "$SYMMETA_CHECKPOINT" | sha256sum -c -
    cp "$here/CASES.json" "$out/cases.input.json"
    sha256sum "$out/cases.input.json" > "$out/cases.sha256"
    original_pro_caches "$out"
    export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
    export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
    export FLUFFY_GPU_EXECUTION=baige NCCL_P2P_DISABLE=1
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER
    libs="$project/setups/libero90-full-action-20261119-r1/desktop-libs02"
    IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
    for path in "${paths[@]}"; do
        if [[ -n $path && ! -f "$path/libtorch_python.so" ]]; then drivers+=":$path"; fi
    done
    export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$drivers"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
    PYTHONPATH="$code/src" "$project/setups/robocasa365-native-20261101-r1/env/bin/python" \
        -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation \
        --baige-container --output "$out/gpu.json"
    "$SIM_PYTHON" -Bu - "$out/gpu.json" <<'PY'
import json, sys
from mujoco.egl import GLContext
from OpenGL import GL
with open(sys.argv[1]) as f:
    device = json.load(f)["devices"][0]["cuda_name"]
if "RTX PRO 6000" not in device.upper() and device != "NVIDIA RPBZZZ6":
    raise RuntimeError(f"Expected the requested Pro6K, got {device}")
context = GLContext(16, 16)
try:
    context.make_current()
    vendor, renderer = [(GL.glGetString(k) or b"").decode() for k in (GL.GL_VENDOR, GL.GL_RENDERER)]
    if "NVIDIA" not in vendor.upper():
        raise RuntimeError(f"Unexpected EGL vendor {vendor} / {renderer}")
    print(f"EGL vendor={vendor} renderer={renderer}")
finally:
    context.free()
PY
    "$POLICY_PYTHON" -Bu "$here/eval.py" run --manifest "$out/cases.input.json" --output "$out"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    [[ $# == 2 || $# == 3 ]] || { echo "Usage: bash run.sh DEPLOYED_COMMIT FRESH_ATTEMPT [WALL_SECONDS]" >&2; exit 2; }
    pin=$1 attempt=$2 wall=${3:-5400}
    [[ $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
    [[ $wall =~ ^[1-9][0-9]{0,3}$ ]] && (( wall > 10 && wall <= 5400 )) || exit 2
    [[ $(hostname) == job-*-master-0 && -n ${KUBERNETES_SERVICE_HOST:-} ]] || {
        echo "Requires a Baige Distributed Training master container" >&2; exit 2;
    }
    start=$(date +%s)
    out="$project/outputs/symmeta-original-pro-20261123-r1/$attempt"
    mkdir -p "$(dirname "$out")"
    mkdir "$out"
    trap 'rc=$?; printf "%s %s %s\n" "$rc" "$start" "$(date +%s)" > "$out/exit.txt"' EXIT
    printf 'commit=%s\nhost=%s\ncheckout=%s\nwall_seconds=%s\n' \
        "$pin" "$(hostname)" "$code" "$wall" > "$out/launch.txt"
    df -h "$out" > "$out/storage.txt"
    remain=$((start + wall - 8 - $(date +%s)))
    (( remain > 0 ))
    timeout --kill-after=6s "$remain" bash -euo pipefail -c \
        'source "$1"; paired_payload "$2" "$3"' _ "$here/run.sh" "$pin" "$out" > "$out/run.log" 2>&1
fi
