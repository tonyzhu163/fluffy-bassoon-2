#!/usr/bin/env bash
set -euo pipefail

start=$(date +%s)
here=$(cd "$(dirname "$0")" && pwd -P)
root=/mowenjie/fluffy-bassoon-2
campaign=xemb-icml-campaign-20261121-r1
legacy=/mowenjie/fluffy-bassoon
wrapper_pin=${1:?pass the deployed wrapper commit}
attempt=${2:?pass a fresh attempt name}
deps=${3:-$root/setups/$campaign/source-s2ab-v2}
[[ $wrapper_pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[a-z0-9]+$ ]]
[[ $(hostname) == job-* ]]
out="$root/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT

payload() {
    local here=$1 out=$2 root=$3 legacy=$4 campaign=$5 deps=$6 wrapper_pin=$7
    local dep_pin=437da7f677632e9495c813aaa52071e207ba834e
    local code release sim libs glvnd path driver_paths="" probe_python python
    local -a paths
    code=$(git -C "$here" rev-parse --show-toplevel)
    deps=$(cd "$deps" && pwd -P)
    [[ ${deps##*/} == source-s2ab-v2 && $deps != "$code" ]]
    [[ $(git -C "$deps" rev-parse --show-toplevel) == "$deps" ]]
    [[ $(git -C "$deps" rev-parse HEAD) == "$dep_pin" ]]
    [[ $(git -C "$code" rev-parse HEAD) == "$wrapper_pin" ]]
    # Check tracked wrapper bytes explicitly, including when outputs are ignored.
    local file rel
    for file in "$here"/{run.sh,smoke.py}; do
        rel=${file#"$code/"}
        [[ $(git -C "$code" rev-parse "$wrapper_pin:$rel") == "$(git hash-object "$file")" ]]
    done
    [[ -z $(git --no-optional-locks -C "$deps" status --porcelain) ]]
    export SMOKE_DEPS="$deps" SMOKE_DEP_PIN="$dep_pin" SMOKE_WRAPPER_PIN="$wrapper_pin"
    release="$root/sources/libero-pro2-e5500c69"
    sim="$root/setups/libero-meta-learning-20261117-r1"
    export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
    export CROSSARM_RELEASE="$release" OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
    export PYTHONPATH="$root/setups/$campaign/cv2-headless-overlay:$deps/src:$release:$sim/simulator-source:$deps/outputs/libero-crossarm-20261120-r1:$deps/outputs/libero-xemb-ur5e-sysid-20261120-r1:$deps/outputs/libero-xemb-ur5e-scale-20261120-r1:$deps/outputs/$campaign"
    export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
    export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige XLA_PYTHON_CLIENT_PREALLOCATE=false
    export NCCL_P2P_DISABLE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
    libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
    glvnd="$libs/prefix/usr/lib/x86_64-linux-gnu"
    IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
    for path in "${paths[@]}"; do
        if [[ -n $path && ! -f $path/libtorch_python.so ]]; then driver_paths+=":$path"; fi
    done
    export LD_LIBRARY_PATH="$glvnd$driver_paths" __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
    probe_python="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
    python="$legacy/envs/openpi-215abfb/bin/python"
    "$probe_python" -Bu "$deps/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training \
        --baige-container --baige-exclusive-training --output "$out/gpu.json" > "$out/probe.log" 2>&1
    "$probe_python" -B - "$out/gpu.json" <<'PY'
import json
import sys
from pathlib import Path

probe = json.loads(Path(sys.argv[1]).read_text())
device, = probe["devices"]
name = device["cuda_name"].upper()
if probe["status"] != "passed" or ("RTX PRO 6000" not in name and name != "NVIDIA RPBZZZ6"):
    raise SystemExit("Expected one qualified RTX PRO 6000-series card")
PY
    for policy in pi0 pi05; do
        "$python" -Bu "$here/smoke.py" "$policy" "$out/result.json" > "$out/$policy.log" 2>&1
    done
}
export -f payload
# One wall clock for pin checks, GPU probe, both model loads and all four episodes.
remaining=$((start + 712 - $(date +%s)))
((remaining > 0))
timeout --kill-after=5s "${remaining}s" bash -euo pipefail -c 'payload "$@"' _ \
    "$here" "$out" "$root" "$legacy" "$campaign" "$deps" "$wrapper_pin"
