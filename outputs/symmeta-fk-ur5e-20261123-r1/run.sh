#!/usr/bin/env bash
set -euo pipefail
start=$(date +%s)
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
root=/mowenjie/fluffy-bassoon-2
run=symmeta-fk-ur5e-20261123-r1
[[ $# == 4 ]] || { echo "Usage: bash run.sh DEPLOYED_COMMIT NEW_ATTEMPT GPU_COUNT WALL_SECONDS" >&2; exit 2; }
pin=$1 attempt=$2 count=$3 wall=$4
[[ $pin =~ ^[[:xdigit:]]{40}$ && $attempt =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
[[ $count =~ ^[1-8]$ && $wall =~ ^[1-9][0-9]{0,4}$ ]]
(( wall > 10 && count * wall <= 28800 ))
[[ $(hostname) == job-*-master-0 && "$here" == "$code/outputs/$run" ]]
out="$root/outputs/$run/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$start" "$(date +%s)" > "$out/exit.txt"' EXIT
export here code root out count wall pin
payload() {
    local campaign=xemb-icml-campaign-20261121-r1
    local legacy=/mowenjie/fluffy-bassoon
    local sim="$root/setups/libero-meta-learning-20261117-r1" release="$root/sources/libero-pro2-e5500c69"
    local repair="$root/setups/$campaign/source-ff2ddf0a"
    local repair_pin=ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c
    local libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
    local overlay="$root/setups/$campaign/cv2-headless-overlay"
    local python="$legacy/envs/openpi-215abfb/bin/python"
    local d expected file path driver_paths="" device shard rc=0
    local -a paths pids=()
    for d in "$code" "$repair"; do
        case "$d" in "$code") expected=$pin;; "$repair") expected=$repair_pin;; esac
        [[ $(git -C "$d" rev-parse HEAD) == "$expected" ]]
        [[ -z $(git --no-optional-locks -C "$d" status --porcelain) ]]
    done
    for file in "$here"/{run.sh,paired.py}; do
        [[ $(git -C "$code" rev-parse "$pin:${file#"$code/"}") == "$(git hash-object "$file")" ]]
    done
    [[ $(git -C "$release" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
    export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
    export CROSSARM_RELEASE="$release" OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
    export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige XLA_PYTHON_CLIENT_PREALLOCATE=false
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
    for path in "${paths[@]}"; do
        if [[ -n "$path" && ! -f "$path/libtorch_python.so" ]]; then driver_paths+=":$path"; fi
    done
    export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$driver_paths"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
    export PAIRED_DEPS="$repair"
    export PYTHONPATH="$overlay:$repair/src:$release:$sim/simulator-source:$repair/outputs/libero-crossarm-20261120-r1:$repair/outputs/libero-xemb-ur5e-sysid-20261120-r1:$repair/outputs/libero-xemb-ur5e-scale-20261120-r1:$repair/outputs/$campaign"
    cd "$here"
    printf '%s\n' "commit=$pin" "tree=$code" "host=$(hostname)" "gpus=$count" "wall_seconds=$wall" \
        "runner=$repair@$repair_pin" > "$out/config.txt"
    df -h "$out" > "$out/storage.txt"
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((count-1)))
    "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$repair/scripts/[train]gpu_probe.py" \
        --expected-count "$count" --purpose training --baige-container --baige-exclusive-training \
        --output "$out/gpu.json" > "$out/probe.log" 2>&1
    "$python" -B - "$out/gpu.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    probe = json.load(f)
for device in probe["devices"]:
    name = device["cuda_name"].upper()
    if "RTX PRO 6000" not in name and name != "NVIDIA RPBZZZ6":
        raise SystemExit(f"Expected qualified RTX PRO 6000-series GPU, found {name}")
PY
    JAX_PLATFORMS=cpu "$python" -Bu "$here/paired.py" prepare "$out"
    for ((device=0; device<count; device++)); do
        (
            export CUDA_VISIBLE_DEVICES="$device" MUJOCO_EGL_DEVICE_ID="$device"
            for ((shard=device; shard<8; shard+=count)); do
                "$python" -Bu "$here/paired.py" shard "$out" --shard "$shard" > "$out/shard-$shard.log" 2>&1
            done
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
    (( rc == 0 )) || return "$rc"
    JAX_PLATFORMS=cpu "$python" -Bu "$here/paired.py" merge "$out"
}
export -f payload
remaining=$((start + wall - 6 - $(date +%s)))
(( remaining > 0 ))
timeout --kill-after=5s "${remaining}s" bash -euo pipefail -c payload > "$out/run.log" 2>&1
