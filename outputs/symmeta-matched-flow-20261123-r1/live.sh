#!/usr/bin/env bash
set -euo pipefail

[[ $# == 5 ]] || { echo "Usage: live.sh prepare|run DEPLOYMENT_COMMIT RUN_ROOT FRESH_ATTEMPT PREDICTIONS_SHA256" >&2; exit 2; }
mode=$1 pin=$2 run_root=$3 attempt=$4 predictions_sha=$5
[[ $mode == prepare || $mode == run ]]
[[ $pin =~ ^[0-9a-f]{40}$ && $predictions_sha =~ ^[0-9a-f]{64}$ ]]
[[ $run_root == /* && $attempt =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]
[[ $(uname -s) == Linux ]]
if [[ $mode == run ]]; then [[ $(hostname) == job-*-master-0 ]]; fi
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
run_root=$(cd "$run_root" && pwd -P)
out="$run_root/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$start" "$(date +%s)" > "$out/exit.txt"' EXIT

check_tree() {
    local directory=$1 revision=$2 head status
    head=$(git -C "$directory" rev-parse HEAD)
    status=$(git --no-optional-locks -C "$directory" status --porcelain)
    if [[ $head != "$revision" || -n $status ]]; then
        echo "Wrong revision or dirty checkout: $directory (HEAD $head)" >&2
        return 1
    fi
}

payload() {
    local root=/mowenjie/fluffy-bassoon-2 legacy=/mowenjie/fluffy-bassoon
    local setup="$root/setups/xemb-icml-campaign-20261121-r1"
    local sim="$root/setups/libero-meta-learning-20261117-r1"
    local libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
    local python="$legacy/envs/openpi-215abfb/bin/python"
    local release="$root/sources/libero-pro2-e5500c69" deps="$setup/source-ff2ddf0a"
    local deps_pin=ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c
    local release_pin=e5500c690c1498eca1b7050350654964190cdf70
    local openpi_pin=215abfb217dbac7d5f1273282331b9b1866c0479
    local paths path drivers="" panel_sha device shard pid rc=0
    local -a pids=()
    check_tree "$code" "$pin"
    check_tree "$deps" "$deps_pin"
    check_tree "$release" "$release_pin"
    check_tree "$legacy/third_party/openpi-215abfb" "$openpi_pin"
    cd "$here"
    export PAIRED_DEPS="$deps" CROSSARM_RELEASE="$release" OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
    export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
    export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false NCCL_P2P_DISABLE=1
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    unset LIBGL_ALWAYS_SOFTWARE
    IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
    for path in "${paths[@]}"; do
        if [[ -n $path && ! -f "$path/libtorch_python.so" ]]; then drivers+=":$path"; fi
    done
    export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$drivers"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
    export PYTHONPATH="$setup/cv2-headless-overlay:$deps/src:$release:$sim/simulator-source:$deps/outputs/libero-crossarm-20261120-r1:$deps/outputs/libero-xemb-ur5e-sysid-20261120-r1:$deps/outputs/libero-xemb-ur5e-scale-20261120-r1:$deps/outputs/xemb-icml-campaign-20261121-r1"
    printf '%s\n' "mode=$mode" "host=$(hostname)" "tree=$code" "commit=$pin" "runner=$deps@$deps_pin" \
        "release=$release@$release_pin" "openpi=$openpi_pin" "predictions_sha256=$predictions_sha" \
        "run_root=$run_root" > "$out/config.txt"
    df -h "$out" > "$out/storage.txt"

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$python" -Bu live.py prepare "$out" \
        "$run_root/evidence/predictions.json" "$predictions_sha"
    panel_sha=$(sha256sum "$out/panel.json")
    panel_sha=${panel_sha%% *}
    printf '%s  panel.json\n' "$panel_sha" > "$out/panel.sha256"
    if [[ $mode == prepare ]]; then return 0; fi

    printf '%s\n' 'gpus=2' 'wall_seconds=5400' >> "$out/config.txt"
    export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige
    export CUDA_VISIBLE_DEVICES=0,1
    "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$deps/scripts/[train]gpu_probe.py" \
        --expected-count 2 --purpose evaluation --baige-container --output "$out/gpu.json" > "$out/probe.log" 2>&1
    "$python" -B - "$out/gpu.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    receipt = json.load(stream)
devices = receipt["devices"]
if receipt["status"] != "passed" or len(devices) != 2:
    raise SystemExit("Expected two scheduler-attested devices")
for index, d in enumerate(devices):
    name = d["cuda_name"].upper()
    if name != "NVIDIA RPBZZZ6" and "RTX PRO 6000" not in name:
        raise SystemExit(f"Expected Pro6K, found {name}")
    if d["allocation_ordinal"] != index or d["visibility_token"] != str(index):
        raise SystemExit("Unexpected CUDA ordinal mapping")
PY
    for device in 0 1; do
        (
            export CUDA_VISIBLE_DEVICES="$device" MUJOCO_EGL_DEVICE_ID="$device"
            for ((shard=device; shard<8; shard+=2)); do
                "$python" -Bu live.py shard "$out" "$panel_sha" "$shard" > "$out/shard-$shard.log" 2>&1
            done
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
    (( rc == 0 )) || return "$rc"
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$python" -Bu live.py merge "$out" "$panel_sha"
}

export mode pin run_root out here code predictions_sha
export -f payload check_tree
if [[ $mode == run ]]; then
    remaining=$((start + 5400 - 8 - $(date +%s)))
    (( remaining > 0 ))
    timeout --kill-after=5s "${remaining}s" bash -euo pipefail -c payload > "$out/run.log" 2>&1
else
    payload > "$out/run.log" 2>&1
fi
