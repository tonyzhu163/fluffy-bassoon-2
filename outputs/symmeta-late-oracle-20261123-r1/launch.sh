#!/usr/bin/env bash
set -euo pipefail
self=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
export HELPER_TREE=/mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-d568535b758f
helper_pin=d568535b758f2b97b382f9b3e280376e769b5942
source "$HELPER_TREE/outputs/symmeta-perceiver-retry-20261123-r1/live.sh"
attempts="$root/outputs/symmeta-late-oracle-20261123-r1"

execute() {
    local mode=$1 pin=$2 out=$3 count=$4 expected_panel=$5
    check_tree "$self" "$pin"
    check_tree "$HELPER_TREE" "$helper_pin"
    environment
    export PAIRED_DEPS="$setup/source-ff2ddf0a"
    check_tree "$PAIRED_DEPS" ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c
    check_tree "$setup/source-s2ab-v2" 437da7f677632e9495c813aaa52071e207ba834e
    check_tree "$CROSSARM_RELEASE" e5500c690c1498eca1b7050350654964190cdf70
    check_tree "$legacy/third_party/openpi-215abfb" 215abfb217dbac7d5f1273282331b9b1866c0479
    export PYTHONPATH="$setup/cv2-headless-overlay:$PAIRED_DEPS/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$PAIRED_DEPS/outputs/libero-crossarm-20261120-r1:$PAIRED_DEPS/outputs/libero-xemb-ur5e-sysid-20261120-r1:$PAIRED_DEPS/outputs/libero-xemb-ur5e-scale-20261120-r1:$PAIRED_DEPS/outputs/$campaign:$HELPER_TREE/outputs/symmeta-perceiver-retry-20261123-r1"
    export NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib" XDG_CACHE_HOME="$out/cache"
    mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$python" -Bu "$self/driver.py" prepare "$out"
    sha256sum "$out/panel.json" | cut -d' ' -f1 > "$out/panel.sha256"
    if [[ -n $expected_panel ]]; then
        [[ $(cat "$out/panel.sha256") == "$expected_panel" ]] || { echo "Panel differs from qualified CPU input"; return 1; }
    fi
    if [[ $mode == cpu ]]; then
        export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1
        export MUJOCO_EGL_DEVICE_ID=0 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
        "$python" -Bu "$self/driver.py" cpu "$out"
        return
    fi
    if [[ $mode == prepare ]]; then return; fi
    export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
    export __EGL_VENDOR_LIBRARY_FILENAMES="$root/setups/libero90-full-action-20261119-r1/desktop-libs02/nvidia-vendor.json"
    unset LIBGL_ALWAYS_SOFTWARE
    if [[ -z ${CUDA_VISIBLE_DEVICES+x} ]]; then
        export CUDA_VISIBLE_DEVICES
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((count-1)))
    fi
    PYTHONPATH="$HELPER_TREE/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
        "$HELPER_TREE/scripts/[train]gpu_probe.py" --expected-count "$count" --purpose evaluation \
        --baige-container --output "$out/gpu.json" > "$out/probe.log" 2>&1
    "$python" -Bu "$self/driver.py" devices "$out" --cards "$count"
    local lane token egl pid rc=0
    local -a pids=() selectors=()
    mapfile -t selectors < <("$python" -Bc 'import json,sys
for d in json.load(open(sys.argv[1])): print(d["visibility_token"],d["egl_device"])' "$out/devices.json")
    [[ ${#selectors[@]} == "$count" ]]
    for ((lane=0;lane<count;lane++)); do
        read -r token egl <<< "${selectors[lane]}"
        CUDA_VISIBLE_DEVICES="$token" MUJOCO_EGL_DEVICE_ID="$egl" "$python" -Bu "$self/driver.py" \
            lane "$out" --lane "$lane" --cards "$count" > "$out/lane-$lane.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
    (( rc == 0 )) || return "$rc"
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$python" -Bu "$self/driver.py" merge "$out"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    [[ $# == 5 || $# == 6 ]] || { echo 'Usage: launch.sh cpu|prepare|run CODE_PIN FRESH_ATTEMPT CARDS WALL_SECONDS [QUALIFIED_PANEL_SHA256]' >&2; exit 2; }
    mode=$1 pin=$2 attempt=$3 count=$4 wall=$5 expected_panel=${6:-}
    [[ $(uname -s) == Linux && $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ && $wall =~ ^[1-9][0-9]{0,4}$ ]]
    if [[ $mode == run ]]; then
        [[ $(hostname) == job-*-master-0 && $count =~ ^[1-4]$ ]]
        [[ $expected_panel =~ ^[0-9a-f]{64}$ ]]
        (( count*wall < 10800 ))
    else
        [[ ( $mode == prepare || $mode == cpu ) && $count == 0 ]]
        export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
    fi
    (( wall > 5 ))
    out="$attempts/$attempt"
    mkdir "$out"
    start=$(date +%s)
    trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
    printf '%s\n' "host=$(hostname)" "mode=$mode" "pin=$pin" "cards=$count" "wall_seconds=$wall" "expected_panel_sha256=$expected_panel" > "$out/config.txt"
    df -h "$out" > "$out/storage.txt"
    timeout --kill-after=5s "$((wall-5))s" bash -euo pipefail -c 'source "$1"; shift; execute "$@"' \
        bash "$self/launch.sh" "$mode" "$pin" "$out" "$count" "$expected_panel" > "$out/run.log" 2>&1
fi
