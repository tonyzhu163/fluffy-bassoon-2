#!/usr/bin/env bash
set -euo pipefail
self=$(realpath "${BASH_SOURCE[0]}")
bin=$(dirname "$self")
pin=d568535b758f2b97b382f9b3e280376e769b5942
deployed=/mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-d568535b758f
source "$deployed/outputs/symmeta-perceiver-retry-20261123-r1/live.sh"

prepare_panel() {
    local panel=$1 out=$2
    check_tree "$code" "$pin"
    environment
    check_tree "$CROSSARM_RELEASE" e5500c690c1498eca1b7050350654964190cdf70
    check_tree "$legacy/third_party/openpi-215abfb" 215abfb217dbac7d5f1273282331b9b1866c0479
    "$python" -IB "$bin/fragment.py" snapshot "$out"
    # Pinned atomic() only accepts byte-identical content at an existing panel.json.
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu invoke "$panel" live.py prepare "$panel" "$out" \
        --predictions-sha c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b \
        > "$out/$panel/prepare.log" 2>&1
}

work() {
    local panel=$1 out=$2 selector token egl shard rc failed=0
    prepare_panel "$panel" "$out"
    export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
    export __EGL_VENDOR_LIBRARY_FILENAMES="$root/setups/libero90-full-action-20261119-r1/desktop-libs02/nvidia-vendor.json"
    unset LIBGL_ALWAYS_SOFTWARE
    # Keep an explicit scheduler mask, even an invalid/empty one; the probe decides.
    if [[ -z ${CUDA_VISIBLE_DEVICES+x} ]]; then export CUDA_VISIBLE_DEVICES=0; fi
    PYTHONPATH="$code/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
        "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container \
        --output "$out/gpu.json" > "$out/probe.log" 2>&1
    PYTHONPATH="$code/src" "$python" -Bu "$here/paired_devices.py" "$out" 1 "$pin"
    selector=$("$python" -IB "$bin/fragment.py" selector "$out" "$panel")
    read -r token egl <<< "$selector"
    export CUDA_VISIBLE_DEVICES="$token" MUJOCO_EGL_DEVICE_ID="$egl"
    for shard in {0..7}; do
        rc=0
        invoke "$panel" live.py shard "$panel" "$out" --shard "$shard" --lane 0 \
            > "$out/$panel/shard-$shard.log" 2>&1 || rc=$?
        echo "$rc" > "$out/$panel/shard-$shard.rc"
        if (( rc )); then failed=1; fi
    done
    "$python" -IB "$bin/fragment.py" finish "$out" "$panel"
    return "$failed"
}

main_panel() {
    [[ $# == 1 && $(uname -s) == Linux && $(hostname) =~ ^job-[a-z0-9]+-master-0$ ]]
    local panel=$1 batch host out start rc=0
    case "$panel" in gpu09test|gpu09iiwa|gpu06kinova3|gpu08pi0fix) ;; *) return 2 ;; esac
    batch=$(dirname "$bin")
    [[ $bin == "$batch/bin" && $(dirname "$batch") == "$root/outputs/${here##*/}" && ${batch##*/} =~ ^fragment-[A-Za-z0-9_-]+$ ]]
    host=$(hostname)
    out="$batch/$panel-$host"
    mkdir -- "$out"
    start=$(date +%s)
    printf '%s\n' "panel=$panel" "host=$host" "commit=$pin" 'wall_seconds=14400' \
        "cuda_mask=${CUDA_VISIBLE_DEVICES-<unset>}" "nvidia_mask=${NVIDIA_VISIBLE_DEVICES-<unset>}" > "$out/config.txt"
    df -h "$out" > "$out/storage.txt"
    timeout --kill-after=5s 14400s bash -euo pipefail -c 'source "$1"; work "$2" "$3"' \
        bash "$self" "$panel" "$out" > "$out/run.log" 2>&1 || rc=$?
    find "$out" -mindepth 1 \( -type f -o -type d \) -exec chmod a-w {} + || rc=1
    printf '%s %s %s\n' "$rc" "$start" "$(date +%s)" > "$out/.exit"
    chmod a-w "$out/.exit"
    mv -T "$out/.exit" "$out/exit.txt"
    chmod a-w "$out"
    return "$rc"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main_panel "$@"; fi
