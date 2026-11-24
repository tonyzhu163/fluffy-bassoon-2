#!/usr/bin/env bash
# Root submits gpu mode; a four-card job assigns one whole trajectory triplet to each lane.
set -euo pipefail
mode=${1:?cpu or gpu}
export EVAL_PIN=${2:?evaluation commit}
attempt=${3:?fresh attempt}
[[ $(uname -s) == Linux && $EVAL_PIN =~ ^[0-9a-f]{40}$ && $attempt =~ ^[[:alnum:]_-]+$ ]]
source "$(dirname "$0")/environment.sh"
out="$shared/$attempt"
if [[ $mode == gpu ]]; then
    reserved="$shared/reserve-$attempt"
    [[ ! -e $out && -s $reserved/reservation.json ]]
    "$python" - "$reserved/reservation.json" "$EVAL_PIN" "${9}" "${10}" "${11}" <<'CHECK'
import json, sys
r = json.load(open(sys.argv[1]))
assert [r["pin"], str(r["start"]), str(r["stop"]), r["lane"]] == sys.argv[2:]
CHECK
    mv -T "$reserved" "$out"
else
    mkdir "$out"
fi
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/.exit.txt.slot"; mv "$out/.exit.txt.slot" "$out/exit.txt"' EXIT
exec > >(tee "$out/console.log") 2>&1
printf '%s\n' "$EVAL_PIN" "$(hostname)" "$@" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
scratch="$out"
if [[ $mode == gpu ]]; then
    scratch=$(mktemp -d "/tmp/plate-$attempt-XXXXXXXX")
    export XDG_CACHE_HOME="$scratch/cache" JAX_COMPILATION_CACHE_DIR="$scratch/jax"
    [[ $(stat -c '%d' "$scratch") != $(stat -c '%d' "$out") ]]
    df -h "$scratch" >> "$out/storage.txt"
    df -i "$out" >> "$out/storage.txt"
fi
export TMPDIR="$scratch/tmp" NUMBA_CACHE_DIR="$scratch/numba" MPLCONFIGDIR="$scratch/mpl"
mkdir "$TMPDIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
if [[ $mode == cpu ]]; then
    [[ $# == 3 && $attempt == cpu* ]]
    export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    taskset -c 40,41 "$python" -Bu "$here/panel.py" "$out" --fixture
else
    [[ $mode == gpu && $# == 12 && $(hostname) == job-*-master-0 ]]
    fit_pin=$4; fit_attempt=$5; fit_sha=$6; prepared=$7; manifest_sha=$8; first=$9; stop=${10}; lane=${11}; wall=${12}
    [[ $fit_pin =~ ^[0-9a-f]{40}$ && $fit_attempt =~ ^gpu[[:alnum:]_-]+$ && $fit_sha =~ ^[0-9a-f]{64}$ ]]
    [[ $prepared =~ ^prepare[0-9]+$ && $manifest_sha =~ ^[0-9a-f]{64}$ && $first =~ ^[0-9]+$ && $stop =~ ^[0-9]+$ ]]
    [[ $wall =~ ^[0-9]+$ && $wall -gt 0 && $wall -le 32400 && $lane =~ ^(all|[0-3])$ ]]
    # One launch may compute a disjoint part of a registered cohort. Readback still requires the full prefix.
    bound=64
    while ((first >= bound)); do bound=$((2*bound)); done
    [[ $first -lt $stop && $stop -le $bound && $((first % 4)) -eq 0 && $((stop % 4)) -eq 0 ]]
    fit_shared="$shared/../fit"
    fit="$fit_shared/$fit_attempt"
    fit_tree="$fit_shared/code-$fit_pin"
    [[ $(git -C "$fit_tree" rev-parse HEAD) == "$fit_pin" && -z $(git -C "$fit_tree" status --porcelain --untracked-files=no) ]]
    read -r rc _ < "$fit/exit.txt"; [[ $rc == 0 ]]
    [[ $(sha256sum "$fit/fit.json" | cut -d' ' -f1) == "$fit_sha" ]]
    manifest="$shared/$prepared/panel/manifest.json"
    [[ $(sha256sum "$manifest" | cut -d' ' -f1) == "$manifest_sha" ]]
    export PYTHONPATH="$fit_tree/outputs/frozen-vla-plate-transfer-20261124-r1/fit:$fit_tree/outputs/frozen-vla-cotraining-20261124-r1/model:$PYTHONPATH"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json" FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
    n=1
    if [[ $lane == all ]]; then n=4; fi
    if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
        # The exclusive Baige container can export only NVIDIA_VISIBLE_DEVICES.
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n-1)))
    fi
    gpu_python="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
    PYTHONPATH="$code/src" "$gpu_python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count "$n" --baige-container --purpose evaluation --output "$scratch/gpu.json"
    cat "$scratch/gpu.json" > "$out/.gpu.json.slot"
    cmp "$scratch/gpu.json" "$out/.gpu.json.slot"
    mv "$out/.gpu.json.slot" "$out/gpu.json"
    "$python" - "$out/gpu.json" "$n" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["status"] == "passed" and len(p["devices"]) == int(sys.argv[2])
assert all(d["cuda_name"].upper() == "NVIDIA GEFORCE RTX 4090" for d in p["devices"]), p["devices"]
PY
    cpuset=$("$python" -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')
    IFS=, read -ra cpu_ids <<< "$cpuset"
    declare -a gpu_ids pids
    IFS=, read -ra gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    for ((rank=0; rank<n; rank++)); do
        shard=$lane
        if [[ $lane == all ]]; then shard=$rank; fi
        [[ -d $out/shard$shard ]]
        egl=$("$python" - "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" "$rank" <<'PY'
import runpy, sys
print(runpy.run_path(sys.argv[1])["egl_device"](int(sys.argv[2])))
PY
)
        i=$((2*rank % ${#cpu_ids[@]}))
        j=$(((i+1) % ${#cpu_ids[@]}))
        cores="${cpu_ids[i]},${cpu_ids[j]}"
        extra=()
        if [[ -n ${EVAL_RESUME_FROM:-} ]]; then
            [[ $EVAL_RESUME_FROM =~ ^gpu[[:alnum:]_-]+$ ]]
            extra+=(--resume "$shared/$EVAL_RESUME_FROM/shard$shard")
        fi
        remaining=$((wall - $(date +%s) + start))
        [[ $remaining -gt 0 && $egl =~ ^[0-9]+$ ]]
        q="$out/shard$shard"
        (
            export MUJOCO_EGL_DEVICE_ID="$egl" CUDA_VISIBLE_DEVICES="${gpu_ids[rank]}"
            printf '%s\n' "$EVAL_PIN" "$fit_pin" "$fit_sha" "$manifest_sha" "$first" "$stop" "$shard" "$egl" "$CUDA_VISIBLE_DEVICES" > "$q/run.txt"
            timeout --kill-after=25s "$remaining" taskset -c "$cores" "$python" -Bu "$here/storage.py" run "$scratch/shard$shard" "$q" "$python" -Bu "$here/panel.py" "$scratch/shard$shard" --fit "$fit" --fit-sha "$fit_sha" --fit-pin "$fit_pin" "${extra[@]}" --manifest "$manifest" --manifest-sha "$manifest_sha" --start "$first" --stop "$stop" --shard "$shard"
        ) > "$q/console.log" 2>&1 &
        pids+=("$!")
    done
    rc=0
    for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
    exit "$rc"
fi
