#!/usr/bin/env bash
set -euo pipefail
[[ $# == 6 && $(uname -s) == Linux && $(hostname) == job-*-master-0 ]]
export ZERO_PIN=$1
attempt=$2; lane=$3; first=$4; stop=$5; wall=$6
[[ $attempt =~ ^gpu[[:alnum:]_-]+$ && $lane =~ ^[0-3]$ && $first =~ ^[0-9]+$ && $stop =~ ^[0-9]+$ && $wall =~ ^[0-9]+$ ]]
(( first >= 64 && first % 4 == 0 && stop > first && stop % 4 == 0 && stop <= 256 && wall > 0 && wall <= 3420 ))
source "$(dirname "$0")/environment.sh"
cores=$("$python" -c 'import os; a=sorted(os.sched_getaffinity(0)); b=[x for x in a if x not in (31,40,41)]; print(",".join(map(str,(b or a)[:2])))')
taskset -pc "$cores" "$$"
reserved="$shared/reserve-$attempt"
out="$shared/$attempt"
[[ ! -e $out && -s $reserved/reservation.json ]]
"$python" - "$reserved/reservation.json" "$ZERO_PIN" "$first" "$stop" "$lane" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert [r["pin"], str(r["start"]), str(r["stop"]), r["lane"]] == sys.argv[2:]
PY
mv -T "$reserved" "$out"
start=$(date +%s)
scratch=
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$start" "$(date +%s)" > "$out/.exit.txt.slot"; mv "$out/.exit.txt.slot" "$out/exit.txt"; if [[ $rc == 0 && -n $scratch ]]; then rm -rf -- "$scratch"; fi' EXIT
exec > >(tee "$out/console.log") 2>&1
printf '%s\n' "$ZERO_PIN" "$(hostname)" "$@" > "$out/launch.txt"
read -r fit_rc _ < "$fit/exit.txt"; [[ $fit_rc == 0 ]]
[[ $(sha256sum "$fit/fit.json" | cut -d' ' -f1) == "$fit_sha" ]]
[[ $(sha256sum "$manifest" | cut -d' ' -f1) == "$manifest_sha" ]]
scratch=$(mktemp -d "/tmp/plate-zero-$attempt-XXXXXXXX")
[[ $(stat -c '%d' "$scratch") != "$(stat -c '%d' "$out")" ]]
df -h "$scratch" "$out" > "$out/storage.txt"
df -i "$out" >> "$out/storage.txt"
export XDG_CACHE_HOME="$scratch/cache" JAX_COMPILATION_CACHE_DIR="$scratch/jax"
export TMPDIR="$scratch/tmp" NUMBA_CACHE_DIR="$scratch/numba" MPLCONFIGDIR="$scratch/mpl"
mkdir "$TMPDIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json" FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda
unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
gpu_python="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
PYTHONPATH="$code/src" "$gpu_python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 --baige-container --purpose evaluation --output "$scratch/gpu.json"
cat "$scratch/gpu.json" > "$out/.gpu.json.slot"
cmp "$scratch/gpu.json" "$out/.gpu.json.slot"
mv "$out/.gpu.json.slot" "$out/gpu.json"
"$python" - "$out/gpu.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["status"] == "passed" and [x["cuda_name"].upper() for x in p["devices"]] == ["NVIDIA GEFORCE RTX 4090"]
PY
export MUJOCO_EGL_DEVICE_ID=$("$python" - "$code/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import runpy, sys
print(runpy.run_path(sys.argv[1])["egl_device"](0))
PY
)
[[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
q="$out/shard$lane"
printf '%s\n' "$ZERO_PIN" "$EVAL_PIN" "$fit_pin" "$fit_sha" "$manifest_sha" "$first" "$stop" "$lane" "$MUJOCO_EGL_DEVICE_ID" "$CUDA_VISIBLE_DEVICES" > "$q/run.txt"
remaining=$((wall - $(date +%s) + start))
(( remaining > 0 ))
timeout --kill-after=25s "$remaining" "$python" -Bu "$zero_here/lanes.py" run "$scratch/shard$lane" "$q" \
    "$python" -Bu "$zero_here/branch.py" "$scratch/shard$lane" --fit "$fit" --fit-sha "$fit_sha" --fit-pin "$fit_pin" \
    --manifest "$manifest" --manifest-sha "$manifest_sha" --start "$first" --stop "$stop" --shard "$lane" \
    > "$q/console.log" 2>&1
