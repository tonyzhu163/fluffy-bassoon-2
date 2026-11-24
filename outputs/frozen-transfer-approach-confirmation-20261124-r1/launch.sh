#!/usr/bin/env bash
# Both entries are fresh, append-only Linux attempts. Only the parent dispatches the GPU entry.
set -euo pipefail
study_dir=$(cd "$(dirname "$0")" && pwd -P)
source "$study_dir/../frozen-transfer-progress-20261124-r1/runtime.sh"
here=$study_dir
shared="$root/outputs/$(basename "$here")"
[[ $# == 2 && $2 =~ ^[0-9a-f]{40}$ ]]
pin=$2
[[ $(uname -s) == Linux ]]
if git -C "$repo" symbolic-ref -q HEAD >/dev/null; then exit 1; fi
pinned_tree "$repo" "$pin"

case "$1" in
  cpu)
    out="$shared/cpu01"
    [[ $(hostname) != job-* ]]
    ;;
  gpu)
    [[ $(hostname) == job-*-master-0 ]]
    out="$shared/gpu01"
    qual="$shared/cpu01"
    [[ $(cat "$qual/exit.txt" | cut -d' ' -f1) == 0 && -s "$qual/readback.json" ]]
    ;;
  *) exit 2 ;;
esac
mkdir -p "$shared"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
printf 'pin=%s host=%s attempt=%s start=%s\n' "$pin" "$(hostname)" "$1" "$start" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
exec > "$out/run.log" 2>&1
runtime
if [[ $1 == cpu ]]; then
    export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
    export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless MUJOCO_EGL_DEVICE_ID=0
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    cmd=(taskset -c 30,31 "$python" -Bu)
    extra=(--cpu)
else
    "$python" -B - "$qual/qualified.json" "$qual/readback.json" "$pin" <<'PY'
import json, sys
v, r = [json.load(open(p)) for p in sys.argv[1:3]]
assert r == dict(status="reproduced", episodes=8, pin=sys.argv[3])
assert v["fixture"] and v["pin"] == sys.argv[3] and v["status"] == "passed" and v["episodes"] == 8
for key in ("drawer", "moka"):
    task = v["tasks"][key]
    assert task["cut"] == 105 and set(task["rows"]) == {"20"}
    for name, b in task["rows"]["20"]["modes"].items():
        assert b["controls"] == 110 and (name == "tuned" or b["handback"]["t"] == 105)
PY
    export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json" CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER EGL_PLATFORM
    PYTHONPATH="$repo/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
        "$repo/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
    MUJOCO_EGL_DEVICE_ID=$("$python" -B - "$out/gpu.json" "$repo/outputs/reground-rotation-reference-20261123-r1/reference.py" <<'PY'
import json, runpy, sys
p=json.load(open(sys.argv[1]))
d, = p["devices"]
assert p["status"] == "passed" and "RTX 4090" in d["cuda_name"].upper()
print(runpy.run_path(sys.argv[2])["egl_device"](0))
PY
)
    [[ $MUJOCO_EGL_DEVICE_ID =~ ^[0-9]+$ ]]
    export MUJOCO_EGL_DEVICE_ID
    cmd=("$python" -Bu)
    extra=()
fi
"${cmd[@]}" "$here/confirmation.py" "$out" "$pin" "${extra[@]}"
"${cmd[@]}" "$here/audit.py" "$out" "$pin" "${extra[@]}" > "$out/readback.json"
