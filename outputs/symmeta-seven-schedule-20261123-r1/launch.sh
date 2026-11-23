#!/usr/bin/env bash
set -euo pipefail

[[ $# == 2 ]] || { echo "Usage: launch.sh EXPECTED_DEPLOYMENT_COMMIT FRESH_ATTEMPT" >&2; exit 2; }
pin=$1 attempt=$2 host=$(hostname)
[[ $host == job-*-master-0 ]] || { echo "Requires a single-node Baige Distributed Training job" >&2; exit 2; }
[[ $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^gpu[0-9a-z]+$ ]] || { echo "Invalid commit or attempt" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
head=$(git -C "$code" rev-parse HEAD)
status=$(git --no-optional-locks -C "$code" status --porcelain)
[[ $head == "$pin" && -z $status ]] || { echo "Expected clean $pin, found $head: $status" >&2; exit 2; }
# outputs/ is ignored. Verify the actual launch and training files are in this commit.
for source in "$here/launch.sh" "$here/run.py" "$here/paired.py" "$here/../xemb-icml-campaign-20261121-r1/stage1_offline_gate.py"; do
    rel=$(realpath --relative-to="$code" "$source")
    [[ $(git -C "$code" rev-parse "$pin:$rel") == "$(git hash-object "$source")" ]] || exit 2
done

root=/mowenjie/fluffy-bassoon-2
run="$root/outputs/$(basename "$here")"
mkdir -p "$run"
out="$run/$attempt"
mkdir "$out" || { echo "Attempt must be new: $out" >&2; exit 2; }
start=$(date +%s)
trap 'rc=$?; end=$(date +%s); printf "%s %s %s\n" "$rc" "$end" "$start" > "$out/exit.txt"; echo "$end" > "$out/gpu-end.txt"' EXIT
echo "$start" > "$out/gpu-start.txt"
printf '%s %s %s %s\n' "$pin" "$host" "$$" "$code" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"

python=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
probe_python="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
export PYTHONPATH="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay:$code/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_P2P_DISABLE=1
# Use the same driver-library selection as the Stage 1 launcher.
libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
driver_paths=""
for path in "${paths[@]}"; do
    if [[ -n $path && ! -f "$path/libtorch_python.so" ]]; then driver_paths+=":$path"; fi
done
export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$driver_paths"

"$probe_python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training \
    --baige-container --baige-exclusive-training --output "$out/gpu.json" > "$out/probe.log" 2>&1
"$python" -B - "$out/gpu.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    probe = json.load(stream)
device, = probe["devices"]
name = device["cuda_name"].upper()
if probe["status"] != "passed" or (name not in {"NVIDIA GEFORCE RTX 4090", "NVIDIA RPBZZZ6"}
                                    and "RTX PRO 6000" not in name):
    raise SystemExit(f"Expected an allocated RTX 4090 or Pro6K, got {name}")
PY
"$python" -Bu "$here/run.py" --revision "$pin" --output "$out/results" > "$out/run.log" 2>&1
