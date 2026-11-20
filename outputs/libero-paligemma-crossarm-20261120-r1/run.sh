#!/usr/bin/env bash
# Run inside a single exclusive Baige Distributed Training container.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ! $2 =~ ^[a-z0-9]+$ ]]; then
    echo 'Usage: bash run.sh FULL_GIT_PIN FRESH_ATTEMPT [reader]' >&2
    exit 2
fi
pin=$1
attempt=$2
mode=${3:-pipeline}
[[ $mode == pipeline || $mode == reader ]]
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
release="$root/sources/libero-pro2-e5500c69"
setup="$root/setups/libero-meta-learning-20261117-r1"
probe_python="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
run="$root/outputs/$(basename "$here")"
host=$(hostname)

[[ $(uname -s) == Linux && $host == job-* ]] || {
    echo 'Use the assigned Baige Distributed Training container.' >&2
    exit 2
}
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
for file in motor.py motor_data.py motor_weights.py demos.py oracle_fallback.py sim.py adapt_eval.py run.sh; do
    git -C "$here" ls-files --error-unmatch -- "$file" >/dev/null
done

mkdir -p "$run"
out="$run/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT
exec > >(tee -a "$out/run.log") 2>&1
trap 'printf "Launcher stopped at line %s (exit %s).\n" "$LINENO" "$?" >&2' ERR
printf '%s %s %s %s\n' "$pin" "$host" "$$" "$start" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
df -i "$out" >> "$out/storage.txt"
if command -v quota >/dev/null; then quota -s >> "$out/storage.txt" 2>&1 || true; fi
[[ $(git -C "$release" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
test -x "$setup/env/bin/python"
test -x "$probe_python"
weights=${WEIGHTS:-/mowenjie/.cache/huggingface/hub/models--google--paligemma-3b-mix-224/snapshots/d1d8734c9c3ad0ccfeea4afc270faa356c2ba515}
test -f "$weights/model.safetensors.index.json"

export LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false MOTOR_CODE_REVISION="$pin"
export CROSSARM_SIM_PYTHON="$setup/env/bin/python" CROSSARM_RELEASE="$release"
export PYTHONPATH="$code/src:$release:$setup/simulator-source:$here"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export FLUFFY_GPU_EXECUTION=baige
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    visible=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ $visible == GPU-* && $visible == "${NVIDIA_VISIBLE_DEVICES:?}" && $visible != *$'\n'* ]]
    export CUDA_VISIBLE_DEVICES=0
fi
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $gpu != *$'\n'* ]]
case "$gpu" in
    *6000D*|*RPBZZZ6*) export NCCL_P2P_DISABLE=1 ;;
    *4090*) ;;
    *) echo "Unqualified accelerator: $gpu" >&2; exit 2 ;;
esac
libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
glvnd="$libs/prefix/usr/lib/x86_64-linux-gnu"
for lib in libEGL.so.1 libOpenGL.so.0 libX11.so.6; do test -r "$glvnd/$lib"; done
test -r "$libs/nvidia-vendor.json"
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
driver_paths=""
for path in "${paths[@]}"; do
    if [[ -n $path && ! -f $path/libtorch_python.so ]]; then driver_paths+=":$path"; fi
done
export LD_LIBRARY_PATH="$glvnd:$libs/prefix/lib/x86_64-linux-gnu$driver_paths"
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
"$probe_python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training \
    --baige-container --baige-exclusive-training --output "$out/gpu.json"
if [[ $mode == reader ]]; then
    "$probe_python" -Bu "$here/pretrain_reader.py" --out "$out/pretrain" --revision "$pin"
    exit 0
fi

"$probe_python" - "$weights" "$out/weight-source.json" <<'PYCODE'
import json
import sys
from pathlib import Path
from motor_weights import verified_snapshot
with verified_snapshot(sys.argv[1]) as (_, receipt):
    Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2))
PYCODE
archive="$run/assets01/goal5-base.hdf5"
collect() {
    local bank=$1 poses=$2 rc=0
    local -a ids
    IFS=, read -ra ids <<< "$poses"
    "$setup/env/bin/python" -Bu "$here/oracle_fallback.py" --matching --archive "$archive" \
        --output "$out/$bank" --poses "${ids[@]}" --bodies Panda Sawyer --seed 20261120 || rc=$?
    if (( rc )); then
        # The collector also signals individually unsuccessful contacts. Inspect the complete bank.
        "$probe_python" - "$out/$bank/index.json" <<'PYCODE'
import json
import sys
from pathlib import Path
index = json.loads(Path(sys.argv[1]).read_text())
rows = index['attempts']
if any(row.get('error') for row in rows):
    raise RuntimeError('Collector runtime error. See individual failure receipts.')
coverage = {(row['body'], row['pose']) for row in rows}
expected = {(body, pose) for body in index['bodies'] for pose in index['poses']}
if coverage != expected:
    raise RuntimeError('Collector stopped before it attempted all layouts.')
PYCODE
    fi
}
collect demos_motor "$(seq -s, 0 15)"
"$probe_python" -Bu "$here/motor.py" train --weights "$weights" --demos "$out/demos_motor" \
    --output "$out/motor" --steps 300 --seed 20261120 --microbatch 4
common=(--motor-output "$out/motor" --weights "$weights" --revision "$pin")
stage() {
    "$probe_python" -Bu "$here/adapt_eval.py" "$1" "${common[@]}" --out "$out/$1" "${@:2}"
}
stage qualify
collect demos_adapter 24,25,26,27
joint_poses=$("$probe_python" - "$out/demos_adapter" <<'PYCODE'
from pathlib import Path
import sys
root = Path(sys.argv[1])
poses = sorted(set.intersection(*({int(p.stem) for p in (root / body).glob("*.npz")}
                                  for body in ("Panda", "Sawyer"))))
if not poses:
    raise ValueError("No common successful source demonstrations for the residual.")
print(",".join(str(p) for p in poses))
PYCODE
)
stage fit --qualification "$out/qualify" --demos "$out/demos_adapter" --demo-poses "$joint_poses"
stage dev --fit "$out/fit"
stage reserved --fit "$out/fit" --dev "$out/dev"
