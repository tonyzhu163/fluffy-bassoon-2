#!/usr/bin/env bash
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
pin=${1:?commit}
mode=${2:?collect or train}
id=${3:?fresh output id}
shift 3
[[ $id =~ ^[a-z0-9-]+$ ]]
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
git -C "$code" diff --quiet HEAD
base=$root/outputs/robocasa-sft-reproduction-20261109-r1
study=$root/outputs/robocasa-feedback-advantage-20261117-r1
test "$(git -C "$base/sources/openpi" rev-parse HEAD)" = ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8
test "$(git -C "$base/sources/robocasa" rev-parse HEAD)" = 29f7ce8814c1547f5af762a0997fbd4b64848dd7
out=$study/$id
mkdir -p "$study"
test ! -e "$out"
receipt=$out-receipt
mkdir "$receipt"
exec > >(tee "$receipt/run.log") 2>&1
started=$(date +%s)
trap 'rc=$?; echo "{\"exit_code\":$rc,\"started\":$started,\"ended\":$(date +%s)}" > "$receipt/clock.json"' EXIT
echo "$pin $(hostname) $mode $*" > "$receipt/identity.txt"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONHASHSEED=73 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.72
export JAX_COMPILATION_CACHE_DIR="$base/jax-cache" OPENPI_DATA_HOME="$base/openpi-cache"
export PYTHONPATH="$base/sources/openpi/src:$base/sources/openpi/packages/openpi-client/src:$base/sources/robocasa:$code/src:$here"
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
if [[ $(hostname) == aibox-* ]]; then
    test "${CGPU_COUNT:?}" = 1 && test "${CGPU0_COMPUTE_LIMIT:?}" = 100
    test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = "${NVIDIA_VISIBLE_DEVICES:?}"
    export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
    export LD_LIBRARY_PATH="$root/outputs/robocasa-bowl-dpt-labels-20261110-r1/dev-graphics-01/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    if [[ $mode == train ]]; then
        probe=(--purpose training --direct-hostname "$(hostname)")
    else
        probe=(--purpose measurement --direct-hostname "$(hostname)")
    fi
else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    if [[ $mode == train ]]; then
        probe=(--purpose training --baige-container --baige-exclusive-training)
    else
        probe=(--purpose evaluation --baige-container)
    fi
fi
"$py" -B "$code/scripts/[train]gpu_probe.py" --expected-count 1 --output "$receipt/gpu.json" "${probe[@]}" > "$receipt/probe.log"
case "$mode" in
    collect)
        "$py" -Bu "$here/collect_panel.py" "$out" "$root/inputs/robocasa365-public-pi05-c484-75000-20261101-r1" "$@"
        ;;
    train)
        panel=${1:?panel dir}; shift
        "$py" -Bu "$here/train_and_eval.py" --panel "$panel" --out "$out" "$@"
        "$py" -Bu "$here/audit.py" --panel "$panel" --fit "$out" --output "$out/AUDIT.json"
        ;;
    *)
        exit 2
        ;;
esac
