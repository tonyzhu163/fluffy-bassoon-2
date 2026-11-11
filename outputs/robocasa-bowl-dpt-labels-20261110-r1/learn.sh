#!/usr/bin/env bash
set -Eeuo pipefail
pin=$1 mode=$2 out=$3; shift 3
root=/mowenjie/fluffy-bassoon-2
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
study=$root/outputs/robocasa-bowl-dpt-labels-20261110-r1
base=$root/outputs/robocasa-sft-reproduction-20261109-r1
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
git -C "$code" diff --quiet HEAD
export RLINF_PATH=$root/sources/critic-selected-a1229239-r1
test "$(git -C "$RLINF_PATH" rev-parse HEAD)" = 8f9bfa7661928ef8c17494b04f4d7bf3125634e1
export PYTHONPATH="$code/src:$RLINF_PATH:$code/outputs/robocasa-three-ppo-dpt-lora-20261110-r1:$code/outputs/robocasa-compact-bowls-20261109-r1:$code/outputs/robocasa-meta-feedback-20261109-r1:$code/outputs/robocasa-sft-reproduction-20261109-r1:$base/sources/openpi/src:$base/sources/openpi/packages/openpi-client/src:$base/sources/robocasa"
export ROBOCASA_PI05_TORCH_MODEL=$root/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$root/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
export OPENPI_DATA_HOME=$root/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export PILOT_COMMIT=$pin MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NCCL_P2P_DISABLE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=73
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
case "$out" in "$study"/*) ;; *) exit 2 ;; esac
receipt=$out-receipt
mkdir -p "$(dirname "$out")"
mkdir "$receipt"
exec > >(tee "$receipt/run.log") 2>&1
begin=$(date +%s)
trap 'rc=$?; echo "{\"exit_code\":$rc,\"started\":$begin,\"ended\":$(date +%s)}" > "$receipt/clock.json"' EXIT
if [[ $(hostname) == aibox-* ]]; then
    test "${CGPU_COUNT:?}" = 1 && test "${CGPU0_COMPUTE_LIMIT:?}" = 100
    test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = "${NVIDIA_VISIBLE_DEVICES:?}"
    export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
    export LD_LIBRARY_PATH="$study/dev-graphics-01/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    probe=(--purpose measurement --direct-hostname "$(hostname)")
    if [[ $mode == fit ]]; then probe=(--purpose training --direct-hostname "$(hostname)"); fi
else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    probe=(--purpose evaluation --baige-container)
    if [[ $mode == fit ]]; then probe=(--purpose training --baige-container --baige-exclusive-training); fi
fi
"$py" -B "$code/scripts/[train]gpu_probe.py" --expected-count 1 --output "$receipt/gpu.json" "${probe[@]}" > "$receipt/probe.log"
case "$mode" in
    fit) train=$1; valid=$2; shift 2; mkdir "$out"
         "$py" -Bu "$here/fit_bowl.py" "$code" "$train" "$valid" "$out" "$@" ;;
    eval) checkpoint=$1; source=$2; admission=$3; shift 3
          "$py" -Bu "$here/eval_bowl.py" "$code" "$checkpoint" "$source" "$admission" "$out" "$@" ;;
    direction) "$py" -Bu "$here/paired_direction.py" "$code" "$1" "$2" "$3" "$out" ;;
    trace) "$py" -Bu "$here/trace_motor.py" "$code" "$1" "$2" "$3" "$out" ;;
    common-prefix) "$py" -Bu "$here/common_prefix.py" "$1" "$2" "$out" ;;
    control-replay) "$py" -Bu "$here/qualify_controls.py" "$1" "$ROBOCASA_PI05_NATIVE_NORM" "$out" --seed "$2" ;;
    *) exit 2 ;;
esac
