#!/usr/bin/env bash
set -euo pipefail
export TASK_SOURCE=${1:?commit}
attempt=${2:?attempt}
root=/mowenjie/fluffy-bassoon-2
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
[[ $(hostname) == job-*-master-0 && $(git -C "$code" rev-parse HEAD) == "$TASK_SOURCE" ]]
git -C "$code" diff --quiet HEAD
run=$root/outputs/robotwin-e1-replay-20261110-r1
out=$run/$attempt
scratch=/tmp/$HOSTNAME-sapien
mkdir "$out" "$scratch"
exec > >(tee "$out/worker.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
export NATIVE_SOURCE=$root/sources/robotwin-motus-native-fa8d97b0-20261109-r1
[[ $(git -C "$NATIVE_SOURCE" rev-parse HEAD) == fa8d97b0f33bb894eae16df5a8175490f5b977a7 ]]
py=$root/envs/motus-reference-openpi-20261109-r2/bin/python
export PYTHONPATH=$code/src FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
 uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
 [[ $uuid == GPU-* && $uuid != *$'\n'* && $uuid == "${NVIDIA_VISIBLE_DEVICES:?}" ]]; export CUDA_VISIBLE_DEVICES=0
fi
"$py" "$code/scripts/[train]gpu_probe.py" --baige-container --purpose evaluation --expected-count 1 --output "$out/gpu.json"
export LYCO_BAIGE_GPU_MAP=$out/gpu.json
mkdir -p /root/.sapien/physx
[[ -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]] || ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
git -C "$NATIVE_SOURCE" archive HEAD envs task_config description script | tar -x -C "$scratch"
"$py" "$code/outputs/robotwin-motus-release-baseline-20261109-r1/prepare_assets.py" "$scratch/assets" "$out/assets.json"
export PYTHONPATH=$scratch:$scratch/description/utils:$scratch/script:$PYTHONPATH
cd "$scratch"
timeout -k 15 1600 "$py" -Bu "$code/outputs/robotwin-e1-replay-20261110-r1/replay.py" "$out" "$run/input" "$scratch" > "$out/replay.log" 2>&1
if grep -Eq 'OIDN Error|unsupported device type|invalid handle' "$out/replay.log"; then exit 1; fi
