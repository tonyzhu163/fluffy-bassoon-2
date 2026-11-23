#!/usr/bin/env bash
# Run inside the single-GPU Baige development machine.
set -euo pipefail
source "$(dirname -- "$0")/activate.sh"
root=$REGROUND_RUNTIME_ROOT
probe="${REGROUND_GPU_PROBE:-$root/probe-source/scripts/[train]gpu_probe.py}"
[[ $(sha256sum "$probe" | cut -d' ' -f1) == f441331456e5f26d19b237d1e1037b8fb64c6725f640f28d61cada7ed6f66117 ]]
mapfile -t uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ ${#uuids[@]} == 1 ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES="${uuids[0]}" python "$probe" --direct-hostname "$(hostname)" \
    --expected-count 1 --output "$root/dev-gpu-probe.json"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi
python -u "$root/runtime/qualify.py" --device cuda --output "$root/dev-gpu-smoke.json"
