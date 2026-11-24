#!/usr/bin/env bash
# One or more native CPU episodes of frozen SmolVLA. Works unchanged on any Linux host with Mesa EGL.
# Usage: bash run.sh SCRATCH TAG ARM LAYOUT [LAYOUT ...]
# Optional environment: TASK (default 3), NOISE_OFFSET (default 0), HOOKS (module:Class), OMP_NUM_THREADS (2).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(realpath "${1:?scratch directory from setup.sh}")
tag=${2:?evidence tag}
arm=${3:?arm}
shift 3
test -x "$scratch/env/bin/python"
[[ $(git -C "$scratch/pro2" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
export CROSSARM_RELEASE="$scratch/pro2" LIBERO_CONFIG_PATH="$scratch/libero-config"
export PYTHONPATH="$here:$here/../libero-crossarm-20261120-r1:$scratch/pro2:$scratch/pro2/simulator-source${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=${OMP_NUM_THREADS:-2}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # released init-state files are pickled NumPy arrays
unset CUDA_VISIBLE_DEVICES
exec "$scratch/env/bin/python" -u "$here/harness.py" --scratch "$scratch" --arm "$arm" --task "${TASK:-3}" \
    --noise-offset "${NOISE_OFFSET:-0}" --hooks "${HOOKS:-}" --out "$here/evidence/$tag" --layout "$@"
