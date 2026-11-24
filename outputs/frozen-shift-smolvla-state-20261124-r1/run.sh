#!/usr/bin/env bash
# Usage: bash run.sh SCRATCH TAG ARM HOOK_CLASS NOISE_OFFSET LAYOUT...
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
base="$here/../frozen-shift-smolvla-20261124-r1"
scratch=$(realpath "$1"); tag=$2; arm=$3; hook=$4; offset=$5; shift 5
[[ $(git -C "$scratch/pro2" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
export CROSSARM_RELEASE="$scratch/pro2" LIBERO_CONFIG_PATH="$scratch/libero-config"
export PYTHONPATH="$here:$base:$here/../libero-crossarm-20261120-r1:$scratch/pro2:$scratch/pro2/simulator-source"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_EGL_DEVICE_ID=0
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset CUDA_VISIBLE_DEVICES
exec "$scratch/env/bin/python" -u "$base/harness.py" --scratch "$scratch" --arm "$arm" --task 3 \
    --noise-offset "$offset" --hooks "adapters_state:$hook" --out "$here/evidence/$tag" --layout "$@"
