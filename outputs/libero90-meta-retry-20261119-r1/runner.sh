#!/usr/bin/env bash
# Baige Distributed Training, one assigned GPU per job. From the deployed repo:
# bash outputs/libero90-meta-retry-20261119-r1/runner.sh --stage fitting --shard-index 0 --shards 8
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(cd "$here/../.." && pwd)
shared=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
model_python="$legacy/envs/openpi-215abfb/bin/python"
export BAIGE_PROBE_PYTHON="$shared/setups/robocasa365-native-20261101-r1/env/bin/python"
export LIBERO_SIM_PYTHON=/mowenjie/envs/wam-libero-plus-py310/bin/python
export LIBERO_STD_ROOT="$shared/deps/libero-dsrl-87edbd1-20261029-r1/repo"
export LIBERO_CONFIG_PATH="$shared/cache/libero-dsrl-87edbd1-20261029-r1"
export OPENPI_ROOT="$legacy/third_party/openpi-215abfb"
export OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export LIBERO_SIM_PYTHONPATH="$LIBERO_STD_ROOT:$OPENPI_ROOT/packages/openpi-client/src:$legacy/envs/pi05-plus-client-deps"
glvnd="$shared/setups/libero-gen-history-pilot-20261111-r1/os-libs01/prefix/usr/lib/x86_64-linux-gnu"
for path in "$model_python" "$LIBERO_SIM_PYTHON" "$BAIGE_PROBE_PYTHON"; do
    [[ -x "$path" ]] || { echo "Missing qualified Baige interpreter: $path" >&2; exit 1; }
done
for path in "$LIBERO_CONFIG_PATH/config.yaml" "$OPENPI_ROOT/examples/libero/main.py" "$glvnd/libEGL.so.1" "$glvnd/libGL.so.1" \
    "$LIBERO_STD_ROOT/libero/libero/benchmark/libero_suite_task_map.py" "$code/scripts/[train]gpu_probe.py"; do
    [[ -r "$path" ]] || { echo "Missing qualified runtime file: $path" >&2; exit 1; }
done
# The probe verifies scheduler identity and the one-device CUDA/NVML PCI mapping.
# Some Baige containers rewrite CUDA UUIDs. Use their local ordinal, preserve the scheduler mask.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$code/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH="$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders"
export LD_LIBRARY_PATH="$glvnd:$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_P2P_DISABLE=1
exec "$model_python" -Bu "$here/collect.py" "$@"
