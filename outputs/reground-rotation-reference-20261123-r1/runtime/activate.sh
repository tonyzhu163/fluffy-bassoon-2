# Source with bash. The defaults point to the Baige shared run.
export REGROUND_RUNTIME_ROOT="${REGROUND_RUNTIME_ROOT:-/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1}"
source "$REGROUND_RUNTIME_ROOT/env/bin/activate"
export HF_HOME="$REGROUND_RUNTIME_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export REGROUND_PINNED_OFFLINE=1
export LIBERO_CONFIG_PATH="$REGROUND_RUNTIME_ROOT/libero-config"
export NUMBA_CACHE_DIR="$REGROUND_RUNTIME_ROOT/numba-cache"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REGROUND_RUNTIME_ROOT/runtime/offline:$REGROUND_RUNTIME_ROOT/upstream"
