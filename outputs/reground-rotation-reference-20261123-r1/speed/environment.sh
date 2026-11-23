# Sourced by the launcher and CPU qualification. Fixed to the registered Baige runtime.
export REGROUND_RUNTIME_ROOT=/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1
root=$REGROUND_RUNTIME_ROOT
run=outputs/reground-rotation-reference-20261123-r1
reference_pin=23a6641562a4d833837f27ab0535a20ad94e8ec8
reference_tree="$root/src-$reference_pin"
export REGROUND_REFERENCE_DIR="$reference_tree/$run"
unset REGROUND_UPSTREAM
source "$root/runtime/activate.sh"
py="$root/env/bin/python"
export PYTHONPATH="$REGROUND_REFERENCE_DIR:$PYTHONPATH:$reference_tree/src"
export REGROUND_NEXEC=10 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
libs=/mowenjie/fluffy-bassoon-2/setups/libero90-full-action-20261119-r1/desktop-libs02/prefix
export LD_LIBRARY_PATH="$libs/usr/lib/x86_64-linux-gnu:$libs/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
