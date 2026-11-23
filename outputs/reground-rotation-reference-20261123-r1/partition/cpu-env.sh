# Source in a private Linux CPU fixture, with the pinned run accessible read-only.
set -euo pipefail
: "${FIXTURE:?set FIXTURE to a private scratch directory}"
run=outputs/reground-rotation-reference-20261123-r1
export REGROUND_RUNTIME_ROOT=/mowenjie/fluffy-bassoon-2/$run
refcode=$REGROUND_RUNTIME_ROOT/src-23a6641562a4d833837f27ab0535a20ad94e8ec8
source "$REGROUND_RUNTIME_ROOT/runtime/activate.sh"
export REGROUND_REFERENCE_DIR=$refcode/$run
export PYTHONPATH="$REGROUND_REFERENCE_DIR:$PYTHONPATH:$refcode/src"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR="$FIXTURE/numba" PYTHONPYCACHEPREFIX="$FIXTURE/pycache" PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
libs=/mowenjie/fluffy-bassoon-2/setups/libero90-full-action-20261119-r1/desktop-libs02/prefix
export LD_LIBRARY_PATH="$libs/usr/lib/x86_64-linux-gnu:$libs/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
