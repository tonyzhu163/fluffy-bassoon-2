#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
data=/mowenjie/fluffy-bassoon-2
pin=${1:?expected git revision}
mode=${2:?score or world}
out=${3:?fresh attempt directory}
[[ "$(git -C "$code" rev-parse HEAD)" == "$pin" && -z "$(git -C "$code" status --porcelain)" ]]
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
cd "$code"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib"
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
echo "$pin $(hostname) $$ $mode $start" > "$out/launch.txt"
case "$mode" in
    score)
        export PYTHONPATH="$data/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay:$code/src"
        /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -Bu "$here/score.py" --data-root "$data" --output "$out/scores.json" > "$out/run.log" 2>&1
        ;;
    world)
        export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
        export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
        "$data/setups/libero-meta-learning-20261117-r1/env/bin/python" -Bu "$here/world_control.py" --output "$out/world.json" > "$out/run.log" 2>&1
        ;;
    *) exit 2 ;;
esac
