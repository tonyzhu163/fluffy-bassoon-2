#!/usr/bin/env bash
set -euo pipefail
pin=${1:?deployed SHA}
panel=${2:?archive panel}
attempt=${3:?fresh attempt ID}
[[ "$panel" =~ ^(gpu09iiwa|gpu06kinova3|gpu08pi0fix)$ && "$attempt" =~ ^[[:alnum:]_-]+$ ]]
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
root=/mowenjie/fluffy-bassoon-2
out="$root/outputs/$(basename "$here")/$attempt/$panel"
[[ $(git -C "$code" rev-parse HEAD) == "$pin" ]]
for file in "$here"/{qualify.py,fk.py,run.sh} "$here/../symmeta-camera-command-20261123-r1/world_control.py"; do
    [[ $(git -C "$code" hash-object "$file") == "$(git -C "$code" rev-parse "$pin:$(realpath --relative-to="$code" "$file")")" ]]
done
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec > "$out/run.log" 2>&1
start=$(date +%s)
trap 'echo "$? $start $(date +%s)" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $panel" > "$out/launch.txt"
df -h "$out"
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" NUMBA_CACHE_DIR="$out/cache/numba" MPLCONFIGDIR="$out/cache/matplotlib"
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
sim="$root/setups/libero-meta-learning-20261117-r1"
export CROSSARM_RELEASE="$root/sources/libero-pro2-e5500c69"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
overlay="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay"
reader="$out/reader.json"
if [[ $# == 4 ]]; then
    [[ "$4" =~ ^[[:alnum:]_-]+$ ]]
    reader="$root/outputs/$(basename "$here")/$4/$panel/reader.json"
    sha256sum "$reader"
else
    PYTHONPATH="$overlay" /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -Bu "$here/qualify.py" "$panel" "$reader"
fi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
PYTHONPATH="$CROSSARM_RELEASE:$LIBERO_PRO_ROOT" "$sim/env/bin/python" -Bu "$here/fk.py" "$panel" "$reader" "$out/cases.json"
