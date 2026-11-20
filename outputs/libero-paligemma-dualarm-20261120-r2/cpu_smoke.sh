#!/usr/bin/env bash
set -euo pipefail
root=/mowenjie/fluffy-bassoon-2
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
here="$code/outputs/libero-paligemma-dualarm-20261120-r2"
prev="$root/outputs/libero-paligemma-crossarm-20261120-r1"
setup="$root/setups/libero-meta-learning-20261117-r1"
[[ $(uname -s) == Linux && -z $(git -C "$code" status --porcelain) && $(git -C "$code" rev-parse HEAD) == "$1" ]]
attempt=$2
shift 2
[[ $attempt =~ ^[a-z0-9]+$ ]]
out="$root/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
trap 'echo "$? $(date +%s)" > "$out/exit.txt"' EXIT
git -C "$code" rev-parse HEAD > "$out/pin.txt"
export CROSSARM_RELEASE="$root/sources/libero-pro2-e5500c69" LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
export PYTHONPATH="$code/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$here:$code/outputs/libero-paligemma-crossarm-20261120-r1"
export CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CROSSARM_SIM_PYTHON="$setup/env/bin/python" DUAL_OUTPUT="$out" DUAL_ARCHIVE="$prev/assets01/goal5-base.hdf5"
cd "$out"
if [[ $# == 0 ]]; then
    "$CROSSARM_SIM_PYTHON" -Bu "$here/inspect_contact.py" "$prev/demoqual08/demos/failures/Sawyer-0-clearance.json" "$out/contacts.json"
else
    "$CROSSARM_SIM_PYTHON" -Bu "$here/$1" "${@:2}"
fi
