#!/usr/bin/env bash
# One lane or the summary in an already allocated Baige job; no job submission. Environment as symmeta_adapter.sh.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
project=/mowenjie/fluffy-bassoon-2
export REGROUND_RUNTIME_ROOT=${REGROUND_RUNTIME_ROOT:-$project/outputs/reground-rotation-reference-20261123-r1}
root=$REGROUND_RUNTIME_ROOT
source "$root/runtime/activate.sh"
campaign=xemb-icml-campaign-20261121-r1
export SYMMETA_SOURCE="$project/setups/$campaign/source-ff2ddf0a"
export SYMMETA_CHECKPOINT="$project/outputs/$campaign/gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack"
export SYMMETA_PYTHON=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
export SYMMETA_CV2="$project/setups/$campaign/cv2-headless-overlay"
# Current source first: its embodiment.py adds the wrist rotation and otherwise equals the archive's.
export PYTHONPATH="$PYTHONPATH:$code/src:$SYMMETA_SOURCE/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl REGROUND_NEXEC=10
export TOKENIZERS_PARALLELISM=false
export SYMMETA_CODE_REV=$(git -C "$code" rev-parse HEAD)
git -C "$SYMMETA_SOURCE" diff --quiet HEAD -- src outputs
[[ $(git -C "$SYMMETA_SOURCE" rev-parse HEAD) == ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c ]]
[[ $(git -C "$root/upstream" rev-parse HEAD) == 16a98486f560676b9036d1a6d067b9dd51a5917a ]]
if [[ ${1:-} == lane || ${1:-} == probe ]]; then
    [[ $(hostname) == job-*-master-0 && -n ${KUBERNETES_SERVICE_HOST:-} ]]
    : "${SYMMETA_EXPECTED_CODE:?set to the verified deployment revision}"
    [[ $SYMMETA_CODE_REV == "$SYMMETA_EXPECTED_CODE" ]]
    git -C "$code" diff --quiet HEAD -- "${here#"$code"/}" outputs/reground-rotation-reference-20261123-r1 src/fluffy_bassoon/xemb/embodiment.py \
        outputs/symmeta-original-pro-20261123-r1/eval_reader.py
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER
    libs="$project/setups/libero90-full-action-20261119-r1/desktop-libs02"
    if [[ -f "$libs/nvidia-vendor.json" ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
        export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu:$libs/prefix/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    fi
    export NCCL_P2P_DISABLE=1
    # The probe attests every device in the scheduler container once, before any lane loads CUDA.
    if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
        IFS=, read -ra devices <<< "${NVIDIA_VISIBLE_DEVICES:?missing scheduler UUIDs}"
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 "$((${#devices[@]} - 1))")
    fi
    if [[ $1 == probe ]]; then
        [[ $# == 3 && $3 =~ ^[1-9][0-9]*$ ]] || { echo "usage: probe ATTEMPT EXPECTED_GPUS" >&2; exit 2; }
        exec "$root/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" --baige-container \
            --purpose evaluation --expected-count "$3" --output "$2/gpu.json"
    fi
else
    export CUDA_VISIBLE_DEVICES="" LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MUJOCO_EGL_DEVICE_ID=0
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
fi
exec "$root/env/bin/python" -Bu "$here/arms.py" "$@"
