#!/usr/bin/env bash
set -euo pipefail
# Baige Distributed Training, one exclusive Pro6K / RTX 4090.
pin=$1
attempt=$2
runtime=$3
entry=$4
shift 4
[[ $attempt =~ ^[a-z0-9]+$ && $entry =~ ^[a-z][a-z0-9_]+\.py$ ]]
[[ $runtime == sim || $runtime == vlm ]]
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
release="$root/sources/libero-pro2-e5500c69"
setup="$root/setups/libero-meta-learning-20261117-r1"
probe_python="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
run="$root/outputs/$(basename "$here")"
host=$(hostname)

[[ $(uname -s) == Linux && $host == job-* ]] || {
    echo 'Use the assigned Baige Distributed Training container.' >&2
    exit 2
}
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
git -C "$here" ls-files --error-unmatch -- "$entry" >/dev/null
mkdir -p "$run"
out="$run/$attempt"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT
exec > >(tee -a "$out/run.log") 2>&1
trap 'printf "Launcher stopped at line %s (exit %s).\n" "$LINENO" "$?" >&2' ERR
printf '%s %s %s %s\n' "$pin" "$host" "$$" "$start" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
df -i "$out" >> "$out/storage.txt"
if command -v quota >/dev/null; then quota -s >> "$out/storage.txt" 2>&1 || true; fi
[[ $(git -C "$release" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
test -x "$setup/env/bin/python"
test -x "$probe_python"
weights=${WEIGHTS:-/mowenjie/.cache/huggingface/hub/models--google--paligemma-3b-mix-224/snapshots/d1d8734c9c3ad0ccfeea4afc270faa356c2ba515}
test -f "$weights/model.safetensors.index.json"

export LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false MOTOR_CODE_REVISION="$pin"
export CROSSARM_SIM_PYTHON="$setup/env/bin/python" CROSSARM_RELEASE="$release"
export PYTHONPATH="$code/src:$release:$setup/simulator-source:$here:$code/outputs/libero-paligemma-crossarm-20261120-r1"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export FLUFFY_GPU_EXECUTION=baige
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    visible=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ $visible == GPU-* && $visible == "${NVIDIA_VISIBLE_DEVICES:?}" && $visible != *$'\n'* ]]
    export CUDA_VISIBLE_DEVICES=0
fi
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $gpu != *$'\n'* ]]
case "$gpu" in
    *6000D*|*RPBZZZ6*) export NCCL_P2P_DISABLE=1 ;;
    *4090*) ;;
    *) echo "Unqualified accelerator: $gpu" >&2; exit 2 ;;
esac
libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
glvnd="$libs/prefix/usr/lib/x86_64-linux-gnu"
for lib in libEGL.so.1 libOpenGL.so.0 libX11.so.6; do test -r "$glvnd/$lib"; done
test -r "$libs/nvidia-vendor.json"
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
driver_paths=""
for path in "${paths[@]}"; do
    if [[ -n $path && ! -f $path/libtorch_python.so ]]; then driver_paths+=":$path"; fi
done
export LD_LIBRARY_PATH="$glvnd:$libs/prefix/lib/x86_64-linux-gnu$driver_paths"
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
"$probe_python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training \
    --baige-container --baige-exclusive-training --output "$out/gpu.json"
prev="$root/outputs/libero-paligemma-crossarm-20261120-r1"
export DUAL_OUTPUT="$out" DUAL_ARCHIVE="$prev/assets01/goal5-base.hdf5" DUAL_WEIGHTS="$weights"
python="$CROSSARM_SIM_PYTHON"
if [[ $runtime == vlm ]]; then python="$probe_python"; fi
cd "$out"
"$python" -Bu "$here/$entry" "$@"
