#!/usr/bin/env bash
set -euo pipefail
pin=${1:?deployed git SHA}
attempt=${2:?fresh attempt name}
[[ $attempt =~ ^[[:alnum:]_-]+$ ]]
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
data=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
out=$data/outputs/$(basename "$here")/$attempt
[[ $(hostname) == job-* && $(git -C "$code" rev-parse HEAD) == "$pin" ]]
[[ -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
for file in adapter.py panel.py native.py evidence/palette.json; do
    git -C "$here" ls-files --error-unmatch -- "$file" >/dev/null
done
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
start=$(date +%s)
trap 'echo "$? $start $(date +%s)" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $start" > "$out/launch.txt"
df -h "$out"
setup=$data/setups/libero-meta-learning-20261117-r1
export LIBERO_PRO_ROOT=$setup/simulator-source LIBERO_CONFIG_PATH=$setup/libero-config
export CROSSARM_RELEASE=$data/sources/libero-pro2-e5500c69
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C "$legacy/third_party/openpi-215abfb" rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
test -d "$legacy/inputs/pi05-libero-openpi-20261027-r1/params"
export PYTHONPATH="$code/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$code/outputs/libero-crossarm-20261120-r1:$code/outputs/libero-xemb-ur5e-sysid-20261120-r1"
export OPENPI_DATA_HOME=$legacy/cache/openpi-215abfb
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $gpu != *$'\n'* ]]
case "$gpu" in
    *6000D*|*RPBZZZ6*) export NCCL_P2P_DISABLE=1 ;;
    *4090*) ;;
    *) echo "Unexpected GPU $gpu"; exit 2 ;;
esac
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ $uuid == GPU-* && $uuid == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
    export CUDA_VISIBLE_DEVICES=0
fi
libs=$data/setups/libero90-full-action-20261119-r1/desktop-libs02
glvnd=$libs/prefix/usr/lib/x86_64-linux-gnu
IFS=: read -ra dirs <<< "${LD_LIBRARY_PATH:-}"
clean=
for d in "${dirs[@]}"; do if [[ -n $d && ! -f $d/libtorch_python.so ]]; then clean+=":$d"; fi; done
export LD_LIBRARY_PATH="$glvnd:$libs/prefix/lib/x86_64-linux-gnu$clean"
export __EGL_VENDOR_LIBRARY_FILENAMES=$libs/nvidia-vendor.json
export FLUFFY_GPU_EXECUTION=baige
"$data/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json"
timeout --kill-after=15s 105m "$legacy/envs/openpi-215abfb/bin/python" -Bu "$here/native.py" "$out" "$here/evidence/palette.json"
