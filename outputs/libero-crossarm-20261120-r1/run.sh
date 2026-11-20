#!/usr/bin/env bash
# Run inside a single exclusive Baige Distributed Training container.
set -euo pipefail

if [[ $# != 3 || $2 != pilot || ! $3 =~ ^[a-z0-9]+$ ]]; then
    echo 'Usage: bash run.sh FULL_GIT_PIN pilot FRESH_ATTEMPT' >&2
    exit 2
fi
pin=$1
attempt=$3
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
release="$root/sources/libero-pro2-e5500c69"
setup="$root/setups/libero-meta-learning-20261117-r1"
motor="$legacy/envs/openpi-215abfb/bin/python"
probe_python="$root/setups/robocasa365-native-20261101-r1/env/bin/python"
run="$root/outputs/$(basename "$here")"
host=$(hostname)

[[ $(uname -s) == Linux && $host == job-* ]] || {
    echo 'Use the assigned Baige Distributed Training container.' >&2
    exit 2
}
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
for file in pilot.py run.sh runner_native.py arm_env.py sim_reset.py learner.py; do
    git -C "$here" ls-files --error-unmatch -- "$file" >/dev/null
done

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
[[ $(git -C "$legacy/third_party/openpi-215abfb" rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
test -x "$motor"
test -x "$setup/env/bin/python"
test -x "$probe_python"
test -d "$legacy/inputs/pi05-libero-openpi-20261027-r1/params"

export LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
export OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export CROSSARM_SIM_PYTHON="$setup/env/bin/python" CROSSARM_RELEASE="$release"
export PYTHONPATH="$code/src:$release:$setup/simulator-source:$here"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige XLA_PYTHON_CLIENT_PREALLOCATE=false
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
timeout --kill-after=20s 90m "$motor" -Bu "$here/pilot.py" --output "$out/pilot"
