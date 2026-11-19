#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
release="$root/sources/libero-pro2-e5500c69"
sim="$root/setups/libero-meta-learning-20261117-r1"
libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
pin=${1:?Supply the verified source revision}
attempt=${2:?Supply a fresh attempt identity}
[[ $(hostname) == job-* && "$attempt" =~ ^[a-z0-9]+$ ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
[[ $(git -C "$release" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C "$legacy/third_party/openpi-215abfb" rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
out="$root/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $start" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $start" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
echo ok > "$out/write-check"
rm "$out/write-check"
visible=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ "$visible" == GPU-* && "$visible" != *$'\n'* ]]
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    [[ "$visible" == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
    export CUDA_VISIBLE_DEVICES=0
fi
case $(nvidia-smi --query-gpu=name --format=csv,noheader) in
    *6000D*|*RPBZZZ6*) export NCCL_P2P_DISABLE=1 ;;
    *4090*) ;;
    *) echo 'Unexpected accelerator model' >&2; exit 1 ;;
esac
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export PYTHONPATH="$code/src:$release:$sim/simulator-source"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
glvnd="$libs/prefix/usr/lib/x86_64-linux-gnu"
test -r "$glvnd/libEGL.so.1" && test -r "$glvnd/libX11.so.6"
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
driver_paths=""
for path in "${paths[@]}"; do
    if [[ -n "$path" && ! -f "$path/libtorch_python.so" ]]; then driver_paths+=":$path"; fi
done
export LD_LIBRARY_PATH="$glvnd:$libs/prefix/lib/x86_64-linux-gnu$driver_paths"
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json" > "$out/probe.log" 2>&1
"$sim/env/bin/python" -B - <<'PY' > "$out/egl.log"
from mujoco.egl import GLContext
from OpenGL import GL
context = GLContext(16, 16)
try:
    context.make_current()
    vendor, renderer = (GL.glGetString(k) for k in (GL.GL_VENDOR, GL.GL_RENDERER))
    print(vendor, renderer)
    assert vendor and b'nvidia' in vendor.lower() and renderer
finally:
    context.free()
PY
timeout --kill-after=20s 85m "$legacy/envs/openpi-215abfb/bin/python" -Bu "$here/gate.py" --output "$out/native" > "$out/run.log" 2>&1
test -s "$out/native/pilot/complete.json"
"$legacy/envs/openpi-215abfb/bin/python" -B "$here/score.py" "$out/native/pilot" > "$out/score.json"
