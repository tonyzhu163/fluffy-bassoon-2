#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
release="$root/sources/libero-pro2-e5500c69"
sim="$root/setups/libero-meta-learning-20261117-r1"
run="$root/outputs/$(basename "$here")"
python="$legacy/envs/openpi-215abfb/bin/python"
sim_python="$sim/env/bin/python"

pin=$1
stage=$2
attempt=$3
host=$(hostname)
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
[[ $(git -C "$release" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C "$legacy/third_party/openpi-215abfb" rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]

out="$run/$attempt"
mkdir -p "$run"
mkdir "$out"
df -h "$out" > "$out/storage.txt"
printf 'write probe\n' > "$out/write-probe"
rm "$out/write-probe"
printf '%s %s %s %s\n' "$pin" "$host" "$$" "$stage" > "$out/launch.txt"
start=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT

export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export CROSSARM_RELEASE="$release"
export OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$code/src:$release:$sim/simulator-source:$code/outputs/libero-crossarm-20261120-r1:$here"

if [[ "$stage" == "test" || "$stage" == "gate0-cpu" ]]; then
    export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
    export LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
    if [[ "$stage" == "test" ]]; then
        export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
        test_site="$root/setups/robocasa365-native-20261101-r1/env/lib/python3.11/site-packages"
        timeout --kill-after=15s 10m "$python" -Bu -c '
import sys
sys.path.append(sys.argv[1])
import pytest
raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q", sys.argv[2]]))
' "$test_site" "$code/tests/test_xemb_ur5e.py" > "$out/run.log" 2>&1
    else
        export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" NUMBA_CACHE_DIR="$out/cache/numba"
        mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR"
        timeout --kill-after=20s 20m "$sim_python" -Bu "$here/gate0_cpu_ur5e.py" \
            --output "$out/gate0" > "$out/run.log" 2>&1
    fi
    exit 0
fi

if [[ "$host" == job-* ]]; then
    probe=(--baige-container --baige-exclusive-training)
    if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
        visible=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
        [[ "$visible" == GPU-* && "$visible" == "${NVIDIA_VISIBLE_DEVICES:?}" && "$visible" != *$'\n'* ]]
        export CUDA_VISIBLE_DEVICES=0
    fi
else
    [[ "$stage" == "gate1-smoke" ]] || {
        echo 'Use Baige Distributed Training for full Gate 1 evaluation.' >&2
        exit 2
    }
    [[ $CGPU_COUNT == 1 && $NVIDIA_VISIBLE_DEVICES == "$NVIDIA_VISIBLE_GPUS_UUID" && $NVIDIA_VISIBLE_DEVICES != *,* ]]
    probe=(--direct-hostname "$host")
    export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
fi

case $(nvidia-smi --query-gpu=name --format=csv,noheader) in
    *6000D*|*RPBZZZ6*) export NCCL_P2P_DISABLE=1 ;;
esac
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0

if [[ "$host" == job-* ]]; then
    libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
    glvnd="$libs/prefix/usr/lib/x86_64-linux-gnu"
    test -r "$glvnd/libEGL.so.1"
    test -r "$glvnd/libOpenGL.so.0"
    test -r "$glvnd/libX11.so.6"
    test -r "$libs/nvidia-vendor.json"
    IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
    driver_paths=""
    for path in "${paths[@]}"; do
        if [[ -n "$path" && ! -f "$path/libtorch_python.so" ]]; then driver_paths+=":$path"; fi
    done
    export LD_LIBRARY_PATH="$glvnd:$libs/prefix/lib/x86_64-linux-gnu$driver_paths"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
fi

date +%s > "$out/gpu-start.txt"
"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose training "${probe[@]}" --output "$out/gpu.json" > "$out/probe.log" 2>&1

"$sim/env/bin/python" -B - <<'PY' > "$out/egl.log" 2>&1
from mujoco.egl import GLContext
from OpenGL import GL

context = GLContext(16, 16)
try:
    context.make_current()
    vendor = GL.glGetString(GL.GL_VENDOR)
    renderer = GL.glGetString(GL.GL_RENDERER)
    print(vendor, renderer)
    if not vendor or b"nvidia" not in vendor.lower() or not renderer:
        raise RuntimeError("Require NVIDIA EGL rendering on the assigned device")
finally:
    context.free()
PY

case "$stage" in
    gate1-smoke)
        time_limit=15m
        command=("$here/gate1_ur5e.py" --output "$out/smoke" --panel smoke)
        ;;
    gate1)
        time_limit=60m
        command=("$here/gate1_ur5e.py" --output "$out/gate1" --panel development)
        ;;
    *)
        echo "Unsupported stage $stage" >&2
        exit 2
        ;;
esac

timeout --kill-after=20s "$time_limit" "$python" -Bu "${command[@]}" > "$out/run.log" 2>&1
date +%s > "$out/gpu-end.txt"
