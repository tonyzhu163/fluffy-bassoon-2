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
export OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$code/src:$release:$sim/simulator-source:$here"

if [[ "$stage" == "test" || "$stage" == "cpu-smoke" ]]; then
    export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
    export LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export MUJOCO_EGL_DEVICE_ID=1 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
    if [[ "$stage" == "test" ]]; then
        export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
        # Import the test harness after the qualified motor's own numerical packages.
        test_site="$root/setups/robocasa365-native-20261101-r1/env/lib/python3.11/site-packages"
        timeout --kill-after=15s 10m "$python" -Bu -c '
import sys
sys.path.append(sys.argv[1])
import pytest
raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q", sys.argv[2]]))
' "$test_site" "$code/tests/test_xemb_sysid.py" > "$out/run.log" 2>&1
    else
        timeout --kill-after=15s 15m "$python" -Bu "$here/collect.py" \
            --output "$out/archive" --panel smoke --modes active_probe --allow-mesa > "$out/collect.log" 2>&1
        timeout --kill-after=15s 15m "$python" -Bu "$here/train_readers.py" \
            --archive "$out/archive/archive.json" --output "$out/train" --steps 20 --batch-size 4 --seeds 41 > "$out/train.log" 2>&1
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
    [[ "$stage" == smoke ]] || {
        echo 'Use Baige Distributed Training for full Gate 1, collection, training, and live evaluation.' >&2
        exit 2
    }
    [[ $CGPU_COUNT == 1 && $NVIDIA_VISIBLE_DEVICES == "$NVIDIA_VISIBLE_GPUS_UUID" && $NVIDIA_VISIBLE_DEVICES != *,* ]]
    probe=(--direct-hostname "$host")
    export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
fi

if [[ "$host" != job-* ]]; then
    exec 9>"$root/outputs/libero-bowl-decisions-20261118-r1/device.lock"
    flock -w 3600 9
    idle=false
    for ((check=0; check<30; check++)); do
        if [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) &&
              $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) -le 5 ]]; then
            idle=true
            break
        fi
        sleep 2
    done
    [[ "$idle" == true ]]
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

if [[ "$stage" == smoke ]]; then
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
fi

case "$stage" in
    smoke)
        time_limit=20m
        command=("$here/gate1_check.py" --output "$out/smoke" --panel smoke)
        ;;
    gate1)
        time_limit=60m
        command=("$here/gate1_check.py" --output "$out/gate1" --panel development)
        ;;
    pilot-collect)
        time_limit=45m
        command=("$here/collect.py" --output "$out/archive" --panel poc --modes active_probe,passive_attempt1 --passive-max-controls 15)
        ;;
    pilot-train)
        time_limit=45m
        command=("$here/train_readers.py" --archive "$4" --output "$out/train" --steps 1500 --batch-size 32 --seeds 41)
        ;;
    pilot-eval)
        time_limit=60m
        command=("$here/eval_live.py" --archive "$4" --train-dir "$5" --output "$out/live" --panel poc --seed-ckpt 41)
        ;;
    collect)
        time_limit=90m
        command=("$here/collect.py" --output "$out/archive" --panel full --modes active_probe,passive_attempt1)
        ;;
    train)
        time_limit=30m
        command=("$here/train_readers.py" --archive "$4" --output "$out/train" --steps 1500 --batch-size 32 --seeds 41,42,43 --gate-yaw-deg 20.0)
        ;;
    eval-live)
        time_limit=240m
        command=("$here/eval_live.py" --archive "$4" --train-dir "$5" --output "$out/live" --panel development --seed-ckpt 41)
        ;;
    *)
        exit 2
        ;;
esac

timeout --kill-after=20s "$time_limit" "$python" -Bu "${command[@]}" > "$out/run.log" 2>&1
date +%s > "$out/gpu-end.txt"
