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

if [[ "$host" == job-* ]]; then
    probe=(--baige-container --baige-exclusive-training)
    if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
        visible=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
        [[ "$visible" == GPU-* && "$visible" == "${NVIDIA_VISIBLE_DEVICES:?}" && "$visible" != *$'\n'* ]]
        export CUDA_VISIBLE_DEVICES=0
    fi
else
    [[ "$stage" == smoke || "$stage" == smoke-eval || "$stage" == fit-history ]] || {
        echo 'Use Baige Distributed Training for full adapter/history GPU runs.' >&2
        exit 2
    }
    [[ $CGPU_COUNT == 1 && $NVIDIA_VISIBLE_DEVICES == "$NVIDIA_VISIBLE_GPUS_UUID" && $NVIDIA_VISIBLE_DEVICES != *,* ]]
    probe=(--direct-hostname "$host")
    export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
fi

out="$run/$attempt"
case "$stage" in
    smoke)
        time_limit=20m
        command=("$here/train_adapter.py" "$run/data/index-smoke.json" "$out/training"
            --steps 12 --batch 4 --validation-batches 2 --checkpoint-every 12)
        ;;
    smoke-eval)
        time_limit=20m
        command=("$here/evaluate.py" --stage adapter --checkpoint "$4" --output "$out/native" --panel smoke)
        ;;
    fit-adapter)
        time_limit=120m
        command=("$here/train_adapter.py" "$run/data/index.json" "$out/training")
        if [[ $# == 4 ]]; then command+=(--resume "$4"); fi
        ;;
    eval-adapter)
        time_limit=120m
        command=("$here/evaluate.py" --stage adapter --checkpoint "$4" --output "$out/native" --panel development)
        ;;
    collect-history)
        time_limit=120m
        command=("$here/evaluate.py" --stage collect-history --checkpoint "$4" --output "$out/archive" --panel development)
        ;;
    fit-history)
        time_limit=30m
        command=("$here/train_history.py" --archive "$4" --adapter-checkpoint "$5" --output "$out/history-fit")
        ;;
    eval-history)
        time_limit=120m
        command=("$here/evaluate.py" --stage history --checkpoint "$4" --history-checkpoint "$5" --archive "$6" --output "$out/history-eval")
        ;;
    *)
        exit 2
        ;;
esac

mkdir -p "$run"
mkdir "$out"
df -h "$out" > "$out/storage.txt"
printf 'write probe\n' > "$out/write-probe"
rm "$out/write-probe"
printf '%s %s %s %s\n' "$pin" "$(hostname)" "$$" "$stage" > "$out/launch.txt"
start=$(date +%s)
trap 'rc=$?; printf "%s %s %s\n" "$rc" "$(date +%s)" "$start" > "$out/exit.txt"' EXIT

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
date +%s > "$out/gpu-start.txt"

case $(nvidia-smi --query-gpu=name --format=csv,noheader) in
    *6000D*|*RPBZZZ6*) export NCCL_P2P_DISABLE=1 ;;
esac
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$code/src:$release:$sim/simulator-source"

if [[ "$host" == job-* ]]; then
    glvnd="$root/setups/libero-gen-history-pilot-20261111-r1/os-libs01/prefix/usr/lib/x86_64-linux-gnu"
    test -f "$glvnd/libEGL.so.1"
    test -f "$glvnd/libGL.so.1"
    export LD_LIBRARY_PATH="$glvnd${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

"$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose training "${probe[@]}" --output "$out/gpu.json" > "$out/probe.log" 2>&1

if [[ "$host" == job-* && "$stage" == smoke ]]; then
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

timeout --kill-after=20s "$time_limit" "$python" -Bu "${command[@]}" > "$out/run.log" 2>&1
date +%s > "$out/gpu-end.txt"
