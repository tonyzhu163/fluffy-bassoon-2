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
export OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$code/src:$release:$sim/simulator-source:$here"

if [[ "$stage" == "test" || "$stage" == "gate0-cpu" ]]; then
    export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
    export LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}" __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
    if [[ "$stage" == "test" ]]; then
        export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
        test_site="$root/setups/robocasa365-native-20261101-r1/env/lib/python3.11/site-packages"
        timeout --kill-after=15s 10m "$python" -Bu -c '
import sys
sys.path.append(sys.argv[1])
import pytest
raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q", sys.argv[2]]))
' "$test_site" "$code/tests/test_xemb_tcp.py" > "$out/run.log" 2>&1
    else
        export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" NUMBA_CACHE_DIR="$out/cache/numba"
        mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR"
        timeout --kill-after=20s 20m "$sim_python" -Bu "$here/gate0_cpu.py" \
            --output "$out/gate0" --workers 4 > "$out/run.log" 2>&1
    fi
    exit 0
fi

echo "Unsupported stage $stage (GPU stages will be added after Gate 0)" >&2
exit 2
