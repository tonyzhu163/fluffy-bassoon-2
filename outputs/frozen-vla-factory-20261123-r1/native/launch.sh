#!/usr/bin/env bash
# One fresh one-card Baige Distributed Training allocation. Set the platform maximum to two hours.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
setup=$root/setups/xemb-icml-campaign-20261121-r1
deps=$setup/source-ff2ddf0a
sim=$root/setups/libero-meta-learning-20261117-r1
libs=$root/setups/libero90-full-action-20261119-r1/desktop-libs02
python=$legacy/envs/openpi-215abfb/bin/python
probe_python=$root/setups/robocasa365-native-20261101-r1/env/bin/python
release=$root/sources/libero-pro2-e5500c69
motor=$legacy/inputs/pi05-libero-openpi-20261027-r1

qualify() {
    local missing=0 path
    if [[ $(hostname) != job-*-master-0 || -z ${KUBERNETES_SERVICE_HOST:-} ]]; then
        echo "Requires a Baige Distributed Training master-0 container" >&2
        missing=1
    fi
    for path in "$deps" "$release" "$motor/READY.json" "$python" "$sim/env/bin/python" \
        "$probe_python" "$libs/nvidia-vendor.json" "$sim/preparation.json" \
        "$root/outputs/symmeta-camera-command-20261123-r1/orbitjob01/replays"; do
        if [[ ! -e $path ]]; then echo "Missing: $path" >&2; missing=1; fi
    done
    return "$missing"
}

check_tree() {
    [[ $(git -C "$1" rev-parse HEAD) == "$2" && -z $(git -C "$1" status --porcelain) ]] ||
        { echo "Wrong or dirty deployment: $1; expected $2" >&2; return 1; }
}

if [[ ${1:-} == qualify && $# == 1 ]]; then qualify; exit; fi
[[ $# == 3 && $1 == run && $2 =~ ^[a-f0-9]{40}$ && $3 =~ ^[a-zA-Z0-9_-]+$ ]] ||
    { echo "Usage: launch.sh qualify | run DEPLOYMENT_SHA NEW_ATTEMPT" >&2; exit 2; }
qualify
check_tree "$code" "$2"
check_tree "$deps" ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c
check_tree "$release" e5500c690c1498eca1b7050350654964190cdf70
check_tree "$legacy/third_party/openpi-215abfb" 215abfb217dbac7d5f1273282331b9b1866c0479
out="$root/outputs/frozen-vla-factory-20261123-r1/native/$3"
mkdir -p "$(dirname "$out")"
mkdir "$out"
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
printf '%s\n' "host=$(hostname)" "deployment=$2" "wall_seconds=7200" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
export ORBIT_DEPLOYMENT=$2 PAIRED_DEPS=$deps CROSSARM_RELEASE=$release
export OPENPI_DATA_HOME=$legacy/cache/openpi-215abfb LIBERO_PRO_ROOT=$sim/simulator-source
export LIBERO_CONFIG_PATH=$sim/libero-config MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=$libs/nvidia-vendor.json
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false NCCL_P2P_DISABLE=1
export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige
unset LIBGL_ALWAYS_SOFTWARE
if [[ -z ${CUDA_VISIBLE_DEVICES+x} ]]; then export CUDA_VISIBLE_DEVICES=0; fi
IFS=: read -ra current <<< "${LD_LIBRARY_PATH:-}"
drivers=""
for path in "${current[@]}"; do
    if [[ -n $path && ! -e "$path/libtorch_python.so" ]]; then drivers+=":$path"; fi
done
export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$drivers"
PYTHONPATH="$code/src" timeout 300 "$probe_python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose evaluation --baige-container --output "$out/gpu.json" > "$out/probe.log" 2>&1
export PYTHONPATH="$setup/cv2-headless-overlay:$deps/src:$release:$LIBERO_PRO_ROOT:$deps/outputs/libero-crossarm-20261120-r1:$deps/outputs/libero-xemb-ur5e-sysid-20261120-r1:$deps/outputs/libero-xemb-ur5e-scale-20261120-r1:$deps/outputs/xemb-icml-campaign-20261121-r1"
remaining=$((start + 7190 - $(date +%s)))
((remaining > 0))
timeout --kill-after=5s "${remaining}s" "$python" -Bu "$here/rollout.py" "$out" > "$out/run.log" 2>&1
