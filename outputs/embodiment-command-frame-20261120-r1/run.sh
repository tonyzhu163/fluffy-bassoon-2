#!/usr/bin/env bash
set -euo pipefail
pin=${1:?expected full verified revision}
attempt=${2:?expected fresh attempt id}
[[ $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[a-z][a-z0-9]+$ && $(uname -s) == Linux ]]
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
shared=/mowenjie/fluffy-bassoon-2
out="$shared/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
trap 'rc=$?; printf "%s %s\n" "$rc" "$(date -u +%FT%TZ)" > "$out/exit.txt"' EXIT
setup="$shared/setups/libero-meta-learning-20261117-r1"
export CROSSARM_RELEASE="$shared/sources/libero-pro2-e5500c69"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
export LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
export CROSSARM_SIM_PYTHON="$setup/env/bin/python"
fit_python="$shared/setups/robocasa365-native-20261101-r1/env/bin/python"
test -x "$CROSSARM_SIM_PYTHON" && test -x "$fit_python"
{
    hostname
    date -u +%FT%TZ
    echo "$pin"
    df -h "$out"
    df -i "$out"
    if command -v quota >/dev/null; then quota -s || true; fi
    if [[ -r /sys/fs/cgroup/cpu.max ]]; then cat /sys/fs/cgroup/cpu.max; fi
    if [[ -r /sys/fs/cgroup/memory.max ]]; then cat /sys/fs/cgroup/memory.max; fi
    if command -v nvidia-smi >/dev/null; then nvidia-smi -L || true; fi
} | tee "$out/allocation.txt"
"$fit_python" -Bu -c 'import torch; print("torch", torch.__version__, "gpus", torch.cuda.device_count()); assert torch.cuda.device_count() == 0'
export PYTHONPATH="$code/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1
export MUJOCO_EGL_DEVICE_ID=0 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
mkdir "$out/cache" "$out/tmp"
export XDG_CACHE_HOME="$out/cache" TMPDIR="$out/tmp" NUMBA_CACHE_DIR="$out/cache/numba"
"$CROSSARM_SIM_PYTHON" -Bu "$here/collect.py" --out "$out/collection"
"$fit_python" -Bu "$here/fit.py" --tapes "$out/collection/tapes.jsonl" --out "$out/fit"
