#!/usr/bin/env bash
set -euo pipefail
pin=${1:?expected full verified Git revision}
attempt=${2:?expected fresh attempt name}
[[ $(uname -s) == Linux && $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[a-z][a-z0-9]+$ ]]
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" ]]
if git -C "$code" symbolic-ref -q HEAD; then exit 1; fi
git -C "$code" diff --quiet HEAD --
shared=/mowenjie/fluffy-bassoon-2
out="$shared/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
trap 'rc=$?; printf "%s %s\n" "$rc" "$(date -u +%FT%TZ)" > "$out/exit.txt"' EXIT
setup="$shared/setups/libero-meta-learning-20261117-r1"
mesa="$shared/setups/embodiment-command-frame-20261120-r1/mesa-runtime"
export CROSSARM_RELEASE="$shared/sources/libero-pro2-e5500c69"
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
export LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
export CROSSARM_SIM_PYTHON="$setup/env/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
mkdir "$out/tmp" "$out/cache"
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" NUMBA_CACHE_DIR="$out/cache/numba"
{
    hostname
    date -u +%FT%TZ
    echo "source $pin"
    df -hT "$out"
    df -i "$out"
    if command -v quota >/dev/null; then quota -s || true; else echo "quota command unavailable"; fi
    cat /sys/fs/cgroup/{cpu.max,memory.max}
    "$CROSSARM_SIM_PYTHON" -Bu -c 'import glob, numpy, sys; print("Python", sys.version, "NumPy", numpy.__version__); cards=glob.glob("/dev/nvidia[0-9]*"); print("NVIDIA cards", cards); assert not cards'
} | tee "$out/allocation.txt"
(cd "$here"; sha256sum run.sh collect.py score.py) > "$out/source.sha256"
(cd "$mesa"; echo '78686e3a44f2daf8a67b5150672ba828afbd4b33f6d4df87d6b4e11a1c10f4af  SHA256SUMS' | sha256sum -c -; sha256sum --quiet -c SHA256SUMS)
export LD_LIBRARY_PATH="$mesa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBGL_DRIVERS_PATH="$mesa/dri" __EGL_VENDOR_LIBRARY_FILENAMES="$mesa/50_mesa.json"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 LIBGL_ALWAYS_SOFTWARE=1
export PYTHONPATH="$code/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT"
"$CROSSARM_SIM_PYTHON" -Bu - <<'PY' | tee "$out/renderer.log"
import mujoco
from OpenGL import GL

context = mujoco.GLContext(16, 16)
try:
    context.make_current()
    renderer = GL.glGetString(GL.GL_RENDERER)
    print("GL_RENDERER", renderer)
    assert renderer and b"llvmpipe" in renderer
finally:
    context.free()
PY
echo "Collection started $(date -u +%FT%TZ)"
"$CROSSARM_SIM_PYTHON" -Bu "$here/collect.py" --out "$out/collection"
echo "Collection finished $(date -u +%FT%TZ)"
(cd "$out"; wc -lc collection/tapes.jsonl; sha256sum collection/tapes.jsonl > tapes.sha256)
"$CROSSARM_SIM_PYTHON" -Bu "$here/score.py" --self-test | tee "$out/self-test.log"
echo "Frozen scoring started $(date -u +%FT%TZ)"
"$CROSSARM_SIM_PYTHON" -Bu "$here/score.py" --tapes "$out/collection/tapes.jsonl" --out "$out/score"
echo "Frozen scoring finished $(date -u +%FT%TZ)"
