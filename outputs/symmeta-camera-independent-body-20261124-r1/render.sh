#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
pin=${1:?parent deployed git SHA}
attempt=${2:?fresh attempt}
[[ $(uname -s) == Linux && $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[a-z0-9_-]+$ ]]
[[ $(git -C "$here" rev-parse HEAD) == "$pin" && -z $(git --no-optional-locks -C "$here" status --porcelain --untracked-files=no) ]]
root=/mowenjie/fluffy-bassoon-2
run="$root/outputs/$(basename "$here")"
out="$run/$attempt"
mkdir -p "$run"
exec 9>"$run/qualification.lock"
flock -n 9
mkdir "$out"
touch "$out/write-test"
df -h "$out" > "$out/storage.txt"
start=$(date +%s)
used=$(awk '{s+=$1} END {print int(s)}' "$run"/*/wall.seconds 2>/dev/null || true)
remaining=$((1800 - ${used:-0}))
((remaining > 0))
trap 'rc=$?; end=$(date +%s); echo $((end-start)) > "$out/wall.seconds"; echo "$rc $start $end" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $start $remaining" > "$out/launch.txt"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void LIBGL_ALWAYS_SOFTWARE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" MPLCONFIGDIR="$out/cache/mpl" NUMBA_CACHE_DIR="$out/cache/numba"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
sim="$root/setups/libero-meta-learning-20261117-r1"
export CROSSARM_RELEASE="$root/sources/libero-pro2-e5500c69"
export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export PYTHONPATH="$CROSSARM_RELEASE:$LIBERO_PRO_ROOT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
# Check CPU activity immediately before consuming either reserved core.
taskset -c 36,37 python3 - <<'PY' > "$out/idle.json"
import json,time
def counts():
    with open('/proc/stat') as file:
        return {s[0]:[int(x) for x in s[1:9]] for line in file if (s:=line.split())[0] in ('cpu36','cpu37')}
a=counts(); time.sleep(2); b=counts()
delta={k:[j-i for i,j in zip(a[k],b[k])] for k in a}
print(json.dumps(delta))
# Allow sporadic kernel ticks; reject a busy core.
assert set(delta)=={'cpu36','cpu37'} and all(x[3]>.9*sum(x) for x in delta.values()),delta
PY
timeout -k 2s "$((remaining-4))s" taskset -c 36,37 "$sim/env/bin/python" -Bu "$here/bank.py" "$out" > "$out/render.log" 2>&1
