#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
pin=${1:?full deployed pin}
attempt=${2:?fresh attempt name}
root=/mowenjie/fluffy-bassoon-2
run=$root/outputs/$(basename "$here")
[[ $pin =~ ^[0-9a-f]{40}$ && $(git -C "$here" rev-parse HEAD) == "$pin" && $attempt =~ ^[0-9a-z_-]+$ ]]
[[ $(uname -s) == Linux && ! -n $(git --no-optional-locks -C "$here" status --porcelain --untracked-files=no) ]]
mkdir -p "$run"
exec 9>"$run/prep.lock"
flock -n 9
out=$run/$attempt
mkdir "$out"
used=$(awk '{s+=$1} END {print int(s)}' "$run"/*/wall.seconds 2>/dev/null || true)
remaining=$((2100-${used:-0}))
((remaining > 5))
start=$(date +%s)
trap 'rc=$?; end=$(date +%s); echo $((end-start)) > "$out/wall.seconds"; echo "$rc $start $end" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $start $remaining" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
sim=$root/setups/libero-meta-learning-20261117-r1
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export LIBERO_PRO_ROOT="$sim/simulator-source" CROSSARM_RELEASE="$root/sources/libero-pro2-e5500c69" LIBERO_CONFIG_PATH="$sim/libero-config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export PYTHONPATH="$CROSSARM_RELEASE:$LIBERO_PRO_ROOT" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" MPLCONFIGDIR="$out/cache/mpl" NUMBA_CACHE_DIR="$out/cache/numba"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
python3 - <<'PY' > "$out/idle.json"
import json,time
cores=[f'cpu{i}' for i in range(42,46)]
def sample():
    with open('/proc/stat') as f:
        return {s[0]:[int(x) for x in s[1:9]] for line in f if (s:=line.split())[0] in cores}
a=sample(); time.sleep(2); b=sample()
d={k:[j-i for i,j in zip(a[k],b[k])] for k in a}
print(json.dumps(d)); assert set(d)==set(cores) and all(x[3]>.9*sum(x) for x in d.values()), d
PY
pids=()
for slot in 0 1 2 3; do
    timeout -k 2s "$((remaining-4))s" taskset -c "$((42+slot))" "$sim/env/bin/python" -Bu "$here/render_source.py" "$out" "$slot" > "$out/render-$slot.log" 2>&1 &
    pids+=($!)
done
rc=0
for child in "${pids[@]}"; do wait "$child" || rc=1; done
exit "$rc"
