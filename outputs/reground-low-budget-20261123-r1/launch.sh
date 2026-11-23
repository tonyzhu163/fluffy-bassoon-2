#!/usr/bin/env bash
# One exclusive Baige Distributed Training GPU. Parent owns submission and aggregate ledger.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
[[ $# -ge 6 ]] || { echo "usage: launch.sh CODE_SHA FRESH_ATTEMPT WALL_SECONDS fit|panel32|fit-panel32|full SEED BUDGET..." >&2; exit 2; }
pin=$1 id=$2 wall=$3 phase=$4 seed=$5
shift 5
[[ $pin =~ ^[0-9a-f]{40}$ && $id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ && $wall =~ ^[1-9][0-9]*$ ]]
[[ $phase =~ ^(fit|panel32|fit-panel32|full)$ && $seed =~ ^[0-4]$ ]]
for n in "$@"; do [[ $n =~ ^(15|100|1000)$ ]]; done
[[ $(hostname) == job-*-master-0 && -n ${KUBERNETES_SERVICE_HOST:-} && ${WORLD_SIZE:-1} == 1 ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" ]]
if git -C "$code" symbolic-ref -q HEAD >/dev/null; then echo "Detached job checkout required" >&2; exit 2; fi
git -C "$code" diff --quiet HEAD -- "${here#"$code"/}" outputs/reground-rotation-reference-20261123-r1 ':(literal)scripts/[train]gpu_probe.py' src/fluffy_bassoon/baige_gpu.py
export REGROUND_RUNTIME_ROOT=${REGROUND_RUNTIME_ROOT:-/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1}
root=${REGROUND_RUNTIME_ROOT%/*}/reground-low-budget-20261123-r1
export LOW_ATTEMPT="$root/attempts/$id"
mkdir -p "$root/attempts"
mkdir "$LOW_ATTEMPT"
start=$(date +%s)
finish() {
    local rc=$? elapsed
    elapsed=$(( $(date +%s) - start ))
    printf '{"exit_code":%d,"elapsed_gpu_seconds":%d,"gpus":1}\n' "$rc" "$elapsed" > "$LOW_ATTEMPT/exit.tmp"
    mv "$LOW_ATTEMPT/exit.tmp" "$LOW_ATTEMPT/exit.json"
}
trap finish EXIT
source "$REGROUND_RUNTIME_ROOT/runtime/activate.sh"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PYTHONPATH:$code/src" REGROUND_NEXEC=10
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NCCL_P2P_DISABLE=1
unset MUJOCO_EGL_DEVICE_ID LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER
export TMPDIR="$LOW_ATTEMPT/tmp" NUMBA_CACHE_DIR="$root/numba-cache" XDG_CACHE_HOME="$root/app-cache"
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME"
libs=/mowenjie/fluffy-bassoon-2/setups/libero90-full-action-20261119-r1/desktop-libs02
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu:$libs/prefix/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
# Match the reference probe on a container where the scheduler exposes exactly one whole GPU.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
py="$REGROUND_RUNTIME_ROOT/env/bin/python"
"$py" -B - "$LOW_ATTEMPT/attempt.json" "$pin" "$start" "$wall" "$phase" "$seed" "$@" <<'PY'
import json, socket, sys
from pathlib import Path
path,pin,start,wall,phase,seed,*budgets=sys.argv[1:]
Path(path).write_text(json.dumps(dict(code=pin,host=socket.gethostname(),start=int(start),
    wall_seconds=int(wall),max_gpu_seconds=int(wall),phase=phase,seed=int(seed),budgets=list(map(int,budgets))))+"\n")
PY
remaining() {
    seconds=$((start + wall - $(date +%s) - 5))
    (( seconds > 0 )) || exit 124
}
remaining
timeout --kill-after=5s "${seconds}s" "$py" -Bu "$code/scripts/[train]gpu_probe.py" --baige-container --purpose training \
    --baige-exclusive-training --expected-count 1 --output "$LOW_ATTEMPT/gpu.json" > "$LOW_ATTEMPT/probe.log" 2>&1
remaining
timeout --kill-after=5s "${seconds}s" "$py" -Bu "$here/runner.py" "$phase" "$seed" "$@" > "$LOW_ATTEMPT/run.log" 2>&1
