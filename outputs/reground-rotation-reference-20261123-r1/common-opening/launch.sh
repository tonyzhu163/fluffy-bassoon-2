#!/usr/bin/env bash
# One already allocated Baige RTX 4090. Never submit or alter a scheduler job.
set -euo pipefail
[[ $# == 6 ]] || { echo 'usage: launch.sh CODE_PIN FRESH_ID WALL_SECONDS PHASE SEED SHARD' >&2; exit 2; }
pin=$1 id=$2 wall=$3 phase=$4 seed=$5 shard=$6
[[ $pin =~ ^[0-9a-f]{40}$ && $id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,64}$ ]]
[[ $wall =~ ^[1-9][0-9]{3}$ && $wall -ge 1200 && $wall -le 2700 ]]
[[ $seed =~ ^[0-4]$ && $shard =~ ^([0-9]|1[0-7])$ && $phase =~ ^(part-a|capture|base|symmeta|reground)$ ]]
[[ $(hostname) == job-*-master-0 && -n ${KUBERNETES_SERVICE_HOST:-} && ${WORLD_SIZE:-1} == 1 ]]
[[ ${NVIDIA_VISIBLE_DEVICES:-} =~ ^GPU-[0-9a-fA-F-]+$ ]]
export REFERENCE_STARTED=$(date +%s) REFERENCE_CODE_PIN=$pin REFERENCE_PHASE=common-opening-$phase
export SYMMETA_CODE_REV=$pin GIT_OPTIONAL_LOCKS=0
here=$(cd "$(dirname "$0")" && pwd -P)
run=outputs/reground-rotation-reference-20261123-r1
root=/mowenjie/fluffy-bassoon-2/$run
attempt=$root/common-opening/attempts/$id
ledger=$root/reference-native/attempts/common-opening-$id
mkdir -p "$root/common-opening/attempts"
mkdir "$attempt"
mkdir "$ledger"
finish() {
    local rc=$? end used
    end=$(date +%s)
    used=$((end - REFERENCE_STARTED))
    printf '{"exit_code":%d,"end_utc":%d,"allocated_gpu_seconds":%d}\n' "$rc" "$end" "$used" > "$ledger/exit.tmp"
    mv "$ledger/exit.tmp" "$ledger/exit.json"
    if [[ ! -f $ledger/reservation.json ]]; then
        printf '{"admitted":false,"max_gpu_seconds":%d}\n' "$used" > "$ledger/charge.tmp"
        mv "$ledger/charge.tmp" "$ledger/reservation.json"
    fi
}
trap finish EXIT
payload() {
    set -euo pipefail
    local code
    code=$(git -C "$here" rev-parse --show-toplevel)
    [[ $here == "$code/$run/common-opening" && $(git -C "$code" rev-parse HEAD) == "$pin" ]]
    if git -C "$code" symbolic-ref -q HEAD; then return 1; fi
    git -C "$code" ls-files --error-unmatch "$run/common-opening/"{runner.py,opening.py,launch.sh,env.sh} >/dev/null
    [[ -z $(git -C "$code" status --porcelain --untracked-files=all -- "$run/common-opening") ]]
    git -C "$code" diff --quiet HEAD -- "$run/speed/runtime.sha256"
    (cd "$root" && sha256sum --strict --quiet -c "$here/../speed/runtime.sha256")
    source "$here/env.sh"
    # Share the native accounting lock and the partition admission gate on the project filesystem.
    "$root/env/bin/python" -B - "$ledger" "$wall" "$seed" <<'PY'
import os, socket, sys
from pathlib import Path
from types import SimpleNamespace
import reference as ref
path, wall, seed = sys.argv[1:]
path = Path(path)
cap = float(os.environ.get("REFERENCE_CAP_GPUH", "40"))
ref.require(0 < cap <= 40, "Existing reference ceiling is at most 40 GPUh")
owner = path / "budget-owner"
owner.mkdir()
ref.write(owner / "owner.json", dict(attempt=path.name, host=socket.gethostname()))
gate = path.parent / ".partition-reserving"
owner.rename(gate)
try:
    ref.reserve(SimpleNamespace(attempt=path, gpus=1, wall=int(wall), seeds=[int(seed)], cap=cap))
finally:
    gate.rename(owner)
PY
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER MUJOCO_EGL_DEVICE_ID
    libs=/mowenjie/fluffy-bassoon-2/setups/libero90-full-action-20261119-r1/desktop-libs02
    export __EGL_VENDOR_LIBRARY_FILENAMES=$libs/nvidia-vendor.json
    export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu:$libs/prefix/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    "$root/env/bin/python" -Bu "$science/scripts/[train]gpu_probe.py" --baige-container --purpose training \
        --baige-exclusive-training --expected-count 1 --output "$attempt/gpu.json"
    export SYMMETA_ACCELERATOR='NVIDIA GeForce RTX 4090'
    exec "$root/env/bin/python" -Bu "$here/runner.py" "$phase" "$seed" "$shard" "$attempt"
}
export -f payload
export here run root ledger attempt wall seed shard phase pin
remaining=$((REFERENCE_STARTED + wall - 15 - $(date +%s)))
(( remaining > 0 ))
timeout --kill-after=10s "${remaining}s" bash -c payload > "$attempt/launch.log" 2>&1
