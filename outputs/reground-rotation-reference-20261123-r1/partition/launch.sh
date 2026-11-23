#!/usr/bin/env bash
# In an already allocated Baige Distributed Training container.
set -euo pipefail
export REFERENCE_STARTED=$(date +%s) GIT_OPTIONAL_LOCKS=0 PYTHONDONTWRITEBYTECODE=1
[[ $# -ge 5 ]] || { echo 'usage: launch.sh TRANSPORT_SHA FRESH_ID CARDS WALL_SECONDS PHASE --namespace NAME --seeds ... [--shards FIRST STOP] [--descending]' >&2; exit 2; }
pin=$1 id=$2 gpus=$3 wall=$4 phase=$5
shift 5
[[ $pin =~ ^[a-f0-9]{40}$ && $id =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}$ && $gpus =~ ^[1-5]$ ]]
[[ $wall =~ ^[1-9][0-9]{1,5}$ && $wall -gt 20 && $phase =~ ^(babble|collect|fit)$ ]]
[[ $(hostname) == job-*-master-0 && -n ${KUBERNETES_SERVICE_HOST:-} && ${WORLD_SIZE:-1} == 1 ]]
: "${REFERENCE_CAP_GPUH:?supply the cumulative reference ceiling}"
here=$(cd "$(dirname "$0")" && pwd -P)
run=outputs/reground-rotation-reference-20261123-r1
root=/mowenjie/fluffy-bassoon-2/$run
export REFERENCE_PHASE=$phase REFERENCE_CODE_PIN=$pin
refpin=23a6641562a4d833837f27ab0535a20ad94e8ec8
refcode=$root/src-$refpin
attempt="$root/reference-native/attempts/$id"
mkdir "$attempt"  # atomically own this fresh ID before writing any receipts
finish() {
    local rc=$? end used
    end=$(date +%s)
    used=$(((end - REFERENCE_STARTED) * gpus))
    printf '{"exit_code":%s,"end_utc":%s,"allocated_gpu_seconds":%s}\n' "$rc" "$end" "$used" > "$attempt/exit.tmp"
    mv "$attempt/exit.tmp" "$attempt/exit.json"
    if [[ ! -f $attempt/reservation.json ]]; then
        # A rejected/failed preflight still consumed allocation time. The native
        # ledger scans reservation.json and charges exit.json when it is present.
        printf '{"admitted":false,"phase":"%s","code":"%s","host":"%s","gpus":%s,"wall_seconds":%s,"started_utc":%s,"max_gpu_seconds":%s}\n' \
            "$phase" "$pin" "$(hostname)" "$gpus" "$wall" "$REFERENCE_STARTED" "$used" > "$attempt/charge.tmp"
        mv "$attempt/charge.tmp" "$attempt/reservation.json"
    fi
}
trap finish EXIT
verify_tree() {
    local tree=$1 expected=$2
    shift 2
    [[ $(git -C "$tree" rev-parse HEAD) == "$expected" ]]
    ! git -C "$tree" symbolic-ref -q HEAD >/dev/null || return 1
    git -C "$tree" diff --quiet HEAD -- "$@"
}
payload() {
    set -euo pipefail
    local code py arg file
    code=$(git -C "$here" rev-parse --show-toplevel)
    [[ $here == "$code/$run/partition" ]]
    for arg in "$@"; do
        case $arg in
            --namespace|--seeds|--shards|--descending) ;;
            --*|-*) echo "Unexpected option $arg" >&2; exit 2 ;;
        esac
    done
    verify_tree "$code" "$pin" "$run/partition" "$run/speed/runtime.sha256"
    for file in launch.sh partition.py collect.py fit.py; do
        git -C "$code" ls-files --error-unmatch "$run/partition/$file" >/dev/null
    done
    [[ -z $(git -C "$code" ls-files --others -- ":(glob)$run/partition/*.py" ":(glob)$run/partition/*.pyc" ":(glob)$run/partition/*/__init__.py") ]]
    verify_tree "$refcode" "$refpin" "$run" src ':(literal)scripts/[train]gpu_probe.py'
    (cd "$root" && sha256sum --strict --check --quiet "$here/../speed/runtime.sha256")
    export REGROUND_RUNTIME_ROOT=$root REGROUND_UPSTREAM=$root/upstream REGROUND_REFERENCE_DIR=$refcode/$run
    source "$root/runtime/activate.sh"
    export NUMBA_CACHE_DIR=$attempt/numba-cache
    export PYTHONPATH="$REGROUND_REFERENCE_DIR:$PYTHONPATH:$refcode/src"
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export NCCL_P2P_DISABLE=1 REGROUND_NEXEC=10
    unset MUJOCO_EGL_DEVICE_ID LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER
    libs=/mowenjie/fluffy-bassoon-2/setups/libero90-full-action-20261119-r1/desktop-libs02
    export __EGL_VENDOR_LIBRARY_FILENAMES=$libs/nvidia-vendor.json
    export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu:$libs/prefix/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 "$((gpus - 1))")
    fi
    py=$root/env/bin/python
    args=(--attempt "$attempt" --phase "$REFERENCE_PHASE" --gpus "$gpus" --wall "$wall" --cap "$REFERENCE_CAP_GPUH" "$@")
    "$py" -Bu "$here/partition.py" reserve "${args[@]}"
    "$py" -Bu "$refcode/scripts/[train]gpu_probe.py" --baige-container --purpose training \
        --baige-exclusive-training --expected-count "$gpus" --output "$attempt/gpu.json"
    exec "$py" -Bu "$here/partition.py" dispatch "${args[@]}"
}
export here run root refpin refcode pin attempt gpus wall
export -f payload verify_tree
remaining=$((REFERENCE_STARTED + wall - 15 - $(date +%s)))
(( remaining > 0 ))
timeout --kill-after=10s "${remaining}s" bash -c 'payload "$@"' partition "$@" > "$attempt/launch.log" 2>&1
