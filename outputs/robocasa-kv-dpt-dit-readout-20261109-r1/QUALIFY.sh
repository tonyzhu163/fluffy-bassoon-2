#!/usr/bin/env bash
# One allocated Pro6K, sequential, 15 minutes per frozen candidate including export.
set -Eeuo pipefail
ROOT=/mowenjie/fluffy-bassoon-2
R=robocasa-kv-dpt-dit-readout-20261109-r1
OLD=outputs/robocasa-three-task-kv-velocity-meta-20261105-r1
PIN=87984718c9199797d31a158c47225cc695623afe
SCI=$ROOT/sources/steering-packed-87984718
HERE=$(cd "$(dirname "$0")/../.." && pwd)
test "$(git -C "$HERE" rev-parse HEAD)" = "${1:?runner commit}"
test "$(git -C "$SCI" rev-parse HEAD)" = "$PIN"
[[ "$HOSTNAME" == job-*-master-0 ]]
ARCH=/mnt/kv-readout/$HOSTNAME
mountpoint -q /mnt/kv-readout
test "$(cat /mnt/kv-readout/dev-write-readback-01.txt)" = reached
mkdir "$ARCH"
TMP=$(mktemp -d "/tmp/$R.XXXXXXXX")
test "$(stat -c %d "$TMP")" != "$(stat -c %d "$ROOT")"
test "$(df -P -k "$TMP" | awk 'END{print $4}')" -gt 230686720
echo write-readback >"$ARCH/volume-probe"
test "$(cat "$ARCH/volume-probe")" = write-readback
export STEERING_CODE=$SCI STEERING_COMMIT=$PIN RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1
export PYTHONPATH=$SCI/src:$RLINF_PATH OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
# The readout independently checks this input volume; all growing outputs go to separate scratch.
export STEERING_MIN_FREE_GIB=128 STEERING_WAVE_ALLOWANCE_SECONDS=300
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
if [[ ! -v CUDA_VISIBLE_DEVICES ]]; then
    # Baige exports the assigned UUID even when it omits the optional CUDA mask.
    export CUDA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:?scheduler must supply the allocated UUID}
fi
"$PY" -B "$SCI/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$ARCH/GPU-PROBE.json"
cp "$ARCH/GPU-PROBE.json" "$TMP/"
"$PY" -c 'import torch; d=torch.cuda.get_device_name(0); print(d); assert ("6000" in d or d=="NVIDIA RPBZZZ6") and torch.cuda.device_count()==1'
rc=0
for m in dpt dit; do
    out=$TMP/$m
    mkdir "$out"
    case "$m" in
        dpt) sha=7a0c4f6c3fbb92f54c3b6f0d9a36828e8d16fce2e5970b73a23ce77e8286f217 ;;
        dit) sha=7ccaae7048df38a90ced30530002469f1f73e1ee9cb37a17b07c8137cabb3721 ;;
    esac
    fit=$ROOT/$OLD/fits/kv_bridge/$m
    echo "$sha  $fit/candidate.pt" | sha256sum -c -
    date -Is > "$ARCH/$m-start.txt"
    start=$(date +%s)
    # Reserve time for export. Persist finished banks even if a later bank times out.
    timeout -k 10 745 "$PY" -B "$HERE/$OLD/READOUT.py" --code "$SCI" --commit "$PIN" --fit "$fit" --output "$out" --probe "$TMP/GPU-PROBE.json" --interface kv_bridge --objective "$m" --split qualification --qualification-phase deployment --deadline "$((start+730))" --attempt "qual01-$m" >"$out/WORKER.log" 2>&1 &
    worker=$!
    while kill -0 "$worker" 2>/dev/null; do
        tail -c 3500 "$out/WORKER.log" >"$ARCH/$m-current.log"
        if [[ -f "$out/PROGRESS.json" ]]; then cp "$out/PROGRESS.json" "$ARCH/$m-progress.json"; fi
        sleep 12
    done
    status=0; wait "$worker" || status=$?
    tail -n 12 "$out/WORKER.log"
    for f in PROGRESS.json OPERATIONAL-CHECKPOINT.json RESULT.json WORKER.log; do
        if [[ -f "$out/$f" ]]; then cp "$out/$f" "$ARCH/$m-$f"; fi
    done
    # Exclude the already preserved immutable candidate; retain raw attempts and traces.
    tar --exclude=selected-candidate.pt -C "$out" -cf "$TMP/$m.tar" .
    timeout 110 cp "$TMP/$m.tar" "$ARCH/$m.tar"
    local_sum=$(sha256sum "$TMP/$m.tar" | cut -d' ' -f1)
    echo "$local_sum  $ARCH/$m.tar" | timeout 110 sha256sum -c -
    echo "$local_sum  $m.tar" > "$ARCH/$m.sha256"
    echo "worker_status=$status finished=$(date -Is)" > "$ARCH/$m-exit.txt"
    if [[ "$status" != 0 || ! -f "$out/RESULT.json" ]]; then rc=1; fi
done
exit "$rc"
