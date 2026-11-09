#!/usr/bin/env bash
# Matched frozen Drawer pilot, at most 15 minutes per arm. One allocated Pro6K.
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
# Raw output stays on local scratch; stock RoboCasa briefly writes XML beside its meshes.
# Keep its input-volume floor and check it again for every bank.
export GIT_OPTIONAL_LOCKS=0
export STEERING_MIN_FREE_GIB=220 STEERING_WAVE_ALLOWANCE_SECONDS=300
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
if [[ ! -v CUDA_VISIBLE_DEVICES ]]; then
    # Use the original runtime's sole container ordinal only after NVML proves it
    # is the scheduler's single assigned UUID. The independent probe verifies PCI and CUDA UUID.
    visible_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    test "$visible_uuid" = "${NVIDIA_VISIBLE_DEVICES:?scheduler must supply the allocated UUID}"
    [[ "$visible_uuid" == GPU-* && "$visible_uuid" != *$'\n'* && "$visible_uuid" != *,* ]]
    export CUDA_VISIBLE_DEVICES=0
fi
"$PY" -B "$SCI/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$ARCH/GPU-PROBE.json"
cp "$ARCH/GPU-PROBE.json" "$TMP/"
"$PY" -c 'import torch; d=torch.cuda.get_device_name(0); print(d); assert ("6000" in d or d=="NVIDIA RPBZZZ6") and torch.cuda.device_count()==1'
rc=0
# Closed readout is an input: patch only an exact copy on job-local scratch.
cp "$HERE/$OLD/READOUT.py" "$HERE/$OLD/FITTED-CHECK.py" "$TMP/"
(cd "$TMP"; echo "612f4bfb185d84d3e974742b949914e64f6e95804d96e5c11ec5a8bbca587507  READOUT.py" | sha256sum -c -;
 patch --batch --fuzz=0 READOUT.py < "$HERE/outputs/$R/drawer-pilot.patch")
# DPT full is already complete; do not repeat it. No success outcomes selected the cases.
for arm in dit-full dpt-reset dit-reset; do
    m=${arm%-*}; mode=${arm#*-}
    out=$TMP/$arm
    mkdir "$out"
    case "$m" in
        dpt) sha=7a0c4f6c3fbb92f54c3b6f0d9a36828e8d16fce2e5970b73a23ce77e8286f217 ;;
        dit) sha=7ccaae7048df38a90ced30530002469f1f73e1ee9cb37a17b07c8137cabb3721 ;;
    esac
    fit=$ROOT/$OLD/fits/kv_bridge/$m
    echo "$sha  $fit/candidate.pt" | sha256sum -c -
    date -Is > "$ARCH/$arm-start.txt"
    start=$(date +%s)
    # Reserve time for export. Persist finished banks even if a later bank times out.
    timeout -k 10 745 "$PY" -B "$TMP/READOUT.py" --code "$SCI" --commit "$PIN" --fit "$fit" --output "$out" --probe "$TMP/GPU-PROBE.json" --interface kv_bridge --objective "$m" --split qualification --qualification-phase "drawer-$mode" --deadline "$((start+730))" --attempt "pilot-$arm" >"$out/WORKER.log" 2>&1 &
    worker=$!
    while kill -0 "$worker" 2>/dev/null; do
        tail -c 3500 "$out/WORKER.log" >"$ARCH/$arm-current.log"
        # Journal rows distinguish initialization from issued and completed native actions.
        find "$out" -path '*/rank-00/six-episode-*.json' -exec cp '{}' "$ARCH/$arm-current-journal.json" \;
        if [[ -f "$out/PROGRESS.json" ]]; then cp "$out/PROGRESS.json" "$ARCH/$arm-progress.json"; fi
        sleep 12
    done
    status=0; wait "$worker" || status=$?
    tail -n 12 "$out/WORKER.log"
    for f in PROGRESS.json OPERATIONAL-CHECKPOINT.json RESULT.json WORKER.log; do
        if [[ -f "$out/$f" ]]; then cp "$out/$f" "$ARCH/$arm-$f"; fi
    done
    # Exclude the already preserved immutable candidate; retain raw attempts and traces.
    tar --exclude=selected-candidate.pt -C "$out" -cf "$TMP/$arm.tar" .
    timeout 110 cp "$TMP/$arm.tar" "$ARCH/$arm.tar"
    local_sum=$(sha256sum "$TMP/$arm.tar" | cut -d' ' -f1)
    echo "$local_sum  $ARCH/$arm.tar" | timeout 110 sha256sum -c -
    echo "$local_sum  $arm.tar" > "$ARCH/$arm.sha256"
    echo "worker_status=$status finished=$(date -Is)" > "$ARCH/$arm-exit.txt"
    if [[ "$status" != 0 || ! -f "$out/RESULT.json" ]]; then rc=1; fi
done
exit "$rc"
