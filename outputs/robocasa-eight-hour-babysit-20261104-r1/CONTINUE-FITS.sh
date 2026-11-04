#!/usr/bin/env bash
set -Eeuo pipefail
: "${BABYSIT_CODE:?}" "${BABYSIT_COMMIT:?}" "${BABYSIT_METHODS:?}" "${BABYSIT_ATTEMPT:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-eight-hour-babysit-20261104-r1
LEARNER=$ROOT/sources/three-task-joint-d71cb532
PIN=d71cb532f6cf4933a4a5e236e1b085bce74ce969
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
read -ra METHODS <<< "$BABYSIT_METHODS"
GPUS=${#METHODS[@]}
(( GPUS >= 1 && GPUS <= 4 ))
OUT=$ROOT/$RUN/continuations/$BABYSIT_ATTEMPT
test ! -e "$OUT"
mkdir -p "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
START=$(date +%s)
finish() {
  rc=$?
  printf '{"started_unix":%s,"ended_unix":%s,"exit_code":%s,"hostname":"%s","gpus":%s}\n' "$START" "$(date +%s)" "$rc" "$HOSTNAME" "$GPUS" > "$OUT/CLOCK.json"
}
trap finish EXIT
test "$(git -C "$BABYSIT_CODE" rev-parse HEAD)" = "$BABYSIT_COMMIT"
test "$(git -C "$LEARNER" rev-parse HEAD)" = "$PIN"
for code in "$BABYSIT_CODE" "$LEARNER"; do test -z "$(git -C "$code" status --porcelain)"; done
export PYTHONPATH=$BABYSIT_CODE/src:$ROOT/sources/critic-selected-a1229239-r1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Keep the scheduler's NVIDIA assignment; select its local CUDA ordinals only.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS-1)))}
"$PY" -B "$BABYSIT_CODE/scripts/[train]gpu_probe.py" --expected-count "$GPUS" --purpose training \
  --baige-container --baige-exclusive-training --output "$OUT/GPU-PROBE.json"
export LYCO_BAIGE_GPU_MAP=$OUT/GPU-PROBE.json
export BABYSIT_ALLOCATION_GPUS=$GPUS
"$PY" -B - <<'PY'
import datetime,json,os,socket,time
from pathlib import Path
gpus=int(os.environ['BABYSIT_ALLOCATION_GPUS'])
assert socket.gethostname().startswith('job-')
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0])/int(cpu[1]) >= 16*gpus
assert int(Path('/sys/fs/cgroup/memory.max').read_text()) >= 64*gpus*2**30
shm=os.statvfs('/dev/shm');assert shm.f_blocks*shm.f_frsize >= 8*gpus*2**30
root=Path('/mowenjie/fluffy-bassoon-2');disk=os.statvfs(root)
assert disk.f_bavail*disk.f_frsize > 220*2**30
ledger=json.loads((root/'outputs/robocasa-eight-hour-babysit-20261104-r1/META-ALLOCATION-LEDGER.json').read_text())
reserved=ledger['preparation_GPU_hours']
job=socket.gethostname().removesuffix('-master-0')
included=False
for row in ledger['allocations']:
    began=datetime.datetime.fromisoformat(row['created_shanghai']).timestamp()
    end=datetime.datetime.fromisoformat(row['released_shanghai']).timestamp() if row.get('released_shanghai') else time.time()
    hours=row['GPUs']*(end-began)/3600
    reserved += hours if row.get('released_shanghai') else max(hours,row['max_reservation_GPU_hours'])
    included |= row['job_id']==job
assert reserved + (0 if included else 2*gpus) <= ledger['approved_GPU_hours']==96
PY
timeout --kill-after=30s 6900 "$PY" -B "$BABYSIT_CODE/$RUN/CONTINUE-FITS.py" \
  --allocation-probe "$OUT/GPU-PROBE.json" --learner-code "$LEARNER" \
  --wrapper-commit "$BABYSIT_COMMIT" --source-commit "$PIN" --methods "${METHODS[@]}" \
  --assembly "$ROOT/outputs/robocasa-three-task-meta-baseline-20261104-r2/joint/preparation/ASSEMBLY.json" \
  --assembly-sha256 fc3f388ef3feddde4c44ebd45ae9d7e73dab4a0fc58252c506315697be2a4758 \
  --output "$ROOT/outputs/robocasa-three-task-meta-baseline-20261104-r2/joint/fits" \
  --expected-hostname "$HOSTNAME" --attempt "$BABYSIT_ATTEMPT" --max-elapsed-seconds 6600
