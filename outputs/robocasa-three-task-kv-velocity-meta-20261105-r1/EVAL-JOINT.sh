#!/usr/bin/env bash
set -Eeuo pipefail
BEGAN=$(date +%s)
INTERFACE=${1:?interface} OBJECTIVE=${2:?objective} ATTEMPT=${3:?fresh-attempt}
case "$INTERFACE" in kv_bridge|velocity_bridge) ;; *) exit 2 ;; esac
case "$OBJECTIVE" in ad|dpt|dit|ic_iql) ;; *) exit 2 ;; esac
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COMMIT=$(git -C "$CODE" rev-parse HEAD)
test -z "$(git -C "$CODE" status --porcelain)"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-three-task-kv-velocity-meta-20261105-r1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
export STEERING_INTERFACE=$INTERFACE STEERING_OBJECTIVE=$OBJECTIVE STEERING_CODE=$CODE STEERING_COMMIT=$COMMIT
export STEERING_MIN_FREE_GIB=411 STEERING_WAVE_ALLOWANCE_SECONDS=1200
export RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1 PYTHONPATH=$CODE/src:$ROOT/sources/critic-selected-a1229239-r1
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$ROOT/$RUN/logs" "$ROOT/$RUN/probes"
exec > >(tee "$ROOT/$RUN/logs/$ATTEMPT.log") 2>&1
"$PY" - <<'PY'
import datetime,hashlib,json,os,shutil,time
from pathlib import Path
root=Path('/mowenjie/fluffy-bassoon-2');run=root/'outputs/robocasa-three-task-kv-velocity-meta-20261105-r1'
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
# This file must be written only after the direct human budget decision and
# shared-storage admission. Its absence keeps the scientific benchmark held.
admission=json.loads((run/'READOUT-ADMISSION.json').read_text())
assert admission['approved'] and admission['human_budget_authorization']
assert admission['implementation_commit']==os.environ['STEERING_COMMIT']
ledger=json.loads((root/'outputs/robocasa-eight-hour-babysit-20261104-r1/META-ALLOCATION-LEDGER.json').read_text())
assert admission['approved_GPU_hours']==ledger['approved_GPU_hours'] and ledger['approved_GPU_hours']<=190
assert admission['scientific_contract_sha256']==sha(run/'CONTRACT.json')
q=run/'migrations/pfs-members-01/FITTED-QUALIFICATION-SUMMARY04.json'
assert admission['fitted_qualification_sha256']==sha(q) and json.loads(q.read_text())['passed']
p=run/'migrations/pfs-members-01/PACKED-ARCHIVE-QUALIFICATION05.json'
assert admission['packed_storage_qualification_sha256']==sha(p) and json.loads(p.read_text())['passed']
key=os.environ['STEERING_INTERFACE']+'/'+os.environ['STEERING_OBJECTIVE'];fit=run/'fits'/key
status=json.loads((fit/'training-status.json').read_text())
assert status['converged'] and status['stop_reason']=='validation_plateau'
assert admission['selected_candidates'][key]['sha256']==sha(fit/'candidate.pt')
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));reserved=ledger['preparation_GPU_hours'];own=False
for row in ledger['allocations']:
    end=datetime.datetime.fromisoformat(row['released_shanghai']) if row.get('released_shanghai') else now
    actual=row['GPUs']*(end-datetime.datetime.fromisoformat(row['created_shanghai'])).total_seconds()/3600
    reserved+=actual if row.get('released_shanghai') else max(actual,row['max_reservation_GPU_hours'])
    own |= row['job_id']==os.environ['HOSTNAME'].removesuffix('-master-0')
assert reserved+(0 if own else 7.5)<=ledger['approved_GPU_hours']
assert shutil.disk_usage(root).free/2**30>411
watch=json.loads((run/'monitor/WATCH-RECEIPT.json').read_text());assert watch['packed_archive_enabled']
assert watch['source_commit']==admission['implementation_commit']
# The CPU watcher is in a different allocated container: use its shared-PFS
# heartbeat rather than treating its PID as visible in this GPU namespace.
heartbeat=json.loads((run/'monitor/HOT-STORAGE.json').read_text())
assert time.time()-heartbeat['checked_unix']<600 and heartbeat['packed_archive_enabled']
print(json.dumps(dict(admitted=True,model=key,maximum_job_GPU_hours=7.5,committed_GPU_hours=reserved,
    free_PFS_GiB=shutil.disk_usage(root).free/2**30,hot_reservation_GiB=40,main_cases_per_task=50,episodes_per_case=6)),flush=True)
PY
PROBE=$ROOT/$RUN/probes/$ATTEMPT.json
"$PY" -B "$CODE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$PROBE"
DEADLINE=$(( BEGAN + 450*60 - 120 ))
for SPLIT in ID OOD; do
    REMAINING=$(( BEGAN + 450*60 - 30 - $(date +%s) ))
    test "$REMAINING" -gt 0
    timeout --signal=TERM --kill-after=15s "${REMAINING}s" "$PY" -B "$CODE/$RUN/READOUT.py" \
        --code "$CODE" --commit "$COMMIT" --fit "$ROOT/$RUN/fits/$INTERFACE/$OBJECTIVE" \
        --output "$ROOT/$RUN/readouts/$INTERFACE/$OBJECTIVE/$SPLIT" --probe "$PROBE" \
        --interface "$INTERFACE" --objective "$OBJECTIVE" --split "$SPLIT" \
        --deadline "$DEADLINE" --attempt "$ATTEMPT"
    test -f "$ROOT/$RUN/readouts/$INTERFACE/$OBJECTIVE/$SPLIT/RESULT.json" || exit 0
done
