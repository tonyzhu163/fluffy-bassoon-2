#!/usr/bin/env bash
set -Eeuo pipefail
BEGAN=$(date +%s)
INTERFACE=${1:?interface} OBJECTIVE=${2:?objective} ATTEMPT=${3:?fresh-attempt}
case "$INTERFACE" in kv_bridge|velocity_bridge) ;; *) exit 2 ;; esac
test "$OBJECTIVE" = ad
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COMMIT=$(git -C "$CODE" rev-parse HEAD)
test -z "$(git -C "$CODE" status --porcelain)"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-three-task-kv-velocity-meta-20261105-r1
SCIENCE=$ROOT/sources/steering-packed-87984718
SCICOMMIT=87984718c9199797d31a158c47225cc695623afe
test "$(git -C "$SCIENCE" rev-parse HEAD)" = "$SCICOMMIT"
test -z "$(git -C "$SCIENCE" status --porcelain)"
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
MINUTES=${STEERING_READOUT_MINUTES:-300}
export STEERING_INTERFACE=$INTERFACE STEERING_OBJECTIVE=$OBJECTIVE STEERING_COMMIT=$COMMIT
export STEERING_READOUT_MINUTES=$MINUTES STEERING_OPERATIONAL_CODE=$CODE
export STEERING_STORAGE_ACCOUNTING=$ROOT/$RUN/STORAGE-ACCOUNTING12.json
export STEERING_MIN_FREE_GIB=220 STEERING_WAVE_ALLOWANCE_SECONDS=1200
export RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1 PYTHONPATH=$SCIENCE/src:$ROOT/sources/critic-selected-a1229239-r1
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$ROOT/$RUN/logs" "$ROOT/$RUN/probes" "$ROOT/$RUN/operational-implementations"
test ! -e "$ROOT/$RUN/logs/$ATTEMPT.log"
exec > >(tee "$ROOT/$RUN/logs/$ATTEMPT.log") 2>&1
"$PY" - <<'PY'
import datetime,hashlib,importlib.util,json,os,time
from pathlib import Path
root=Path('/mowenjie/fluffy-bassoon-2');run=root/'outputs/robocasa-three-task-kv-velocity-meta-20261105-r1'
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
a=json.loads((run/'READOUT-ADMISSION.json').read_text())
assert a['approved'] and a['human_budget_authorization'] and a['approved_GPU_hours']==150
assert a['implementation_commit']==os.environ['STEERING_COMMIT']
assert a['scientific_code_commit']=='87984718c9199797d31a158c47225cc695623afe'
assert a['scientific_contract_sha256']==sha(run/'CONTRACT.json')
assert a['storage_accounting_sha256']==sha(Path(os.environ['STEERING_STORAGE_ACCOUNTING']))
for field,name in [('fitted_qualification_sha256','FITTED-QUALIFICATION-SUMMARY04.json'),('packed_storage_qualification_sha256','PACKED-ARCHIVE-QUALIFICATION05.json')]:
    p=run/'migrations/pfs-members-01'/name
    assert a[field]==sha(p) and json.loads(p.read_text())['passed']
key=os.environ['STEERING_INTERFACE']+'/'+os.environ['STEERING_OBJECTIVE']
assert key in a['selected_candidates'] and set(a['selected_candidates'])=={'kv_bridge/ad','velocity_bridge/ad'}
fit=run/'fits'/key;status=json.loads((fit/'training-status.json').read_text())
assert status['converged'] and a['selected_candidates'][key]['sha256']==sha(fit/'candidate.pt')
ledger=json.loads((root/'outputs/robocasa-eight-hour-babysit-20261104-r1/META-ALLOCATION-LEDGER.json').read_text())
assert ledger['approved_GPU_hours']==150
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
reserved=ledger['preparation_GPU_hours'];own=False;job=os.environ['HOSTNAME'].removesuffix('-master-0')
for row in ledger['allocations']:
    end=datetime.datetime.fromisoformat(row['released_shanghai']) if row.get('released_shanghai') else now
    actual=row['GPUs']*(end-datetime.datetime.fromisoformat(row['created_shanghai'])).total_seconds()/3600
    reserved+=actual if row.get('released_shanghai') else max(actual,row['max_reservation_GPU_hours'])
    own|=row['job_id']==job
    if not row.get('released_shanghai') and row.get('interface')==os.environ['STEERING_INTERFACE'] and row.get('objective')=='ad':
        assert row['job_id']==job, 'Do not overlap a readout writer'
minutes=int(os.environ['STEERING_READOUT_MINUTES']);assert 30<=minutes<=450
assert reserved+(0 if own else minutes/60)<=150
watch=json.loads((run/'monitor/WATCH-RECEIPT.json').read_text())
assert watch['source_commit']==a['storage_watcher_commit'] and watch['packed_archive_enabled']
hot=json.loads((run/'monitor/HOT-STORAGE.json').read_text())
assert hot['packed_archive_enabled'] and time.time()-hot['checked_unix']<600
spec=importlib.util.spec_from_file_location('storage_resume_guard',Path(os.environ['STEERING_OPERATIONAL_CODE'])/'outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/STORAGE-GUARD.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
print(json.dumps(dict(admitted=True,model=key,maximum_job_GPU_hours=minutes/60,committed_GPU_hours=reserved,forward_storage=guard.require_capacity())),flush=True)
PY
PROBE=$ROOT/$RUN/probes/$ATTEMPT.json
"$PY" -B "$SCIENCE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$PROBE"
export STEERING_OPERATIONAL_ATTEMPT=$ATTEMPT
"$PY" - <<'PY'
import hashlib,json,os,time
from pathlib import Path
r=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1')
receipt=dict(implementation_commit=os.environ['STEERING_COMMIT'],scientific_code_commit='87984718c9199797d31a158c47225cc695623afe',attempt=os.environ['STEERING_OPERATIONAL_ATTEMPT'],storage_policy_sha256=hashlib.sha256(Path(os.environ['STEERING_STORAGE_ACCOUNTING']).read_bytes()).hexdigest(),change='Forward storage accounting only; original scientific code and immutable binding retained',created_unix=time.time())
with (r/'operational-implementations'/(receipt['attempt']+'.json')).open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
PY
DEADLINE=$(( BEGAN + MINUTES*60 - 120 ))
for SPLIT in ID OOD; do
    REMAINING=$(( BEGAN + MINUTES*60 - 30 - $(date +%s) ))
    test "$REMAINING" -gt 0
    timeout --signal=TERM --kill-after=15s "${REMAINING}s" "$PY" -B "$CODE/$RUN/READOUT.py" \
        --code "$SCIENCE" --commit "$SCICOMMIT" --fit "$ROOT/$RUN/fits/$INTERFACE/$OBJECTIVE" \
        --output "$ROOT/$RUN/readouts/$INTERFACE/$OBJECTIVE/$SPLIT" --probe "$PROBE" \
        --interface "$INTERFACE" --objective "$OBJECTIVE" --split "$SPLIT" \
        --deadline "$DEADLINE" --attempt "$ATTEMPT"
    test -f "$ROOT/$RUN/readouts/$INTERFACE/$OBJECTIVE/$SPLIT/RESULT.json" || exit 0
done
