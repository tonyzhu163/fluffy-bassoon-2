#!/usr/bin/env bash
set -Eeuo pipefail
: "${STEERING_CODE:?}" "${STEERING_COMMIT:?}" "${STEERING_ATTEMPT:?}" "${STEERING_QUALIFICATION:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-three-task-kv-velocity-meta-20261105-r1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
test "$(git -C "$STEERING_CODE" rev-parse HEAD)" = "$STEERING_COMMIT"
test -z "$(git -C "$STEERING_CODE" status --porcelain)"
export RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1
export PYTHONPATH=$STEERING_CODE/src:$RLINF_PATH
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
mkdir -p "$ROOT/$RUN/probes" "$ROOT/$RUN/logs"
exec > >(tee "$ROOT/$RUN/logs/$STEERING_ATTEMPT.log") 2>&1
"$PY" - <<'PY'
from pathlib import Path
import datetime,json,os
r=Path('/mowenjie/fluffy-bassoon-2');d=json.loads((r/'outputs/robocasa-eight-hour-babysit-20261104-r1/META-ALLOCATION-LEDGER.json').read_text())
assert d['approved_GPU_hours']==150
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));reserved=d['preparation_GPU_hours'];own=False
for row in d['allocations']:
    actual=row['GPUs']*((datetime.datetime.fromisoformat(row['released_shanghai']) if row.get('released_shanghai') else now)-datetime.datetime.fromisoformat(row['created_shanghai'])).total_seconds()/3600
    reserved+=actual if row.get('released_shanghai') else max(actual,row['max_reservation_GPU_hours'])
    own|=row['job_id']==os.environ['HOSTNAME'].removesuffix('-master-0')
assert reserved+(0 if own else 72)<=150
v=os.statvfs(r);assert v.f_bavail*v.f_frsize>220*2**30
q=json.loads(Path(os.environ['STEERING_QUALIFICATION']).read_text());assert q['passed']
print(json.dumps(dict(approved_GPU_hours=150,maximum_new_reservation_GPU_hours=72,committed_GPU_hours=reserved,free_PFS_GiB=v.f_bavail*v.f_frsize/2**30)))
PY
PROBE=$ROOT/$RUN/probes/$STEERING_ATTEMPT.json
"$PY" -B "$STEERING_CODE/scripts/[train]gpu_probe.py" --expected-count 8 --purpose training --baige-container --baige-exclusive-training --output "$PROBE"
timeout --signal=TERM --kill-after=120s 31500 "$PY" -B "$STEERING_CODE/$RUN/MATRIX.py" --code "$STEERING_CODE" --commit "$STEERING_COMMIT" --attempt "$STEERING_ATTEMPT" --probe "$PROBE" --qualification "$STEERING_QUALIFICATION" --execution-batch "${STEERING_EXECUTION_BATCH:-8}"
