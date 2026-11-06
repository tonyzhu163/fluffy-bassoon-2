#!/usr/bin/env bash
set -Eeuo pipefail
BEGAN=$(date +%s)
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COMMIT=$(git -C "$CODE" rev-parse HEAD)
test -z "$(git -C "$CODE" status --porcelain)"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=$ROOT/outputs/steering-effect-refit-20261106-r1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
ATTEMPT=${STEERING_ATTEMPT:?Require a fresh allocation identity}
MINUTES=${STEERING_MINUTES:?Require the admitted scheduler runtime}
export RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1
export PYTHONPATH=$CODE/src:$RLINF_PATH
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$RUN/probes" "$RUN/logs"
test ! -e "$RUN/logs/$ATTEMPT.log"
exec > >(tee "$RUN/logs/$ATTEMPT.log") 2>&1
export STEERING_MINUTES=$MINUTES STEERING_ATTEMPT=$ATTEMPT
"$PY" - <<'PY'
from pathlib import Path
import datetime,json,os,shutil
run=Path('/mowenjie/fluffy-bassoon-2/outputs/steering-effect-refit-20261106-r1')
ledger=json.loads((run/'ALLOCATIONS.json').read_text())
minutes=int(os.environ['STEERING_MINUTES']);assert 20<=minutes<=480
assert ledger['stage']=='refit' and ledger['attempt']==os.environ['STEERING_ATTEMPT']
assert ledger['combined_hard_ceiling_GPU_hours']==150 and ledger['parent_conservative_GPU_hours']==140.2711111111111
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
committed=ledger['parent_conservative_GPU_hours'];own=False
for row in ledger['allocations']:
    actual=row['GPUs']*((datetime.datetime.fromisoformat(row['released_shanghai']) if row.get('released_shanghai') else now)-datetime.datetime.fromisoformat(row['created_shanghai'])).total_seconds()/3600
    committed+=actual if row.get('released_shanghai') else max(actual,row['max_reservation_GPU_hours'])
    own|=row.get('attempt')==os.environ['STEERING_ATTEMPT']
if not own:committed+=(minutes+5)/60
assert committed<=150
assert shutil.disk_usage(run).free>286*2**30
print(json.dumps(dict(stage='corrected_KV_AD_refit',maximum_allocation_minutes=minutes,committed_GPU_hours=committed,hard_ceiling_GPU_hours=150,free_GiB=shutil.disk_usage(run).free/2**30)))
PY
PROBE=$RUN/probes/$ATTEMPT.json
"$PY" -B "$CODE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training --baige-container --baige-exclusive-training --output "$PROBE"
DEADLINE=$(( BEGAN + MINUTES*60 - 300 ))
timeout --signal=TERM --kill-after=120s "$(( MINUTES*60 - 120 ))" "$PY" -B "$CODE/outputs/steering-effect-refit-20261106-r1/REFIT.py" \
    --code "$CODE" --commit "$COMMIT" --probe "$PROBE" --output "$RUN/fits/kv_bridge/ad" \
    --max-seconds "$(( MINUTES*60 - 600 ))" --deadline "$DEADLINE"
