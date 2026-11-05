#!/usr/bin/env bash
set -Eeuo pipefail
BEGAN=$(date +%s)
INTERFACE=${1:?interface}
case "$INTERFACE" in kv_bridge|velocity_bridge) ;; *) exit 2 ;; esac
PHASE=${2:-deployment}
case "$PHASE" in
    deployment) ATTEMPT=fitted-deploy-pfs03-$INTERFACE; DIRECTORY=qualification-fitted-pfs03 ;;
    ood-full) ATTEMPT=fitted-oodfull-pfs04-$INTERFACE; DIRECTORY=qualification-fitted-oodfull-pfs04 ;;
    reset) ATTEMPT=fitted-reset-pfs04-$INTERFACE; DIRECTORY=qualification-fitted-reset-pfs04 ;;
    *) exit 2 ;;
esac
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
COMMIT=$(git -C "$CODE" rev-parse HEAD)
test -z "$(git -C "$CODE" status --porcelain)"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-three-task-kv-velocity-meta-20261105-r1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
OUT=$ROOT/$RUN/$DIRECTORY/$INTERFACE
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
exec > >(tee "$OUT/WORKER.log") 2>&1
export STEERING_CODE=$CODE STEERING_COMMIT=$COMMIT STEERING_MIN_FREE_GIB=375
export RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1
export PYTHONPATH=$CODE/src:$RLINF_PATH
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ROBOCASA_NATIVE_OUTPUT=$OUT/native-$ATTEMPT
export ROBOCASA_PI05_TORCH_MODEL=$ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
"$PY" - <<'PY'
import datetime,json,os,shutil
from pathlib import Path
from hydra import compose,initialize_config_dir
with initialize_config_dir(config_dir=str(Path(os.environ['STEERING_CODE'])/'configs/native_ppo'),version_base=None):
    configuration=compose(config_name='robocasa365_pi05_eval')
assert configuration.rollout.model is not None and configuration.env.eval is not None
root=Path('/mowenjie/fluffy-bassoon-2')
ledger=json.loads((root/'outputs/robocasa-eight-hour-babysit-20261104-r1/META-ALLOCATION-LEDGER.json').read_text())
assert ledger['approved_GPU_hours']==150
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
reserved=ledger['preparation_GPU_hours'];own=False
for row in ledger['allocations']:
    end=datetime.datetime.fromisoformat(row['released_shanghai']) if row.get('released_shanghai') else now
    actual=row['GPUs']*(end-datetime.datetime.fromisoformat(row['created_shanghai'])).total_seconds()/3600
    reserved+=actual if row.get('released_shanghai') else max(actual,row['max_reservation_GPU_hours'])
    own |= row['job_id']==os.environ['HOSTNAME'].removesuffix('-master-0')
assert reserved+(0 if own else .25)<=150
free=shutil.disk_usage(root).free/2**30
assert free>375
print(json.dumps(dict(source_commit=os.environ['STEERING_COMMIT'],GPU_hour_limit=.25,
    maximum_committed_GPU_hours=reserved,free_PFS_GiB=free,minimum_free_PFS_GiB=375,
    actual_Hydra_config_composed=True,scientific_benchmark_held=True)),flush=True)
PY
PROBE=$OUT/GPU-PROBE.json
"$PY" -B "$CODE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$PROBE"
DEADLINE=$(( BEGAN + 780 ))
REMAINING=$(( BEGAN + 840 - $(date +%s) ))
test "$REMAINING" -gt 0
timeout --signal=TERM --kill-after=15s "${REMAINING}s" "$PY" -B "$CODE/$RUN/READOUT.py" \
    --code "$CODE" --commit "$COMMIT" --fit "$ROOT/$RUN/fits/$INTERFACE/ad" \
    --output "$OUT" --probe "$PROBE" --interface "$INTERFACE" --objective ad \
    --split qualification --qualification-phase "$PHASE" --deadline "$DEADLINE" --attempt "$ATTEMPT"
