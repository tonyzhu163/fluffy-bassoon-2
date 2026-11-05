#!/usr/bin/env bash
set -Eeuo pipefail
: "${SOURCE5_DIAG_CODE:?}" "${SOURCE5_DIAG_COMMIT:?}" "${SOURCE5_DIAG_MANIFEST_SHA:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-eight-hour-babysit-20261104-r1
EVAL=$ROOT/sources/opendrawer-ablation-3m-eval-r3
VENDOR=$ROOT/sources/critic-selected-a1229239-r1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
test "$(git -C "$SOURCE5_DIAG_CODE" rev-parse HEAD)" = "$SOURCE5_DIAG_COMMIT"
test "$(git -C "$EVAL" rev-parse HEAD)" = 81caf87060c9d966e0256e94da2ae705366ec33f
test "$(git -C "$VENDOR" rev-parse HEAD)" = 8f9bfa7661928ef8c17494b04f4d7bf3125634e1
for CODE in "$SOURCE5_DIAG_CODE" "$EVAL" "$VENDOR"; do test -z "$(git -C "$CODE" status --porcelain)"; done
SOURCE5_DIAG_TASK=${SOURCE5_DIAG_TASK:-PickPlaceCounterToCabinet}
case "$SOURCE5_DIAG_TASK" in
  PickPlaceCounterToCabinet) PREFIX=counter ;;
  TurnOnSinkFaucet) PREFIX=faucet ;;
  *) exit 2 ;;
esac
SOURCE5_DIAG_GROUP=${SOURCE5_DIAG_GROUP:-$PREFIX-r1}
[[ $SOURCE5_DIAG_GROUP =~ ^$PREFIX-r[1-9][0-9]*$ ]]
GROUP=$ROOT/$RUN/reset-qualification/$SOURCE5_DIAG_GROUP
OUT=$GROUP/gpu-01
python3 - "$ROOT/$RUN" <<'PY'
import datetime,json,os,socket,sys,time
from pathlib import Path
run=Path(sys.argv[1]);job=socket.gethostname().removesuffix('-master-0')
budget=json.loads((run/'SOURCE5-BUDGET.json').read_text());assert budget['approved'] and budget['GPU_hours_ceiling']==70
ledger=json.loads((run/'SOURCE5-ALLOCATION-LEDGER.json').read_text())
reserved=diagnosed=remaining=0;included=False
for row in ledger['allocations']:
    began=datetime.datetime.fromisoformat(row.get('accounting_start_shanghai',row['created_shanghai'])).timestamp()
    end=datetime.datetime.fromisoformat(row['released_shanghai']).timestamp() if row.get('released_shanghai') else time.time()
    charge=row['GPUs']*(end-began)/3600
    amount=charge if row.get('released_shanghai') else max(charge,row['max_reservation_GPU_hours'])
    reserved+=amount
    if row['kind']=='diagnostic':diagnosed+=amount
    if row['job_id']==job:
        assert row['kind']=='diagnostic' and row['GPUs']==1 and row['max_reservation_GPU_hours']==.5
        included=True
    elif not row.get('released_shanghai'):
        written=0
        for parent,_,files in os.walk(run/row['output'],followlinks=False):
            for name in files:
                path=Path(parent)/name;assert not path.is_symlink()
                try:written+=path.stat().st_size
                except FileNotFoundError:pass
        allowance=row['storage_reservation_GiB']*2**30;assert written<=allowance
        remaining+=allowance-written
for row in ledger.get('pending_submissions',[]):
    reserved+=row['max_reservation_GPU_hours']
    if row['kind']=='diagnostic':diagnosed+=row['max_reservation_GPU_hours']
    if row['job_id']==job:included=True
    else:
        allowance=row['storage_reservation_GiB']*2**30
        written=sum(p.stat().st_size for p in (run/row['output']).rglob('*') if p.is_file() and not p.is_symlink())
        assert written<=allowance
        remaining+=allowance-written
assert reserved+(0 if included else .5)<=70 and diagnosed+(0 if included else .5)<=2
fs=os.statvfs(run);free=fs.f_bavail*fs.f_frsize;required=222*2**30+remaining
assert free>=required
print(json.dumps(dict(passed=True,reserved_GPU_hours=reserved+(0 if included else .5),diagnosed_GPU_hours=diagnosed+(0 if included else .5),free_GiB=free/2**30,required_GiB=required/2**30)))
PY
if [[ ${1:-} == --admit ]]; then exit 0; fi
mkdir "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
START=$(date +%s)
finish() {
  rc=$?
  printf '{"started_unix":%s,"ended_unix":%s,"exit_code":%s,"hostname":"%s","gpus":1}\n' "$START" "$(date +%s)" "$rc" "$HOSTNAME" > "$OUT/CLOCK.json"
}
trap finish EXIT
echo "$SOURCE5_DIAG_MANIFEST_SHA  $GROUP/MANIFEST.json" | sha256sum -c -
export TASK_CODE_PIN=$SOURCE5_DIAG_COMMIT
export PYTHONPATH=$SOURCE5_DIAG_CODE/$RUN:$EVAL/outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1:$EVAL/outputs/robocasa-flow-sde-checkpoint-selection-20261104-r1:$SOURCE5_DIAG_CODE/src:$VENDOR
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
  [[ $UUIDS == GPU-* && $UUIDS != *$'\n'* ]]
  export CUDA_VISIBLE_DEVICES=0
fi
"$PY" -B "$SOURCE5_DIAG_CODE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$OUT/GPU-PROBE.json"
DEADLINE=$((START+1650))
timeout --kill-after=30s 1650 "$PY" -B "$SOURCE5_DIAG_CODE/$RUN/SOURCE5-RESET-QUALIFY.py" run --task "$SOURCE5_DIAG_TASK" --group "$SOURCE5_DIAG_GROUP" --output "$OUT" --deadline "$DEADLINE"
