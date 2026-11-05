#!/usr/bin/env bash
set -Eeuo pipefail
: "${SOURCE5_RECOVERY_CODE:?}" "${SOURCE5_RECOVERY_COMMIT:?}" "${SOURCE5_RECOVERY_PLAN_SHA:?}" "${SOURCE5_RECOVERY_TASK:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-eight-hour-babysit-20261104-r1
GROUP=$ROOT/$RUN/source5-tests/$SOURCE5_RECOVERY_TASK/attempt-02
PLAN=$GROUP/PLAN.json
OUT=$GROUP/node-00
EVAL=$ROOT/sources/opendrawer-ablation-3m-eval-r3
ENGINE=$ROOT/sources/opendrawer-ablation-3m-engine-r2
VENDOR=$ROOT/sources/critic-selected-a1229239-r1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
test "$(git -C "$SOURCE5_RECOVERY_CODE" rev-parse HEAD)" = "$SOURCE5_RECOVERY_COMMIT"
test "$(git -C "$EVAL" rev-parse HEAD)" = 81caf87060c9d966e0256e94da2ae705366ec33f
test "$(git -C "$ENGINE" rev-parse HEAD)" = 02ce594015eaa8f972f875d7f37e6972f4ada495
test "$(git -C "$VENDOR" rev-parse HEAD)" = 8f9bfa7661928ef8c17494b04f4d7bf3125634e1
for CODE in "$SOURCE5_RECOVERY_CODE" "$EVAL" "$ENGINE" "$VENDOR"; do test -z "$(git -C "$CODE" status --porcelain)"; done
echo "$SOURCE5_RECOVERY_PLAN_SHA  $PLAN" | sha256sum -c -
python3 - "$ROOT/$RUN" "$SOURCE5_RECOVERY_TASK" "$PLAN" <<'PY'
from pathlib import Path
import datetime,json,os,socket,sys,time
run=Path(sys.argv[1]);task=sys.argv[2];plan=json.loads(Path(sys.argv[3]).read_text())
budget=json.loads((run/'SOURCE5-BUDGET.json').read_text());ledger=json.loads((run/'SOURCE5-ALLOCATION-LEDGER.json').read_text())
assert budget['approved'] and budget['GPU_hours_ceiling']==ledger['approved_GPU_hours']==70
maximum=plan['maximum_GPU_hours']
assert maximum in (.5,1.2)
assert plan['task']==task and 1<=len(plan['remaining_banks'])<=(8 if maximum==.5 else 24)
assert (plan['scheduler_minutes'],plan['command_seconds'])==((30,1620) if maximum==.5 else (72,4140))
assert plan['scientific_comparison_unchanged'] and not plan['automatic_retry']
job=socket.gethostname().removesuffix('-master-0');reserved=tests=remaining=0;included=False
for row in ledger['allocations']+ledger.get('pending_submissions',[]):
    start=datetime.datetime.fromisoformat(row.get('accounting_start_shanghai',row.get('last_confirmed_queued_shanghai',row['created_shanghai']))).timestamp()
    end=datetime.datetime.fromisoformat(row['released_shanghai']).timestamp() if row.get('released_shanghai') else time.time()
    charge=row['GPUs']*(end-start)/3600
    amount=charge if row.get('released_shanghai') else max(charge,row['max_reservation_GPU_hours'])
    reserved+=amount
    if row['kind']=='endpoint_test':tests+=amount
    if row['job_id']==job:
        assert row['kind']=='endpoint_test' and row['task']==task and row['GPUs']==1 and row['max_reservation_GPU_hours']==maximum
        included=True
    elif not row.get('released_shanghai'):
        written=0
        for parent,_,files in os.walk(run/row['output'],followlinks=False):
            for name in files:
                path=Path(parent)/name
                assert not path.is_symlink()
                try:written+=path.stat().st_size
                except FileNotFoundError:pass
        allowance=row['storage_reservation_GiB']*2**30;assert written<=allowance
        remaining+=allowance-written
shared=json.loads((run/'SOURCE5-SHARED-RESERVATIONS.json').read_text())
assert shared['combined_base_floor_GiB']==220 and shared['meta_readouts_held']
pilot=shared.get('external_RoboTwin_pilot_growth_GiB',0)
assert pilot in (0,125) and (shared['RoboTwin_large_writes_held'] or pilot==125)
registered={row['output'] for row in ledger['allocations']+ledger.get('pending_submissions',[])}
current=f'source5-tests/{task}/attempt-02'
for intent in shared['owned_intents']:
    if intent['output']==current or intent['output'] in registered:continue
    reserved+=intent['max_reservation_GPU_hours'];tests+=intent['max_reservation_GPU_hours']
    remaining+=intent['storage_reservation_GiB']*2**30
remaining+=(shared['external_fit_only_growth_GiB']+shared.get('external_scratch_growth_GiB',0)+pilot)*2**30
assert reserved+(0 if included else maximum)<=70 and tests+(0 if included else maximum)<=8
fs=os.statvfs(run);free=fs.f_bavail*fs.f_frsize;required=225*2**30+remaining
assert free>=required
print(json.dumps(dict(passed=True,reserved_GPU_hours=reserved+(0 if included else maximum),test_reserved_GPU_hours=tests+(0 if included else maximum),free_GiB=free/2**30,required_GiB=required/2**30)))
PY
if [[ ${1:-} == --admit ]]; then exit 0; fi
mkdir "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
START=$(date +%s)
finish() { rc=$?; printf '{"started_unix":%s,"ended_unix":%s,"exit_code":%s,"hostname":"%s","gpus":1}\n' "$START" "$(date +%s)" "$rc" "$HOSTNAME" > "$OUT/CLOCK.json"; }
trap finish EXIT
export TASK_CODE_PIN=02ce594015eaa8f972f875d7f37e6972f4ada495
export PYTHONPATH=$SOURCE5_RECOVERY_CODE/$RUN:$EVAL/outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1:$EVAL/outputs/robocasa-flow-sde-checkpoint-selection-20261104-r1:$EVAL/outputs/robocasa-flow-sde-endpoint-test512-20261104-r1:$ENGINE/src:$VENDOR
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ $UUIDS == GPU-* && $UUIDS != *$'\n'* ]]
    export CUDA_VISIBLE_DEVICES=0
fi
"$PY" -B "$ENGINE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$OUT/GPU-PROBE.json"
COMMAND_SECONDS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["command_seconds"])' "$PLAN")
DEADLINE=$((START+COMMAND_SECONDS))
timeout --kill-after=30s "$COMMAND_SECONDS" "$PY" -B "$SOURCE5_RECOVERY_CODE/$RUN/SOURCE5-CONTINUE-TEST.py" run --plan "$PLAN" --deadline "$DEADLINE"
