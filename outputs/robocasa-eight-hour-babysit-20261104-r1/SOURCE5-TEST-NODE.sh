#!/usr/bin/env bash
set -Eeuo pipefail
: "${SOURCE5_TEST_CODE:?}" "${SOURCE5_TEST_COMMIT:?}" "${SOURCE5_TEST_TASK:?}" "${SOURCE5_TEST_MANIFEST_SHA:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=outputs/robocasa-eight-hour-babysit-20261104-r1
EVAL=$ROOT/sources/opendrawer-ablation-3m-eval-r3
ENGINE=$ROOT/sources/opendrawer-ablation-3m-engine-r2
VENDOR=$ROOT/sources/critic-selected-a1229239-r1
PY=$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
test "$(git -C "$SOURCE5_TEST_CODE" rev-parse HEAD)" = "$SOURCE5_TEST_COMMIT"
test "$(git -C "$EVAL" rev-parse HEAD)" = 81caf87060c9d966e0256e94da2ae705366ec33f
test "$(git -C "$ENGINE" rev-parse HEAD)" = 02ce594015eaa8f972f875d7f37e6972f4ada495
test "$(git -C "$VENDOR" rev-parse HEAD)" = 8f9bfa7661928ef8c17494b04f4d7bf3125634e1
for CODE in "$SOURCE5_TEST_CODE" "$EVAL" "$ENGINE" "$VENDOR"; do test -z "$(git -C "$CODE" status --porcelain)"; done
case "$SOURCE5_TEST_TASK" in PickPlaceCounterToCabinet|CoffeeSetupMug|TurnOnSinkFaucet) ;; *) exit 2 ;; esac
GROUP=$ROOT/$RUN/source5-tests/$SOURCE5_TEST_TASK/attempt-01
MANIFEST=$GROUP/MANIFEST.json
OUT=$GROUP/node-00
python3 - "$ROOT/$RUN" "$SOURCE5_TEST_TASK" <<'PY'
import datetime,json,os,socket,sys,time
from pathlib import Path
run=Path(sys.argv[1]);task=sys.argv[2];job=socket.gethostname().removesuffix('-master-0')
budget=json.loads((run/'SOURCE5-BUDGET.json').read_text())
assert budget['approved'] and budget['GPU_hours_ceiling']==70 and budget['independent_endpoint_gate_authorized']
ledger=json.loads((run/'SOURCE5-ALLOCATION-LEDGER.json').read_text())
assert ledger['approved_GPU_hours']==70
reserved=test_reserved=remaining=0;included=False
for row in ledger['allocations']:
    began=datetime.datetime.fromisoformat(row.get('accounting_start_shanghai',row['created_shanghai'])).timestamp()
    end=datetime.datetime.fromisoformat(row['released_shanghai']).timestamp() if row.get('released_shanghai') else time.time()
    charge=row['GPUs']*(end-began)/3600
    amount=charge if row.get('released_shanghai') else max(charge,row['max_reservation_GPU_hours'])
    reserved+=amount
    if row['kind']=='endpoint_test':test_reserved+=amount
    if row['job_id']==job:
        assert row['kind']=='endpoint_test' and row['task']==task and row['GPUs']==1 and row['max_reservation_GPU_hours']==2.5
        included=True
    elif not row.get('released_shanghai'):
        written=0
        for parent,_,files in os.walk(run/row['output'],followlinks=False):
            for name in files:
                path=Path(parent)/name
                assert not path.is_symlink()
                try:written+=path.stat().st_size
                except FileNotFoundError:pass
        allowance=row['storage_reservation_GiB']*2**30
        assert written<=allowance
        remaining+=allowance-written
for row in ledger.get('pending_submissions',[]):
    reserved+=row['max_reservation_GPU_hours']
    if row['kind']=='endpoint_test':test_reserved+=row['max_reservation_GPU_hours']
    if row['job_id']==job:
        assert row['kind']=='endpoint_test' and row['task']==task and row['GPUs']==1 and row['max_reservation_GPU_hours']==2.5
        included=True
        continue
    allowance=row['storage_reservation_GiB']*2**30
    written=sum(p.stat().st_size for p in (run/row['output']).rglob('*') if p.is_file() and not p.is_symlink())
    assert written<=allowance
    remaining+=allowance-written
assert reserved+(0 if included else 2.5)<=70
assert test_reserved+(0 if included else 2.5)<=8
fs=os.statvfs(run);free=fs.f_bavail*fs.f_frsize
required=(200+20+15)*2**30+remaining
assert free>=required, 'Need the shared floor, every independent active reservation and test headroom'
print(json.dumps(dict(passed=True,free_GiB=free/2**30,required_GiB=required/2**30,
    campaign_reserved_GPU_hours=reserved+(0 if included else 2.5),test_reserved_GPU_hours=test_reserved+(0 if included else 2.5))))
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
echo "$SOURCE5_TEST_MANIFEST_SHA  $MANIFEST" | sha256sum -c -
export TASK_CODE_PIN=02ce594015eaa8f972f875d7f37e6972f4ada495
export PYTHONPATH=$SOURCE5_TEST_CODE/$RUN:$EVAL/outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1:$EVAL/outputs/robocasa-flow-sde-checkpoint-selection-20261104-r1:$EVAL/outputs/robocasa-flow-sde-endpoint-test512-20261104-r1:$ENGINE/src:$VENDOR
export OPENPI_DATA_HOME=$ROOT/setups/robocasa365-native-20261101-r1/cache/.cache/openpi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_P2P_DISABLE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl FLUFFY_GPU_EXECUTION=baige PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
  [[ $UUIDS == GPU-* && $UUIDS != *$'\n'* ]]
  export CUDA_VISIBLE_DEVICES=0
fi
"$PY" -B "$ENGINE/scripts/[train]gpu_probe.py" --expected-count 1 --purpose evaluation --baige-container --output "$OUT/GPU-PROBE.json"
DEADLINE=$((START+8700))
timeout --kill-after=30s 8700 "$PY" -B "$SOURCE5_TEST_CODE/$RUN/SOURCE5-TEST.py" run --task "$SOURCE5_TEST_TASK" --manifest "$MANIFEST" --output "$OUT" --deadline "$DEADLINE"
"$PY" -B "$SOURCE5_TEST_CODE/$RUN/SOURCE5-TEST.py" combine --task "$SOURCE5_TEST_TASK" --manifest "$MANIFEST" --output "$GROUP"
