#!/usr/bin/env bash
set -Eeuo pipefail
: "${SOURCE5_CODE_DIR:?}" "${SOURCE5_CODE_COMMIT:?}"
ROOT=/mowenjie/fluffy-bassoon-2
RUN=$ROOT/outputs/robocasa-eight-hour-babysit-20261104-r1
TARGET=${1:?PickPlaceCounterToCabinet or TurnOnSinkFaucet}
ATTEMPT=${SOURCE5_ATTEMPT:-01}
case "$ATTEMPT" in 01|02) ;; *) exit 2 ;; esac
export TASK_META=$SOURCE5_CODE_DIR TASK_CODE_PIN=$SOURCE5_CODE_COMMIT
export TASK_ATTEMPT=source5-40-r$((10#$ATTEMPT)) TASK_SEGMENT=source5-40
export TASK_OUTPUT=$RUN/source5/$TARGET/attempt-$ATTEMPT
export ROBOCASA_SIM_PPO_ACCOUNTING=$RUN/source5/$TARGET/accounting
case "$TARGET" in
  PickPlaceCounterToCabinet) export ROBOCASA_FLOW_TRAIN_SEED=2200000 ROBOCASA_FLOW_EVAL_SEED=620100 ;;
  TurnOnSinkFaucet) export ROBOCASA_FLOW_TRAIN_SEED=2800000 ROBOCASA_FLOW_EVAL_SEED=680100 ;;
  *) exit 2 ;;
esac
export ROBOCASA_FLOW_TASK=$TARGET
export TASK_CONFIG=robocasa365_pi05_flow_sde_ppo_reuse4
export TASK_GPUS=8 TASK_CPUS=168 TASK_RAM_GIB=1280 TASK_SHM_GIB=32 TASK_WALL_SECONDS=13200
export ROBOCASA_SIM_PPO_ACTOR_PLACEMENT=0-1 ROBOCASA_SIM_PPO_ROLLOUT_PLACEMENT=0-7
export ROBOCASA_SIM_PPO_MICRO_BATCH=16 ROBOCASA_SIM_PPO_WORLD_SIZE=8
export ROBOCASA_FLOW_MAX_EPOCHS=40 ROBOCASA_FLOW_UPDATE_EPOCHS=4
export ROBOCASA_FLOW_FULL_MODEL_EXPORT_STEPS='[40]'
export ROBOCASA_FLOW_VAL_INTERVAL=10 ROBOCASA_FLOW_SAVE_INTERVAL=10
export ROBOCASA_FLOW_NOISE_LEVEL=0.3 ROBOCASA_FLOW_IGNORE_LAST=true
export ROBOCASA_FLOW_MAX_ACTIONS=4100000 ROBOCASA_FLOW_TRAIN_ENVS=128 ROBOCASA_FLOW_ROUNDS=1
export ROBOCASA_FLOW_SAMPLING_RECORD=full ROBOCASA_SIM_PPO_GLOBAL_ESCROW=true
export ROBOCASA_FLOW_ACTOR_BATCH_DROP='[]'
export RLINF_PATH=$ROOT/sources/critic-selected-a1229239-r1
export PYTHONPATH=$TASK_META/src:$RLINF_PATH
export ROBOCASA_SIM_PPO_OUTPUT=$TASK_OUTPUT TASK_DEADLINE=1
export ROBOCASA_PI05_TORCH_MODEL=$ROOT/inputs/robocasa365-public-pi05-c484-torch-20261101-r2
export ROBOCASA_PI05_NATIVE_NORM=$ROOT/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json
if [[ $ATTEMPT == 02 ]]; then
  : "${SOURCE5_RESUME_REFERENCE_SHA:?}"
  export TASK_WALL_SECONDS=10500
  export ROBOCASA_FLOW_FRESH_EVAL_PROCESSES=true
  export ROBOCASA_SIM_PPO_RESUME=$RUN/source5/$TARGET/attempt-01/$TARGET/checkpoints/global_step_10
  export ROBOCASA_SIM_PPO_BEFORE_REFERENCE=$RUN/source5/$TARGET/RESUME-REFERENCE-01.json
  export ROBOCASA_SIM_PPO_BEFORE_SHA256=$SOURCE5_RESUME_REFERENCE_SHA
fi
if [[ ${2:-} == --config ]]; then
  exec "$ROOT/setups/robocasa365-native-20261101-r1/env/bin/python" -B "$TASK_META/scripts/[train]robocasa_flow_ppo.py" --config-name "$TASK_CONFIG" --cfg job --resolve
fi
python3 - "$RUN" "$TARGET" "$ATTEMPT" <<'PY'
import datetime,json,os,socket,sys,time
from pathlib import Path
run=Path(sys.argv[1]);target=sys.argv[2];attempt=sys.argv[3];p=run/'SOURCE5-BUDGET.json'
if not p.exists():raise SystemExit('Source budget approval record is missing')
d=json.loads(p.read_text())
assert d.get('approved') is True and d.get('GPU_hours_ceiling')==70
assert d.get('independent_endpoint_gate_authorized') is True
ledger=json.loads((run/'SOURCE5-ALLOCATION-LEDGER.json').read_text())
assert ledger['approved_GPU_hours']==70
cap=24 if attempt=='02' else 30
storage=110 if attempt=='02' else 150
if attempt=='02':
    group='counter-r1' if target=='PickPlaceCounterToCabinet' else 'faucet-r1'
    qualification=json.loads((run/'reset-qualification'/group/'gpu-01/RESULT.json').read_text())
    import hashlib
    qm=run/'reset-qualification'/group/'MANIFEST.json'
    assert qualification['manifest_sha256']==hashlib.sha256(qm.read_bytes()).hexdigest()
    assert json.loads(qm.read_text())['task']==target
    assert qualification['passed'] and qualification['original_cases']==64 and qualification['repeated_cases']==8
    assert qualification['fresh_eval_processes'] and qualification['original_pairing_check_retained'] and qualification['SGD']==0
    reference=Path(os.environ['ROBOCASA_SIM_PPO_BEFORE_REFERENCE'])
    assert hashlib.sha256(reference.read_bytes()).hexdigest()==os.environ['SOURCE5_RESUME_REFERENCE_SHA']
    ref=json.loads(reference.read_text());assert ref['pending_evaluation_step']==10
    assert Path(ref['checkpoint'])==Path(os.environ['ROBOCASA_SIM_PPO_RESUME'])
job=socket.gethostname().removesuffix('-master-0');reserved=task_reserved=0;included=False
for row in ledger['allocations']:
    began=datetime.datetime.fromisoformat(row.get('accounting_start_shanghai',row['created_shanghai'])).timestamp()
    end=datetime.datetime.fromisoformat(row['released_shanghai']).timestamp() if row.get('released_shanghai') else time.time()
    hours=row['GPUs']*(end-began)/3600
    reserved += hours if row.get('released_shanghai') else max(hours,row['max_reservation_GPU_hours'])
    if row['task']==target and row['kind'] in ('source_train','source_resume'):
        task_reserved += hours if row.get('released_shanghai') else max(hours,row['max_reservation_GPU_hours'])
    if row['job_id']==job:
        assert row['task']==target and row['GPUs']==8 and row['max_reservation_GPU_hours']==cap
        included=True
for row in ledger.get('pending_submissions',[]):
    reserved += row['max_reservation_GPU_hours']
    if row['task']==target and row['kind'] in ('source_train','source_resume'):
        task_reserved += row['max_reservation_GPU_hours']
    if row['job_id']==job:
        assert row['task']==target and row['GPUs']==8 and row['max_reservation_GPU_hours']==cap
        included=True
assert reserved + (0 if included else cap) <= 70
assert task_reserved + (0 if included else cap) <= 30
space=os.statvfs(run)
# Account for future growth of every other active source. Already-written
# bytes are reflected in free space; avoid reserving those bytes twice.
remaining=0;active_sources=[]
for row in ledger['allocations']+ledger.get('pending_submissions',[]):
    if row.get('released_shanghai') or row['job_id']==job:
        continue
    directory=run/row['output'];written=0
    for parent,dirs,files in os.walk(directory,followlinks=False):
        for name in files:
            path=Path(parent)/name
            if path.is_symlink():raise RuntimeError('Active source contains an unexpected symlink')
            try:written+=path.stat().st_size
            except FileNotFoundError:pass  # an atomic producer rename
    allowance=row['storage_reservation_GiB']*2**30
    assert written<=allowance, 'Active source exceeded its measured storage reservation'
    remaining+=allowance-written
    active_sources.append(dict(job_id=row['job_id'],written_bytes=written,reserved_remaining_bytes=allowance-written))
required=(200+storage+20)*2**30+remaining
free=space.f_bavail*space.f_frsize
assert free>=required, 'Need the200GiB floor, new source reservation, other active reservations and20GiB headroom'
print(json.dumps(dict(passed=True,checked_unix=time.time(),free_GiB=free/2**30,required_GiB=required/2**30,active_sources=active_sources,reserved_GPU_hours=reserved+(0 if included else cap),task_reserved_GPU_hours=task_reserved+(0 if included else cap))))

PY
if [[ ${2:-} == --admit ]]; then exit 0; fi
exec bash "$TASK_META/outputs/robocasa-ppo-grpo-meta-pipeline-20261104-r1/COMMAND.sh"
