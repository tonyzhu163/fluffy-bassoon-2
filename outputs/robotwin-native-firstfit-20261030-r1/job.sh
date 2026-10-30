#!/usr/bin/env bash
# Launch artifact only: invokes the sole owner's qualified native trainer.
# Scheduler request: one GPU, 8 CPU, 32 GiB RAM, 8 GiB shared memory, 20 min.
set -euo pipefail
TASK_PIN=1aeea4f9fcc307bb147def92374bb81a3e2cb50e
TASK_RUN=robotwin-native-firstfit-20261030-r1
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=${1:-$TASK_ROOT/worktrees/$TASK_RUN}
TASK_OUT=${2:-$TASK_ROOT/outputs/$TASK_RUN}
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_TRAIN_INDEX=$TASK_ROOT/outputs/robotwin-native-history-20261030-r1/train/index.json
TASK_VALIDATION_INDEX=$TASK_ROOT/outputs/robotwin-native-history-20261030-r1/validation/index.json
TASK_TRAIN_SHA=4e8d4e64eb1845a234f710342f168d3a4cdb86366d64dd6c7cef6c7c5895a4af
TASK_VALIDATION_SHA=194ae0fd4588e7bd55c3cd7a0fc6eedfb010bcb8043e311eeedd8adff0b2472b
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_PIN"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
exec > >(tee "$TASK_OUT/console.log") 2>&1
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"' EXIT
export PYTHONPATH="$TASK_CODE/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FLUFFY_GPU_EXECUTION=baige
# Retain the allocation's visibility. The qualified image uses local device0
# when NVIDIA_VISIBLE_DEVICES isolates one card but CUDA's variable is absent.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cp "${BASH_SOURCE[0]}" "$TASK_OUT/launch-script.sh"
# 970s includes allocation probe, data loading, every fit and artifact audit.
# The outer990s TERM plus10s KILL reserve bounds emergency execution at1000s.
timeout --signal=TERM --kill-after=10s 990 "$TASK_PYTHON" - \
  "$TASK_PIN" "$TASK_CODE" "$TASK_OUT" "$TASK_TRAIN_INDEX" "$TASK_TRAIN_SHA" \
  "$TASK_VALIDATION_INDEX" "$TASK_VALIDATION_SHA" <<'PYFIRSTFIT'
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import zipfile

pin,code_arg,out_arg,train_arg,train_sha,validation_arg,validation_sha=sys.argv[1:]
code,out=Path(code_arg).resolve(),Path(out_arg).resolve()
train_path,validation_path=Path(train_arg),Path(validation_arg)
started=time.monotonic(); deadline=started+970.
host=socket.gethostname()
if sys.platform!='linux' or not host.startswith('job-'):
    raise RuntimeError('Run only in the actual allocated Linux scheduler container')
logs=out/'logs'; logs.mkdir()
portable=out/'portable'; portable.mkdir()
inputs=out/'inputs'; inputs.mkdir()
teacher_sha='f2fb39571cc3d0e809dfdc454f7f6df7f48e585fc972fd4c2fb8c83d36e63a42'
records=[]; failure=None; input_receipts={}; artifact_audits=[]


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def remaining():
    return deadline-time.monotonic()


def run_owned(command,label,limit):
    log_path=logs/f'{label}.log'
    start=time.monotonic()
    print(json.dumps(dict(stage='start',label=label,remaining_seconds=remaining(),command=command)),flush=True)
    with log_path.open('w') as stream:
        process=subprocess.Popen(command,cwd=code,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        timed_out=False
        try:
            code_value=process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out=True
            # This group was created here for this exact owned subprocess.
            os.killpg(process.pid,signal.SIGTERM)
            try:
                code_value=process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                code_value=process.wait(timeout=5)
    result=dict(label=label,command=command,returncode=code_value,timed_out=timed_out,
        process_seconds=time.monotonic()-start,log=str(log_path.relative_to(out)))
    print(json.dumps(dict(stage='end',**result)),flush=True)
    write_json(out/f'{label}-process.json',result)
    return result


try:
    if digest(train_path)!=train_sha or digest(validation_path)!=validation_sha:
        raise ValueError('Reviewed train/validation index SHA256 differs')
    train=json.loads(train_path.read_text()); validation=json.loads(validation_path.read_text())
    if (train['partition']!='train' or validation['partition']!='validation'
            or validation['validation_iteration']!=10
            or set(train['case_ids']) & set(validation['case_ids'])
            or train['teacher_recovery_sha256']!=teacher_sha
            or validation['teacher_recovery_sha256']!=teacher_sha
            or train['interface']!=validation['interface']
            or validation['source_method']!='ppo'):
        raise ValueError('Require the qualified disjoint train/PPO10-validation banks with the same SAC5 teacher')
    for label,path,sha,metadata in (('train',train_path,train_sha,train),
                                   ('validation',validation_path,validation_sha,validation)):
        shutil.copy2(path,inputs/f'{label}-index.json')
        input_receipts[label]=dict(path=str(path),sha256=sha,partition=metadata['partition'],
            source_method=metadata['source_method'],teacher_method=metadata['teacher_method'],
            teacher_recovery_sha256=metadata['teacher_recovery_sha256'],
            counts=metadata['counts'],case_ids=metadata['case_ids'],
            validation_iteration=metadata.get('validation_iteration'))
    write_json(out/'input-data.json',input_receipts)
    write_json(out/'allocation-context.json',dict(hostname=host,pid=os.getpid(),
        requested_resources=dict(gpus=1,cpus=8,memory_gib=32,shared_memory_gib=8,scheduler_minutes=20),
        source_commit=pin,started_at_utc=datetime.now(timezone.utc).isoformat(),
        global_budget_seconds=970,execution='four independent sequential processes',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        additional_simulator_actions=0,additional_vla_queries=0))
    probe=run_owned([sys.executable,str(code/'scripts/[train]gpu_probe.py'),'--expected-count','1',
        '--output',str(out/'gpu-probe.json'),'--purpose','training','--baige-container'],
        'gpu-probe',min(60.,remaining()-70.))
    if probe['returncode']!=0 or probe['timed_out']:
        raise RuntimeError('Actual allocation/CUDA probe failed; no fitting started')
    for objective in ('ad','dpt'):
        for mode in ('full','static'):
            label=f'{objective}-{mode}'; model_out=out/f'fit-{label}'
            limit=min(270.,remaining()-70.)
            if limit<60.:
                records.append(dict(label=label,status='not_started_global_wall_budget',completed=False,
                    requested_updates=100,committed_updates=0,model_directory=str(model_out.relative_to(out))))
                continue
            command=[sys.executable,'-m','fluffy_bassoon.meta_rl.robotwin_history_train',
                '--train-index',str(train_path),'--train-sha256',train_sha,
                '--validation-index',str(validation_path),'--validation-sha256',validation_sha,
                '--output',str(model_out),'--source-commit',pin,'--expected-hostname',host,
                '--objective',objective,'--mode',mode,'--device','cuda','--threads','4',
                '--seed','0','--updates','100','--batch-size','8','--max-window','128',
                '--width','128','--layers','2','--heads','4','--context-length','300',
                '--learning-rate','3e-4','--validation-batches','4','--eval-every','25',
                '--milestones','25','100','--max-seconds','240']
            result=run_owned(command,label,limit)
            summary_path=model_out/'summary.json'
            summary=json.loads(summary_path.read_text()) if summary_path.is_file() else None
            recovery=model_out/'latest.pt'
            row=dict(label=label,objective=objective,mode=mode,seed=0,requested_updates=100,
                process=result,summary=summary,completed=bool(summary and summary['completed']),
                status='completed' if summary and summary['completed'] else 'bounded_partial_fit',
                model_directory=str(model_out.relative_to(out)),recovery_present=recovery.is_file())
            records.append(row)
            if result['returncode']!=0 and not result['timed_out']:
                row['status']='failed'; raise RuntimeError(f'{label} failed; inspect its owned process log')
            if not recovery.is_file():
                row['status']='failed_no_committed_recovery'
                raise RuntimeError(f'{label} has no complete committed recovery')
            for candidate in sorted(model_out.glob('candidate*.pt')):
                shutil.copy2(candidate,portable/f'{label}--{candidate.name}')
            write_json(out/'fit-progress.json',records)
    if digest(train_path)!=train_sha or digest(validation_path)!=validation_sha:
        raise ValueError('Reviewed immutable dataset indexes changed during fitting')
except Exception as error:
    failure=dict(type=type(error).__name__,message=str(error))
    traceback.print_exc()
finally:
    # Read-only metadata/CRC audit inside this same allocation. No policy is
    # instantiated or evaluated here; CUDA fitting already occurred above.
    import torch
    for row in records:
        directory=out/row['model_directory']; recovery=directory/'latest.pt'
        if not recovery.is_file():
            continue
        try:
            saved=torch.load(recovery,map_location='cpu',weights_only=False)
            contract=saved['contract']; config=contract['config']; update=int(saved['update'])
            finite=all(bool(value.isfinite().all()) for value in saved['policy'].values())
            with zipfile.ZipFile(recovery) as archive:
                crc_ok=archive.testzip() is None
            optimizer_steps=sorted(set(float(value['step']) for value in saved['optimizer']['state'].values()))
            expected_config=dict(width=128,layers=2,heads=4,context_length=300)
            if (saved['format']!='robotwin_native_history_resume_v1' or not finite or not crc_ok
                    or not 1<=update<=100 or contract['source_commit']!=pin
                    or contract['seed']!=0 or contract['mode']!=row['mode']
                    or contract['objective']!=row['objective']
                    or any(config[key]!=value for key,value in expected_config.items())
                    or contract['budget']['updates']!=100 or contract['budget']['batch_size']!=8
                    or contract['budget']['max_window']!=128
                    or not saved['optimizer']['state']
                    or not {'sampler','python','numpy','torch','cuda'}<=set(saved['rng'])
                    or any(not math.isfinite(step) or not 0<step<=update for step in optimizer_steps)):
                raise ValueError('Complete recovery audit failed')
            row['committed_updates']=update
            row['fit_positions']=int(saved['fit_positions'])
            row['optimizer_state_count']=len(saved['optimizer']['state'])
            row['optimizer_step_values']=optimizer_steps
            row['recovery_sha256']=digest(recovery)
            row['recovery_bytes']=recovery.stat().st_size
            row['recovery_integrity_pass']=True
            if row.get('completed') and update!=100:
                raise ValueError('Completed summary differs from committed update counter')
            artifact_audits.append(dict(model=row['label'],update=update,finite_policy=finite,
                recovery_zip_crc_ok=crc_ok,rng_keys=sorted(saved['rng']),
                optimizer_state_count=len(saved['optimizer']['state']),optimizer_step_values=optimizer_steps))
        except Exception as error:
            row['recovery_integrity_pass']=False
            failure=failure or dict(type=type(error).__name__,message=str(error),model=row['label'])
    completed=sum(bool(row.get('completed') and row.get('recovery_integrity_pass')) for row in records)
    status='failed' if failure else ('completed_all_four' if completed==4 else 'bounded_partial_fit')
    report=dict(status=status,source_commit=pin,models=records,recovery_audits=artifact_audits,
        completed_models=completed,requested_models=4,elapsed_seconds=time.monotonic()-started,
        input_data=input_receipts,failure=failure,
        additional_simulator_actions=0,additional_vla_queries=0,
        scientific_claim='offline native AD/DPT fitting only; no robot success or ICRL claim',
        accounting='fit_positions are repeated supervised positions, not new robot decisions; unique source cost is shared')
    write_json(out/'summary.json',report)
    files=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name not in ('console.log','artifact-hashes.json','exit-code.txt'):
            files.append(dict(path=str(path.relative_to(out)),bytes=path.stat().st_size,sha256=digest(path)))
    write_json(out/'artifact-hashes.json',dict(source_commit=pin,files=files))
    print(json.dumps(dict(status=status,completed_models=completed,elapsed_seconds=report['elapsed_seconds'],
                         additional_simulator_actions=0,additional_vla_queries=0)),flush=True)
sys.exit(1 if failure else 0)
PYFIRSTFIT
