#!/usr/bin/env python3
"""Bounded two-learner/six-simulators-each test in one exclusive allocation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path('/mowenjie/fluffy-bassoon-2')
METHODS = ('ppo', 'ppo_reset')
ACTIONS_PER_ARM = 10_800
MAX_SECONDS = 2700
GIB = 1 << 30


def commands(worktree, output, source, *, python=sys.executable):
    model = ROOT / 'cache/robotwin/pi05-motus-effc6e1a-r1'
    native = ROOT / 'sources/robotwin-baseline-smoke-20261028-r1'
    shared = [python, str(worktree/'scripts/diagnostics/robotwin_noise_compare.py'),
        '--source-commit', source, '--model-path', str(model), '--norm-stats',
        str(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'),
        '--robotwin-root', str(native/'RoboTwin'), '--rlinf-root', str(native/'RLinf'),
        '--source-steps', str(ACTIONS_PER_ARM), '--source-iterations', '1',
        '--source-eval-iterations', '1', '--benchmark-episode-limits',
        '--ppo-trials-per-update', '6', '--collector-lanes', '6',
        '--collector-group-seconds', '600', '--train-episodes', '4',
        '--ppo-epochs', '5', '--ppo-minibatch-trials', '16', '--learning-rate', '1e-6',
        '--ppo-max-update-kl', '.1', '--native-horizon', '32', '--episode-limit', '600',
        '--seed', '20261030', '--record-corpus', '--final-observation-only',
        '--capture-qualification', str(ROOT/'outputs/robotwin-capture-qual-20261029-r1/qualification/qualification.json'),
        '--native-episodes', '0', '--source-validation-episodes', '0',
        '--stage-source-validation-only', '--wall-seconds', '2340',
        '--boundary-reserve-seconds', '180']
    return {method: shared + ['--methods', method, '--output', str(output/method),
        '--campaign-root', str(output/('ledger-'+method)),
        '--campaign-id', output.name+'-'+method,
        '--campaign-max-actions', str(ACTIONS_PER_ARM)] for method in METHODS}


def sample_gpu(uuid):
    line = subprocess.check_output(['nvidia-smi', '--id='+uuid,
        '--query-gpu=memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    used, total, utilization = map(float, line.split(','))
    return dict(used_bytes=int(used*(1<<20)), total_bytes=int(total*(1<<20)),
                utilization_percent=utilization, used_gib=used/1024)


def owned_members(proc, output_token):
    """A new session is retained by its descendants after its leader aborts."""
    if str(output_token) not in ' '.join(map(str,proc.args)):
        raise RuntimeError('Missing original owned-session identity token')
    result=[]
    listing=subprocess.check_output(['ps','-e','-o','pid=,sid=,pgid=,uid=,stat='],text=True)
    for line in listing.splitlines():
        pid,sid,pgid,uid,state=line.split()
        if int(sid)==proc.pid and not state.startswith('Z'):
            if int(uid)!=os.getuid() or int(pgid)!=proc.pid:
                raise RuntimeError('Owned session identity changed; refusing signal')
            result.append(int(pid))
    return result


def stop_owned(processes, output, *, grace_seconds=60, term_seconds=10):
    """Clean only recorded new sessions, including orphaned native workers."""
    for method, proc in processes.items():
        owned_members(proc,output/method)  # Validate before the first signal.
        if proc.poll() is None:
            try:
                command = Path(f'/proc/{proc.pid}/cmdline').read_bytes().replace(b'\0', b' ')
            except FileNotFoundError:
                if proc.poll() is not None:
                    continue
                raise
            if str(output/method).encode() not in command:
                raise RuntimeError('Owned-process identity changed; refusing signal')
            try:os.kill(proc.pid, signal.SIGINT)
            except ProcessLookupError:pass
    def remaining():
        return [(method,proc) for method,proc in processes.items()
                if owned_members(proc,output/method)]
    def wait_until(seconds):
        deadline=time.monotonic()+seconds
        while remaining() and time.monotonic()<deadline:time.sleep(.1)
    wait_until(grace_seconds)
    for method,proc in remaining():
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    wait_until(term_seconds)
    for method,proc in remaining():
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    wait_until(5)
    if remaining():raise RuntimeError('Owned worker session survived forced cleanup')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--worktree',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-probe',type=Path,required=True)
    parser.add_argument('--maximum-seconds',type=int,default=2520)
    args=parser.parse_args()
    if (not 60<=args.maximum_seconds<=2520 or args.output.exists()
            or subprocess.check_output(['git','-C',str(args.worktree),'rev-parse','HEAD'],text=True).strip()!=args.source_commit):
        parser.error('Fresh pinned output and bounded runtime required')
    probe=json.loads(args.gpu_probe.read_text())
    if probe.get('status')!='passed' or len(probe.get('devices',[]))!=1:
        raise RuntimeError('One attested allocated GPU is required')
    args.output.mkdir(parents=True,exist_ok=False)
    cmds=commands(args.worktree,args.output,args.source_commit)
    admission=dict(source_commit=args.source_commit,initialization='fresh controllers; seed20261030',
        maximum_new_source_actions=2*ACTIONS_PER_ARM,maximum_scheduler_gpu_seconds=MAX_SECONDS,
        learners=2,simulators_per_learner=6,global_trials_per_update=6,episodes_per_trial=4,
        update_attempts=2,maximum_optimizer_steps=10,target_used_gib=[70,80],
        abort_above_used_gib=85,commands=cmds,active_production_modified=False,
        prior_qualification_envelope_reused=False)
    (args.output/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    processes={}; streams={}; started=time.monotonic(); samples=[]; reason=None
    try:
        for method, command in cmds.items():
            streams[method]=(args.output/(method+'-console.log')).open('w')
            env=dict(os.environ,FB2_ROBOTWIN_RENDER_RECORDS=str(args.output/method/'renderer-bindings'))
            processes[method]=subprocess.Popen(command,cwd=args.worktree,env=env,
                stdout=streams[method],stderr=subprocess.STDOUT,start_new_session=True)
        with (args.output/'gpu-memory.jsonl').open('w',buffering=1) as trace:
            while any(proc.poll() is None for proc in processes.values()):
                failed=[method for method,proc in processes.items() if proc.poll() not in (None,0)]
                if failed:
                    reason='arm_failed:'+','.join(failed);break
                row=sample_gpu(probe['devices'][0]['uuid'])
                row.update(elapsed_seconds=time.monotonic()-started,
                    utc=datetime.now(timezone.utc).isoformat(),
                    parent_pids={method:proc.pid for method,proc in processes.items() if proc.poll() is None})
                trace.write(json.dumps(row)+'\n'); samples.append(row)
                if row['used_bytes']>min(85*GIB,.9*row['total_bytes']):
                    reason='memory_headroom_limit'; break
                if row['elapsed_seconds']>=args.maximum_seconds:
                    reason='wall_time_limit'; break
                time.sleep(1)
    except BaseException as error:
        reason=f'{type(error).__name__}: {error}'
        raise
    finally:
        # A native parent abort can leave a live session even after poll()
        # returns. Always inspect our sessions before releasing the allocation.
        stop_owned(processes,args.output)
        for proc in processes.values():
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        for stream in streams.values():stream.close()
        result=dict(admission=admission,stop_reason=reason,
            elapsed_seconds=time.monotonic()-started,
            return_codes={method:proc.poll() for method,proc in processes.items()},
            peak_used_gib=max((row['used_gib'] for row in samples),default=None),
            peak_utilization_percent=max((row['utilization_percent'] for row in samples),default=None))
        (args.output/'packed-result.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if not reason and all(proc.returncode==0 for proc in processes.values()) else 3


if __name__=='__main__':
    raise SystemExit(main())
