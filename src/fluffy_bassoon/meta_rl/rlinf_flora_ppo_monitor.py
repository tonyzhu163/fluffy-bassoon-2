"""Readable logs and predeclared DEV-only stopping; no dashboard or model."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone


def stopping_decision(evaluations,*,baseline_success_rate):
    """Same heuristic per arm; repeated10 placements are not50 independent scenes."""
    if not evaluations:return dict(stop=False,reason='no_evaluation')
    if not math.isfinite(baseline_success_rate) or not 0<=baseline_success_rate<=1:raise ValueError('Fixed native baseline rate required')
    for i,row in enumerate(evaluations):
        if (row.get('metric_source')!='simulator_DEV_final_success' or row.get('episodes')!=50
                or type(row.get('successes')) is not int or not 0<=row['successes']<=50
                or row.get('unique_nominal_states')!=10 or row.get('full_boundary_checkpoint') is not True
                or not row.get('checkpoint_sha256') or row.get('health')!='healthy'
                or row['iteration']!=i+1):
            raise ValueError('Healthy ordered simulator DEV evaluations/full checkpoints required; never score staleness as convergence')
    if len(evaluations)>=3 and all(r['successes']>=48 for r in evaluations[-3:]):
        return dict(stop=True,reason='ceiling_48of50_three_consecutive',interpretation='Heuristic on10 repeated DEV placements, not statistical convergence proof')
    if evaluations[-1]['iteration']<6:return dict(stop=False,reason='minimum6_not_reached')
    if len(evaluations)>=4:
        best=[];value=baseline_success_rate
        for row in evaluations:value=max(value,row['successes']/50);best.append(value)
        improvement=best[-1]-best[-4]
        if improvement<.04-1e-12:
            reason='futility' if best[-1]-baseline_success_rate<.10-1e-12 else 'plateau'
            return dict(stop=True,reason=reason,window_improvement_percentage_points=100*improvement,
                interpretation='Pilot resource heuristic; not statistical proof of convergence')
    return dict(stop=False,reason='continuing')


class ProgressLog:
    """Atomic status plus append-only JSONL and readable heartbeats, every30s."""
    def __init__(self,output,arm,*,contract_sha256,interval=30.,clock=time.time):
        if arm not in ('S','F') or not 0<interval<=60:raise ValueError('Matched arm and at-most60s heartbeat interval required')
        self.root=Path(output);self.root.mkdir(parents=True,exist_ok=True)
        self.clock=clock;self.arm=arm;self.interval=interval;self.lock=threading.RLock();self.stop_event=threading.Event()
        self.state=dict(arm=arm,contract_sha256=contract_sha256,phase='preparing',iteration=0,
            episodes_complete=0,labels_complete=0,policy_updates=0,physical_actions=0,aggregate_gpu_seconds=0,
            phase_seconds=dict(rollout=0,flora=0,update=0,eval=0),gpu_memory=dict(current_bytes=None,peak_bytes=None),
            ppo_diagnostics=dict(KL=None,clip_fraction=None,value_loss=None),last_full_recovery=None,
            representative_clean_videos=[],last_progress_unix=clock())
        old=self.root/'status.json'
        if old.exists():
            saved=json.loads(old.read_text())
            if saved['contract_sha256']!=contract_sha256:raise ValueError('Do not overwrite another run contract')
            self.state=saved # Visibility resumes saved totals; authoritative charges remain owner/journals.
        self.thread=None
    def emit(self,*,event='progress',**updates):
        with self.lock:
            now=self.clock()
            if event!='heartbeat':self.state.update(updates);self.state['last_progress_unix']=now
            row=deepcopy(self.state);row.update(event=event,heartbeat_unix=now,progress_age_seconds=now-self.state['last_progress_unix'])
            data=json.dumps(row,sort_keys=True,allow_nan=False)+'\n'
            temporary=self.root/'status.tmp';temporary.write_text(data);temporary.replace(self.root/'status.json')
            with (self.root/'metrics.jsonl').open('a') as stream:stream.write(data);stream.flush()
            stamp=datetime.fromtimestamp(now,timezone.utc).isoformat(timespec='seconds')
            details=' '.join(f'{key}={row[key]}' for key in ('evaluation_successes','best_success_rate',
                'plateau_progress','stop_reason','phase_seconds','ppo_diagnostics','gpu_memory') if row.get(key) is not None)
            print(f"{stamp} arm={self.arm} phase={row['phase']} iter={row['iteration']} episodes={row['episodes_complete']} labels={row['labels_complete']} actions={row['physical_actions']} gpu_s={row['aggregate_gpu_seconds']} progress_age_s={row['progress_age_seconds']:.1f} event={event} {details}",flush=True)
            return row
    def start(self):
        if self.thread is not None:raise ValueError('One progress emitter per owned arm')
        self.emit()
        def heartbeat():
            while not self.stop_event.wait(self.interval):self.emit(event='heartbeat')
        self.thread=threading.Thread(target=heartbeat,daemon=True);self.thread.start();return self
    def close(self):
        self.stop_event.set()
        if self.thread is not None:self.thread.join(timeout=self.interval+1)
        self.emit(event='closed')


def whole_iteration_admission(*,charged_actions,charged_gpu_seconds,projected_actions,projected_gpu_seconds,
        maximum_gpu_seconds=None):
    values=(charged_actions,charged_gpu_seconds,projected_actions,projected_gpu_seconds)
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):raise ValueError('Actual and conservative finite charges required')
    if maximum_gpu_seconds is not None and (not math.isfinite(maximum_gpu_seconds) or maximum_gpu_seconds<=0):
        raise ValueError('An optional owner-bound finite phase ceiling must be positive')
    # Human removed the aggregate10h cap for these two jobs; actual GPU charges
    # remain logged. Each admitted scheduler job still has <=8 GPUs and <8h.
    return charged_actions+projected_actions<=1000000 and (maximum_gpu_seconds is None or
        charged_gpu_seconds+projected_gpu_seconds<=maximum_gpu_seconds)
