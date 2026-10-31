#!/usr/bin/env python3
"""Read all12 immutable demo commands on allocated CPU; no simulator/model."""
import argparse
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('allocation','zero-gpu-proof','demos'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    args=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('Allocated Linux only, never Mac array work')
    if not math.isfinite(args.deadline_unix):raise ValueError('Finite shared original clock required')
    from fluffy_bassoon.meta_rl.robotwin_replay_diagnosis import Audit,json_write
    audit=Audit(args.deadline_unix)
    ticket=audit.json(dict(path=str(args.allocation),sha256=args.allocation_sha256))
    proof=audit.json(dict(path=str(args.zero_gpu_proof),sha256=args.zero_gpu_proof_sha256))
    def git(*values):
        audit.guard()
        return subprocess.check_output(['git',*values],text=True,
            timeout=max(.1,min(10,args.deadline_unix-time.time()))).strip()
    if (git('rev-parse','HEAD')!=args.source_commit or git('status','--porcelain')
            or ticket.get('source_commit')!=args.source_commit or ticket.get('hostname')!=socket.gethostname()
            or ticket.get('phase_id')!='frozen-pi05-endtoend-20261031-r1' or ticket.get('gpu_count')!=0
            or not 1<=ticket['maximum_cpu_worker_seconds']<=240
            or ticket['prior_phase_cpu_charge']<827 or ticket['maximum_phase_cpu_worker_seconds']!=1800
            or ticket['prior_phase_cpu_charge']+ticket['maximum_cpu_worker_seconds']>1800
            or Path(ticket['schedule_output']).resolve()!=args.output.resolve()
            or args.output.resolve()!=Path(ticket['output']).resolve()/'schedule'
            or proof.get('passed') is not True or proof.get('hostname')!=socket.gethostname()
            or proof.get('device_count_before_mask')!=0 or proof.get('cuda_initialized_before_mask') is not False
            or proof.get('actual_cpu_cores',0)<2 or proof.get('actual_memory_bytes',0)<(16<<30)
            or ticket.get('zero_gpu_attestation_sha256')!=args.zero_gpu_proof_sha256):
        raise ValueError('Exact original CPU residual/clean source/actual0GPU2CPU16GiB proof required')
    if (ticket['deadline_unix']!=args.deadline_unix
            or not ticket['started_unix']<=time.time()<args.deadline_unix<=
                   ticket['started_unix']+ticket['maximum_cpu_worker_seconds']):
        raise ValueError('Original clock includes setup/tests/preflight; no clock reset')
    # Imports/real-array reads only after actual CPU admission.
    from fluffy_bassoon.meta_rl.robotwin_demo_capacity import validate_demo_manifest,read_episode
    from fluffy_bassoon.meta_rl.robotwin_event_schedule import build_schedule
    demos=validate_demo_manifest(audit.json(dict(path=str(args.demos),sha256=args.demos_sha256)))
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for episode in demos['episodes']:
        audit.guard();ref=demos['actual_whole_files'][episode['id']]
        path=audit.bound(ref['path'],ref['sha256'])
        _,targets=read_episode(path,episode,demos['schema'])
        rows.append(dict(episode_id=episode['id'],split=episode['split'],source=ref,
                         schedule=build_schedule(targets)))
    accepted=all(row['schedule']['admissible'] for row in rows)
    report=dict(format='robotwin_all12_event_schedule_preflight_v1',status='passed' if accepted else 'rejected_schedule',
        source_commit=args.source_commit,demos_sha256=args.demos_sha256,rows=rows,input_files=audit.files,
        source_frames_retained=True,demos_filtered=0,new_robot_actions=0,model_forwards=0,SGD=0,
        interpretation='Schedule admissibility only, not teacher success/physical replay or GPU admission')
    audit.guard();json_write(args.output/'result.json',report)
    if any(k.split('.')[0] in ('torch','jax','rlinf','openpi','sapien') for k in sys.modules):
        raise RuntimeError('Forbidden model imports in schedule-only phase')
    print(json.dumps(dict(status=report['status'],demos=len(rows),filtered=0)))
    if not accepted:raise RuntimeError('Retained fixed bank contains an overlong schedule; no filtering/GPU execution')


if __name__=='__main__':main()
