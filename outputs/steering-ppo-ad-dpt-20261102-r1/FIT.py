"""One independent convergence fit; scheduler boundaries preserve full recovery."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--assembly',type=Path,required=True)
    p.add_argument('--objective',choices=('ad','dpt','dit','ic_iql'),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--context-length',type=int,default=300)
    p.add_argument('--max-eval-horizon',type=int,default=1050);a=p.parse_args()
    import torch
    from fluffy_bassoon.meta_rl.vla_history import file_sha256
    if sys.platform!='linux' or torch.cuda.device_count()!=1:
        raise ValueError('Use exactly one allocated Linux GPU per independently resumable fit')
    if a.context_length<5*((a.max_eval_horizon+49)//50):
        raise ValueError('The declared five physical attempts must fit the model memory')
    code=Path(__file__).resolve().parents[2]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    if head!=a.source_commit or subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip():
        raise ValueError('Require exact clean pinned source')
    binding=json.loads(a.assembly.read_text())
    if binding.get('completed') is not True:raise ValueError('Source corpus is not complete')
    if binding.get('producer_recipe',{}).get('finalized') is not True:
        raise ValueError('Producer iterations/envs per GPU must be finalized before a dependent fit')
    horizons=binding['source_physical_horizons']
    if not horizons or any(type(h) is not int or h<50 or h%50 for h in horizons):
        raise ValueError('Bind actual complete source horizons')
    window_episodes=max(5,math.ceil(5*a.max_eval_horizon/min(horizons)))
    if window_episodes*(max(horizons)//50)>a.context_length:
        raise ValueError('Complete source windows covering the longest eval history must fit memory')
    train=binding['datasets'][a.objective+'_train'];validation=binding['datasets'][a.objective+'_validation']
    for item in (train,validation):
        if file_sha256(item['path'])!=item['sha256']:raise ValueError('Bound dataset changed')
    status=a.output/'training-status.json'
    if status.exists() and json.loads(status.read_text()).get('converged'):
        raise ValueError('Fit already converged; publish its own evaluation, never retrain it')
    probes=a.output.parent/'allocation-probes';probes.mkdir(parents=True,exist_ok=True)
    probe=probes/(a.attempt+'.json')
    if probe.exists():raise ValueError('Never reuse an allocation attempt identity')
    subprocess.run([sys.executable,'-B',str(code/'scripts/[train]gpu_probe.py'),
        '--expected-count','1','--baige-container','--purpose','training','--output',str(probe)],check=True)
    trainer='history_value_train' if a.objective in ('dit','ic_iql') else 'vla_history_train'
    args=[sys.executable,'-B','-m','fluffy_bassoon.meta_rl.'+trainer,
        '--train-data',train['path'],'--train-sha256',train['sha256'],
        '--validation-data',validation['path'],'--validation-sha256',validation['sha256'],
        '--output-dir',str(a.output),'--mode','full','--width','128','--layers','2','--heads','4',
        '--context-length',str(a.context_length),'--backbone','decoder','--max-episodes-per-window',str(window_episodes),
        '--batch-size','8','--learning-rate','0.0003','--eval-every','100','--validation-batches','16',
        '--seed','0','--threads','4','--device','cuda:0','--checkpoint-selection','validation-nll',
        '--until-converged','--minimum-updates','1000','--validation-patience','10',
        '--validation-relative-delta','0.001','--max-elapsed-seconds','12000',
        '--source-commit',a.source_commit,'--expected-hostname',socket.gethostname()]
    if a.objective in ('dit','ic_iql'):args+=['--objective',a.objective]
    if (a.output/'latest.pt').exists():args+=['--resume',str(a.output/'latest.pt')]
    subprocess.run(args,check=True)
    state=json.loads(status.read_text())
    if not state.get('converged'):
        print('Operational checkpoint only; preserve optimizer/sampler/all RNG/plateau and resume identical contract.',flush=True)
        raise SystemExit(75)
    print(json.dumps(dict(converged=True,objective=a.objective,candidate=str(a.output/'candidate.pt'),
        candidate_sha256=file_sha256(a.output/'candidate.pt'),latest_sha256=file_sha256(a.output/'latest.pt'),
        source=a.source_commit,attempt=a.attempt,independent_evaluation_ready=True)),flush=True)


if __name__=='__main__':main()
