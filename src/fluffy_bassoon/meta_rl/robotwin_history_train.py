"""Bounded native Transformer AD/DPT full/static fitting; no simulator calls."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from .robotwin_history import (MODES,NativeHistoryConfig,NativeHistoryPolicy,export_policy,
    load_dataset,sample_batch,validate_partitions)
from .vla_history_train import _rng_state,_restore_rng
from .warmstart import atomic_save,checked_load


RESUME_FORMAT='robotwin_native_history_resume_v1'


@torch.no_grad()
def validate(policy,trials,*,mode,batch_size,max_window,batches):
    policy.eval(); sampler=np.random.default_rng(734001)
    values=[float(policy.sequence_nll(*sample_batch(trials,policy,sampler,batch_size=batch_size,
        max_window=max_window,mode=mode),mode=mode)) for _ in range(batches)]
    if not all(math.isfinite(value) for value in values): raise FloatingPointError('Nonfinite native validation NLL')
    return float(np.mean(values))


def fit(train,train_metadata,validation,validation_metadata,output,*,source_commit,mode='full',
        config=None,seed=0,updates=500,batch_size=8,max_window=128,learning_rate=3e-4,
        validation_batches=4,eval_every=25,milestones=(25,100,500),max_seconds=1800.,
        device='cpu',resume_sha256=None):
    validate_partitions(train_metadata,validation_metadata)
    if (mode not in MODES or not 0 <= seed < 2**32 or any(type(n) is not int or n < 1
            for n in (updates,batch_size,max_window,validation_batches,eval_every))
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(max_seconds) or max_seconds <= 0
            or tuple(sorted(set(milestones))) != tuple(milestones)
            or any(type(n) is not int or not 1 <= n <= updates for n in milestones)):
        raise ValueError('Invalid explicit native offline fitting budget')
    config=config or NativeHistoryConfig()
    output=Path(output).resolve()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sampler=np.random.default_rng(seed)
    policy=NativeHistoryPolicy(config).to(device)
    # Only source-training observations enter these fixed buffers.
    policy.fit_observation_normalization(train)
    optimizer=torch.optim.AdamW(policy.parameters(),lr=learning_rate,weight_decay=0.)
    contract=dict(source_commit=source_commit,config=asdict(config),mode=mode,seed=seed,
        interface=train_metadata['interface'],train_index_sha256=train_metadata['index_sha256'],
        validation_index_sha256=validation_metadata['index_sha256'],
        train_metadata=train_metadata,validation_metadata=validation_metadata,
        objective=train_metadata['selected_objective'],validation_objective=validation_metadata['selected_objective'],
        optimizer=dict(name='AdamW',learning_rate=learning_rate,weight_decay=0.,max_grad_norm=1.),
        budget=dict(updates=updates,batch_size=batch_size,max_window=max_window,eval_every=eval_every,
            validation_batches=validation_batches,milestones=list(milestones),max_seconds_per_invocation=max_seconds),
        context='native3090 inputs; each sampled window starts empty; memory capacity and gradient window are separate',
        policy_family='Gaussian physical-noise target NLL; deployed proposals clipped to[-1.5,1.5]',
        normalization='2062-feature mean/std fitted on source training only; fixed for validation/deployment',
        static_control='same architecture/data/seed/budget; no recurrent state or previous noise/reward/boundary; native clocks retained',
        deployment_weights='frozen; context only',status='candidate; NLL does not establish robot improvement or ICRL')
    start_update,fit_positions,elapsed_before,best=0,0,0.,math.inf
    if resume_sha256 is not None:
        saved=checked_load(output/'latest.pt',resume_sha256,weights_only=False)
        if saved.get('format') != RESUME_FORMAT or saved.get('contract') != contract:
            raise ValueError('Native resume contract differs')
        policy.load_state_dict(saved['policy'],strict=True); optimizer.load_state_dict(saved['optimizer'])
        _restore_rng(saved['rng'],sampler)
        start_update,fit_positions,elapsed_before,best=(saved[key] for key in
            ('update','fit_positions','elapsed_seconds','best_validation_nll'))
        if start_update >= updates: raise ValueError('Native fit already reached the requested endpoint')
        for partial in output.glob('*.partial'):
            partial.rename(partial.with_name(partial.name+f'-uncommitted-{time.time_ns()}'))
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'contract.json').write_text(json.dumps(contract,indent=2,allow_nan=False)+'\n')
    started=time.monotonic(); latest_metrics=None
    for update in range(start_update+1,updates+1):
        policy.train(); optimizer.zero_grad(set_to_none=True)
        batch=sample_batch(train,policy,sampler,batch_size=batch_size,max_window=max_window,mode=mode)
        loss=policy.sequence_nll(*batch,mode=mode)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite native supervised loss')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if any(not value.isfinite().all() for value in policy.state_dict().values()):
            raise FloatingPointError('Native fit produced nonfinite parameters')
        fit_positions+=int(batch[2].sum())
        elapsed=elapsed_before+time.monotonic()-started
        terminal=update==updates or time.monotonic()-started>=max_seconds
        if update==1 or update%eval_every==0 or update in milestones or terminal:
            heldout=validate(policy,validation,mode=mode,batch_size=batch_size,max_window=max_window,batches=validation_batches)
            latest_metrics=dict(update=update,train_nll=float(loss.detach()),gradient_norm=float(norm),
                validation_nll=heldout,fit_positions=fit_positions,elapsed_seconds=elapsed)
            with (output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(latest_metrics)+'\n')
            manifest=dict(contract=contract,interface=contract['interface'],mode=mode,update=update,
                validation_nll=heldout,fit_positions=fit_positions,
                unique_training_decisions=sum(len(trial['observations']) for trial in train))
            if update in milestones: export_policy(output/f'candidate-{update:06d}.pt',policy,manifest)
            if heldout<best:
                best=heldout; export_policy(output/'candidate.pt',policy,manifest)
            atomic_save(output/'latest.pt',dict(format=RESUME_FORMAT,contract=contract,update=update,
                policy=policy.state_dict(),optimizer=optimizer.state_dict(),rng=_rng_state(sampler),
                fit_positions=fit_positions,elapsed_seconds=elapsed,best_validation_nll=best))
            print(json.dumps(latest_metrics),flush=True)
        if terminal: break
    summary=dict(completed=update==updates,update=update,fit_positions=fit_positions,metrics=latest_metrics,
        unique_training_decisions=sum(len(trial['observations']) for trial in train),additional_simulator_actions=0,
        additional_vla_queries=0,status='offline fit only; frozen robot evaluation is separate')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return policy.eval().requires_grad_(False),summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('train-index','validation-index','output'): parser.add_argument('--'+name,type=Path,required=True)
    for name in ('train-sha256','validation-sha256','source-commit','expected-hostname'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--objective',choices=('ad','dpt'),required=True)
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--device',default='cpu'); parser.add_argument('--seed',type=int,default=0)
    for name,default in (('updates',500),('batch-size',8),('max-window',128),('eval-every',25),
        ('validation-batches',4),('width',128),('layers',2),('heads',4),('context-length',300),('threads',4)):
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--max-seconds',type=float,default=1800.)
    parser.add_argument('--milestones',type=int,nargs='+',default=[25,100,500])
    parser.add_argument('--resume-sha256')
    args=parser.parse_args(argv)
    if sys.platform!='linux' or socket.gethostname()!=args.expected_hostname or args.threads<1:
        raise ValueError('Fit only on the explicitly identified allocated Linux host')
    root=Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=args.source_commit
            or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()):
        raise ValueError('Require the exact clean reviewed fitting commit')
    torch.set_num_threads(args.threads)
    train,train_metadata=load_dataset(args.train_index,args.train_sha256,args.objective)
    validation_objective='behavior_cloning' if args.objective=='ad' else 'dpt'
    validation,validation_metadata=load_dataset(args.validation_index,args.validation_sha256,validation_objective)
    fit(train,train_metadata,validation,validation_metadata,args.output,source_commit=args.source_commit,
        mode=args.mode,config=NativeHistoryConfig(width=args.width,layers=args.layers,heads=args.heads,
            context_length=args.context_length),seed=args.seed,updates=args.updates,batch_size=args.batch_size,
        max_window=args.max_window,learning_rate=args.learning_rate,validation_batches=args.validation_batches,
        eval_every=args.eval_every,milestones=tuple(args.milestones),max_seconds=args.max_seconds,device=args.device,
        resume_sha256=args.resume_sha256)


if __name__=='__main__': main()
