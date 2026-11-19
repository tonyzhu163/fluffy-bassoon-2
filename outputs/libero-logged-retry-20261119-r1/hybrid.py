"""Evaluate a partially branched archive at matched history and episode budgets."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import torch

from fit import FIT_SEEDS, HERE, LOG_SEEDS, design, fit_group, load


def main(out):
    splits=json.loads((HERE.parent/'libero-meta-learning-20261117-r1/SPLITS.json').read_text())
    x,y,cost,keys,_=load('fitting',splits['development_scaling']['task_subsets']['8'])
    dev,yd,_,devkeys,_=load('development',splits['development'])
    base=json.loads((out.parent/'logged02/scores.json').read_text())
    assert base['keys']==[list(k) for k in devkeys]
    opposite=torch.tensor([devkeys.index((*k[:3],'S20' if k[-1]=='BASE' else 'BASE')) for k in devkeys])
    directory=out/'predictions'
    directory.mkdir()
    jobs=[];ledger={}
    tasks=sorted({k[:2] for k in keys})
    for seed in LOG_SEEDS:
        rng=np.random.default_rng(1000+seed)
        action,subsets=design(keys,seed)
        extra=np.setdiff1d(subsets['log24'],subsets['full16'])
        reduced=np.sort(np.r_[subsets['full16'],rng.choice(extra,5,replace=False)])
        for regime,ids,count in [('mix32',subsets['log32'],8),('mix21',reduced,6)]:
            audit_tasks=rng.choice(len(tasks),count,replace=False)
            audit=[int(rng.choice([i for i,index in enumerate(ids) if keys[index][:2]==tasks[t]])) for t in audit_tasks]
            rows=np.r_[np.arange(len(ids)),audit]
            arms=np.r_[action[ids],1-action[ids[audit]]]
            observed=y[ids[rows],arms].copy()
            stem=f'{regime}-{seed}'
            ledger[stem]=dict(episodes=len(ids)+len(rows),controls=int(cost[ids,0].sum()+cost[ids[rows],1+arms].sum()),
                observed=[[int(ids[r]),int(a),int(v)] for r,a,v in zip(rows,arms,observed)])
            for f in FIT_SEEDS:
                jobs.append((f'{stem}-{f}',f,ids,(rows,arms),x,observed,dev,opposite,directory))
    with ProcessPoolExecutor(max_workers=3) as pool:
        for name in pool.map(fit_group,jobs):
            print(name,flush=True)
    results={}
    for stem,_,_,_,_,_,_,_,_ in jobs:
        pred=json.loads((directory/f'{stem}.json').read_text())
        scored={}
        for k,p in pred.items():
            a=np.array(p)
            choice=np.where(a[:,0]==a[:,1],.5,a[:,0]<a[:,1])
            scored[k]=yd[:,0]+choice*(yd[:,1]-yd[:,0])
        regime,s,f=stem.split('-')
        paired={}
        for other in (('log32','full32') if regime=='mix32' else ('log24','full16')):
            rival=np.array(base['runs'][f'{other}-{s}-{f}']['native']['retained'])
            delta=scored['retained']-rival
            paired[other]=dict(gains=delta.clip(min=0).sum(),losses=(-delta).clip(min=0).sum(),net=delta.sum())
        results[stem]=dict(native={k:v.tolist() for k,v in scored.items()},paired=paired)
    (out/'hybrid.json').write_text(json.dumps(dict(designs=ledger,runs=results))+'\n')


if __name__=='__main__':
    main(Path(sys.argv[1]))
