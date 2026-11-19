"""Aggregate paired choices across physical cases and log / initialization draws."""
import json
from pathlib import Path
import sys
import numpy as np

REGIMES=('log32','full32','log24','full16')
SEEDS=(101,102,103,104)
FITS=(41,42)


def stats(x):
    a=np.asarray(x,dtype=float)
    return dict(mean=float(a.mean()),min=float(a.min()),max=float(a.max()))


def main(worker,out):
    result=json.loads((worker/'scores.json').read_text())
    design=json.loads((worker/'design.json').read_text())
    observed=design['designs']
    archive=worker.parents[1]/'libero-decision-pretraining-20261119-r1/reader01/summary.json'
    reference=json.loads(archive.read_text())
    def lag_map(panel,keys):
        readings={tuple(r['case']):int(r['choices']['legacy_lag']=='paced') for r in reference[panel]['detail']}
        return np.array([readings[tuple(k)] for k in keys])
    train_lag=lag_map('fitting',design['keys'])
    dev_lag=lag_map('dev_without_model',result['keys'])
    y=np.array(result['development_hits'],float)
    task_keys=sorted({tuple(k[:2]) for k in result['keys']})
    subsets={f'{suite}/{task}':np.array([k[:2]==[suite,task] for k in result['keys']]) for suite,task in task_keys}
    arms={}
    for re in REGIMES:
        all_obs=[];controls=[];tables=[]
        scores={};per_log={};per_fit={}
        for s in SEEDS:
            d=observed[f'{re}-{101 if re=="full32" else s}']
            ledger=d['observed'];controls.append(d['controls'])
            # A public lag estimate chooses one of two tables. The posterior mean
            # counts only revealed outcomes, using a Beta(1/2, 1/2) prior in each cell.
            count=np.zeros((2,2));sums=count.copy()
            for i,a,hit in ledger:
                count[train_lag[i],a]+=1
                sums[train_lag[i],a]+=hit
            means=(sums+.5)/(count+1)
            slot=means[dev_lag,1]-means[dev_lag,0]
            prob=np.where(slot==0,.5,slot>0)
            tables.append(float((y[:,0]+prob*(y[:,1]-y[:,0])).sum()))
            all_obs.append(count.tolist())
            for f in FITS:
                r=result['runs'][f'{re}-{s}-{f}']
                for m,tot in r['success'].items():scores.setdefault(m,[]).append(tot)
                per_log.setdefault(s,[]).append(r['success']['retained'])
                per_fit.setdefault(f,[]).append(r['success']['retained'])
        arms[re]=dict(episodes=d['episodes'],controls=stats(controls),native={k:stats(v) for k,v in scores.items()},
            public_lag_table=stats(tables),observed_by_lag_action=all_obs,
            mean_by_log={k:float(np.mean(v)) for k,v in per_log.items()},
            mean_by_fit={k:float(np.mean(v)) for k,v in per_fit.items()})
    comparisons={}
    for a,b in [('full32','log32'),('full16','log24')]:
        changes=[];gains=[];loss=[];memory=[]
        for s in SEEDS:
            for f in FITS:
                ra=result['runs'][f'{a}-{s}-{f}']['native']
                rb=result['runs'][f'{b}-{s}-{f}']['native']
                delta=np.array(ra['retained'])-np.array(rb['retained'])
                changes.append(delta)
                gains.append(float(delta.clip(min=0).sum()));loss.append(float((-delta).clip(min=0).sum()))
                memory.append(float(delta.sum()-sum(ra['cleared'])+sum(rb['cleared'])))
        v=np.stack(changes)
        comparisons[f'{a}_minus_{b}']=dict(native=stats(v.sum(axis=1)),wins=stats(gains),losses=stats(loss),
            history_gain_difference=stats(memory),by_task={t:stats(v[:,mask].sum(axis=1)) for t,mask in subsets.items()})
    summary=dict(regimes=arms,comparisons=comparisons,bench=dict(tasks=len(task_keys),cases=len(y),
        discordant=int((y[:,0]!=y[:,1]).sum()),both_fail=int(np.all(y==0,axis=1).sum()),
        both_succeed=int(np.all(y==1,axis=1).sum()),default=int(y[:,0].sum()),paced=int(y[:,1].sum()),
        known_lag=sum(result['lag']),ceiling=result['motor_ceiling']))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main(*map(Path,sys.argv[1:]))
