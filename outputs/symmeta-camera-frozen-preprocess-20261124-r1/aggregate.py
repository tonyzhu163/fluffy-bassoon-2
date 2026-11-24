"""Standalone task aggregation and identity-checked joins to sealed camreader output."""
import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

METRICS = ('yaw_mae_deg','abs_log_gain_error','gripper_wrong_fraction')


def read(path):
    with gzip.open(path,'rt') as f:
        return json.load(f)


def losses(row):
    return np.array([abs((row['yaw_deg']-row['true_yaw_deg']+180)%360-180),
                     abs(np.log(row['xy_scale']/row['true_gain'])),
                     float(row['gripper_sign']!=row['true_sign'])])


@lru_cache(None)
def draws(n):
    return np.random.default_rng(1124).integers(n, size=(20000,n))


def estimate(values):
    tasks = sorted({task for task,seed in values})
    seeds = [41,42,43]
    matrix = np.stack([[np.mean(values[task,seed],axis=0) for seed in seeds] for task in tasks])
    per_task = matrix.mean(axis=1)
    average = per_task.mean(axis=0)
    # Independently recompute each point from individual task/seed scalar lists.
    for j, estimate_j in enumerate(average):
        independent = statistics.mean(statistics.mean(statistics.mean(float(x[j]) for x in values[t,s])
                                for s in seeds) for t in tasks)
        np.testing.assert_allclose(independent,estimate_j,atol=1e-12)
    cis = np.percentile(per_task[draws(len(tasks))].mean(axis=1),[2.5,97.5],axis=0)
    per_seed = matrix.mean(axis=0)
    return {m:dict(mean=average[j],ci95=cis[:,j].tolist(),
                   task=dict(zip(tasks,per_task[:,j].tolist())),
                   seed=dict(zip(map(str,seeds),per_seed[:,j].tolist()))) for j,m in enumerate(METRICS)}


def identity(r,other,check_H=True):
    fields = ['index','task','seed','view','source_sha256','native_view_sha256','observed_sha256',
              'embodiment','regime','true_yaw_deg','true_gain','true_sign','panel','camera_deg']
    if check_H:
        fields.append('calibration_sha256')
    if any(r[k]!=other[k] for k in fields):
        raise ValueError([(k,r[k],other[k]) for k in fields if r[k]!=other[k]])


def aggregate(path, refit=None):
    rows = read(path)
    panel = rows[0]['panel']
    assert {r['panel'] for r in rows}=={panel}
    own = {(r['index'],r['seed'],r['view'],r['model']):r for r in rows}
    assert len(own)==len(rows)
    competitors = {}
    if refit:
        competitor_rows = [r for r in read(refit) if r['calibration']=='true' and r['model'] in ('A','B','shipped')
                           and r['regime']!='familiar']
        competitors = {(r['index'],r['seed'],r['view'],r['model']):r for r in competitor_rows}
        assert len(competitors)==len(competitor_rows)
    primary = ('roll-45','roll-20','roll+20','roll+45') if panel=='digital' else ('orbit-','orbit+')
    group = defaultdict(lambda:defaultdict(list))
    joined = defaultdict(int)
    for r in rows:
        if r['model']!='frozen_warp':
            continue
        i,s,v = r['index'],r['seed'],r['view']
        own_matches = {model:own[i,s,v,model] for model in ('shipped','frozen_warp_Aflow')}
        if v!='nominal':
            own_matches['frozen_warp_opposite']=own[i,s,v,'frozen_warp_opposite']
        if (i,s,v,'shipped_corrected') in own:
            own_matches['shipped_corrected']=own[i,s,v,'shipped_corrected']
        for m,other in own_matches.items():
            identity(r,other,check_H=m!='frozen_warp_opposite')
        alternatives = {}
        if competitors and (i,s,v,'A') in competitors:
            for m in ('A','B','shipped'):
                alt = competitors[i,s,v,m]
                identity(r,alt)
                alternatives['refit_'+m if m!='shipped' else 'camreader_shipped']=alt
            joined[v]+=1
        measures = {model:losses(other) for model,other in (own_matches|alternatives).items()}
        measures['frozen_warp']=losses(r)
        for m in ['shipped','shipped_corrected','refit_A','refit_B']:
            if m in measures:
                measures['warp_minus_'+m]=losses(r)-measures[m]
        if v!='nominal':
            measures['wrong_minus_true']=measures['frozen_warp_opposite']-losses(r)
        measures['warp_minus_upright']=losses(r)-losses(own[i,s,'nominal','frozen_warp'])
        measures['Aflow_minus_nativeDIS']=measures['frozen_warp_Aflow']-losses(r)
        if alternatives:
            here,other=own_matches['shipped'],alternatives['camreader_shipped']
            measures['camreader_shipped_minus_our_raw']=losses(other)-losses(here)
        splits = ['all']
        if panel=='physical':
            splits += ['historical' if r['original_orbit'] else 'additional',r['regime']]
        for split in splits:
            for target in ([v,'primary'] if v in primary else [v]):
                for m,val in measures.items():
                    group[split,target,m][r['task'],s].append(val)
    if refit and not joined:
        raise ValueError('No matched refitted supports')
    out = {}
    for (split,view,m),values in group.items():
        out.setdefault(split,{}).setdefault(view,{})[m]=estimate(values)
    return dict(panel=panel,supports=len({r['index'] for r in rows}),tasks=sorted({r['task'] for r in rows}),
                source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                refit=str(refit) if refit else None,
                refit_sha256=hashlib.sha256(refit.read_bytes()).hexdigest() if refit else None,
                refit_joined_support_seed_pairs_by_view=dict(joined), bootstrap_seed=1124,draws=20000,
                metrics=out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('predictions',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--refit',type=Path)
    a=p.parse_args()
    result=aggregate(a.predictions,a.refit)
    with a.output.open('x') as f:
        json.dump(result,f,separators=(',',':'),allow_nan=False)
        f.write('\n')
