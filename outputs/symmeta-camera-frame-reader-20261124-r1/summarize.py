"""Paired, equal-task estimates. Keep all repeated views, seeds and controls inside tasks."""
from collections import defaultdict
from functools import lru_cache
import numpy as np


def wrap(x):
    return (x+180)%360-180


def error(row):
    return abs(wrap(row['yaw_deg']-row['true_yaw_deg']))


@lru_cache(maxsize=None)
def draws(n):
    return np.random.default_rng(1124).integers(n,size=(20000,n))


def estimate(values):
    """Values keyed by task and seed, containing matched support/view contributions."""
    t=sorted({t for t,s in values})
    seed=sorted({s for t,s in values})
    mat=np.asarray([[np.mean(values[task,s]) for s in seed] for task in t])
    task_mean=mat.mean(axis=1)
    per_seed=mat.mean(axis=0)
    return dict(mean=float(task_mean.mean()),ci95=np.quantile(task_mean[draws(len(t))].mean(axis=1),[.025,.975]).tolist(),
                per_seed=dict(zip(map(str,seed),per_seed.tolist())),seed_sd=float(per_seed.std(ddof=1)),
                per_task=dict(zip(t,task_mean.tolist())))


def summarize(rows):
    by={(r['index'],r['seed'],r['view'],r['model'],r['calibration']):r for r in rows}
    if len(by)!=len(rows):
        raise ValueError('Duplicate support/seed/view/model/calibration')
    output={}
    splits=['all','ood']
    if rows[0]['panel']=='physical':
        splits.extend(['original','additional'])
    for split in splits:
        subset=[r for r in rows if
                (split!='ood' or r['regime']!='familiar') and
                (split!='original' or r['original_orbit']) and
                (split!='additional' or not r['original_orbit'])]
        if not subset:
            continue
        grouped=defaultdict(lambda:defaultdict(list))
        support_diff=defaultdict(list)
        for r in subset:
            idx,seed,view,model,cal=(r[k] for k in ('index','seed','view','model','calibration'))
            key=(r['task'],seed)
            def add(metric,val):
                grouped[view,metric][key].append(val)
                if view in ('roll+20','roll-20','roll+45','roll-45'):
                    grouped['primary_nonquarter',metric][key].append(val)
            mae=error(r)
            add(f'{model}_{cal}',mae)
            if cal=='true':
                add(f'{model}_yaw_minus_upright',mae-error(by[idx,seed,'nominal',model,'true']))
                add(f'{model}_abs_log_gain_error',abs(np.log(r['xy_scale']/r['true_gain'])))
                add(f'{model}_gripper_accuracy',r['gripper_sign']==r['true_sign'])
            if cal=='opposite':
                good=by[idx,seed,view,model,'true']
                add(f'{model}_wrong_minus_true',mae-error(good))
                add(f'{model}_calibration_yaw_change',abs(wrap(r['yaw_deg']-good['yaw_deg'])))
            if cal=='identity_token':
                ref=by[idx,seed,'nominal',model,'true']
                if any(r[k]!=ref[k] for k in ('yaw_deg','xy_scale','gripper_sign')):
                    raise ValueError('Inverse quarter-turn with identity encoding changed predictions')
            if model=='A' and cal=='true':
                diff=mae-error(by[idx,seed,view,'B','true'])
                add('A_minus_B',diff)
                add('A_minus_shipped_nominal',mae-error(by[idx,seed,'nominal','shipped','true']))
                support_diff[view,idx].append((seed,diff))
                add('known_map_oracle',0.)
                add('identity_map_yaw_error',abs(wrap(r['true_yaw_deg'])))
        metrics={}
        for (view,name),values in grouped.items():
            metrics.setdefault(view,{})[name]=estimate(values)
        output[split]=dict(supports=len({r['index'] for r in subset}),tasks=sorted({r['task'] for r in subset}),
                  views=metrics,support_A_minus_B={f'{v}/{i}':dict(per_seed=dict(x),mean=float(np.mean([d for _,d in x]))) for (v,i),x in support_diff.items()})
    return output
