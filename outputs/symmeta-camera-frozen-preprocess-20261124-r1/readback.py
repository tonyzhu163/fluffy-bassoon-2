"""Independent readback of the primary yaw contrasts from individual prediction exports."""
from collections import defaultdict
import gzip
import hashlib
from itertools import combinations_with_replacement
import json
from math import factorial, prod
from pathlib import Path
from statistics import mean
import sys


def read(path):
    return json.load(gzip.open(path,'rt'))


def yaw(r):
    return abs((r['true_yaw_deg']-r['yaw_deg']+180)%360-180)


def exact_interval(task_effects):
    # Exhaust the multinomial weights for a whole-task nonparametric bootstrap.
    n=len(task_effects)
    weighted=[]
    for sample in combinations_with_replacement(range(n),n):
        count=factorial(n)//prod(factorial(sample.count(i)) for i in set(sample))
        weighted.append((mean(task_effects[i] for i in sample),count))
    total=n**n
    assert sum(c for _,c in weighted)==total
    sorted_dist=sorted(weighted)
    def quantile(q):
        cum=0
        for x,n_occurs in sorted_dist:
            cum+=n_occurs
            if cum>=q*total:
                return x
    return [quantile(.025),quantile(.975)]


def check(pred,summary):
    rows=read(pred)
    s=json.loads(summary.read_text())
    assert s['source_sha256']==hashlib.sha256(pred.read_bytes()).hexdigest()
    panel=s['panel']
    views={'roll+20','roll-20','roll+45','roll-45'} if panel=='digital' else {'orbit+','orbit-'}
    pair=defaultdict(dict)
    for row in rows:
        if row['view'] in views:
            key=(row['suite'],row['task_id']),row['index'],row['seed'],row['view']
            assert row['model'] not in pair[key]
            pair[key][row['model']]=row
    assert len(pair)==s['supports']*3*len(views)
    scores={}
    for a,b,field in [('frozen_warp','shipped','warp_minus_shipped'),
                      ('frozen_warp_opposite','frozen_warp','wrong_minus_true'),
                      ('frozen_warp_Aflow','frozen_warp','Aflow_minus_nativeDIS')]:
        by_task=defaultdict(list)
        for (task,idx,seed,view),models in pair.items():
            left,right=models[a],models[b]
            assert all(left[k]==right[k] for k in ('observed_sha256','source_sha256','sensor_sha256','true_yaw_deg'))
            by_task[task].append(yaw(left)-yaw(right))
        task_means=[mean(x) for x in by_task.values()]
        val=mean(task_means)
        ref=s['metrics']['all']['primary'][field]['yaw_mae_deg']
        assert abs(ref['mean']-val)<1e-10
        assert all(abs(ref['task'][f'{t[0]}/{t[1]}']-mean(x))<1e-10 for t,x in by_task.items())
        scores[field]=dict(mean=val,exact_task_bootstrap_ci95=exact_interval(task_means),
                           registered_20000_draw_ci95=ref['ci95'])
    return dict(supports=s['supports'],tasks=len(by_task),predictions=len(rows),scores=scores)


def historical(proof,full):
    old={(r['index'],r['seed'],r['model'],r['view']):r for r in read(proof)}
    names=('source_sha256','observed_sha256','calibration_sha256','before_sha256','after_sha256',
           'dis_sha256','sensor_sha256','yaw_deg','xy_scale','gripper_sign')
    diffs={k:0 for k in names}
    matches=0
    for row in read(full):
        key=tuple(row[k] for k in ('index','seed','model','view'))
        if key in old:
            matches+=1
            for k in names:
                diffs[k]+=int(row[k]!=old[key][k])
    assert matches==len(old)
    return dict(paired_rows=matches,mismatches=diffs)


def verify_refits(panel, pred, refit, output):
    released=json.loads(refit.with_name(f'{panel}-receipt.json').read_text())
    assert released['sha256']==hashlib.sha256(refit.read_bytes()).hexdigest()
    seal_path=refit.parents[1]/'fit01/sealed.json'
    assert released['seal']==hashlib.sha256(seal_path.read_bytes()).hexdigest()
    seal=json.loads(seal_path.read_text())
    frozen={(r['index'],r['seed'],r['view'],r['model']):r for r in read(pred)}
    selected=[]
    task_effects=defaultdict(lambda:defaultdict(list))
    views={'roll+20','roll-20','roll+45','roll-45'} if panel=='digital' else {'orbit+','orbit-'}
    for r in read(refit):
        if r['regime']=='familiar' or r['calibration']!='true' or r['model'] not in ('A','B'):
            continue
        name=f"{r['model']}-seed{r['seed']}-step{seal['steps']}.msgpack"
        assert Path(r['checkpoint_path']).name==name
        assert r['checkpoint_sha256']==seal['checkpoint_sha256'][name]
        key=r['index'],r['seed'],r['view']
        for fmodel in ('frozen_warp','frozen_warp_Aflow'):
            f=frozen[(*key,fmodel)]
            assert all(r[k]==f[k] for k in ('source_sha256','native_view_sha256','observed_sha256',
                     'calibration_sha256','suite','task_id','embodiment','true_yaw_deg','true_gain','true_sign'))
            if r['view'] in views:
                task_effects[f"{fmodel}_minus_{r['model']}"][r['task']].append(yaw(f)-yaw(r))
        selected.append(r)
    assert len(selected)==2*len([r for r in frozen.values() if r['model']=='frozen_warp'])
    with gzip.open(output/f'{panel}-refit-factors.json.gz','wt',compresslevel=4) as f:
        json.dump(selected,f,separators=(',',':'),allow_nan=False)
    comparisons={}
    for contrast,bytask in task_effects.items():
        means=[mean(v) for v in bytask.values()]
        comparisons[contrast]=dict(mean=mean(means),exact_task_bootstrap_ci95=exact_interval(means),
                                   per_task={k:mean(v) for k,v in bytask.items()})
    return dict(receipt=released, refit_seal=seal, full_factor_rows=len(selected),primary_yaw=comparisons)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('evidence',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('--digital-refit',type=Path)
    ap.add_argument('--physical-refit',type=Path)
    a=ap.parse_args()
    result={}
    for panel in ('digital','physical'):
        d=a.evidence/panel
        proof=a.evidence/'proof'
        if not d.is_dir():
            d=a.evidence/(panel+'01')
            proof=a.evidence/'proof01'
        pred=d/f'{panel}-predictions.json.gz'
        result[panel]=check(pred,d/f'{panel}-summary.json')
        result[f'{panel}_proof_overlap']=historical(proof/pred.name,pred)
        refit=getattr(a,panel+'_refit')
        if refit:
            result[panel+'_refit']=verify_refits(panel,pred,refit,a.output)
    (a.output/'readback.json').write_text(json.dumps(result,indent=2)+'\n')
