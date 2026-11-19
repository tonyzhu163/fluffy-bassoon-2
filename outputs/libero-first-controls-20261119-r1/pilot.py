"""Fit opening cadence choices on source tasks, checking exact causal branch equivalence."""
from itertools import product
from pathlib import Path
import json
import sys
import numpy as np

MODES=('default','paced','short_wait','gripper_wait')


def load(panel):
    rows=[]
    variants={}
    for case in json.loads((panel/'progress.json').read_text()):
        name=case['trial']
        traces={}
        hits=[]
        for mode in ('support',*MODES):
            path=panel/name/mode
            rec=json.loads((path/'result.json').read_text())
            with np.load(path/'public.npz') as d:
                keys=('issued','positions','gripper','rewards','initial_position','opening')
                vals=[d[k][:5].copy() if k in keys[:4] else d[k].copy() for k in keys]
                assert len(vals[0])==5 and np.any(d['rewards']>0)==rec['success']
            if mode in MODES:
                assert rec==case['queries'][mode]
                hits.append(int(rec['success']))
            else:
                assert rec==case['support']
            traces[mode]=vals
        for mode in MODES[1:]:
            assert case['queries'][mode]['identity']==case['queries'][MODES[0]]['identity']
            for a,b in zip(traces[mode],traces[MODES[0]]):
                np.testing.assert_array_equal(a,b,err_msg=name+'/'+mode)
        def features(tape):
            a,p,_,_,initial,_=tape
            dist=np.linalg.norm(np.diff(np.vstack((initial,p)).astype(float),axis=0),axis=1).cumsum()
            command=np.linalg.norm(a[:,:3].astype(float),axis=1).cumsum()
            return np.divide(dist,command,out=np.zeros_like(dist),where=command>0).tolist()
        c=case['queries']['default']['case']
        row=dict(trial=name,task=f"{c['suite']}/{c['task_id']}",pose=c['init_state_id'],timing=c['variant'],
                 prior=features(traces['support']),current=features(traces['default']),hits=hits)
        rows.append(row)
        variants.setdefault((row['task'],row['pose']),{})[row['timing']]=case['queries']['default']
    assert len(variants)==16 and len(rows)==32
    parity={k:sum(v['BASE']['identity'][k]==v['S20']['identity'][k] for v in variants.values())
            for k in ('physics','public','prompt')}
    parity['proposal']=sum(v['BASE']['opening_sha256']==v['S20']['opening_sha256'] for v in variants.values())
    return rows,parity


def fit(rows,key,n,objective):
    x=np.array([r[key][n-1] for r in rows])
    y=np.array([r['hits'] for r in rows])
    slow=np.array([r['timing']=='S20' for r in rows])
    const=int(y.sum(axis=0).argmax())
    best=(int(y[:,const].sum()),True,0.) if objective=='native' else (-1,False,0.)
    model=dict(cut=None,slow=const,fast=const)
    values=np.unique(x)
    for lo,hi in zip(values,values[1:]):
        cut=float((lo+hi)/2)
        group=x<=cut
        a=int(y[group].sum(axis=0).argmax())
        b=int(y[~group].sum(axis=0).argmax())
        score=int(y[group,a].sum()+y[~group,b].sum()) if objective=='native' else int((group==slow).sum())
        rank=(score,False,float(hi-lo))
        if rank>best:
            best=rank
            model=dict(cut=cut,slow=a,fast=b)
    return model


def decide(model,x):
    return model['slow'] if model['cut'] is None or x<=model['cut'] else model['fast']


def study(train,test,n,objective):
    models={k:fit(train,k,n,objective) for k in ('prior','current')}
    fixed=int(np.sum([r['hits'] for r in train],axis=0).argmax())
    oracle={g:int(np.sum([r['hits'] for r in train if r['timing']==g],axis=0).argmax()) for g in ('BASE','S20')}
    other={(r['task'],r['pose'],r['timing']):r for r in test}
    trials=[]
    for r in test:
        donor=other[r['task'],r['pose'],'S20' if r['timing']=='BASE' else 'BASE']
        modes={k:decide(m,r[k][n-1]) for k,m in models.items()}
        modes.update(fixed=fixed,opposite=decide(models['prior'],donor['prior'][n-1]),given_timing=oracle[r['timing']])
        trials.append(dict(trial=r['trial'],task=r['task'],timing=r['timing'],choices={k:MODES[a] for k,a in modes.items()},
                           hits={k:r['hits'][a] for k,a in modes.items()},
                           correct_class={k:(m['cut'] is not None and (r[k][n-1]<=m['cut'])==(r['timing']=='S20')) for k,m in models.items()}))
    return dict(models=models,scores={k:sum(t['hits'][k] for t in trials) for k in trials[0]['hits']},
                timing_labels_correct={k:sum(t['correct_class'][k] for t in trials) for k in models},trials=trials)


if __name__=='__main__':
    root=Path(sys.argv[1])
    train,first=load(root/'fitting')
    test,second=load(root/'development')
    assert {r['task'] for r in train}.isdisjoint(r['task'] for r in test)
    out=dict(parity=dict(fit=first,development=second),development=test,fit=train,results={})
    for objective,n in product(('timing','native'),range(1,6)):
        out['results'][f'{objective}{n}']=study(train,test,n,objective)
    with Path(sys.argv[2]).open('x') as f: print(json.dumps(out,indent=2),file=f)
