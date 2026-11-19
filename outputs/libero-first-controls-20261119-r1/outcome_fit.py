"""Learn the binary cadence decision directly from archived native returns."""
from pathlib import Path
import json,sys
import numpy as np


def ratio(path):
    with np.load(path/'public.npz') as a:
        diff=np.linalg.norm(a['positions'][:7].astype(float)-a['initial_position'],axis=1)
        return float(diff[6]/max(1e-12,max(diff[:6])))


def fit(panel):
    rows=json.loads((panel/'progress.json').read_text())
    x=np.array([ratio(panel/r['trial']/'support') for r in rows])
    rewards=np.array([[r['queries'][mode]['success'] for mode in ('default','paced')] for r in rows])
    fixed=int(rewards.sum(axis=0).argmax())
    score=int(rewards[:,fixed].sum())
    best=(score,True,0.)
    model=dict(cut=None,low=fixed,high=fixed,train_score=score)
    values=np.unique(x)
    for a,b in zip(values,values[1:]):
        cut=float((a+b)/2)
        left=x<=cut
        lo=int(rewards[left].sum(axis=0).argmax())
        hi=int(rewards[~left].sum(axis=0).argmax())
        value=int(rewards[left,lo].sum()+rewards[~left,hi].sum())
        rank=(value,False,float(b-a))
        if rank>best:
            best=rank
            model=dict(cut=cut,low=lo,high=hi,train_score=value)
    return dict(model=model,fixed_score=score,train_examples=len(rows))


def action(m,x):
    return ('default','paced')[m['low'] if m['cut'] is None or x<=m['cut'] else m['high']]


def evaluate(fit,archive,live):
    m=fit['model']
    dev=json.loads((archive/'progress.json').read_text())
    result={'archived_prior':sum(r['queries'][action(m,ratio(archive/r['trial']/'support'))]['success'] for r in dev)}
    rows=json.loads(live.read_text())
    scored=[]
    for r in rows:
        option=action(m,r['prior_score'])
        record=dict(task=r['suite']+'/'+str(r['task']),regime=r['timing'],prior=option,hit=r['hits'][option])
        for key,opening in [('ten_first','paced'),('default_first','default')]:
            mode=action(m,r['choices'][key]['score'])
            if mode==opening:
                record[key]=r['hits'][opening]
            elif mode==r['choices'][key]['mode']:
                record[key]=r['hits'][key]
            else:
                record[key]=None
        scored.append(record)
    result['fresh']=scored
    return result


if __name__=='__main__':
    if sys.argv[1]=='fit':
        result=fit(Path(sys.argv[2]))
    else:
        result=evaluate(json.loads(Path(sys.argv[2]).read_text()),Path(sys.argv[3]),Path(sys.argv[4]))
    print(json.dumps(result,indent=2))
