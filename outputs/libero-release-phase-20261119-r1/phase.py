"""Test event-aligned wrist features on fresh fixed-command simulator executions."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import json
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'libero-retry-value-20261119-r1'))
from experiment import completion_preferences, histories, prior, read_tape
from timing import PILOT,execute

SOURCE=prior.ROOT/'libero-retry-value-20261119-r1/timing01/episodes'
MODES=('summary','clock','phase')
DIMS=(6,8,11)
POSES=range(15,20)
QUERIES=range(24,28)
ORDERS=[(p,15+(p-14)%5,q) for p,q in product(POSES,QUERIES)]


def features(tape):
    actions,states=tape['issued'],tape['states']
    base=prior.features(actions,states)['history']
    width=np.abs(states[:,-2]-states[:,-1])
    closed=actions[:,-1]>0
    eligible=np.flatnonzero(np.convolve(closed.astype(int),np.ones(11,int),'valid')==11)
    extra=np.zeros(5)
    if len(eligible):
        diff=width[eligible+10]-width[eligible]
        if diff.max()>0:
            t=int(eligible[diff.argmax()])
            opened=np.flatnonzero(~closed[:t+1])
            start=int(opened[-1])+1 if len(opened) else 0
            z=states[:,2]
            extra[:]=(t,t-start,z[t]-z[start],z[start:t+1].max()-z[t],z[t]-z[max(start,t-10)])
    return np.r_[base,extra]


def fit(out):
    xs=[];ys=[];off=[]
    for p,g,m in product(range(10,15),(0,1),('normal','early','late')):
        tape=read_tape(SOURCE/f'p{p}-g{g}-{m}.npz')
        xs.append(features(tape))
        hits=tape['hits'].any(axis=0)
        ys.append(hits[g]);off.append(hits[1-g])
    x=np.stack(xs)
    mean=x.mean(axis=0)
    sd=x.std(axis=0)
    scale=np.where(sd<1e-6,1.,sd)*np.sqrt([6]*6+[2]*2+[3]*3)
    y=np.array(ys,dtype=float)
    bank=dict(mean=mean,scale=scale,base=y.mean(),off=(sum(off)+.5)/(len(y)+1),floor=.5/(len(y)+1))
    for mode,n in zip(MODES,DIMS):
        X=(x[:,:n]-mean[:n])/scale[:n]
        bank[mode]=np.linalg.solve(X.T@X+len(X)*.01*np.eye(n),X.T@(y-y.mean()))
    np.savez(out/'model.npz',**bank)


def prob(bank,tapes):
    x=np.stack([features(t) for t in tapes.values()])
    X=(x-bank['mean'])/bank['scale']
    return {m:np.clip(bank['base']+X[:,:n]@bank[m],bank['floor'],1-bank['floor']) for m,n in zip(MODES,DIMS)}


def policies(d,tapes,bank,probs):
    result={k:v for k,v in completion_preferences(d,tapes).items() if k in ('completion_joint','table')}
    for name,per_tape in probs.items():
        own=per_tape[d['keys']]
        rewards=d['rewards']
        own_likelihood=np.where(rewards,own,1-own)
        other_likelihood=np.where(rewards,bank['off'],1-bank['off'])
        odds=own_likelihood*other_likelihood[:,::-1]
        result[name]=np.where(np.isclose(odds[:,0],odds[:,1],rtol=0,atol=1e-10),.5,odds[:,0]>odds[:,1])
    return result


def panel(folder,which,outcomes,bank):
    tapes={(p,g,0,b):read_tape(folder/f'p{p}-g{g}-{which[int(b)]}.npz')
           for p,g,b in product(POSES,(0,1),(False,True))}
    d=histories(tapes,outcomes,1,orders=ORDERS)
    probs=prob(bank,tapes)
    ps=policies(d,tapes,bank,probs)
    return tapes,d,probs,ps


def analyze(folder,which,outcomes,bank,records,saved):
    tapes,d,probs,ps=panel(folder,which,outcomes,bank)
    assert all(np.array_equal(p,saved[k]) for k,p in ps.items())
    idx={k:i for i,k in enumerate(tapes)}
    donor=np.zeros(len(d['h']))
    for a,b in product((False,True),repeat=2):
        replace=np.array([idx[p,g,j,(a,b)[g]] for p,g,j,_ in tapes])
        donor+=policies(dict(d,keys=replace[d['keys']]),tapes,bank,probs)['phase']/4
    ps['phase_donor']=donor
    y=np.array([t['hits'][:,g].any() for (p,g,j,b),t in tapes.items()])
    fired={(r['pose'],r['goal'],r['mode']) for r in records if r['start'] is not None}
    eligible=np.array([all((p,g,m) in fired for m in which if m!='normal') for p,g,j,b in tapes])
    eligible=eligible[d['keys']].all(axis=1)
    def score(mask):
        native=d['native'][mask]
        final=d['final'][mask]
        def values(hit):
            return {k:hit[:,1]+p[mask]*(hit[:,0]-hit[:,1]) for k,p in ps.items()}
        total=values(native)
        paired={}
        for k in ('summary','clock','completion_joint','table','phase_donor'):
            v=total['phase']-total[k]
            paired[k]=dict(gains=float(v.clip(min=0).sum()),losses=float((-v).clip(min=0).sum()))
        return dict(n=int(mask.sum()),ceiling=float(native.max(axis=1).sum()),
                    native={k:float(v.sum()) for k,v in total.items()},
                    final={k:float(v.sum()) for k,v in values(final).items()},
                    correct={k:float(np.where(d['h']==0,p,1-p)[mask].sum()) for k,p in ps.items()},paired=paired)
    return dict(all=score(np.ones(len(y[d['keys']]),bool)),zero=score(~d['rewards'].any(axis=1)),
                triggered=score(eligible),triggered_zero=score(eligible & ~d['rewards'].any(axis=1)),
                by_query={q:score(d['q']==q) for q in QUERIES},
                brier={k:float(np.mean((v-y)**2)) for k,v in probs.items()})


def main(out):
    fit(out)
    with np.load(out/'model.npz') as z:
        bank=dict(z)
    folder=out/'episodes'
    folder.mkdir()
    scenarios={'stages':('early','late'),'early':('normal','early'),'late':('normal','late')}
    with ProcessPoolExecutor(max_workers=4) as pool:
        practices=list(pool.map(execute,[(folder,p,g,m) for p,g,m in product(POSES,(0,1),('normal','early','late'))]))
        empty={q:(np.zeros((2,2)),np.zeros((2,2))) for q in QUERIES}
        saved={}
        for name,which in scenarios.items():
            tapes,d,probs,ps=panel(folder,which,empty,bank)
            saved[name]={k:v.tolist() for k,v in ps.items()}
        with (out/'choices.json').open('x') as f:
            json.dump(saved,f)
        jobs=[]
        for q,g in product(QUERIES,(0,1)):
            path=PILOT/'livepair01/episodes/default_support'/f'b{q-24}-BASE'/f'goal{g}'/f's{q}-BASE-g{g}-gain1'
            jobs.append((folder,q,g,'query',path))
        queries=list(pool.map(execute,jobs))
    outcomes={}
    for q in QUERIES:
        hits=[read_tape(folder/f'p{q}-g{g}-query.npz')['hits'] for g in (0,1)]
        outcomes[q]=(np.stack([h.any(axis=0) for h in hits]),np.stack([h[-1] for h in hits]))
    report=dict(practices=practices,queries=queries)
    frozen=json.loads((out/'choices.json').read_text())
    for name,which in scenarios.items():
        report[name]=analyze(folder,which,outcomes,bank,practices,frozen[name])
        print(name,report[name]['all'],flush=True)
    (out/'phase.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main(Path(sys.argv[1]))
