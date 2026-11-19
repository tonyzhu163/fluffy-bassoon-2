"""Source-fitted completion probabilities from commands and sensed motion."""
from pathlib import Path
import sys

import numpy as np
OLD=Path(__file__).resolve().parents[1]/'libero-interrupted-practice-20261119-r1'
sys.path.insert(0,str(OLD))
import score as prior

MODES=('commands','motion','joint','selected')
BANDS=(.25,1.,4.)


def normalized_dist(a,b,scale):
    return np.mean(((a[:,None]-b[None,:])/scale)**2,axis=-1)


def scale(x):
    s=x.std(axis=0)
    return np.where(s<1e-6,1.,s)


def fit(archive,destination):
    data=prior.load(archive)
    src=[r for (f,p,g,j,b),r in data.items() if f=='bowl' and p in range(32,36)]
    assert len(src)==64 and len({r['pose'] for r in src})==4
    C=np.array([r['commands'] for r in src]); S=np.array([r['history'] for r in src])
    y=np.array([r['ever'][r['goal']] for r in src],dtype=float)
    poses=np.array([r['pose'] for r in src])
    folds=[]
    for p in np.unique(poses):
        fitmask=poses!=p
        folds.append((normalized_dist(C[~fitmask],C[fitmask],scale(C[fitmask])),
                      normalized_dist(S[~fitmask],S[fitmask],scale(S[fitmask])),y[fitmask],y[~fitmask]))
    weights=[];bands=[];losses=[]
    for options in [(0.,),(1.,),(.5,),(.25,.5,.75)]:
        best=None
        for w in options:
            for bw in BANDS:
                errors=[]
                for dc,ds,ys,targets in folds:
                    prob=[prior.neighbor(d,ys,bw) for d in (1-w)*dc+w*ds]
                    errors.extend((prob-targets)**2)
                candidate=(float(np.mean(errors)),w,bw)
                if best is None or candidate<best:best=candidate
        loss,w,bw=best
        weights.append(w);bands.append(bw);losses.append(loss)
    old=prior.fit(src)
    table=np.full((2,2,2),.5)
    for k,v in old['table'].items():table[tuple(map(int,k))]=v
    np.savez(destination,commands=C,motion=S,y=y,cscale=scale(C),sscale=scale(S),weights=weights,
             bands=bands,losses=losses,off=old['off'],jaw=old['jaw'],table=table)


def predict(model,commands,states):
    f=prior.features(commands,states)
    dc=normalized_dist(f['commands'][None,:],model['commands'],model['cscale'])[0]
    ds=normalized_dist(f['history'][None,:],model['motion'],model['sscale'])[0]
    out={m:prior.neighbor((1-w)*dc+w*ds,model['y'],bw)
         for m,w,bw in zip(MODES,model['weights'],model['bands'])}
    cut,low,high=model['jaw']
    out['jaw']=high if f['history'][0]>cut else low
    return out


def choose(values,first,rewards,off):
    likelihood=np.full((2,2),float(off))
    likelihood[0,first],likelihood[1,1-first]=values
    return prior.decide(rewards,likelihood)


if __name__=='__main__':fit(*map(Path,sys.argv[1:]))
