"""Source-only completion readouts for paired interrupted practices and unchanged native queries."""
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs')
BOWL=ROOT/'libero-matched-zero-corrections-20261119-r1/matched01/episodes'
BOTTLE=ROOT/'libero-first-attempt-20261119-r1/transfer02/episodes/bottle'


def features(actions,states):
    width=np.abs(np.diff(states[:,-2:],axis=1)).ravel()
    closed=actions[:,-1]>0
    ix=np.convolve(closed.astype(int),np.ones(11,dtype=int),'valid')==11
    # Jaw widening during intended closure is directly observable on a physical robot.
    widening=(width[10:]-width[:-10])[ix]
    peak=widening.max(initial=0.)
    summary=np.array([peak, np.quantile(widening,.95) if len(widening) else 0.,
                      width[closed].mean() if closed.any() else 0.,width[closed].std() if closed.any() else 0.,
                      np.ptp(states[:,2]),np.max(np.linalg.norm(states[:,:2]-states[0,:2],axis=1))])
    return dict(history=summary,commands=actions.reshape(30,10,7).mean(axis=1).ravel())


def load(directory):
    data={}
    for line in (directory/'branches.jsonl').read_text().splitlines():
        r=json.loads(line)
        f,p,g,j,b=(r[k] for k in ('family','pose','goal','draw','interrupted'))
        with np.load(directory/f'{f}-p{p}-g{g}-d{j}-{int(b)}.npz') as t:
            assert np.array_equal(t['hits'].any(axis=0),r['ever'])
            r.update(features(t['issued'],t['states']))
        data[f,p,g,j,b]=r
    return data


def smooth(labels):
    return (sum(labels)+.5)/(len(labels)+1)


def fit(source):
    y=np.array([t['ever'][t['goal']] for t in source],dtype=float)
    poses=np.array([t['pose'] for t in source])
    models={}
    for mode in ('commands','history'):
        X=np.array([t[mode] for t in source])
        scale=X.std(axis=0)
        scale[scale<1e-6]=1.
        x=X/scale
        distances=np.mean((x[:,None]-x[None,:])**2,axis=-1)
        errors=[]
        for bw in (.25,1.,4.):
            pred=[neighbor(d[poses!=p],y[poses!=p],bw) for d,p in zip(distances,poses)]
            errors.append(np.mean((y-pred)**2))
        models[mode]=(x,scale,y,(.25,1.,4.)[int(np.argmin(errors))])
    x=np.array([t['history'][0] for t in source])
    vals=np.unique(x)
    best=None
    for cut in np.r_[-1.,(vals[:-1]+vals[1:])/2,vals[-1]+1.]:
        split=x>cut
        low,high=smooth(y[~split]),smooth(y[split])
        if high>low:continue
        loss=np.mean((y-np.where(split,high,low))**2)
        if best is None or loss<best[0]:best=(loss,float(cut),float(low),float(high))
    models['jaw']=best[1:]
    models['prior']=[smooth([t['ever'][g] for t in source if t['goal']==g]) for g in (0,1)]
    models['off']=smooth([t['ever'][1-t['goal']] for t in source])
    return models


def neighbor(d,y,bw):
    w=np.exp(-(d-d.min())/bw)
    return (w@y+.5)/(w.sum()+1)


def predict(m,r):
    result={}
    for mode in ('commands','history'):
        x,scale,y,bw=m[mode]
        d=np.mean((x-r[mode]/scale)**2,axis=1)
        result[mode]=float(neighbor(d,y,bw))
    cut,low,high=m['jaw']
    result.update(jaw=high if r['history'][0]>cut else low,prior=m['prior'][r['goal']])
    return result


def query(family,p):
    hits=[]
    for g in (0,1):
        if family=='bowl':
            split='fitting' if p<40 else 'evaluation'
            with np.load(BOWL/split/f'p{p}'/f'g{(4,8)[g]}-query5/tape.npz') as z:native=z['native']
        else:
            paths=list((BOTTLE/f's{p}-BASE').glob(f'support-g{g}-*'))
            paths=[x for x in paths if not json.loads((x/'result.json').read_text())['paced']]
            assert len(paths)==1
            rec=json.loads((paths[0]/'result.json').read_text())
            assert rec['scene']==p and rec['goal']==g and rec['variant']=='BASE'
            native=np.load(paths[0]/'evaluator-predicates.npy')
        assert native.shape==(300,2)
        hits.append(native)
    hits=np.stack(hits)
    return hits.any(axis=1),hits[:,-1]


def decide(rewards, likelihood):
    p=np.prod(np.where(np.asarray(rewards)[:,None],likelihood,1-likelihood),axis=0)
    return float(p[0]>p[1]) if not np.isclose(p[0],p[1],rtol=0,atol=1e-10) else .5


def score(data, model, family, poses, draws):
    poses=list(poses)
    panel={k:r for k,r in data.items() if k[0]==family and k[1] in poses}
    assert len(panel)==len(poses)*2*draws*2
    pred={k:predict(model,r) for k,r in panel.items()}
    calibrated={m:dict(brier=float(np.mean([(v[m]-panel[k]['ever'][k[2]])**2 for k,v in pred.items()])),
                       correct=sum((v[m]>=.5)==panel[k]['ever'][k[2]] for k,v in pred.items()))
                for m in ('commands','history','jaw','prior')}
    rows=[]
    for i,p in enumerate(poses):
        s,q=poses[(i+1)%len(poses)],poses[(i+2)%len(poses)]
        if family=='bottle':q-=6
        hit,final=query(family,q)
        for first,j,k in product((0,1),range(draws),range(draws)):
            def matrices(b,c):
                keys=((family,p,first,j,b),(family,s,1-first,k,c))
                actual=np.array([panel[key]['ever'] for key in keys])
                fields={}
                for mode in ('commands','history','jaw','prior'):
                    mat=np.full((2,2),model['off'])
                    for idx,key in enumerate(keys):mat[idx,key[2]]=pred[key][mode]
                    fields[mode]=mat
                return actual,fields
            branches={bc:matrices(*bc) for bc in product((False,True),repeat=2)}
            for (b,c),(actual,fields) in branches.items():
                for hidden in (0,1):
                    rewards=actual[:,hidden]
                    policy={m:decide(rewards,mat) for m,mat in fields.items()}
                    policy['true']=decide(rewards,actual)
                    policy['donor']=np.mean([decide(rewards,mats['history']) for _,mats in branches.values()])
                    policy['opposite']=decide(rewards,branches[not b,not c][1]['history'])
                    rows.append(dict(base=(p,s,q,first,j,k,*map(int,rewards)),branch=(b,c),hidden=hidden,zero=not rewards.any(),
                                     hit=hit[:,hidden],final=final[:,hidden],policy=policy,
                                     identified=int(np.all(actual==rewards[:,None],axis=0).sum())==1))
    def summarize(rs):
        methods=list(rs[0]['policy']) if rs else []
        def total(m,field):
            return float(sum((v:=r['policy'][m])*r[field][0]+(1-v)*r[field][1] for r in rs))
        bounds={}
        for name in ('starts','executions'):
            groups=defaultdict(list)
            for r in rs:groups[r['base'] if name=='starts' else (r['base'],r['branch'])].append(r['hit'])
            bounds[name]=int(sum(np.sum(v,axis=0).max() for v in groups.values()))
        reversals=defaultdict(set)
        for r in rs:
            if r['identified'] and r['hit'].sum()==1 and r['hit'][r['hidden']]:
                reversals[r['base']].add(r['hidden'])
        return dict(n=len(rs),native={m:total(m,'hit') for m in methods},final={m:total(m,'final') for m in methods},
                    correct={m:float(sum(r['policy'][m] if r['hidden']==0 else 1-r['policy'][m] for r in rs)) for m in methods},
                    bounds=bounds,oracle=int(sum(r['hit'].any() for r in rs)),reversal_groups=sum(len(v)==2 for v in reversals.values()))
    pairs=[(r,panel[(*k[:-1],True)]) for k,r in panel.items() if not k[-1]]
    counts=dict(pairs=len(pairs),triggered=sum(b['start'] is not None for a,b in pairs),
                loss_on_interrupt=sum(b['lost_contact'] is True for a,b in pairs),loss_on_sham=sum(a['lost_contact'] is True for a,b in pairs),
                sham_own=sum(a['ever'][a['goal']] for a,b in pairs),interrupt_own=sum(b['ever'][b['goal']] for a,b in pairs),
                off_target=sum(r['ever'][1-r['goal']] for r in panel.values()))
    return dict(physical=counts,completion=calibrated,all=summarize(rows),zero=summarize([r for r in rows if r['zero']]),
                by_query={str(q):summarize([r for r in rows if r['base'][2]==q]) for q in sorted({r['base'][2] for r in rows})})


if __name__=='__main__':
    inp,out=map(Path,sys.argv[1:])
    data=load(inp)
    model=fit([r for (f,p,g,j,b),r in data.items() if f=='bowl' and p in range(32,36)])
    config=dict(jaw=model['jaw'],bandwidths={m:model[m][-1] for m in ('commands','history')},prior=model['prior'],off=model['off'])
    results=dict(fit=config)
    for label,family,poses,draws in [('source','bowl',range(32,36),4),('bowl','bowl',range(44,48),4),('bottle','bottle',range(6,12),1)]:
        results[label]=score(data,model,family,poses,draws)
        print(label,results[label]['physical'],results[label]['zero'])
    (out/'result.json').write_text(json.dumps(results,indent=2,default=lambda x:x.item())+'\n')
