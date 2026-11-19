"""Compare sparse logged native labels with matched branch labels on physical robot retries."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'libero-decision-pretraining-20261119-r1'
sys.path.insert(0,str(OLD))
from reader import Reader, UPDATES, LEARNING_RATE, read_trials

ARCHIVE = Path('/mowenjie/fluffy-bassoon-2/outputs')/OLD.name/'menu02/episodes'
MODES = ('default','paced')
LOG_SEEDS = (101,102,103,104)
FIT_SEEDS = (41,42)


@dataclass
class Public:
    history: torch.Tensor
    lengths: torch.Tensor
    query: torch.Tensor
    reward: torch.Tensor

    def subset(self, ids):
        return Public(*(v[ids] for v in (self.history,self.lengths,self.query,self.reward)))


def load(split,tasks):
    folder=ARCHIVE/split
    modes=json.loads((ARCHIVE/'design.json').read_text())['modes']
    p=read_trials(folder,modes,tasks)
    records=json.loads((folder/'progress.json').read_text())
    assert len(records)==len(p.keys)==32
    rewards=torch.tensor([[r['support']['success'],1.] for r in records],dtype=torch.float32)
    inputs=Public(p.history,p.lengths,p.query,rewards)
    ys=p.outcomes[:,[modes.index(m) for m in MODES]].float().numpy()
    controls=np.array([[r['support']['controls'],*(r['queries'][m]['controls'] for m in MODES)] for r in records])
    return inputs,ys,controls,p.keys,p.delayed


def design(keys,seed):
    rng=np.random.default_rng(seed)
    action=rng.integers(0,2,len(keys))
    two=[];three=[]
    for task in sorted({k[:2] for k in keys}):
        ids=[i for i,k in enumerate(keys) if k[:2]==task]
        start=rng.choice([0,3])
        pair=[i for i in ids if (keys[i][2]==start)==(keys[i][3]=='BASE')]
        assert len(pair)==2
        two.extend(pair);three.extend(pair+[int(rng.choice(sorted(set(ids)-set(pair))))])
    all_ids=np.arange(len(keys))
    return action,dict(log32=all_ids,full32=all_ids,log24=np.array(sorted(three)),full16=np.array(sorted(two)))


def forward(model,x,retain,order=None):
    if order is None:
        order=torch.arange(len(x.query))
    reward=x.reward[order] if retain else torch.zeros_like(x.reward)
    return model(x.history[order],x.lengths[order],torch.cat([x.query,reward],dim=1),retain=retain)


def train(x,labels,rows,actions,seed,current):
    torch.manual_seed(seed)
    stats=SimpleNamespace(history=x.history,lengths=x.lengths,query=torch.cat([x.query,x.reward],dim=1))
    model=Reader(stats,2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)
    for _ in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        pred=forward(model,x,False)[rows,actions]
        loss=F.binary_cross_entropy_with_logits(pred,labels)
        if not current:
            full=forward(model,x,True)[rows,actions]
            loss=(loss+F.binary_cross_entropy_with_logits(full,labels))/2
        loss.backward()
        optimizer.step()
    return model.eval()


def fit_group(spec):
    stem,seed,subset,actions,train_x,observed,dev,opposite,out=spec
    torch.set_num_threads(1)
    # The worker receives only revealed labels. It never receives the complete outcome matrix.
    rows,arms=actions
    x=train_x.subset(subset)
    rows=torch.tensor(rows);arms=torch.tensor(arms)
    labels=torch.tensor(observed)
    pred={}
    for current in (False,True):
        model=train(x,labels,rows,arms,seed,current)
        with torch.inference_mode():
            for name,keep,order in ([('current',False,None)] if current else
                                      [('retained',True,None),('cleared',False,None),('swapped',True,opposite)]):
                prob=forward(model,dev,keep,order).sigmoid()
                pred[name]=prob.tolist()
    (out/f'{stem}.json').write_text(json.dumps(pred)+'\n')
    return stem


def main(out):
    split=json.loads((HERE.parent/'libero-meta-learning-20261117-r1/SPLITS.json').read_text())
    fit_tasks=split['development_scaling']['task_subsets']['8']
    for suite,tasks in fit_tasks.items():
        assert not set(tasks)&set(split['development'][suite]+split['test'][suite])
    x,y,cost,keys,lag=load('fitting',fit_tasks)
    dev,yd,_,devkeys,dlag=load('development',split['development'])
    assert set(k[:2] for k in keys).isdisjoint(k[:2] for k in devkeys)
    opposite=torch.tensor([devkeys.index((*k[:3],'S20' if k[-1]=='BASE' else 'BASE')) for k in devkeys])
    directory=out/'predictions'
    directory.mkdir()
    tasks=[];designs={}
    for seed in LOG_SEEDS:
        action,subsets=design(keys,seed)
        for regime,ids in subsets.items():
            if regime=='full32' and seed!=LOG_SEEDS[0]:
                continue
            if regime.startswith('log'):
                rows=np.arange(len(ids));arms=action[ids]
            else:
                rows=np.repeat(np.arange(len(ids)),2);arms=np.tile([0,1],len(ids))
            observed=y[ids[rows],arms].copy()
            key=f'{regime}-{seed}'
            designs[key]=dict(histories=len(ids),episodes=len(ids)+len(rows),
                              controls=int(cost[ids,0].sum()+cost[ids[rows],1+arms].sum()),
                              observed=[[int(ids[r]),int(a),int(v)] for r,a,v in zip(rows,arms,observed)])
            for fitseed in FIT_SEEDS:
                tasks.append((f'{key}-{fitseed}',fitseed,ids,(rows,arms),x,observed,dev,opposite,directory))
    (out/'design.json').write_text(json.dumps(dict(keys=keys,designs=designs),indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=3) as pool:
        for name in pool.map(fit_group,tasks):
            print(name,flush=True)
    def preference(probs):
        probs=np.array(probs)
        return np.where(probs[:,0]==probs[:,1],.5,(probs[:,1]>probs[:,0]).astype(float))
    def native(choice):
        return (1-choice)*yd[:,0]+choice*yd[:,1]
    fixed=int(y.mean(axis=0).argmax())
    by_timing=[int(y[lag==v].mean(axis=0).argmax()) if (lag==v).any() else fixed for v in (False,True)]
    public_lag=native(np.array(by_timing)[dlag.astype(int)])
    report=dict(keys=devkeys,development_hits=yd.tolist(),fitted_lag_actions=by_timing,
                lag=public_lag.tolist(),motor_ceiling=int(yd.max(axis=1).sum()),runs={})
    for seed in LOG_SEEDS:
        for regime in ('log32','full32','log24','full16'):
            primary_seed=LOG_SEEDS[0] if regime=='full32' else seed
            key=f'{regime}-{primary_seed}'
            for fitseed in FIT_SEEDS:
                pred=json.loads((directory/f'{key}-{fitseed}.json').read_text())
                scored={m:native(preference(v)) for m,v in pred.items()}
                success={m:float(v.sum()) for m,v in scored.items()}
                score_brier={m:float(np.mean((np.array(v)-yd)**2)) for m,v in pred.items()}
                report['runs'][f'{regime}-{seed}-{fitseed}']=dict(success=success,brier=score_brier,
                    native={m:v.tolist() for m,v in scored.items()},
                    choices={m:preference(v).tolist() for m,v in pred.items()})
    (out/'scores.json').write_text(json.dumps(report)+'\n')


if __name__=='__main__':
    main(Path(sys.argv[1]))
