"""Train goal-symmetric retry decisions on matched physical outcomes from prior runs."""
from itertools import permutations, product
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'libero-object-history-20261119-r1'))
from evaluate import ARCHIVE, queries, read_practices
from choices import read_tape
from reader import prior, predict, choose

OBJECT = prior.ROOT / 'libero-object-history-20261119-r1'
BANK = ARCHIVE / 'pair01/branches'
MODES = ('rewards', 'commands', 'motion', 'joint')
PENALTIES = (.001, .01, .1)


def archived_panel(family, poses, draws):
    tapes = {(p,g,j,b): read_tape(BANK/f'{family}-p{p}-g{g}-d{j}-{int(b)}.npz')
             for p,g,j,b in product(poses, range(2), range(draws), (False, True))}
    outcomes = {p: prior.query(family, p-6 if family=='bottle' else p) for p in poses}
    return tapes, outcomes


def live_panel(folder, poses, alternate=None, mode='interrupt'):
    tapes = {(p,g,0,b):t for (p,g,b),t in read_practices(folder, poses, alternate, mode).items()}
    outcomes = {p:(x.any(axis=1), x[:,-1]) for p,x in queries(folder, poses).items()}
    return tapes, outcomes


def histories(tapes, outcomes, draws, train=False):
    poses = sorted(outcomes)
    orders = permutations(poses, 3) if train else [
        (p,poses[(i+1)%len(poses)],poses[(i+2)%len(poses)]) for i,p in enumerate(poses)]
    index = {k:i for i,k in enumerate(tapes)}
    complete = {k:t['hits'].any(axis=0) for k,t in tapes.items()}
    rows = []
    for p,s,q in orders:
        for first,j,k,b,c in product(range(2),range(draws),range(draws),(False,True),(False,True)):
            pair = [(p,first,j,b),(s,1-first,k,c)]
            keys = sorted(pair, key=lambda t:t[1])
            for h in (0,1):
                rewards = [complete[key][h] for key in keys]
                rows.append(([index[key] for key in keys], rewards, first, q, h,
                             outcomes[q][0][:,h], outcomes[q][1][:,h]))
    keys, rewards, first, q, h, native, final = (np.asarray(a) for a in zip(*rows))
    raw = [prior.features(t['issued'],t['states']) for t in tapes.values()]
    feats = {k:np.array([t[k] for t in raw]) for k in ('commands','history')}
    return dict(keys=keys, rewards=rewards.astype(int), first=first, q=q, h=h,
                native=native.astype(float), final=final.astype(float), features=feats)


def normalized(data, scales, mode):
    features = data['features']
    blocks = {'rewards':(), 'commands':('commands',), 'motion':('history',), 'joint':('commands','history')}[mode]
    if not blocks:
        return np.empty((len(features['history']),0))
    return np.column_stack([(features[k]-scales[k][0])/scales[k][1]/np.sqrt(features[k].shape[1]*len(blocks))
                            for k in blocks])


def oriented(x, data, goal):
    own = data['keys'][:,goal]
    rival = data['keys'][:,1-goal]
    code = 2*data['rewards'][:,goal] + data['rewards'][:,1-goal]
    return np.column_stack((x[own],x[rival])), code


def fit(data):
    scales = {}
    for key, x in data['features'].items():
        scale=x.std(axis=0)
        scales[key]=np.stack((x.mean(axis=0),np.where(scale<1e-6,1.,scale)))
    bank=dict(scales)
    native=data['native'][:,0]-data['native'][:,1]
    intent=np.where(data['h']==0,1.,-1.)
    y=np.column_stack((native,intent))
    targets=np.vstack((y,-y))
    for mode in MODES:
        x=normalized(data,scales,mode)
        oriented_inputs=[oriented(x,data,g) for g in (0,1)]
        all_x=np.vstack([a for a,c in oriented_inputs])
        all_c=np.concatenate([c for a,c in oriented_inputs])
        for code in range(4):
            keep=all_c==code
            X,Y=all_x[keep],targets[keep]
            for penalty in PENALTIES:
                W=np.zeros((all_x.shape[1]+1,2))
                if len(X):
                    W[0]=Y.mean(axis=0)
                    if X.shape[1]:
                        A=X-X.mean(axis=0)
                        W[1:]=np.linalg.solve(A.T@A + np.eye(A.shape[1])*(penalty*len(A)),A.T@(Y-W[0]))
                        W[0]-=X.mean(axis=0)@W[1:]
                bank[f'{mode}-{code}-{penalty}']=W
    return bank


def preferences(bank,data,mode,penalty):
    x=normalized(data,bank,mode)
    advantage=[]
    for g in (0,1):
        X,code=oriented(x,data,g)
        pred=np.empty((len(X),2))
        for k in range(4):
            keep=code==k
            W=bank[f'{mode}-{k}-{penalty}']
            pred[keep]=W[0]+X[keep]@W[1:]
        advantage.append(pred)
    delta=advantage[0]-advantage[1]
    return np.where(np.isclose(delta,0,rtol=0,atol=1e-10),.5,(delta>0).astype(float))


def completion_preferences(data,tapes):
    with np.load(OBJECT/'object01/model.npz') as archive:
        bank=dict(archive)
    per_tape=[predict(bank,t['issued'],t['states']) for t in tapes.values()]
    methods={}
    for name in ('joint','jaw'):
        values=np.array([p[name] for p in per_tape])[data['keys']]
        methods['completion_'+name]=np.array([choose(v,0,r,bank['off']) for v,r in zip(values,data['rewards'])])
    first=data['first']
    rows=np.arange(len(first))
    r=data['rewards']
    methods['table']=bank['table'][first,r[rows,first],r[rows,1-first]]
    return methods


def evaluate(bank,tapes,outcomes,draws):
    data=histories(tapes,outcomes,draws)
    ps=completion_preferences(data,tapes)
    for mode in MODES:
        for penalty in PENALTIES if mode=='joint' else (.01,):
            pred=preferences(bank,data,mode,penalty)
            suffix='' if penalty==.01 else f'_{penalty}'
            for i,target in enumerate(('utility','intent')):
                ps[f'{target}_{mode}{suffix}']=pred[:,i]
    swapped = dict(data,keys=data['keys'][:,::-1],rewards=data['rewards'][:,::-1])
    flipped = preferences(bank,swapped,'joint',.01)
    np.testing.assert_array_equal(flipped[:,0],1-ps['utility_joint'])
    np.testing.assert_array_equal(flipped[:,1],1-ps['intent_joint'])

    def summarize(mask):
        def score(p, hits):
            return (p*hits[:,0]+(1-p)*hits[:,1])[mask]
        native={k:score(p,data['native']) for k,p in ps.items()}
        diffs={}
        for m in ('utility_commands','intent_joint','completion_joint','completion_jaw','table'):
            d=native['utility_joint']-native[m]
            diffs[m]=dict(gains=d.clip(min=0).sum(),losses=(-d).clip(min=0).sum())
        return dict(n=int(mask.sum()),ceiling=data['native'][mask].max(axis=1).sum(),
                    native={k:v.sum() for k,v in native.items()},
                    final={k:score(p,data['final']).sum() for k,p in ps.items()},
                    correct={k:np.where(data['h']==0,p,1-p)[mask].sum() for k,p in ps.items()},
                    paired=diffs)
    return dict(all=summarize(np.ones(len(data['h']),dtype=bool)), zero=summarize(~data['rewards'].any(axis=1)),
                by_query={int(q):summarize(data['q']==q) for q in sorted(outcomes)})


def main(out):
    tapes, outcomes=archived_panel('bowl',range(32,36),4)
    source=histories(tapes,outcomes,4,train=True)
    assert len(tapes)==64 and len(source['h'])==6144
    np.savez(out/'model.npz',**fit(source))
    with np.load(out/'model.npz') as saved:
        bank=dict(saved)
    counts={str(k):int(v) for k,v in zip(*np.unique(source['native'][:,0]-source['native'][:,1],return_counts=True))}
    report=dict(source=dict(tapes=len(tapes),rows=len(source['h']),native_advantages=counts,
                            query_success={p:pair[0].astype(int).tolist() for p,pair in outcomes.items()}))
    def add(name,panel,draws=1):
        result=evaluate(bank,*panel,draws)
        report[name]=result
        print(name,result['all']['native'],flush=True)
    add('bowl',archived_panel('bowl',range(44,48),4),4)
    add('archived_bottle',archived_panel('bottle',range(6,12),1))
    folder=ARCHIVE/'closed01/episodes'
    add('custom_bottle',live_panel(folder,range(24,30)))
    add('D18_bottle',live_panel(folder,range(24,30),ARCHIVE/'release02/episodes','D18'))
    for name,folder in [('soup','object01/episodes'),('dressing','reciprocal01/episodes/anchor2'),
                        ('ketchup','reciprocal01/episodes/anchor4')]:
        add(name,live_panel(OBJECT/folder,range(6)))
    (out/'scores.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main(Path(sys.argv[1]))
