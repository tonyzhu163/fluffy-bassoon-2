"""Native paired retry outcomes, first-target reversals and common-world checks."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from selector import Reachability

LAYOUTS=('base','cabinet','plate','rack')
KNOWN=(4,8,2,9)
COMPOSITION=((4,9),(8,2))
NEW=(0,1)
METHODS=('selected','fit_constant','balanced','donor','cleared','first','second')


def read(p):
    return json.loads(p.read_text())


def load_panel(root,model=None):
    panel={}
    for f in sorted(root.glob('p*/*/choices.json')):
        b=read(f)
        records={}
        for g in b['goals']:
            d=f.parent/f'g{g}'
            r=read(d/'result.json')
            hits=np.load(d/'native.npy')
            assert hits.shape==(300,len(b['goals']))
            assert r['goal']==g and r['layout']==b['layout'] and r['scene']==b['scene']
            first={str(target):int(np.flatnonzero(col)[0])+1 if col.any() else None for target,col in zip(b['goals'],hits.T)}
            assert first==r['first']
            if model:
                with np.load(d/'initial.npz') as a:
                    assert model.score(g,a['visual'],a['plan'])==b['scores'][str(g)]
                assert f.stat().st_mtime_ns<=(d/'result.json').stat().st_mtime_ns
            records[g]=first
        panel[b['scene'],b['layout']]=dict(meta=b,records=records)
    assert len(panel)==16
    return panel


def next_scene(panel,scene):
    poses=sorted({s for s,_ in panel})
    return poses[(poses.index(scene)+1)%len(poses)]


def split_tie(one,two):
    if one==two:
        return .5
    return float(one>two)


def outcomes(s,q,pair,limit=300):
    result=np.zeros((2,2))
    correct=np.zeros((2,2))
    for i,goal in enumerate(pair):
        for j,hidden in enumerate(pair):
            support_hit=s[goal][str(hidden)]
            feedback=support_hit is not None and support_hit<=limit
            chosen=goal if feedback else pair[1-i]
            hit=q[chosen][str(hidden)]
            correct[i,j]=chosen==hidden
            result[i,j]=hit is not None and hit<=limit
    return result,correct


def fixed_probability(panel,pair):
    if pair==NEW:
        return .5
    totals=np.zeros(2)
    for (s,l),block in panel.items():
        result,_=outcomes(block['records'],panel[next_scene(panel,s),l]['records'],pair)
        totals+=result.sum(axis=1)
    return split_tie(*totals)


def score_pair(source,test,pair):
    fixed=fixed_probability(source,pair)
    blocks=[]
    for (s,l),b in test.items():
        query=test[next_scene(test,s),l]['records']
        current=b['records']
        r,correct=outcomes(current,query,pair)
        scores=b['meta']['scores']
        donor_layout=LAYOUTS[(LAYOUTS.index(l)+1)%len(LAYOUTS)]
        donor=test[s,donor_layout]['meta']['scores']
        sel=split_tie(*(scores[str(g)] for g in pair))
        other=split_tie(*(donor[str(g)] for g in pair))
        probs=dict(selected=sel,fit_constant=fixed,balanced=.5,donor=other,first=1.,second=0.)
        vals={name:(p*r[0]+(1-p)*r[1]).tolist() for name,p in probs.items()}
        query_hits=np.array([[query[g][str(h)] is not None for h in pair] for g in pair])
        vals['cleared']=query_hits.mean(axis=0).tolist()
        only=[current[g][str(g)] is not None for g in pair]
        blocks.append(dict(scene=s,layout=l,choice_probability_first=sel,source_constant_probability_first=fixed,
                           native=vals,correct=(sel*correct[0]+(1-sel)*correct[1]).tolist(),support_hits=only,
                           query_coverage=int(query_hits.any(axis=0).sum()),
                           both_predicates=[all(current[g][str(h)] is not None for h in pair) for g in pair]))
    totals={m:sum(sum(b['native'][m]) for b in blocks) for m in METHODS}
    def fixed_best(bs):
        return max(sum(sum(b['native'][m]) for b in bs) for m in ('first','second'))
    totals.update(test_best_constant=fixed_best(blocks),
                  layout_oracle=sum(fixed_best([b for b in blocks if b['layout']==l]) for l in LAYOUTS),
                  block_oracle=sum(fixed_best([b]) for b in blocks),
                  known_hidden_goal_query=sum(b['query_coverage'] for b in blocks))
    sides=[b['support_hits'] for b in blocks]
    return dict(goals=list(pair),assignments=2*len(blocks),totals=totals,
                reachability=dict(first_only=sides.count([True,False]),second_only=sides.count([False,True]),
                                  both=sides.count([True,True]),neither=sides.count([False,False])),
                differences_by_pose=[sum(sum(b['native']['selected'])-sum(b['native']['fit_constant'])
                                         for b in blocks if b['scene']==s) for s in sorted({b['scene'] for b in blocks})],
                blocks=blocks)


def summarize(root):
    done=read(root/'complete.json')
    assert done=={**done,'fitting':64,'fresh':96}
    model=Reachability(root/'frozen.npz')
    source=load_panel(root/'fitting')
    test=load_panel(root/'fresh',model)
    result=[score_pair(source,test,p) for p in (*combinations(KNOWN,2),NEW)]
    summaries={}
    for name,pairs in [('composition',COMPOSITION),('familiar',tuple(combinations(KNOWN,2))),('unseen',(NEW,))]:
        group=[x for x in result if tuple(x['goals']) in pairs]
        summaries[name]=dict(assignments=sum(x['assignments'] for x in group),
                             totals={k:sum(x['totals'][k] for x in group) for k in group[0]['totals']})
    public={}
    for name,panel in [('fitting',source),('fresh',test)]:
        public[name]={str(s):dict(
            distinct_physics=len({panel[s,l]['meta']['physics'] for l in LAYOUTS}),
            distinct_camera=len({panel[s,l]['meta']['public']['agentview_image'] for l in LAYOUTS}))
            for s in sorted({s for s,_ in panel})}
    return dict(fit=read(root/'fit.json'),complete=done,panels=summaries,pairs=result,layouts=public)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('episodes',type=Path)
    args=p.parse_args()
    print(json.dumps(summarize(args.episodes),indent=2))
