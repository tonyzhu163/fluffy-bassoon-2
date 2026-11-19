"""Paired choice of destination and cadence from two rewards and observed practice executions."""
from collections import defaultdict
from itertools import permutations, product
import json
from pathlib import Path
import sys

import numpy as np
from vision import MODES, fit, predict, read_split

ACTIONS = tuple(product((0,1),(5,10)))


def trials(practice, queries, *, all_orders=False):
    poses = sorted({p for p,g,j in practice})
    cyclic = [(p,poses[(i+1)%len(poses)],poses[(i+2)%len(poses)]) for i,p in enumerate(poses)]
    for p,s,q in (permutations(poses,3) if all_orders else cyclic):
        for first,j,k in product((0,1),range(4),range(4)):
            keys = ((p,first,j), (s,1-first,k))
            hits = np.stack([practice[key]['native'] for key in keys])
            native = np.stack([queries[q,g,c]['native'] for g,c in ACTIONS])
            final = np.stack([queries[q,g,c]['final'] for g,c in ACTIONS])
            for hidden in (0,1):
                yield dict(first=p, second=s, query=q, keys=keys, hidden=hidden, bits=tuple(hits[:,hidden].astype(int)),
                           hits=hits, native=native[:,hidden], final=final[:,hidden])


def query_rule(belief, values):
    score = values @ belief
    ties = np.isclose(score, score.max(), rtol=0, atol=1e-10)
    return ties/ties.sum()


def belief(bits, hits):
    prob = np.ones(2)
    for b,likelihood in zip(bits,hits):
        prob *= likelihood if b else 1-likelihood
    return prob/prob.sum() if prob.sum() else np.full(2,.5)


def train(practice, queries):
    rows = list(trials(practice,queries))
    leaves = defaultdict(list)
    for r in rows:
        leaves[r['keys'][0][1],r['bits']].append(r['native'])
    table = {k:query_rule([1.],np.mean(native,axis=0)[:,None]) for k,native in leaves.items()}
    query = np.stack([(np.sum([r['native'] for (p,goal,cad),r in queries.items() if (goal,cad)==(g,c)],axis=0)+.5)/5
                      for g,c in ACTIONS])
    off = {g:(sum(t['native'][1-g] for t in practice.values() if t['goal']==g)+.5)/17 for g in (0,1)}
    return fit(practice), table, query, off


def decisions(row, scores, table, values, off):
    keys, bits = row['keys'],row['bits']
    answer = {'table':table.get((keys[0][1],bits),query_rule([.5,.5],values)),
              'true_completion':query_rule(belief(bits,row['hits']),values)}
    for mode in (*MODES, 'donor'):
        hits = []
        for p,g,j in keys:
            v = scores[p,g,(j+1)%4]['full'] if mode=='donor' else scores[p,g,j][mode]
            hits.append(np.array([v,off[g]]) if g==0 else np.array([off[g],v]))
        answer[mode] = query_rule(belief(bits,hits),values)
    return answer


def analyze(source, panel):
    model, table, values, off = train(*source)
    practice, queries = panel
    scores = {k:{m:predict(model,t,m) for m in MODES} for k,t in practice.items()}
    class_scores = {}
    for m in MODES:
        pred = np.array([scores[k][m] for k in practice])
        y = np.array([t['native'][t['goal']] for t in practice.values()],dtype=float)
        class_scores[m] = dict(brier=float(np.mean((pred-y)**2)), correct=int(((pred>=.5)==y).sum()))
    rows = []
    matched = defaultdict(lambda: [set(),set()])
    for r in trials(practice, queries):
        probs = decisions(r,scores,table,values,off)
        last = r['keys'][1][1]
        h = r['hidden']
        mask = r['native']
        retain = bool(mask[ACTIONS.index((last,10))])
        default = bool(mask[ACTIONS.index((last,5))])
        switch = bool(mask[2*(1-last):2*(1-last)+2].any())
        same = bool(default or retain)
        # A single consistent hidden goal after seeing the complete executions.
        possible = np.all(r['hits']==np.array(r['bits'])[:,None], axis=0)
        unique = np.flatnonzero(possible)
        group=(r['first'],r['second'],r['query'],last)
        if r['bits']==(0,0) and len(unique)==1:
            assert unique[0] == h
            if same != switch:
                matched[group][0].add('retain' if same else 'switch')
            if (retain and not default and not switch) or (switch and not same):
                matched[group][1].add('retain' if retain else 'switch')
        rows.append(dict(pose=r['query'],last=last,h=h,bits=''.join(map(str,r['bits'])),identified=len(unique)==1,
                         switch_only=switch and not same, same_only=same and not switch,
                         change_motion_only=retain and not default and not switch, oracle=bool(mask.any()),
                         native={m:float(p@mask) for m,p in probs.items()},
                         final={m:float(p@r['final']) for m,p in probs.items()}))
    def summarize(rs):
        return dict(n=len(rs), **{k:sum(r[k] for r in rs) for k in
                   ('identified','switch_only','same_only','change_motion_only','oracle')},
                    **{metric:{m:sum(r[metric][m] for r in rs) for m in rows[0][metric]}
                       for metric in ('native','final')})
    return dict(total=summarize(rows), zero=summarize([r for r in rows if r['bits']=='00']),
                zero_by_query={str(p):summarize([r for r in rows if r['bits']=='00' and r['pose']==p])
                               for p in sorted({r['pose'] for r in rows})},
                reversible_groups=sum(len(v[0])==2 for v in matched.values()),
                requires_new_cadence_groups=sum(len(v[1])==2 for v in matched.values()),groups=8,
                predictor=class_scores, off_target_hits=sum(int(t['native'][1-t['goal']]) for t in practice.values()),
                practice_hits={f'{p}-{g}':sum(t['native'][g].item() for (q,h,j),t in practice.items() if (q,h)==(p,g))
                               for p,g,j in practice if j==0},
                query_hits={f'{p}-{g}-{c}':list(map(int,t['native'])) for (p,g,c),t in queries.items()},
                source_query_probabilities=values.tolist(), source_bandwidths={f'{g}-{m}':v[3] for (g,m),v in model.items()})


if __name__ == '__main__':
    root, out = map(Path,sys.argv[1:])
    train_panel = read_split(root/'fitting')
    fresh = read_split(root/'evaluation')
    for name,panel in [('source',train_panel),('fresh',fresh)]:
        result=analyze(train_panel,panel)
        (out/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(name,result['total'],result['zero'],result['reversible_groups'],result['requires_new_cadence_groups'])
