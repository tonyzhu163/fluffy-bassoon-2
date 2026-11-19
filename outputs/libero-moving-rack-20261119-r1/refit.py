"""Learn a second tree on the completed development panel before prospective testing."""
from itertools import permutations,product
import json
from pathlib import Path
import sys
from pilot import OPTIONS,REGIMES,best,score


def read(*files):
    rows=[json.loads(s) for f in files for s in f.read_text().splitlines()]
    return {(r['pose'],r['regime'],r['option']):r for r in rows}


def learn(panel,out):
    table=read(panel/'fitting.jsonl',panel/'fresh.jsonl')
    # The two members of each training pair use the same opening cadence.
    pairs=list(permutations(range(6),2))+list(permutations(range(6,12),2))
    tree=best(table,pairs,product(OPTIONS,repeat=3))
    constant=best(table,pairs,[(OPTIONS[0],a,a) for a in OPTIONS])
    by_regime={g:max(OPTIONS,key=lambda a:sum(table[p,g,a]['hit'] for p in range(12))) for g in REGIMES}
    model=dict(tree=tree,constant=constant,regime=by_regime,
               scores={k:sum(r['hit'] for r in score(table,pairs,t)) for k,t in [('history',tree),('fixed',constant)]},
               n=2*len(pairs))
    with out.open('x') as f:
        print(json.dumps(model,indent=2),file=f)


def apply(panel,frozen,out):
    data=read(panel/'native.jsonl')
    model=json.loads(frozen.read_text())
    poses=list(range(12,24))
    pairs=list(zip(poses,poses[1:]+poses[:1]))
    rows={k:score(data,pairs,t,swap) for k,t,swap in [
        ('history',model['tree'],False),('fixed',model['constant'],False),('opposite',model['tree'],True)]}
    choices={a:{g:sum(data[q,g,a]['hit'] for q in poses) for g in REGIMES} for a in OPTIONS}
    oracle=sum(any(data[q,g,a]['hit'] for a in OPTIONS) for q,g in product(poses,REGIMES))
    result=dict(rows=rows,fixed_options=choices,oracle=oracle,
        regime_aware=sum(data[q,g,model['regime'][g]]['hit'] for q,g in product(poses,REGIMES)),
        scores={k:{m:sum(x[m] for x in cases) for m in ('hit','final')} for k,cases in rows.items()})
    with out.open('x') as f:
        print(json.dumps(result,indent=2),file=f)


if __name__=='__main__':
    funcs={'learn':learn,'apply':apply}
    funcs[sys.argv[1]](*map(Path,sys.argv[2:]))
