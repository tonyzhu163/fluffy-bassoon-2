"""Separate availability of stage information from outcomes and variation across support poses."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import json
import sys
import numpy as np
from phase import SOURCE,ORDERS


def category(row):
    return row['mode'] if row['start'] is not None else 'normal'


def main(out):
    worker=out.parent/'phase01'
    report=json.loads((worker/'phase.json').read_text())
    choices=json.loads((worker/'choices.json').read_text())
    source=json.loads((SOURCE.parent/'timing.json').read_text())['practices']
    prob={}
    for m in ('early','late','normal'):
        group=[r for r in source if category(r)==m]
        prob[m]=(sum(r['ever'][r['goal']] for r in group)+.5)/(len(group)+1)
    off=(sum(r['ever'][1-r['goal']] for r in source)+.5)/(len(source)+1)
    library={(r['pose'],r['goal'],r['mode']):r for r in report['practices']}
    queries={(r['pose'],r['goal']):r['ever'] for r in report['queries']}
    result=dict(source=prob)
    for name,which in [('stages',('early','late')),('early',('normal','early')),('late',('normal','late'))]:
        rows=[]
        for p,s,q in ORDERS:
            for first,b,c,h in product((0,1),(False,True),(False,True),(0,1)):
                inputs=[library[p,first,which[b]],library[s,1-first,which[c]]]
                bits=tuple(int(r['ever'][h]) for r in inputs)
                value=np.array([queries[q,g][h] for g in (0,1)],dtype=float)
                hit=np.array([r['ever'] for r in inputs])
                compatible=(hit==np.asarray(bits)[:,None]).all(axis=0)
                assert compatible[h]
                exact=compatible[0]/compatible.sum()
                li=np.full((2,2),off)
                for i,r in enumerate(inputs):
                    li[i,r['goal']]=prob[category(r)]
                likelihood=np.prod(np.where(np.asarray(bits)[:,None],li,1-li),axis=0)
                oracle=.5 if np.isclose(*likelihood,atol=1e-10,rtol=0) else float(likelihood[0]>likelihood[1])
                a={k:v[len(rows)] for k,v in choices[name].items()}
                a.update(stage_from_simulator=oracle,perfect_completion=exact)
                rows.append(dict(p=p,base=(p,q,first,*bits),stage=tuple(category(r) for r in inputs),native=value,
                                 zero=not any(bits),outcomes={k:float(value[1]+v*(value[0]-value[1])) for k,v in a.items()}))
        def score(rows):
            total={k:sum(r['outcomes'][k] for r in rows) for k in choices[name]}
            for k in ('stage_from_simulator','perfect_completion'):
                total[k]=sum(r['outcomes'][k] for r in rows)
            bounds={}
            for key in ('base','stage'):
                groups=defaultdict(lambda:np.zeros(2))
                for r in rows:
                    group=r['base'] if key=='base' else r['base']+r['stage']
                    groups[group]+=r['native']
                bounds[key]=sum(x.max() for x in groups.values())
            return dict(n=len(rows),native=total,bounds=bounds)
        result[name]=dict(all=score(rows),zero=score([r for r in rows if r['zero']]),
                          by_support={int(p):score([r for r in rows if r['p']==p]) for p in sorted({r['p'] for r in rows})})
        assert all(result[name]['all']['native'][k]==report[name]['all']['native'][k] for k in choices[name])
    (out/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main(Path(sys.argv[1]))
