"""Apply the bowl-only estimators unchanged to live bottle practices and independent queries."""
import json
from pathlib import Path
import sys

import numpy as np

from score import features,fit,load,score


def main(archive,live,out):
    source=load(archive)
    model=fit([r for (f,p,g,j,b),r in source.items() if f=='bowl' and p in range(32,36)])
    practices,queries={},{}
    for d in sorted(live.glob('p*-g*-*')):
        r=json.loads((d/'result.json').read_text())
        p,g,mode=(r[k] for k in ('pose','goal','mode'))
        with np.load(d/'tape.npz') as t:
            assert r['ever']==t['hits'].any(axis=0).tolist()
            if mode=='query':
                queries[p,g]=(t['hits'].any(axis=0),t['hits'][-1])
            else:
                start=r['start']
                r.update(features(t['issued'],t['states']))
                r['lost_contact']=bool(not all(t['contact'][start:start+10])) if start is not None else None
                practices['live',p,g,0,mode=='interrupt']=r
    def query(family,p):
        return [np.stack([queries[p,g][i] for g in (0,1)]) for i in range(2)]
    assert len(queries)==12
    result=score(practices,model,'live',range(24,30),1,query)
    result['query_hits']={f'{p}-{g}':v[0].tolist() for (p,g),v in queries.items()}
    (out/'closed.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
    print(result['physical'],result['all'],result['zero'])


if __name__=='__main__':main(*map(Path,sys.argv[1:]))
