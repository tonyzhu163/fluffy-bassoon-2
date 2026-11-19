"""Compare closed- and open-loop responses to the same verified released object displacement."""
import json
from pathlib import Path
import sys

import numpy as np
from score import features,fit,load,score


def main(root,out):
    source=load(root/'pair01/branches')
    model=fit([r for (f,p,g,j,b),r in source.items() if f=='bowl' and p in range(32,36)])
    opened=json.loads((root/'drop01/drop.json').read_text())
    open_rows={(r['pose'],r['goal']):r for r in opened['events']}
    panel={};queries={};events=[]
    for p in range(24,30):
        for g in (0,1):
            for b,run,mode in [(False,'closed01','sham'),(True,'release01','D18')]:
                path=root/run/'episodes'/f'p{p}-g{g}-{mode}'
                r=json.loads((path/'result.json').read_text())
                if b:
                    assert r['event']==open_rows[p,g]['event']
                    events.append(dict(pose=p,goal=g,event=r['event'],open_own=open_rows[p,g]['drop'][g],live_own=r['ever'][g]))
                r.update(start=open_rows[p,g]['event']['control']-1 if open_rows[p,g]['event'] else None,lost_contact=None)
                with np.load(path/'tape.npz') as z:
                    assert z['hits'].any(axis=0).tolist()==r['ever']
                    r.update(features(z['issued'],z['states']))
                panel['live',p,g,0,b]=r
            with np.load(root/'closed01/episodes'/f'p{p}-g{g}-query/tape.npz') as z:queries[p,g]=(z['hits'].any(axis=0),z['hits'][-1])
    def query(family,p):
        return [np.stack([queries[p,g][i] for g in (0,1)]) for i in range(2)]
    result=score(panel,model,'live',range(24,30),1,query)
    for k in ('loss_on_interrupt','loss_on_sham'):del result['physical'][k]
    result['events']=events
    (out/'released-live.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
    print(result['physical'],result['all'],result['zero'])


if __name__=='__main__':main(*map(Path,sys.argv[1:]))
