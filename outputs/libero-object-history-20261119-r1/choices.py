"""Freeze all query preferences from public histories and the actually delivered rewards."""
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from reader import MODES,choose,predict


def read_tape(path):
    file=path if path.suffix=='.npz' else path/'tape.npz'
    with np.load(file) as z:return {k:z[k] for k in ('issued','states','hits')}


def choices(model,library,poses,triplets=None):
    cache={}
    for p,g,b,sensor in product(poses,(0,1),(False,True),(False,True)):
        factual,other=library[p,g,b],library[p,g,sensor]
        cache[p,g,b,sensor]=predict(model,factual['issued'],other['states'])
    rows=[]
    if triplets is None:
        triplets=[(p,poses[(i+1)%len(poses)],poses[(i+2)%len(poses)]) for i,p in enumerate(poses)]
    for p,s,q in triplets:
        for first,b,c,h in product((0,1),(False,True),(False,True),(0,1)):
            r=[int(library[p,first,b]['hits'][:,h].any()),int(library[s,1-first,c]['hits'][:,h].any())]
            def preference(mode,u,v):
                values=cache[p,first,b,u][mode],cache[s,1-first,c,v][mode]
                return choose(values,first,r,model['off'])
            ps={m:preference(m,b,c) for m in (*MODES,'jaw')}
            ps['table']=float(model['table'][tuple([first]+r)])
            ps['sensor_donor']=np.mean([preference('joint',u,v) for u,v in product((False,True),repeat=2)])
            rows.append(dict(p=p,s=s,q=q,first=first,b=b,c=c,h=h,r=r,ps=ps))
    return rows


def prospective(out,model_file):
    poses=list(range(6))
    library={(p,g,b):read_tape(out/f'p{p}-g{g}-{("interrupt" if b else "sham")}')
             for p,g,b in product(poses,(0,1),(False,True))}
    with np.load(model_file) as model:rows=choices(model,library,poses)
    actual=[]
    for q in poses:
        # Alternate hidden identity independently of a repeating four-condition sequence.
        # Select the first instruction by the joint reader. Slot zero wins an exact tie.
        first=(q//2)%2;h=q%2
        b,c=((False,False),(True,False),(False,True),(True,True))[q%4]
        row=next(r for r in rows if (r['q'],r['first'],r['b'],r['c'],r['h'])==(q,first,b,c,h))
        actual.append(dict(row,selected=int(row['ps']['joint']<.5)))
    with (out/'choices.json').open('x') as f:json.dump(dict(rows=rows,actual=actual),f,indent=2)
    return actual
