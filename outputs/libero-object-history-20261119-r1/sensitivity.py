"""Vary the practice/query pose ordering and count paired decisions changed by sensor input."""
from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np

from choices import choices
from evaluate import evaluate,queries,read_practices


def paired(rows,q):
    gain=loss=tied=0.
    changed=0
    for r in rows:
        p=r['ps']['joint']
        a=r['ps']['commands']
        hits=q[r['q']][:,:,r['h']].any(axis=1)
        delta=(p-a)*(float(hits[0])-float(hits[1]))
        gain+=max(0.,delta)
        loss+=max(0.,-delta)
        if p!=a:
            changed+=1
            tied+=int(hits[0]==hits[1])
    return dict(n=len(rows),different_choices=changed,changed_with_equal_motor_outcomes=tied,
                gained=gain,lost=loss,net=gain-loss)


if __name__=='__main__':
    worker,out=map(Path,sys.argv[1:])
    with np.load(worker/'model.npz') as z:model=dict(z)
    poses=list(range(6))
    p=read_practices(worker/'episodes',poses)
    q=queries(worker/'episodes',poses)
    report={}
    for name,orders in [('cyclic',None),('all_orders',permutations(poses,3))]:
        rows=choices(model,p,poses,orders)
        result=evaluate(model,p,q,rows)
        report[name]=dict(native=result['all']['native'],zero=result['zero'],
                          change=paired(rows,q),by_query={str(k):paired([r for r in rows if r['q']==k],q) for k in q})
    (out/'sensitivity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['cyclic']['change'],report['all_orders']['change'])
