"""Keep each frozen destination choice and execute the motor's original five-control interval."""
import json
from pathlib import Path
import sys

import numpy as np
from analyze import decisions, train, trials
from vision import MODES, predict, read_split


def score(source, panel):
    model,table,values,off=train(*source)
    practice,query=panel
    scores={k:{m:predict(model,t,m) for m in MODES} for k,t in practice.items()}
    rows=[]
    for r in trials(practice,query):
        ps=decisions(r,scores,table,values,off)
        res={}
        for m,p in ps.items():
            goal=p.reshape(2,2).sum(axis=1)
            res[m]=dict(correct=float(goal[r['hidden']]), native=float(goal @ r['native'][::2]),
                        final=float(goal @ r['final'][::2]))
        rows.append((r['bits'],res,bool(np.allclose(ps['full'],ps['donor']))))
    result={}
    for name,rr in [('all',rows),('zero',[r for r in rows if r[0]==(0,0)])]:
        result[name]=dict(assignments=len(rr), same_donor_choices=sum(r[2] for r in rr),
                          scores={m:{k:sum(r[1][m][k] for r in rr) for k in ('correct','native','final')}
                                  for m in rows[0][1]})
    return result


if __name__=='__main__':
    root,dest=map(Path,sys.argv[1:])
    source=read_split(root/'fitting')
    report={k:score(source,v) for k,v in [('fitting',source),('evaluation',read_split(root/'evaluation'))]}
    (dest/'fixed-default.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['evaluation'])
