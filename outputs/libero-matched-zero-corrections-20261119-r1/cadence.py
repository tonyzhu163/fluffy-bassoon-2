"""Check whether changing cadence adds anything beyond identifying a destination."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from analyze import ACTIONS, trials
from ceilings import ceilings
from vision import GOALS, read_split


def fixed_cadences(queries):
    result = []
    for g in (0,1):
        def hits(c):
            return sum(v['native'][g] for (p,h,cad),v in queries.items() if (h,cad)==(g,c))
        result.append(max((5,10),key=hits))
    return result


def check(rows, fixed):
    groups = defaultdict(dict)
    fixed_rows = []
    for r in rows:
        # Keep one source-selected cadence for each destination.
        indices = [ACTIONS.index((g,c)) for g,c in enumerate(fixed)]
        fixed_rows.append(dict(r, native=r['native'][indices], final=r['final'][indices]))
        if r['bits']!=(0,0) or np.all(~r['hits'],axis=0).sum()!=1:
            continue
        last = r['keys'][-1][1]
        actions = [a for a,hit in zip(ACTIONS,r['native']) if hit]
        kind = None
        if actions == [(last,10)]:
            kind = 'retain_change_cadence'
        elif actions and all(g!=last for g,c in actions):
            kind = 'switch_destination'
        if kind:
            key=(r['first'],r['second'],r['query'],last)
            groups[key].setdefault(kind,dict(seeds=[j for p,g,j in r['keys']],prompted=[GOALS[g] for p,g,j in r['keys']],
                    practice_own_hits=[int(r['hits'][i,g]) for i,(p,g,j) in enumerate(r['keys'])],
                    hidden=GOALS[r['hidden']],successful_query_actions=[(GOALS[g],c) for g,c in actions]))
    reversals = [dict(practice_poses=[p,s],query_pose=q,**v) for (p,s,q,g),v in groups.items() if len(v)==2]
    zero = [r for r in rows if r['bits']==(0,0)]
    zero_fixed = [r for r in fixed_rows if r['bits']==(0,0)]
    return dict(all=ceilings(rows), zero=ceilings(zero), fixed_all=ceilings(fixed_rows),
                fixed_zero=ceilings(zero_fixed), strict_reversals=reversals)


if __name__ == '__main__':
    root,out=map(Path,sys.argv[1:])
    source=read_split(root/'fitting')
    fixed=fixed_cadences(source[1])
    report=dict(cadences=fixed)
    for name,panel in [('fitting',source),('evaluation',read_split(root/'evaluation'))]:
        report[name]={key:check(list(trials(*panel,all_orders=v)),fixed) for key,v in [('cyclic',False),('all_orders',True)]}
    (out/'cadence.json').write_text(json.dumps(report,indent=2)+'\n')
    for name in ('fitting','evaluation'):
        for key,sub in report[name].items():
            print(name,key,len(sub['strict_reversals']),sub['all'],sub['fixed_all'])
