"""Keep practice rewards and issued controls fixed while independently reassigning public motion."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import json
import sys

import numpy as np
from timing import ORDERS,PRACTICE,QUERY
from experiment import histories,methods,read_tape


def main(out):
    worker=out.parent/'timing01'
    folder=worker/'episodes'
    with np.load(out.parent/'value01/model.npz') as file:
        bank=dict(file)
    frozen=json.loads((worker/'choices.json').read_text())
    outcomes={}
    for q in QUERY:
        pair=[read_tape(folder/f'p{q}-g{g}-query.npz')['hits'] for g in (0,1)]
        outcomes[q]=(np.stack([x.any(axis=0) for x in pair]),np.stack([x[-1] for x in pair]))
    report={}
    for mode in ('late','early'):
        tapes={(p,g,0,b):read_tape(folder/f'p{p}-g{g}-{mode if b else "normal"}.npz')
               for p,g,b in product(PRACTICE,(0,1),(False,True))}
        for p,g in product(PRACTICE,(0,1)):
            np.testing.assert_array_equal(tapes[p,g,0,False]['issued'],tapes[p,g,0,True]['issued'])
        d=histories(tapes,outcomes,1,orders=ORDERS)
        actual=methods(bank,d,tapes)
        for k,p in actual.items():
            np.testing.assert_array_equal(p,frozen[mode][k])
        names=['utility_joint','utility_commands','utility_motion','intent_joint','completion_joint','completion_jaw']
        donor={k:np.zeros(len(d['h'])) for k in names}
        idx={key:i for i,key in enumerate(tapes)}
        for a,b in product((False,True),repeat=2):
            replace=np.array([idx[p,g,j,(a,b)[g]] for p,g,j,_ in tapes])
            choice=methods(bank,dict(d,keys=replace[d['keys']]),tapes)
            for k in names:
                donor[k]+=choice[k]/4
        def summary(mask):
            native=d['native'][mask]
            delta=native[:,0]-native[:,1]
            result={}
            for k in names:
                diff=(actual[k][mask]-donor[k][mask])*delta
                result[k]=dict(actual=(native[:,1]+actual[k][mask]*delta).sum(),
                               donor=(native[:,1]+donor[k][mask]*delta).sum(),
                               gains=diff.clip(min=0).sum(),losses=(-diff).clip(min=0).sum())
            start_groups=defaultdict(lambda:np.zeros(2))
            full_groups=defaultdict(lambda:np.zeros(2))
            for i in np.flatnonzero(mask):
                group=(d['q'][i],d['first'][i],*d['rewards'][i])
                start_groups[group]+=d['native'][i]
                full_groups[group+tuple(d['keys'][i])]+=d['native'][i]
            return dict(n=int(mask.sum()),start_reward_bound=sum(v.max() for v in start_groups.values()),
                        assignment_bound=sum(v.max() for v in full_groups.values()),methods=result)
        report[mode]=dict(all=summary(np.ones(len(d['h']),bool)),zero=summary(~d['rewards'].any(axis=1)))
    (out/'donor.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main(Path(sys.argv[1]))
