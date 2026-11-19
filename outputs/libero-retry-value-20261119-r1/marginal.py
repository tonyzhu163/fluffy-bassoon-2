"""Marginalize source retry labels over query poses while keeping all practice inputs fixed."""
import json
from pathlib import Path
import sys

import numpy as np
from experiment import ARCHIVE, OBJECT, archived_panel, fit, histories, live_panel, preferences


def main(out):
    tapes, outcomes=archived_panel('bowl',range(32,36),4)
    data=histories(tapes,outcomes,4,train=True)
    physical=np.stack([v[0] for v in outcomes.values()]).mean(axis=0)
    averaged=dict(data,native=physical[:,data['h']].T)
    np.testing.assert_allclose(averaged['native'].mean(axis=0),data['native'].mean(axis=0))
    bank=fit(averaged)
    with np.load(out.parent/'value01/model.npz') as original:
        for k in bank:
            if k.startswith('joint-'):
                np.testing.assert_allclose(bank[k][:,1],original[k][:,1],atol=1e-12)
        base=dict(original)
    report=dict(source_marginal=physical.tolist())
    def add(name,panel,draws):
        t,y=panel
        d=histories(t,y,draws)
        p=preferences(bank,d,'joint',.01)[:,0]
        previous=preferences(base,d,'joint',.01)[:,0]
        delta=d['native'][:,0]-d['native'][:,1]
        def result(mask):
            wins=(p-previous)*delta
            return dict(n=int(mask.sum()),native=float((d['native'][:,1]+p*delta)[mask].sum()),
                        paired=float((d['native'][:,1]+previous*delta)[mask].sum()),
                        gains=float(wins[mask].clip(min=0).sum()),losses=float((-wins[mask]).clip(min=0).sum()))
        report[name]=dict(all=result(np.ones(len(p),bool)),zero=result(~d['rewards'].any(axis=1)),
                          by_query={int(q):result(d['q']==q) for q in y})
    add('bowl',archived_panel('bowl',range(44,48),4),4)
    add('archived_bottle',archived_panel('bottle',range(6,12),1),1)
    folder=ARCHIVE/'closed01/episodes'
    add('custom_bottle',live_panel(folder,range(24,30)),1)
    add('D18_bottle',live_panel(folder,range(24,30),ARCHIVE/'release02/episodes','D18'),1)
    for name,folder in [('soup','object01/episodes'),('dressing','reciprocal01/episodes/anchor2'),
                        ('ketchup','reciprocal01/episodes/anchor4')]:
        add(name,live_panel(OBJECT/folder,range(6)),1)
    (out/'marginal.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main(Path(sys.argv[1]))
