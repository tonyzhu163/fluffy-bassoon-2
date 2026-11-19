"""Train a binary reward-conditioned continuation from paired native outcomes."""
from itertools import product
import numpy as np


def score(block,query,pair,repeat,probability_first):
    values=np.zeros(2)
    for j,hidden in enumerate(pair):
        for i,g in enumerate(pair):
            reward=int(block[g][str(hidden)] is not None)
            chosen=g if repeat[reward] else pair[1-i]
            chance=probability_first if i==0 else 1-probability_first
            values[j]+=chance*float(query[chosen][str(hidden)] is not None)
    return values


def evaluate(source,test,pair_results):
    from report import next_scene
    candidates=[]
    for repeat in product((False,True),repeat=2):
        native=0.
        total=0
        for pair in ((4,8),(2,9)):
            for (s,l),b in source.items():
                query=source[next_scene(source,s),l]['records']
                native+=score(b['records'],query,pair,repeat,.5).sum()
                total+=2
        candidates.append(dict(repeat=list(repeat),native=float(native),assignments=total))
    def best(rules):
        m=max(r['native'] for r in rules)
        return [r['repeat'] for r in rules if r['native']==m]
    full=best(candidates)
    blind=best([r for r in candidates if r['repeat'][0]==r['repeat'][1]])
    rows=[]
    for p in pair_results:
        pair=p['goals']
        vals=dict(learned_with_selector=0.,blind_with_selector=0.,learned_with_balanced=0.)
        for b in p['blocks']:
            s,l=b['scene'],b['layout'];a=test[s,l]['records'];q=test[next_scene(test,s),l]['records']
            prob=b['choice_probability_first']
            for key,policies,first in [('learned_with_selector',full,prob),('blind_with_selector',blind,prob),
                                        ('learned_with_balanced',full,.5)]:
                vals[key]+=float(np.mean([score(a,q,pair,r,first).sum() for r in policies]))
        rows.append(dict(goals=pair,assignments=p['assignments'],values=vals))
    return dict(candidates=candidates,learned=full,blind=blind,pairs=rows)
