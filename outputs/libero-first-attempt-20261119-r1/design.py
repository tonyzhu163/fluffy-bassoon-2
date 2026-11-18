"""Select the support policy using only physically measured bowl continuations."""
import json
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'libero-bowl-decisions-20261118-r1'
REMOTE = Path('/mowenjie/fluffy-bassoon-2/outputs')/OLD.name
VARIANTS = ('BASE','S20')
BUDGETS = np.arange(24,301)


def lag24(actions, states):
    movement = np.diff(np.asarray(states[:24],dtype=float)[:,:3], axis=0)
    movement -= movement.mean(axis=0)
    commands = np.asarray(actions[:24],dtype=float)[:,:3]
    scores = []
    for lag in range(21):
        indices = np.arange(1,24)-lag
        aligned = np.zeros_like(movement)
        available = indices >= 0
        aligned[available] = commands[indices[available]]
        aligned -= aligned.mean(axis=0)
        norm = np.linalg.norm(aligned)*np.linalg.norm(movement)
        scores.append(float(np.sum(aligned*movement)/norm) if norm else 0.)
    return int(np.argmax(scores))


def use_pacing(model, lag):
    if model['threshold'] is None or lag <= model['threshold']:
        return model['low_paced']
    return model['high_paced']


def scored_trials(model, blocks, *, uniform_option=False, clear=False):
    """Average over the two options only for explicitly randomized policies."""
    rows = []
    for support, query in blocks:
        for variant, assigned in product(VARIANTS, range(2)):
            by_option = []
            for option in (range(2) if uniform_option else [model['option']]):
                lag = support[variant,option,True]['lag']
                paced = use_pacing(model, lag)
                record = support[variant,option,paced]
                hit = record['first'][assigned]
                feedback = BUDGETS >= (hit if hit is not None else 301)
                dest = np.where(feedback,option,1-option)
                query_times = [query[variant,g]['first'][assigned] for g in range(2)]
                times = np.array([t if t is not None else 301 for t in query_times])
                if clear:
                    curve = (times[:,None] <= BUDGETS).mean(axis=0)
                else:
                    curve = (times[dest] <= BUDGETS).astype(float)
                by_option.append(curve)
            rows.append(np.mean(by_option,axis=0))
    return np.stack(rows)


def curve_metrics(rows):
    return dict(assignments=len(rows),successes=float(rows[:,-1].sum()),
                mean_budget_success=float(rows.mean()),
                success_by_budget={str(b):float(rows[:,b-24].sum()) for b in (24,50,100,150,200,250,300)})


def native_record(path):
    rec = json.loads((path/'result.json').read_text())
    hits = np.load(path/'evaluator-predicates.npy')
    assert hits.shape==(300,2)
    first = [int(np.flatnonzero(h)[0])+1 if h.any() else None for h in hits.T]
    assert first==rec['first']
    return rec


def fit(new):
    supports = []
    for scene in range(16,20):
        support = {}
        for variant,option in product(VARIANTS,range(2)):
            a = REMOTE/'pacing01/episodes'/f's{scene}-{variant}-g{option}-paced'
            b = new/f's{scene}-{variant}-g{option}-switch'
            old = native_record(a)
            other = native_record(b)
            assert (old['initial_physics'],old['initial_public'])==(other['initial_physics'],other['initial_public'])
            with np.load(a/'public.npz') as archived, np.load(b/'public.npz') as fresh:
                for key in ('issued','state_after'):
                    np.testing.assert_array_equal(archived[key][:24],fresh[key][:24])
                lag = lag24(archived['issued'],archived['state_after'])
            assert other['lag']==lag
            support[variant,option,True] = dict(old,lag=lag)
            support[variant,option,False] = other
        supports.append(support)
    queries = []
    for idx in range(4):
        query = {}
        for variant in VARIANTS:
            block=REMOTE/'livepair01/episodes/paced_support'/f'b{idx}-{variant}'
            for path in block.glob('goal*/s*-paced'):
                rec=native_record(path)
                assert rec['scene']==24+idx
                query[variant,rec['goal']]=rec
        assert len(query)==4
        queries.append(query)
    blocks=list(product(supports,queries))
    lags=sorted({r['lag'] for s in supports for r in s.values()})
    cuts=[(a+b)/2 for a,b in zip(lags,lags[1:])]
    candidates=[]
    for option in range(2):
        mappings=[(None,p,p) for p in (True,False)]
        mappings += [(cut,p,not p) for cut in cuts for p in (True,False)]
        for cut,low,high in mappings:
            model=dict(option=option,threshold=cut,low_paced=low,high_paced=high)
            metrics=curve_metrics(scored_trials(model,blocks))
            candidates.append(dict(model=model,metrics=metrics))
    def rank(row):
        model,metrics=row['model'],row['metrics']
        return metrics['successes'],metrics['mean_budget_success'],model['threshold'] is None
    best=max(candidates,key=rank)
    constant=max((x for x in candidates if x['model']['threshold'] is None),key=rank)
    return dict(learned=best,fixed=constant,candidates=candidates,training_lags=lags,
                support_scenes=list(range(16,20)),query_scenes=list(range(24,28)))
