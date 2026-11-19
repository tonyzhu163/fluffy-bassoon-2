"""Fit the public completion reader on bowls and score matched bottle retries."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from panel import ACTIONS, read_panel

FEATURES = ('closed lift', 'jaw opening at peak lift', 'closed horizontal travel')
METHODS = ('reward_lag', 'motion', 'rotated_motion', 'true_completion', 'switch_on_zero')


def stump(tapes):
    x = np.array([t['motion'] for t in tapes])
    y = np.array([t['hit'][t['target']] for t in tapes])
    assert len(set(y)) == 2
    best_score, best = .5, dict(constant=bool(y.mean() >= .5))
    for i, feature in enumerate(FEATURES):
        vals = np.unique(x[:, i])
        for low, high in zip(vals, vals[1:]):
            cut = float((low+high)/2)
            for above in (True, False):
                pred = (x[:, i] > cut) == above
                score = (pred[y].mean() + (1-pred[~y]).mean())/2
                if score > best_score:
                    best_score = float(score)
                    best = dict(feature=feature, index=i, cut=cut, success_above=above)
    return dict(model=best, balanced_accuracy=best_score)


def completion(model, tape):
    if 'constant' in model:
        return model['constant']
    return (tape['motion'][model['index']] > model['cut']) == model['success_above']


def slow(tape):
    return tape['lag'] > 8


def fit(source):
    values = {}
    prior = {}
    for lag in (False, True):
        matching = [t for t in source if slow(t) == lag]
        assert matching
        values[lag] = np.mean([t['query'] for t in matching], axis=0).tolist()
        for target in range(2):
            group = [t for t in matching if t['target'] == target]
            assert group
            for reward in (False, True):
                hidden = [h for t in group for h in range(2) if t['hit'][h] == reward]
                # A positive identifies the target when no cross-target hit occurs.
                # For an unseen zero use the balanced prior over hidden assignments.
                p_same = sum(h == target for h in hidden) / len(hidden) if hidden else (1. if reward else .5)
                prior[lag, target, reward] = p_same
    return stump(source), values, prior


def choices(values, target, p_same):
    v = np.array(values)
    score = p_same*v[:, target] + (1-p_same)*v[:, 1-target]
    return np.flatnonzero(np.isclose(score, score.max(), atol=1e-12, rtol=0)).tolist()


def sum_rows(rows):
    return {name:sum(r['score'][name] for r in rows) for name in METHODS}


def grouped_oracle(rows, use_completion):
    groups = defaultdict(list)
    for r in rows:
        key = (r['reward'], r['target'], r['slow'])
        if use_completion:
            key += (r['completed'],)
        groups[key].append(r['native'])
    return int(sum(np.sum(v, axis=0).max() for v in groups.values()))


def evaluate(tapes, model, values, prior):
    rows, conf = [], defaultdict(int)
    for t in tapes:
        target = t['target']
        pred = completion(model, t)
        actual = t['hit'][target]
        assert not t['hit'][1-target], 'cross-target placement would change the reward interpretation'
        conf[f'actual_{int(actual)}_pred_{int(pred)}'] += 1
        matched = sorted((u for u in tapes if u['target'] == target and u['timing'] == t['timing']),
                         key=lambda u:u['index'])
        donor = matched[(matched.index(t)+1)%len(matched)]
        lag = slow(t)
        for h in range(2):
            reward = t['hit'][h]
            probs = dict(reward_lag=prior[lag,target,reward], switch_on_zero=float(reward),
                         motion=1. if reward else float(not pred)/2,
                         rotated_motion=1. if reward else float(not completion(model, donor))/2,
                         true_completion=1. if reward else float(not actual)/2)
            actions = {name:choices(values[lag],target,p) for name,p in probs.items()}
            hits = [q[h] for q in t['query']]
            score = {name:sum(hits[a] for a in act)/len(act) for name,act in actions.items()}
            rows.append(dict(index=t['index'], timing=t['timing'], slow=lag, target=target, hidden=h,
                             reward=reward, completed=actual, predicted=pred, native=hits, score=score,
                             actions=actions, same_possible=any(hits[a] for a,(g,_) in enumerate(ACTIONS) if g==target),
                             switch_possible=any(hits[a] for a,(g,_) in enumerate(ACTIONS) if g!=target)))
    groups = {'all':rows, 'zero':[r for r in rows if not r['reward']]}
    groups.update({kind:[r for r in groups['zero'] if r['completed']==completed] for kind,completed in
                   [('completed_wrong_target',True),('missed_both',False)]})
    results = {}
    for name, data in groups.items():
        results[name] = dict(n=len(data),scores=sum_rows(data),per_assignment_oracle=sum(any(r['native']) for r in data),
                             repeat_only=sum(r['same_possible'] and not r['switch_possible'] for r in data),
                             switch_only=sum(r['switch_possible'] and not r['same_possible'] for r in data),
                             both=sum(r['same_possible'] and r['switch_possible'] for r in data),
                             neither=sum(not any(r['native']) for r in data))
    return dict(tapes=tapes, rows=rows, classification=dict(conf), groups=results,
                hindsight_zero_oracle=dict(reward_target_lag=grouped_oracle(groups['zero'],False),
                                           plus_true_completion=grouped_oracle(groups['zero'],True)))


def main(out):
    bowl = read_panel('bowl')
    classifier, values, prior = fit(bowl)
    config = dict(stump=classifier, actions=ACTIONS, source_returns=values,
                  source_posterior={str(key):v for key,v in prior.items()})
    out.mkdir(parents=True, exist_ok=True)
    with (out/'frozen.json').open('x') as f:
        print(json.dumps(config, indent=2), file=f)
    result = {'bowl_fit':evaluate(bowl, classifier['model'], values, prior),
              'bottle_transfer':evaluate(read_panel('bottle'), classifier['model'], values, prior)}
    (out/'replay.json').write_text(json.dumps(result, indent=2)+'\n')
    for k,v in result.items():
        print(k,v['groups'],v['classification'],v['hindsight_zero_oracle'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    main(p.parse_args().output)
