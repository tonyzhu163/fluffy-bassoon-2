"""Evaluate a frozen threshold controller on actual, pre-executed drawer queries."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def public_feature(path):
    with np.load(path, allow_pickle=False) as f:
        commands = f['requested_action']
        measured = f['measured_state']
    closed = commands[:, 6] >= .5
    edges = np.flatnonzero(~closed[:-1] & closed[1:]) + 1
    if not len(edges):
        return None
    t = edges[0]
    # Two consecutive measured lateral increments, in base coordinates.
    y = measured[:, 1]
    return float((y[t + 1] - y[t]) - (y[t] - y[t - 1]))


def trials(root):
    rows = [json.loads(p.read_text()) for p in root.glob('episode-*.json')]
    out = []
    for pair, sign in itertools.product(range(8), (-1, 1)):
        group = [r for r in rows if r['pair'] == pair and r['sign'] == sign]
        support, = [r for r in group if r['role'] == 'support']
        # Convert evaluator-relative arm names into actual correction in base coordinates.
        query = {r['compensation'] * sign: r for r in group if r['role'] == 'query'}
        assert set(query) == {-1, 0, 1}
        assert len({q['paired_initial_sha256'] for q in query.values()}) == 1
        feature = public_feature(root / f"trial-{support['ordinal']:03d}/public/trajectory.npz")
        out.append(dict(pair=pair, sign=sign, feature=feature, query=query))
    return out


def action(feature, policy):
    threshold, below, above = policy
    return 0 if feature is None else (below if feature < threshold else above)


def score(rows, choose):
    cases = []
    for r in rows:
        a = choose(r)
        result = r['query'][a]
        cases.append(dict(pair=r['pair'], sign=r['sign'], action=a, success=result['success'],
                          contact=result['acquired_handle']))
    return dict(success=sum(c['success'] for c in cases), contact=sum(c['contact'] for c in cases),
                trials=len(cases), cases=cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    data = trials(args.root)
    dev = [r for r in data if r['pair'] < 4]
    test = [r for r in data if r['pair'] >= 4]
    unique = sorted({r['feature'] for r in dev if r['feature'] is not None})
    thresholds = [(x+y)/2 for x,y in zip(unique, unique[1:])]
    # Prefer zero and then a constant correction when the native training scores tie.
    constant = [(0., a, a) for a in (0, -1, 1)]
    options = constant + list(itertools.product(thresholds, (0, -1, 1), (0, -1, 1)))
    def utility(p):
        m = score(dev, lambda r: action(r['feature'], p))
        return m['success'], m['contact']
    best = max(options, key=utility)
    static = max(constant, key=utility)
    donor = {(r['pair'], r['sign']): r['feature'] for r in test}
    choices = dict(own=lambda r: action(r['feature'], best),
                   opposite=lambda r: action(donor[r['pair'], -r['sign']], best),
                   memoryless=lambda r: static[1], zero=lambda r: 0, inverse=lambda r: -r['sign'])
    print(json.dumps(dict(feature='first-close change in measured base-relative lateral increment',
                          dev_policy=best, static=static[1], development_pairs=4, heldout_pairs=4,
                          outcomes={k: score(test,f) for k,f in choices.items()}), indent=2))


if __name__ == '__main__':
    main()
