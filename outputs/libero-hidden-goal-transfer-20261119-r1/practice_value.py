"""Rank optional second attempts among failed first attempts using only public practice inputs."""
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

from reader import KNOWN, Reachability, load_panel, source_scores, tapes
from stopping import fit_tables


def gains(data, rules):
    value = []
    for length in (1, 2):
        p = [rules[length].get(tuple(row[:length]), rules[0][()]) for row in data['reward']]
        value.append((np.array(p)*data['native']).sum(axis=1))
    return value[0], value[1] - value[0]


def keys(reach):
    # Each neighbor estimate is the hit fraction over three fitting trajectories.
    return [tuple(row) for row in np.rint(3 * reach[:, :2]).astype(int)]


def curve(score, gain):
    # At a tied score a coin flip mixes the two endpoint policies.
    total = 0.
    used = 0
    points = [(0., 0.)]
    for v in sorted(set(score), reverse=True):
        mask = score == v
        used += int(mask.sum())
        total += float(gain[mask].sum())
        points.append((used/len(gain), total))
    x, y = np.array(points).T
    area = float(((y[:-1]+y[1:])*np.diff(x)/2).sum())
    return dict(points=points, excess_area_over_random=area-total/2,
                half_budget_gain=float(np.interp(.5, x, y)))


def main(source, target, output, primary):
    panel = load_panel(source / 'fitting')
    source_scores(source, panel, Reachability(source / 'frozen.npz'))
    train = tapes(panel, list(combinations(KNOWN, 3)))
    rules = fit_tables(train)
    _, delta = gains(train, rules)
    neg = train['reward'][:,0] == 0
    bins = {}
    for key, gain in zip(keys(train['reach'][neg]), delta[neg]):
        bins.setdefault(key, []).append(gain)
    ranking = {k:float(np.mean(v)) for k,v in bins.items()}
    triples = [(0, 1, 5)] if primary else [(0, 1, g) for g in KNOWN]
    data = tapes(load_panel(target), triples)
    one, delta = gains(data, rules)
    neg = data['reward'][:,0] == 0
    gain = delta[neg]
    reach = data['reach'][neg]
    prior = float(np.mean([v for group in bins.values() for v in group]))
    trained = np.array([ranking.get(key, prior) for key in keys(reach)])
    report = dict(assignments=len(neg), eligible=int(neg.sum()), one_success=float(one.sum()),
                  matched_random_half_gain=float(gain.sum()/2),
                  source_gain_by_bin={str(k):dict(count=len(bins[k]),mean=v) for k,v in ranking.items()},
                  fitted=curve(trained, gain), second_reach=curve(reach[:,1], gain),
                  average_gain_if_all_negative_probed=float(gain.mean()),
                  unavailable_at_fit=sum(k not in ranking for k in keys(reach)))
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(output.name, {k:report[k] for k in ('assignments', 'eligible', 'one_success', 'matched_random_half_gain')},
          {k:report[k]['half_budget_gain'] for k in ('fitted', 'second_reach')})


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:4]), primary=sys.argv[4]=='primary')
