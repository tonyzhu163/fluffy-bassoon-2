"""Evaluate the cost of asking for a second reward using source-fitted query rules."""
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

from reader import KNOWN, distributions, load_panel, tapes


def observations(root, triples, fitting=False):
    panel = load_panel(root)
    if fitting:
        # This analysis uses only rewards. Source pre-action scores were never saved.
        for block in panel.values():
            block['meta']['scores'] = {str(g): [0.] for g in block['records']}
    return tapes(panel, triples)


def fit_tables(source):
    reward = source['reward']
    native = source['native']
    rules = []
    for length in (0, 1, 2):
        keys = {tuple(row[:length]) for row in reward}
        rule = {}
        for key in keys:
            match = (reward[:, :length] == np.array(key)).all(axis=1)
            rule[key] = distributions(native[match].mean(axis=0, keepdims=True))[0]
        rules.append(rule)
    return rules


def evaluate(data, rules):
    n = len(data['pose'])
    native = data['native']
    bits = data['reward']
    outcomes = []
    for length, rule in enumerate(rules):
        p = np.array([rule.get(tuple(b[:length]), rules[0][()]) for b in bits])
        outcomes.append((p*native).sum(axis=1))
    again = bits[:, 0] == 0
    q = float(again.mean())
    adaptive = np.where(again, outcomes[2], outcomes[1])
    independent = q*outcomes[2] + (1-q)*outcomes[1]
    return dict(assignments=n, num_second_probes=int(again.sum()), mean_probes=1+q,
                blind=float(outcomes[0].sum()), one=float(outcomes[1].sum()),
                two=float(outcomes[2].sum()), stop_on_positive=float(adaptive.sum()),
                same_cost_random=float(independent.sum()),
                by_pose={str(p):float((adaptive-independent)[data['pose']==p].sum()) for p in np.unique(data['pose'])})


if __name__ == '__main__':
    source, target, output = map(Path, sys.argv[1:4])
    rules = fit_tables(observations(source, list(combinations(KNOWN, 3)), fitting=True))
    triples = [(0, 1, 5)] if sys.argv[4] == 'primary' else [(0, 1, g) for g in KNOWN]
    report = evaluate(observations(target, triples), rules)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(report)
