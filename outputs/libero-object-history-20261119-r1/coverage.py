"""Measure what public rewards could identify with perfect knowledge of practice completion."""
import json
from pathlib import Path
import sys

import numpy as np


def outcomes(folder, pose, goal, kind):
    record = json.loads((folder / f'p{pose}-g{goal}-{kind}' / 'result.json').read_text())
    return np.asarray(record['ever'], dtype=int)


def summarize(folder):
    with (folder / 'choices.json').open() as file:
        rows = json.load(file)['rows']
    records = []
    for row in rows:
        first = row['first']
        candidates = np.ones(2, dtype=bool)
        practices = [(row['p'], first, row['b']), (row['s'], 1-first, row['c'])]
        for (p, g, interrupted), feedback in zip(practices, row['r']):
            completed = outcomes(folder, p, g, 'interrupt' if interrupted else 'sham')
            candidates &= completed == feedback
        assert candidates[row['h']]
        goal_probs = candidates / candidates.sum()
        native = np.array([outcomes(folder, row['q'], g, 'query')[row['h']] for g in (0, 1)])
        records.append((not any(row['r']), candidates.sum()==1,
                        goal_probs[row['h']], goal_probs @ native, native.mean(), native[1-row['h']]))
    values = np.asarray(records)
    report = {}
    for name, a in [('all', values), ('zero', values[values[:, 0]==1])]:
        report[name] = dict(n=len(a), uniquely_identified=int(a[:,1].sum()),
                            perfect_completion_goal=float(a[:,2].sum()), perfect_completion_native=float(a[:,3].sum()),
                            random_native=float(a[:,4].sum()), opposite_prompt_native=float(a[:,5].sum()))
    return report


if __name__=='__main__':
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
