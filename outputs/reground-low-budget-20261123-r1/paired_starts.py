"""Exact conditional bootstrap of the low-budget pilot, grouping seeds by native start."""
from collections import Counter
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(sys.argv[1])
BUDGETS = (15, 100, 1000)
SEEDS = (0, 3)


def interval(task_scores, total):
    distribution = Counter({0: 1})
    for scores in task_scores:
        outcomes = Counter(map(sum, product(scores, repeat=len(scores))))
        combined = Counter()
        for left, multiplicity in distribution.items():
            for right, ways in outcomes.items():
                combined[left + right] += multiplicity * ways
        distribution = combined
    counts = sum(distribution.values())
    answer = []
    for percentile in (25, 975):
        cumulative = 0
        for value, weight in sorted(distribution.items()):
            cumulative += weight
            if cumulative * 1000 >= counts * percentile:
                answer.append(100 * value / total)
                break
    return answer


rows = {}
for budget in BUDGETS:
    for seed in SEEDS:
        panel = json.loads((ROOT / f'paired-{budget}-seed{seed}.json').read_text())
        by_start = {(row['task'], row['start']): row for row in panel['rows']}
        assert len(by_start) == panel['summary']['episodes'] == 32
        assert all(row['base_opening_exact'] and row['reground_opening_exact'] for row in by_start.values())
        rows[budget,seed] = by_start
keys = sorted(rows[15,0])
assert all(sorted(panel) == keys for panel in rows.values())
tasks = sorted({task for task, start in keys})
assert len(tasks) == 8 and all(sum(k[0] == task for k in keys) == 4 for task in tasks)
contrasts = []
for budget in BUDGETS:
    for field in ('success','unsafe'):
        for comparator in ('base','reground'):
            differences = [[sum(rows[budget,s][key][f'low_{field}'] - rows[budget,s][key][f'{comparator}_{field}']
                                for s in SEEDS) for key in keys if key[0] == task] for task in tasks]
            contrasts.append(dict(budget=budget, endpoint=field, comparator=comparator,
                                  difference_pp=100 * sum(map(sum, differences)) / 64,
                                  ci95_pp=interval(differences,64)))
print(json.dumps(dict(fixed_seeds=SEEDS, fixed_tasks=tasks, unique_starts=32,
                      bootstrap='Four starts resampled within each fixed task, with both seeds together.',
                      contrasts=contrasts),indent=2))
