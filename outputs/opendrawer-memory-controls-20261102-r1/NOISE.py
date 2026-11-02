"""Summarize the already-audited native draws; no model or new episodes."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def wilson(k, n):
    z = 1.959963984540054
    center = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((k / n * (1 - k / n) + z * z / (4 * n)) / n) ** .5 / (1 + z * z / n)
    return [max(0., center - half), min(1., center + half)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audits', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    tasks, cases = {}, []
    for path in sorted(args.audits.glob('*-native8.json')):
        audit = json.loads(path.read_text())
        assert audit['completed'] and audit['physical_and_public_pairing_exact']
        draws = defaultdict(dict)
        for identity, success in audit['combined_outcomes'].items():
            seed, draw = map(int, identity.split(':'))
            assert type(success) is bool
            draws[seed][draw] = int(success)
        assert len(draws) == 20
        histogram = Counter()
        for seed, outcomes in sorted(draws.items()):
            assert set(outcomes) == set(range(8))
            k = sum(outcomes.values()); histogram[k] += 1
            cases.append(dict(task=audit['task'], native_seed=seed, successes=k, draws=8,
                probability_estimate=k / 8, wilson95=wilson(k, 8),
                observed_class='all_fail' if k == 0 else 'all_success' if k == 8 else 'mixed',
                outcomes=[outcomes[i] for i in range(8)],
                pair_disagreement=2 * k * (8 - k) / (8 * 7)))
        task_cases = [row for row in cases if row['task'] == audit['task']]
        tasks[audit['task']] = dict(cases=20, episodes=160,
            histogram={str(k): histogram[k] for k in range(9)},
            observed_all_fail=histogram[0], observed_all_success=histogram[8],
            observed_mixed=20 - histogram[0] - histogram[8],
            near_half_3_to_5_successes=sum(histogram[k] for k in (3, 4, 5)),
            successes=sum(row['successes'] for row in task_cases),
            mean_success_probability=sum(row['probability_estimate'] for row in task_cases) / 20,
            mean_pair_disagreement=sum(row['pair_disagreement'] for row in task_cases) / 20,
            observed_best_of_8_recoveries=20 - histogram[0],
            audit=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(tasks) == 4
    result = dict(complete=True, tasks=tasks, cases=cases, total_cases=len(cases),
        total_episodes=len(cases) * 8, new_episodes=0,
        selection='20 selected anchor failures per task; H600, development TRAIN cases',
        limits=['Conditional on anchor failure; not an estimate over the whole task distribution.',
                'Observed all-fail/all-success with eight draws does not establish p=0/p=1.',
                'Best-of-8 uses simulator truth for selection; it is an oracle diagnostic, not deployable ICRL or a hard ceiling.',
                'These tasks/horizons/cases differ from the H750/H900 AD/DPT held-out panel.'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'NOISE.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'NOISE.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'native_seed', 'successes', 'draws', 'p_hat', 'wilson95_low', 'wilson95_high', 'class', 'pair_disagreement'])
        for row in cases:
            writer.writerow([row['task'], row['native_seed'], row['successes'], 8, row['probability_estimate'],
                             *row['wilson95'], row['observed_class'], row['pair_disagreement']])
    print(json.dumps(dict(tasks=tasks, total_cases=len(cases), total_episodes=len(cases) * 8)))


if __name__ == '__main__':
    main()
