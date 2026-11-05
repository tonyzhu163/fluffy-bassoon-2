"""Reduce audited paired six-episode waves without evaluating or selecting weights."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def reduce_group(group):
    group = Path(group)
    if (group / 'SUMMARY.json').exists():
        return json.loads((group / 'SUMMARY.json').read_text())
    result = json.loads((group / 'RESULT.json').read_text())
    assert result['completed'] and result['deployment_optimizer_updates'] == 0
    banks = sorted(group.glob('*/*/bank-*/RESULT.json'))
    assert len(banks) == len(result['results'])
    panels, references = {}, []
    for path in banks:
        bank = json.loads(path.read_text())
        audit_path = path.with_name('AUDIT.json')
        if not audit_path.exists():
            return None
        audit = json.loads(audit_path.read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert audit['passed'] and audit['result_sha256'] == digest
        references.append(dict(path=str(path), sha256=digest,
                               audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest()))
        panel = panels.setdefault(bank['task'], {}).setdefault(bank['mode'], {})
        for row in bank['episodes']:
            case = json.dumps(row['case'], sort_keys=True)
            trial = panel.setdefault(case, {})
            assert row['attempt'] not in trial
            trial[row['attempt']] = (int(row['success_once']), int(row['success_at_end']))
    curves, paired = {}, {}
    for task, modes in panels.items():
        assert modes.keys() == {'full', 'episode'} and modes['full'].keys() == modes['episode'].keys()
        cases = sorted(modes['full'])
        assert len(cases) == 50 and all(set(trial) == set(range(1, 7)) for mode in modes.values() for trial in mode.values())
        curves[task] = {}
        for mode in ('full', 'episode'):
            first = np.array([[modes[mode][case][j][0] for j in range(1, 7)] for case in cases])
            end = np.array([[modes[mode][case][j][1] for j in range(1, 7)] for case in cases])
            curves[task][mode] = dict(denominator=50, first_success=first.sum(0).tolist(),
                final_success=end.sum(0).tolist(), ever_success=np.maximum.accumulate(first, axis=1).sum(0).tolist())
        paired[task] = {}
        for query in range(1, 7):
            delta = np.array([modes['full'][case][query][0]-modes['episode'][case][query][0] for case in cases])
            wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
            n = wins + losses
            exact = 1. if n == 0 else min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses)+1))/2**n)
            seed = int.from_bytes(hashlib.sha256((str(group)+'/'+task+'/'+str(query)).encode()).digest()[:8], 'little')
            indices = np.random.default_rng(seed).integers(50, size=(10000, 50))
            interval = np.percentile(delta[indices].mean(1)*100, [2.5, 97.5]).tolist()
            paired[task][str(query-1)] = dict(delta_percentage_points=float(delta.mean()*100),
                wins=wins, losses=losses, exact_McNemar_p=exact,
                paired_case_bootstrap95_percentage_points=interval,
                inference='Descriptive, marginal intervals; no multiplicity-adjusted adaptation claim')
    summary = dict(completed=True, interface=result['binding']['interface'],
        objective=result['binding']['objective'], split=result['binding']['split'],
        selected_candidate_sha256=result['binding']['selected_candidate_sha256'],
        prior_episodes=[0, 1, 2, 3, 4, 5], cases_per_task=50,
        curves=curves, paired_full_minus_reset=paired, source_banks=references,
        episodes=len(panels)*50*6*2, native_parameters_frozen=True, deployment_optimizer_updates=0)
    target = group/'SUMMARY.json';tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2)+'\n');tmp.replace(target)
    return summary
