"""Verify and compare complete matched controls on allocated Linux CPU."""
import json
import math
from pathlib import Path
import runpy
import sys
import numpy as np
import torch

assert sys.platform == 'linux' and not torch.cuda.is_available()
m = runpy.run_path(str(Path(__file__).with_name('CONTROL.py')))
OUT = m['OUT']; ref = m['ref']; read_ref = m['read_ref']; sha = m['sha']
b = json.loads((OUT / 'binding.json').read_text())
for rank in range(8):
    w = json.loads((OUT / 'workers' / f'rank-{rank:02d}.json').read_text())
    assert w['complete'] and w['weights_unchanged'] and w['binding'] == ref(OUT / 'binding.json')


def cluster_interval(effects):
    values = np.asarray(effects, dtype=float)
    rng = np.random.default_rng(20261102)
    samples = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), case_bootstrap95=np.quantile(samples, [.025, .975]).tolist())


def sign_test(effects):
    # Exact paired sign-flip test, also valid for the integer difference-in-differences.
    observed = abs(sum(effects)); distribution = {0: 1}
    for value in (abs(x) for x in effects if x):
        updated = {}
        for total, count in distribution.items():
            for target in (total - value, total + value):
                updated[target] = updated.get(target, 0) + count
        distribution = updated
    return sum(count for total, count in distribution.items() if abs(total) >= observed) / sum(distribution.values())


results = {}; comparisons = []; verified = 0
for family, model in b['models'].items():
    original = read_ref(model['original_result'])
    arms = {arm: {c: {str(k): [] for k in range(1, 6)} for c in ('ID', 'OOD')} for arm in b['arms']}
    for arm in b['arms']:
        for c in ('ID', 'OOD'):
            for i in range(50):
                for k in range(1, 6):
                    old = read_ref(model['episodes'][f'{c}-{i:03d}'][str(k)])
                    if m['reusable'](arm, c, k):
                        r = old
                    else:
                        path = OUT / 'episodes' / family / arm / c / f'case-{i:03d}' / f'p{k}' / 'result.json'
                        r = json.loads(path.read_text())
                        assert r['complete'] and r['family'] == family and r['arm'] == arm
                        assert r['source'] == b['source'] and r['candidate'] == model['selected']['candidate']
                        assert r['original'] == model['episodes'][f'{c}-{i:03d}'][str(k)]
                        assert r['native_case'] == old['native_case'] and r['motor_seeds'] == old['motor_seeds']
                        assert sha(r['records']['path']) == r['records']['sha256']
                        truth = read_ref(r['truth'])
                        assert truth['case'] == r['native_case'] and int(truth['simulator_final_success']) == r['success']
                        pair = r['initial_pairing']; marker = read_ref(pair['canonical'])
                        canonical = Path(pair['canonical']['path']).parent / 'initial.pt'
                        assert sha(canonical) == marker['snapshot_sha256']
                        assert sha(pair['snapshot']['path']) == pair['snapshot']['sha256']
                        report = m['PAIRING']['compare_native_pair'](
                            torch.load(canonical, map_location='cpu', weights_only=False),
                            torch.load(pair['snapshot']['path'], map_location='cpu', weights_only=False))
                        assert report['matched'] and report == pair['report']
                        rows = torch.load(r['records']['path'], map_location='cpu', weights_only=False)
                        assert len(rows) == m['horizon'](c) // 50 and rows[-1]['done'] and rows[-1]['reward'] == r['success']
                        assert r['memory_entry']['cache_length'] == (0 if arm == 'reset' else min(64, (k - 1) * m['horizon'](c) // 50))
                        if arm == 'window64':
                            assert r['final_cache_length'] == min(64, k * m['horizon'](c) // 50)
                        verified += 1
                    arms[arm][c][str(k)].append(int(r['success']))
            for k in range(1, 6):
                control = arms[arm][c][str(k)]
                full = original['cohorts'][c]['attempt_cases'][str(k)]
                effects = [x - y for x, y in zip(full, control, strict=True)]
                comparisons.append(dict(family=family, cohort=c, control=arm, attempt=k,
                    full_successes=sum(full), control_successes=sum(control),
                    full_wins=sum(x > 0 for x in effects), full_losses=sum(x < 0 for x in effects),
                    full_minus_control=cluster_interval(effects), exact_p=sign_test(effects)))
    pooled = {}
    for c in ('ID', 'OOD'):
        full = original['cohorts'][c]['attempt_cases']
        reset = arms['reset'][c]
        effects = [(full['5'][i] - full['1'][i]) - (reset['5'][i] - reset['1'][i]) for i in range(50)]
        average = [sum(full[str(k)][i] - reset[str(k)][i] for k in range(1, 6)) / 5 for i in range(50)]
        pooled[c] = dict(attempt5_minus1_difference_in_differences=cluster_interval(effects),
            difference_in_differences_exact_p=sign_test(effects), across_five_streams_full_minus_reset=cluster_interval(average))
    results[family] = dict(full=original['cohorts'], controls=arms, memory_effect=pooled)
# One declared family for the40 per-attempt exploratory tests, including unchanged endpoints.
ordered = sorted(comparisons, key=lambda row: row['exact_p']); adjusted = 0.
for index, row in enumerate(ordered):
    adjusted = max(adjusted, min(1., row['exact_p'] * (len(ordered) - index)))
    row['holm_p_40_comparisons'] = adjusted
assert verified == 1100
r = dict(complete=True, binding=ref(OUT / 'binding.json'), verified_fresh_episodes=verified,
    evaluation_endpoints=2000, models=results, comparisons=comparisons,
    limits=['50 cases per task; repeated streams remain correlated within each case.',
            'One source task/fit seed; controls are exploratory mechanism diagnostics.',
            'Native noise screen uses different TRAIN cases and horizons.',
            'No new PPO, source-history collection, or Flora optimization.'])
m['write'](OUT / 'RESULT.json', r)
lines = ['# Matched memory controls', '', 'True final simulator success per50 cases; frozen checkpoints and original p1..p5 motor streams.', '',
         '| Model / cohort / arm | p1 | p2 | p3 | p4 | p5 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
for family, result in results.items():
    for c in ('ID', 'OOD'):
        lines.append('| ' + family + ' / ' + c + ' / full | ' + ' | '.join(str(result['full'][c]['attempt_successes'][str(k)]) for k in range(1, 6)) + ' |')
        for arm in b['arms']:
            lines.append('| ' + family + ' / ' + c + ' / ' + arm + ' | ' + ' | '.join(str(sum(result['controls'][arm][c][str(k)])) for k in range(1, 6)) + ' |')
lines += ['', 'Per-attempt exact paired tests, Holm adjustment over40 comparisons, case-bootstrap intervals and the attempt5-minus1 difference in differences are in [RESULT.json](RESULT.json). Reused endpoints add no new episodes. These50-case controls remain exploratory; no broad head-to-head or ICRL claim follows from a small count difference.']
(OUT / 'RESULT.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(dict(complete=True, verified_fresh_episodes=verified, result=ref(OUT / 'RESULT.json'))))
