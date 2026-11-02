"""Read existing paired AD/DPT outcomes on allocated Linux; no new episodes."""
import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
PHASES = ('opendrawer-ad-first-id-ood-20261102-r2',
          'opendrawer-dpt-full-id-ood-20261102-r2',
          'opendrawer-dpt-static-id-ood-20261102-r1')


def reference(ref):
    data = Path(ref['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == ref['sha256']
    return json.loads(data)


def paired(first, second):
    ids = sorted(first.keys() & second.keys())
    wins = losses = 0
    for index in ids:
        a, b = first[index], second[index]
        assert a['native_case'] == b['native_case']
        assert a['motor_seeds'] == b['motor_seeds']
        assert a['initial_pairing']['canonical'] == b['initial_pairing']['canonical']
        wins += a['success'] > b['success']
        losses += a['success'] < b['success']
    discordant = wins + losses
    p = min(1., 2 * sum(math.comb(discordant, j) for j in range(min(wins, losses) + 1))
            / 2 ** discordant) if discordant else 1.
    return dict(cases=len(ids), complete=len(ids) == 50, wins=wins, losses=losses,
                exact_two_sided_p=p, p_values='unadjusted exploratory')


def main():
    import platform
    assert platform.system() == 'Linux'
    rows, files = {}, []
    for phase in PHASES:
        for path in sorted((ROOT / phase / 'episodes').rglob('result.json')):
            record = json.loads(path.read_text())
            truth = reference(record['truth'])
            assert truth['case'] == record['native_case']
            assert bool(truth['simulator_final_success']) == bool(record['success'])
            assert record['initial_pairing']['report']['matched']
            canonical = reference(record['initial_pairing']['canonical'])
            assert canonical
            key = (record['cohort'], record['method'], record['shot'], record['kind'])
            bank = rows.setdefault(key, {})
            assert record['index'] not in bank
            bank[record['index']] = record
            files.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  computation='Allocated Linux CPU; no new episodes, inference or fitting',
                  purpose='Pipeline readout from flat source PPO histories; no ICRL claim',
                  terminal_truth_and_shared_canonical_refs_verified=True,
                  full_native_snapshot_tensor_replay='Owned by original evaluation audit; not repeated here',
                  existing_evaluation_horizons=dict(ID=750, OOD=900), panels={}, result_refs=files)
    for cohort in ('ID', 'OOD'):
        policies = {}
        for method in ('ppo_initial', 'ppo_final', 'ad_full', 'ad_static', 'dpt_full', 'dpt_static'):
            data = rows.get((cohort, method, 'K1' if method.endswith('full') else 'K0', 'query'), {})
            policies[method] = dict(episodes=len(data), successes=sum(r['success'] for r in data.values()),
                                    complete=len(data) == 50)
        comparisons = {}
        for full, static in (('ad_full', 'ad_static'), ('dpt_full', 'dpt_static')):
            a = rows.get((cohort, full, 'K1', 'query'), {})
            for other, shot in ((static, 'K0'), (full, 'K0')):
                b = rows.get((cohort, other, shot, 'query'), {})
                comparisons[full + '_K1_vs_' + other + '_' + shot] = paired(a, b)
        report['panels'][cohort] = dict(policies=policies, paired=comparisons)
    output = ROOT / 'robocasa-interface-screen-20261102-r1/PIPELINE-READOUT.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'result_refs'}, indent=2))


if __name__ == '__main__':
    main()
