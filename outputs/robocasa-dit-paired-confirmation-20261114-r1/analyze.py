"""Verify archived paired cases, then report all predeclared DIT contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

from job import sha
from readout import ARMS, TASKS, HORIZONS, write


def key(case):
    return hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def paired_statistics(outcomes, *, replicates=10000, seed=20261114):
    import numpy as np
    # Every draw selects the complete four-arm case, independently within task.
    assert outcomes.ndim == 3 and outcomes.shape[0] == 5 and outcomes.shape[2] == 4
    assert np.isin(outcomes, [0, 1]).all()
    count = outcomes.shape[1]
    rng = np.random.default_rng(seed)
    boot = np.stack([rows[rng.integers(count, size=(replicates, count))].mean(axis=1) for rows in outcomes])

    def panel(indices):
        observed = outcomes[indices].mean(axis=(0, 1))
        draws = boot[indices].mean(axis=0)
        contrasts = {}
        for control in range(1, 4):
            delta = outcomes[indices, :, 0] - outcomes[indices, :, control]
            contrasts[ARMS[control]] = dict(difference_pp=float(100 * (observed[0] - observed[control])),
                paired_bootstrap95_pp=(100 * np.percentile(draws[:, 0] - draws[:, control], [2.5, 97.5])).tolist(),
                DIT_wins=int((delta > 0).sum()), DIT_losses=int((delta < 0).sum()),
                ties=int((delta == 0).sum()), paired_cases=int(delta.size))
        return dict(tasks=[TASKS[i] for i in indices], cases_per_task=count,
            success_percent=dict(zip(ARMS, (100 * observed).tolist())), contrasts=contrasts)

    return dict(tasks={task: panel([i]) for i, task in enumerate(TASKS)},
        panels={'ID': panel([0, 1, 2]), 'OOD': panel([3, 4])},
        bootstrap_replicates=replicates, bootstrap_seed=seed,
        uncertainty='Task-stratified paired case bootstrap, equal task weighting; marginal95% intervals without multiplicity correction. Conditional on one fixed DIT fit and these five tasks, not training-seed or task-population uncertainty.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--decision-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux', 'Analyze on allocated Linux CPUs'
    import numpy as np
    assert sha(args.decision) == args.decision_sha256
    decision = json.loads(args.decision.read_text())
    launch = json.loads((args.run / 'LAUNCH.json').read_text())
    assert launch['decision_sha256'] == args.decision_sha256 and launch['decision'] == decision
    assert decision['outcome_blind_count_fixed'] and decision['shards'] == 4
    qualification = Path(decision['qualification']['path'])
    assert sha(qualification) == decision['qualification']['sha256']
    qualified = json.loads(qualification.read_text())
    assert qualified['completed'] and qualified['binding']['stage'] == 'qualification'
    count = decision['cases_per_task']
    outcomes = np.full((5, count, 4), -1, dtype=np.int8)
    curves = np.full((5, count, 6), -1, dtype=np.int8)
    sources, seen, worker_states = [], set(), []
    for shard in range(4):
        root = args.run / f'shard-{shard:02d}'
        binding_path = root / 'evaluation/BINDING.json'
        if not binding_path.exists():
            worker_states.append(dict(shard=shard, status='No evaluation binding'))
            continue
        binding = json.loads(binding_path.read_text())
        assert binding['stage'] == 'confirmation' and binding['shard'] == shard and binding['shards'] == 4
        assert binding['cases_per_task'] == count and binding['source_commit'] == launch['source_commit']
        assert binding['source_inputs'] == qualified['binding']['source_inputs']
        for field in ('model_sha256', 'norm_sha256', 'runtime_commit'):
            assert binding['signature'][field] == qualified['binding']['signature'][field]
        assert binding['deployment_optimizer_updates'] == 0 and binding['endpoint_attempt'] == 6
        assert binding['selected_DIT_sha256'] == '7ccaae7048df38a90ced30530002469f1f73e1ee9cb37a17b07c8137cabb3721'
        terminal = root / 'EXIT.json'
        worker_states.append(json.loads(terminal.read_text()) if terminal.exists() else dict(shard=shard, status='No terminal receipt'))
        for ti, task in enumerate(TASKS):
            for first in range(0, count, 8):
                if (ti + first // 8) % 4 != shard:
                    continue
                directory = root / 'evaluation' / task / f'bank-{first:04d}'
                receipt_path = directory / 'ARCHIVE.json'
                paired_path = directory / 'PAIRED.json'
                if not receipt_path.exists() or not paired_path.exists():
                    continue
                receipt = json.loads(receipt_path.read_text())
                assert receipt['every_member_verified'] and receipt['cold_full_readback']
                archive = Path(receipt['archive'])
                assert archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256']
                with tarfile.open(archive, 'r') as bank:
                    raw = bank.extractfile('PAIRED.json').read()
                assert hashlib.sha256(raw).hexdigest() == sha(paired_path)
                paired = json.loads(raw)
                assert paired['task'] == task and paired['first'] == first and set(paired['arms']) == set(ARMS)
                assert paired['reset_qualification'] is None
                expected = set(range(first, min(first + 8, count)))
                assert {entry['case']['slot'] for entry in paired['cases']} == expected
                keys = {key(entry['case']): entry['case'] for entry in paired['cases']}
                for entry in paired['cases']:
                    case = entry['case']
                    assert case == dict(bank='paired_DIT_confirmation_20261114', task=task, split='target',
                        native_seed=72140000 + ti * 10000 + case['slot'], slot=case['slot'], reset_ordinal=0,
                        counted=True, layout_source='native_target_constructor_seed_first_reset')
                    assert entry['horizon'] == HORIZONS[ti]
                for ai, arm in enumerate(ARMS):
                    row = paired['arms'][arm]
                    attempts = range(1, 7) if ai == 0 else (6,)
                    assert row['all_native_evidence_checked'] and set(row['endpoint']) == set(keys)
                    assert row['physical_episodes'] == len(expected) * len(attempts) == len(row['episodes'])
                    assert row['physical_controls'] == row['physical_episodes'] * HORIZONS[ti]
                    covered = set()
                    for ep in row['episodes']:
                        case_id, attempt = key(ep['case']), ep['attempt']
                        assert case_id in keys and ep['case'] == keys[case_id] and attempt in attempts
                        assert ep['completed'] and ep['horizon'] == HORIZONS[ti] and type(ep['success_once']) is bool
                        assert (case_id, attempt) not in covered
                        covered.add((case_id, attempt))
                        if ai == 0:
                            curves[ti, ep['case']['slot'], attempt - 1] = int(ep['success_once'])
                    for case_id, ep in row['endpoint'].items():
                        case = keys[case_id]
                        assert ep['case'] == case and ep['attempt'] == 6 and type(ep['success_once']) is bool
                        assert ai == 0 or ep['reset_from_empty_replay_exact']
                        slot = case['slot']
                        assert (ti, slot, ai) not in seen
                        seen.add((ti, slot, ai))
                        outcomes[ti, slot, ai] = int(ep['success_once'])
                        raw_endpoint = next(x for x in row['episodes'] if key(x['case']) == case_id and x['attempt'] == 6)
                        assert ep['success_once'] == raw_endpoint['success_once']
                sources.append(dict(task=task, first=first, archive=str(archive),
                    archive_sha256=receipt['sha256'], paired_sha256=sha(paired_path)))
    args.output.mkdir(parents=True, exist_ok=False)
    missing = {task: np.flatnonzero((outcomes[ti] < 0).any(axis=1)).tolist() for ti, task in enumerate(TASKS)}
    coverage = dict(planned_cases_per_task=count, missing_paired_slots=missing, workers=worker_states,
        verified_banks=sources, complete=not any(missing.values()),
        exclusions='Only fully audited four-arm banks accepted. Missing/failed cases remain explicit; never replace based on success.')
    write(args.output / 'COVERAGE.json', coverage)
    if not coverage['complete']:
        raise SystemExit('Incomplete fixed panel: wrote coverage; no final effect estimates or significance decision')
    assert np.isin(curves, [0, 1]).all() and np.array_equal(curves[:, :, 5], outcomes[:, :, 0])
    report = dict(completed=True, decision=decision, decision_sha256=args.decision_sha256,
        source_run=str(args.run), verified_physical_episodes=5 * count * 9,
        full_history_success_curves={task: (100 * curves[ti].mean(axis=0)).tolist() for ti, task in enumerate(TASKS)},
        **paired_statistics(outcomes))
    write(args.output / 'RESULT.json', report)
    lines = ['# Frozen DIT paired confirmation', '',
        f'{count} fresh paired cases per task; {5 * count * 9:,} audited physical episodes. Endpoint: success during episode six.', '',
        '| Panel | DIT full | DIT reset | SFT | PPO130 |', '| --- | ---: | ---: | ---: | ---: |']
    for panel, row in report['panels'].items():
        lines.append('| ' + panel + ' | ' + ' | '.join(f'{row["success_percent"][arm]:.1f}%' for arm in ARMS) + ' |')
    lines += ['', '| Panel | DIT contrast | Difference (pp) | Paired 95% interval (pp) | Wins / losses |',
              '| --- | --- | ---: | --- | ---: |']
    for panel, row in report['panels'].items():
        for control, effect in row['contrasts'].items():
            lo, hi = effect['paired_bootstrap95_pp']
            lines.append(f'| {panel} | Full − {control} | {effect["difference_pp"]:+.2f} | [{lo:.2f}, {hi:.2f}] | {effect["DIT_wins"]} / {effect["DIT_losses"]} |')
    lines += ['', report['uncertainty'], '',
        'History benefit and advantage over SFT/PPO support separate claims. These results do not explain why DIT differs, establish seed robustness, or isolate task diversity. OOD is relative to the meta-training corpus. No continue/kill numerical threshold was predeclared.', '']
    (args.output / 'RESULT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(completed=True, cases_per_task=count, panels=report['panels']), indent=2))


if __name__ == '__main__':
    main()
