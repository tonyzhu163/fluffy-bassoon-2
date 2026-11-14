"""Paired three-versus-six task comparisons for all eight fixed-final KV fits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

from labels import sha, write
from matched_readout import TASKS, HORIZONS, METHODS, MODELS, BASELINES, load_baselines

ENDPOINTS = ('full_first', 'full_sixth', 'reset_first', 'reset_sixth')


def key(case):
    return hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def shared_binding(binding):
    # Each shard has its own allocated GPU. Keep the full scientific signature
    # equal without requiring distinct physical devices to have the same UUID.
    identity = {k: v for k, v in binding.items() if k not in ('shard', 'signature')}
    identity['signature'] = {k: v for k, v in binding['signature'].items()
                             if k not in ('cuda_uuid', 'scheduler_uuid')}
    return identity


def paired_statistics(outcomes, *, baselines=None, replicates=10000, seed=20261115):
    import numpy as np
    assert outcomes.shape == (8, 5, 40, 4) and np.isin(outcomes, [0, 1]).all()
    outcomes = outcomes.astype(np.int8)
    rng = np.random.default_rng(seed)
    # The same task-stratified case draws retain all models and both endpoints.
    draws = [rng.integers(40, size=(replicates, 40)) for _ in range(5)]
    boot = np.stack([np.moveaxis(outcomes[:, ti], 0, 1)[draws[ti]].mean(axis=1)
                     for ti in range(5)])
    baseline_boot = None
    if baselines is not None:
        assert baselines.shape == (2, 5, 40) and np.isin(baselines, [0, 1]).all()
        baselines = baselines.astype(np.int8)
        baseline_boot = np.stack([np.moveaxis(baselines[:, ti], 0, 1)[draws[ti]].mean(axis=1) for ti in range(5)])

    def panel(indices):
        observed = outcomes[:, indices].mean(axis=(1, 2))
        samples = boot[indices].mean(axis=0)

        def effect(values, draws):
            return dict(difference_pp=float(100 * values.mean()),
                paired_bootstrap95_pp=(100 * np.percentile(draws, [2.5, 97.5])).tolist())

        models = {}
        for mi, name in enumerate(MODELS):
            rows = outcomes[mi, indices]
            delta = rows[:, :, 1] - rows[:, :, 3]
            history = effect(delta, samples[:, mi, 1] - samples[:, mi, 3])
            history.update(full_wins=int((delta > 0).sum()), full_losses=int((delta < 0).sum()), ties=int((delta == 0).sum()))
            models[name] = dict(success_percent=dict(zip(ENDPOINTS, (100 * observed[mi]).tolist())),
                history_effect=history, first_attempt_disagreements=int((rows[:, :, 0] != rows[:, :, 2]).sum()))
        scaling = {}
        for i, method in enumerate(METHODS):
            small, large = outcomes[i, indices], outcomes[i + 4, indices]
            small_draws, large_draws = samples[:, i], samples[:, i + 4]
            scaling[method] = {endpoint: effect(large[:, :, ei] - small[:, :, ei], large_draws[:, ei] - small_draws[:, ei])
                               for ei, endpoint in enumerate(ENDPOINTS)}
            scaling[method]['history_effect_difference'] = effect(
                (large[:, :, 1] - large[:, :, 3]) - (small[:, :, 1] - small[:, :, 3]),
                (large_draws[:, 1] - large_draws[:, 3]) - (small_draws[:, 1] - small_draws[:, 3]))
        within_corpus = {}
        for offset, size in ((0, 3), (4, 6)):
            comparisons = {}
            for i in range(len(METHODS)):
                for j in range(i + 1, len(METHODS)):
                    left, right = outcomes[offset + i, indices], outcomes[offset + j, indices]
                    left_draws, right_draws = samples[:, offset + i], samples[:, offset + j]
                    comparison = {endpoint: effect(left[:, :, ei] - right[:, :, ei],
                        left_draws[:, ei] - right_draws[:, ei]) for ei, endpoint in enumerate(ENDPOINTS)}
                    for ei, endpoint in enumerate(ENDPOINTS):
                        delta = left[:, :, ei] - right[:, :, ei]
                        comparison[endpoint].update(left_wins=int((delta > 0).sum()),
                            left_losses=int((delta < 0).sum()), ties=int((delta == 0).sum()))
                    comparison['history_effect_difference'] = effect(
                        (left[:, :, 1] - left[:, :, 3]) - (right[:, :, 1] - right[:, :, 3]),
                        (left_draws[:, 1] - left_draws[:, 3]) - (right_draws[:, 1] - right_draws[:, 3]))
                    comparisons[f'{METHODS[i]}_minus_{METHODS[j]}'] = comparison
            within_corpus[str(size)] = comparisons
        result = dict(tasks=[TASKS[i] for i in indices], cases_per_task=40, models=models,
            six_minus_three=scaling, within_corpus_method_differences=within_corpus)
        if baseline_boot is not None:
            base_samples = baseline_boot[indices].mean(axis=0)
            result['baseline_success_percent'] = dict(zip(BASELINES, (100 * baselines[:, indices].mean(axis=(1, 2))).tolist()))
            delta = baselines[1, indices] - baselines[0, indices]
            baseline_gain = effect(delta, base_samples[:, 1] - base_samples[:, 0])
            baseline_gain.update(PPO_wins=int((delta > 0).sum()), PPO_losses=int((delta < 0).sum()), ties=int((delta == 0).sum()))
            result['PPO30_minus_SFT'] = baseline_gain
            for mi, name in enumerate(MODELS):
                comparisons = {}
                for bi, baseline in enumerate(BASELINES):
                    comparisons[baseline] = {}
                    for endpoint, ei in (('full_sixth', 1), ('reset_sixth', 3)):
                        delta = outcomes[mi, indices, :, ei] - baselines[bi, indices]
                        value = effect(delta, samples[:, mi, ei] - base_samples[:, bi])
                        value.update(model_wins=int((delta > 0).sum()), model_losses=int((delta < 0).sum()), ties=int((delta == 0).sum()))
                        comparisons[baseline][endpoint] = value
                models[name]['versus_baselines'] = comparisons
        return result

    return dict(tasks={task: panel([i]) for i, task in enumerate(TASKS)},
        panels={'ID': panel([0, 1, 2]), 'OOD': panel([3, 4])}, bootstrap_replicates=replicates, bootstrap_seed=seed,
        uncertainty='Task-stratified paired case bootstrap preserving all eight models, equal task weighting, marginal95% intervals without multiplicity correction. All six method pairs within each corpus are descriptive comparisons, not a familywise winner test. One fit per cell; no training-seed or task-population uncertainty.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux'
    import numpy as np
    assert sha(args.spec) == args.spec_sha256
    spec = json.loads(args.spec.read_text())
    baseline_spec = load_baselines(spec['baseline_controls'])
    launch = json.loads((args.run / 'LAUNCH.json').read_text())
    assert spec == launch['spec'] and args.spec_sha256 == launch['spec_sha256']
    assert spec['cases_per_task'] == 40 and spec['shards'] == 4 and set(spec['models']) == set(MODELS)
    data = np.full((8, 5, 40, 4), -1, dtype=np.int8)
    curves = np.full((8, 5, 40, 6), -1, dtype=np.int8)
    baseline_data = np.full((2, 5, 40), -1, dtype=np.int8)
    sources, workers, seen = [], [], set()
    common = None
    for shard in range(4):
        root = args.run / f'shard-{shard:02d}'
        binding_path = root / 'evaluation/BINDING.json'
        if not binding_path.exists():
            workers.append(dict(shard=shard, status='No evaluation binding'))
            continue
        binding = json.loads(binding_path.read_text())
        assert binding['models'] == spec['models'] and binding['spec_sha256'] == args.spec_sha256
        assert binding['baseline_controls'] == spec['baseline_controls'] and binding['baselines'] == baseline_spec
        assert binding['source_commit'] == launch['source_commit'] and binding['shard'] == shard and binding['shards'] == 4
        assert binding['deployment_optimizer_updates'] == 0 and binding['cases_per_task'] == 40
        assert binding['full_attempts'] == [1, 2, 3, 4, 5, 6] and binding['reset_physical_attempts'] == [1, 6]
        identity = shared_binding(binding)
        if common is None:
            common = identity
        assert common == identity
        terminal = root / 'EXIT.json'
        workers.append(json.loads(terminal.read_text()) if terminal.exists() else dict(shard=shard, status='No terminal receipt'))
        for ti, task in enumerate(TASKS):
            for first in range(0, 40, 8):
                if (ti + first // 8) % 4 != shard:
                    continue
                bank_root = root / 'evaluation' / task / f'bank-{first:04d}'
                receipt_path, paired_path = bank_root / 'ARCHIVE.json', bank_root / 'PAIRED.json'
                if not receipt_path.exists() or not paired_path.exists():
                    continue
                receipt = json.loads(receipt_path.read_text())
                archive = Path(receipt['archive'])
                assert receipt['every_member_verified'] and receipt['cold_full_readback']
                assert archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256']
                with tarfile.open(archive, 'r') as bank:
                    raw = bank.extractfile('PAIRED.json').read()
                assert hashlib.sha256(raw).hexdigest() == sha(paired_path)
                paired = json.loads(raw)
                assert paired['spec_sha256'] == args.spec_sha256 and set(paired['models']) == set(MODELS)
                assert paired['task'] == task and paired['first'] == first
                assert {item['case']['slot'] for item in paired['cases']} == set(range(first, first + 8))
                cases = {key(item['case']): item['case'] for item in paired['cases']}
                assert len(cases) == 8
                for item in paired['cases']:
                    case = item['case']
                    assert case == dict(bank='matched_three_six_20261114', task=task, split='target',
                        native_seed=73140000 + ti * 10000 + case['slot'], slot=case['slot'], reset_ordinal=0,
                        counted=True, layout_source='native_target_constructor_seed_first_reset')
                    assert item['horizon'] == HORIZONS[ti]
                for mi, name in enumerate(MODELS):
                    assert set(paired['models'][name]) == {'full', 'reset_first', 'reset_sixth'}
                    for arm, attempts, end in [('full', range(1, 7), 6), ('reset_first', (1,), 1), ('reset_sixth', (6,), 6)]:
                        row = paired['models'][name][arm]
                        assert row['all_native_evidence_checked'] and set(row['endpoint']) == set(cases)
                        assert row['physical_episodes'] == len(row['episodes']) == 8 * len(attempts)
                        assert row['physical_controls'] == row['physical_episodes'] * HORIZONS[ti]
                        covered = set()
                        for ep in row['episodes']:
                            case_id, attempt = key(ep['case']), ep['attempt']
                            assert case_id in cases and ep['case'] == cases[case_id] and attempt in attempts
                            assert (case_id, attempt) not in covered
                            covered.add((case_id, attempt))
                            assert ep['completed'] and ep['horizon'] == HORIZONS[ti] and type(ep['success_once']) is bool
                            slot = ep['case']['slot']
                            if arm == 'full':
                                curves[mi, ti, slot, attempt - 1] = int(ep['success_once'])
                            endpoint = 'full_first' if arm == 'full' and attempt == 1 else 'full_sixth' if arm == 'full' and attempt == 6 else arm if arm != 'full' else None
                            if endpoint:
                                ei = ENDPOINTS.index(endpoint)
                                assert (mi, ti, slot, ei) not in seen
                                seen.add((mi, ti, slot, ei))
                                data[mi, ti, slot, ei] = int(ep['success_once'])
                        for case_id, ep in row['endpoint'].items():
                            assert ep['case'] == cases[case_id] and ep['attempt'] == end
                            assert arm == 'full' or ep['reset_from_empty_replay_exact']
                            actual = next(x for x in row['episodes'] if key(x['case']) == case_id and x['attempt'] == end)
                            assert ep['success_once'] == actual['success_once']
                assert set(paired['baselines']) == set(BASELINES)
                for bi, baseline in enumerate(BASELINES):
                    row = paired['baselines'][baseline]
                    assert row['all_native_evidence_checked'] and set(row['endpoint']) == set(cases)
                    assert row['physical_episodes'] == len(row['episodes']) == 8
                    assert row['physical_controls'] == 8 * HORIZONS[ti]
                    expected_teacher = baseline_spec['teacher_task_by_evaluation'][task] if baseline == 'fixed_PPO30' else None
                    assert row['teacher_task'] == expected_teacher
                    covered = set()
                    for ep in row['episodes']:
                        case_id = key(ep['case'])
                        assert case_id in cases and ep['case'] == cases[case_id] and case_id not in covered
                        assert ep['attempt'] == 6 and ep['completed'] and ep['horizon'] == HORIZONS[ti]
                        assert type(ep['success_once']) is bool
                        checked = row['endpoint'][case_id]
                        assert checked['case'] == ep['case'] and checked['attempt'] == 6
                        assert checked['success_once'] == ep['success_once'] and checked['reset_from_empty_replay_exact']
                        slot = ep['case']['slot']
                        assert baseline_data[bi, ti, slot] == -1
                        baseline_data[bi, ti, slot] = int(ep['success_once'])
                        covered.add(case_id)
                sources.append(dict(task=task, first=first, archive=str(archive), archive_sha256=receipt['sha256'], paired_sha256=sha(paired_path)))
    args.output.mkdir(parents=True, exist_ok=False)
    missing = {task: np.flatnonzero((data[:, ti] < 0).any(axis=(0, 2)) | (baseline_data[:, ti] < 0).any(axis=0)).tolist() for ti, task in enumerate(TASKS)}
    complete = not any(missing.values())
    write(args.output / 'COVERAGE.json', dict(completed=complete, planned_cases_per_task=40,
        missing_complete_eight_model_and_baseline_slots=missing, workers=workers, verified_banks=sources,
        exclusions='No outcome-conditioned exclusions. Missing cases remain in the planned denominator.'))
    if not complete:
        raise SystemExit('Incomplete fixed panel: coverage written; no final effect estimates')
    assert np.isin(curves, [0, 1]).all() and np.array_equal(curves[:, :, :, 0], data[:, :, :, 0])
    assert np.array_equal(curves[:, :, :, 5], data[:, :, :, 1])
    report = dict(completed=True, spec_sha256=args.spec_sha256, models=spec['models'],
        verified_physical_episodes=13200, transformer_physical_episodes=12800, baseline_physical_episodes=400,
        baseline_controls=baseline_spec, source_run=str(args.run),
        full_history_success_curves={name: {task: (100 * curves[mi, ti].mean(axis=0)).tolist() for ti, task in enumerate(TASKS)} for mi, name in enumerate(MODELS)},
        **paired_statistics(data, baselines=baseline_data))
    write(args.output / 'RESULT.json', report)
    lines = ['# Matched three- versus six-task KV comparison', '',
        '40 fresh cases per task, all eight fixed-final models and SFT/PPO30 controls, 13,200 audited physical episodes. Native episode-six success and paired history effects are separate from first-attempt competence.', '',
        '| Model | Panel | Full sixth | Reset sixth | History effect (pp) | Paired 95% interval (pp) |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for name in MODELS:
        for panel, values in report['panels'].items():
            row = values['models'][name]
            rate, effect = row['success_percent'], row['history_effect']
            lo, hi = effect['paired_bootstrap95_pp']
            lines.append(f'| {name} | {panel} | {rate["full_sixth"]:.1f}% | {rate["reset_sixth"]:.1f}% | {effect["difference_pp"]:+.1f} | [{lo:.1f}, {hi:.1f}] |')
    lines += ['', '| Method | Panel | Six − three full sixth (pp) | Paired 95% interval | Change in history effect (pp) | Paired 95% interval |',
              '| --- | --- | ---: | --- | ---: | --- |']
    for method in METHODS:
        for panel, values in report['panels'].items():
            row = values['six_minus_three'][method]
            full, history = row['full_sixth'], row['history_effect_difference']
            fl, fh = full['paired_bootstrap95_pp']; hl, hh = history['paired_bootstrap95_pp']
            lines.append(f'| {method} | {panel} | {full["difference_pp"]:+.1f} | [{fl:.1f}, {fh:.1f}] | {history["difference_pp"]:+.1f} | [{hl:.1f}, {hh:.1f}] |')
    lines += ['', '| Corpus | Method contrast | Panel | Full sixth difference (pp) | Paired 95% interval | History-effect difference (pp) | Paired 95% interval |',
              '| --- | --- | --- | ---: | --- | ---: | --- |']
    for size in ('3', '6'):
        for panel, values in report['panels'].items():
            for contrast, row in values['within_corpus_method_differences'][size].items():
                full, history = row['full_sixth'], row['history_effect_difference']
                fl, fh = full['paired_bootstrap95_pp']; hl, hh = history['paired_bootstrap95_pp']
                lines.append(f'| {size} | {contrast.replace("_minus_", " − ")} | {panel} | {full["difference_pp"]:+.1f} | [{fl:.1f}, {fh:.1f}] | {history["difference_pp"]:+.1f} | [{hl:.1f}, {hh:.1f}] |')
    lines += ['', '| Model | Panel | Full sixth − SFT (pp) | Paired 95% interval | Full sixth − PPO30 (pp) | Paired 95% interval |',
              '| --- | --- | ---: | --- | ---: | --- |']
    for name in MODELS:
        for panel, values in report['panels'].items():
            comparison = values['models'][name]['versus_baselines']
            sft, ppo = (comparison[arm]['full_sixth'] for arm in BASELINES)
            sl, sh = sft['paired_bootstrap95_pp']; pl, ph = ppo['paired_bootstrap95_pp']
            lines.append(f'| {name} | {panel} | {sft["difference_pp"]:+.1f} | [{sl:.1f}, {sh:.1f}] | {ppo["difference_pp"]:+.1f} | [{pl:.1f}, {ph:.1f}] |')
    lines += ['', 'PPO30 uses the matching task teacher on ID and the fixed OpenDrawer30 teacher on both OOD tasks. This OOD control measures transfer from one declared source policy, not a multi-task PPO policy or an oracle selected on target outcomes.', '', report['uncertainty'], '',
        'This compares adding task data at fixed fitting compute. It does not isolate diversity from corpus size. OOD is relative to these meta-training tasks, not necessarily base pretraining. First-attempt success and reset disagreements are reported in RESULT.json; initial competence alone does not establish adaptation.', '']
    (args.output / 'RESULT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(completed=True, models=len(MODELS), physical_episodes=13200)))


if __name__ == '__main__':
    main()
