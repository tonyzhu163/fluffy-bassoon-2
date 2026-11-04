"""Audit completed endpoint banks and independently reproduce final paired tests."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    import numpy as np
    import torch
    from scipy.stats import binomtest
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--source-script', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    manifest = read(args.prepared)
    assert manifest['completed'] and manifest['cases'] == 512 and manifest['bank_size'] == 8
    assert manifest['maximum_policy_controls'] == 1228800 and len(manifest['tasks']) == 2
    tasks, all_banks = [], []
    for entry in manifest['tasks']:
        task = entry['task']
        output = args.root / task / 'gpu-r1'
        assert sha(entry['source_config']) == entry['source_config_sha256']
        assert sha(entry['initial_parity_fixture']) == entry['initial_parity_fixture_sha256']
        parity = read(output / 'initial-parity.json')
        assert len(parity) == 1 and parity[0]['passed'] and parity[0]['max_abs_chunk_log_ratio'] == 0
        for checkpoint in entry['checkpoints']:
            assert sha(checkpoint['marker']) == checkpoint['marker_sha256']
            ready = read(args.root / task / 'staging-r1' / str(checkpoint['step']) / 'READY.json')
            assert ready['completed'] and ready['task'] == task and ready['checkpoint'] == checkpoint
            assert ready['prepared_sha256'] == sha(args.prepared)
            assert ready['staging_script_sha256'] == sha(args.source_script)
        journal = read(args.root / task / 'accounting/rank-00/gpu-r1.json')
        calls = journal['calls']
        outcomes = {0: {}, 130: {}}
        seen, pair_rng, image_variation, bank_receipts = set(), {}, 0, []
        phase_starts = {}
        panels = sorted((output / 'panels').glob('*.json'))
        for panel_path in panels:
            panel = read(panel_path)
            step, bank = panel['step'], panel['bank']
            assert step in [0, 130] and 0 <= bank < 64 and (step, bank) not in seen
            seen.add((step, bank))
            phase = f'test-c{step:03d}-b{bank:03d}'
            assert panel_path.name == phase + '.json'
            assert panel['seeds'] == entry['seeds'][bank * 8:(bank + 1) * 8]
            assert panel['rng_seed'] == entry['rng_seeds'][bank]
            episode_path = Path(panel['episode_path'])
            assert episode_path.is_relative_to(output) and sha(episode_path) == panel['episode_sha256']
            episode = read(episode_path)
            assert episode['completed'] and episode['rows'] == panel['episodes']
            assert [r['seed'] for r in episode['rows']] == panel['seeds']
            assert len(episode['rows']) == 8
            assert sum(r['success_once'] for r in episode['rows']) == panel['first_success']
            assert sum(r['success_at_end'] for r in episode['rows']) == panel['success_at_end']
            for row in episode['rows']:
                assert row['episode_id'] == f"{task}/{row['seed']}"
                counts = row['native_counts']
                assert counts['controls'] - counts['reset_controls'] == 600
                if row['success_once']:
                    assert 0 <= row['first_success_control'] <= 600
                assert row['seed'] not in outcomes[step]
                outcomes[step][row['seed']] = bool(row['success_once'])

            initial_path = episode_path.with_name(episode_path.name.replace('-episodes.json', '-initial.json'))
            assert sha(initial_path) == panel['initial_report_sha256']
            initial = read(initial_path)
            current_path = Path(initial['initial']['path'])
            canonical_path = Path(initial['canonical'])
            assert current_path.is_relative_to(output) and canonical_path.is_relative_to(output)
            assert sha(current_path) == initial['initial']['sha256']
            canonical = read(canonical_path)
            reference_path = canonical_path.parent / 'initial.pt'
            assert sha(reference_path) == canonical['snapshot_sha256']
            reference = torch.load(reference_path, map_location='cpu', weights_only=False)
            current = torch.load(current_path, map_location='cpu', weights_only=False)
            for key in ['format', 'case', 'physical', 'task_descriptions']:
                assert current[key] == reference[key], (task, bank, step, key)
            assert reference['case'] == canonical['case']
            a, b = np.asarray(reference['states']), np.asarray(current['states'])
            assert a.shape == b.shape and a.dtype == b.dtype and np.isfinite(a).all()
            assert np.array_equal(a, b)
            report = initial['report']
            assert report['matched'] and report['physical_state_exact'] and report['public_state_exact']
            assert not report['observations_modified']
            images_exact = True
            for key in ['main_images', 'wrist_images', 'extra_view_images']:
                a, b = np.asarray(reference['images'][key]), np.asarray(current['images'][key])
                assert a.shape == b.shape and a.dtype == b.dtype == np.uint8 and a.size
                delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
                actual = report['images'][key]
                assert int(delta.max()) == actual['max_absolute_delta']
                assert int(np.count_nonzero(delta)) == actual['differing_components']
                assert int(delta.size) == actual['total_components']
                assert math.isclose(float(delta.mean()), actual['mean_absolute_delta'], abs_tol=1e-12)
                images_exact = images_exact and not np.any(delta)
            assert bool(images_exact) == panel['images_exact'] == report['images_exact']
            image_variation += not images_exact
            del current, reference

            sampler_paths = sorted((output / 'sampler').glob(phase + '-*.json'))
            assert len(sampler_paths) == 12
            assert [p.name for p in sampler_paths] == [phase + f'-{i:02d}.json' for i in range(1, 13)]
            assert {p.name: sha(p) for p in sampler_paths} == panel['sampler_sha256']
            sampler = [read(p) for p in sampler_paths]
            assert all(r['actions_shape'] == [8, 50, 12] and r['batch_size'] == 8 and r['chunk'] == 50
                       for r in sampler)
            rng = [dict(before=r['before_rng_sha256'], after=r['after_rng_sha256']) for r in sampler]
            saved_rng = read(output / 'banks' / f'bank-{bank:03d}' / 'SAMPLER-RNG-REFERENCE.json')
            assert saved_rng['rng_seed'] == panel['rng_seed'] and saved_rng['rng_hashes'] == rng
            if bank in pair_rng:
                assert pair_rng[bank] == rng
            pair_rng[bank] = rng
            phase_calls = [r for r in calls if r['phase'] == phase]
            assert len(phase_calls) == 12
            assert all(r['resolved'] and r['requested'] == r['executed_or_conservative'] == 400
                       for r in phase_calls)
            phase_starts[(step, bank)] = min(r['started_unix'] for r in phase_calls)
            bank_receipts.append(dict(task=task, step=step, bank=bank, panel_sha256=sha(panel_path),
                initial_snapshot_sha256=sha(current_path), canonical_snapshot_sha256=sha(reference_path),
                episode_sha256=sha(episode_path), sampler_receipts=12, controls=4800,
                physical_and_public_state_exact=True, images_exact=bool(images_exact)))
        for bank in range(64):
            if (0, bank) in seen and (130, bank) in seen:
                observed = sorted([0, 130], key=lambda step: phase_starts[(step, bank)])
                assert observed == entry['bank_order'][bank] == ([0, 130] if bank % 2 == 0 else [130, 0])
        result_path = output / 'RESULT.json'
        finished = result_path.exists() and read(result_path)['completed']
        summary = dict(task=task, completed=finished, audited_policy_banks=len(seen),
                       audited_policy_episodes=8 * len(seen), paired_banks=sum((0, b) in seen and (130, b) in seen for b in range(64)),
                       rendered_initial_variation_banks=int(image_variation), policy_controls=4800 * len(seen))
        summary['completed_pair_ABBA_order_passed'] = True
        if not args.allow_incomplete:
            assert finished and seen == {(s, b) for s in [0, 130] for b in range(64)}
            assert set(outcomes[0]) == set(outcomes[130]) == set(entry['seeds'])
            result = read(result_path)
            assert result['prepared_sha256'] == sha(args.prepared)
            assert result['bank_receipts'] == 128 and result['valid_episodes'] == 1024
            assert result['SGD'] == result['unresolved_intents'] == 0
            assert all(r['resolved'] for r in calls)
            assert sum(r['executed_or_conservative'] for r in calls) == result['all_controls'] == 614400
            assert result['raw_receipt_sha256'] == {p.name: sha(p) for p in panels}
            differences = np.array([int(outcomes[130][s]) - int(outcomes[0][s]) for s in sorted(outcomes[0])])
            wins, losses = int(np.sum(differences == 1)), int(np.sum(differences == -1))
            p_value = float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.
            original = result['final_vs_initial']
            assert wins == original['wins'] and losses == original['losses']
            assert sum(outcomes[0].values()) == original['initial_successes'] == result['first_successes']['0']
            assert sum(outcomes[130].values()) == original['policy_successes'] == result['first_successes']['130']
            assert math.isclose(p_value, original['exact_McNemar_p'], rel_tol=1e-10, abs_tol=1e-15)
            counts = np.array([np.sum(differences == d) for d in [-1, 0, 1]])
            bootstrap = np.random.default_rng(entry['bootstrap_seed']).multinomial(512, counts / 512, size=10000)
            interval = np.quantile((bootstrap[:, 2] - bootstrap[:, 0]) / 512, [.025, .975]) * 100
            np.testing.assert_allclose(interval, original['paired_bootstrap95_percentage_points'], atol=1e-12, rtol=0)
            assert math.isclose(float(differences.mean() * 100), original['delta_percentage_points'], abs_tol=1e-12)
            clock = read(output / 'clock.json')
            assert clock['exit_code'] == 0 and clock['allocated_gpus'] == 1
            stage = read(args.root / task / 'staging-r1/RESULT.json')
            assert stage['completed'] and not stage['source_files_changed']
            summary.update(wins=wins, losses=losses, exact_McNemar_p=p_value, result_sha256=sha(result_path), clock=clock)
        tasks.append(summary)
        all_banks.extend(bank_receipts)
    if not args.allow_incomplete:
        holm = read(args.root / 'HOLM.json')
        assert holm['completed'] and holm['family_alpha'] == .05
        passing = True
        for i, row in enumerate(sorted(tasks, key=lambda r: r['exact_McNemar_p'])):
            threshold = .05 / (2 - i)
            passing = passing and row['exact_McNemar_p'] < threshold
            decision = holm['holm'][row['task']]
            assert decision['threshold'] == threshold and decision['rejected_two_sided'] == passing
            assert decision['improvement'] == (passing and row['wins'] > row['losses'])
    value = dict(completed_for_recorded_banks=True, whole_run_completed=not args.allow_incomplete,
        checked_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        audit_source_sha256=sha(__file__), source_script_sha256=sha(args.source_script),
        prepared_sha256=sha(args.prepared), tasks=tasks, banks=all_banks,
        policy_banks=len(all_banks), policy_episodes=len(all_banks) * 8,
        sampler_receipts=len(all_banks) * 12, CPU_only=True, new_GPU_seconds=0, new_controls=0, SGD=0)
    with args.destination.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'banks'}), flush=True)


if __name__ == '__main__':
    main()
