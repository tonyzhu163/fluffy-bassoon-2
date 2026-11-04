"""Independent CPU closeout of retained baseline training and recovered panels."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        return [str(value.dtype), list(value.shape), hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()]
    if isinstance(value, np.ndarray):
        return [str(value.dtype), list(value.shape), hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def tree_sha(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True).encode()).hexdigest()


def tensor_load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rb') as stream:
            return torch.load(stream, map_location='cpu', weights_only=False)
    return torch.load(path, map_location='cpu', weights_only=False)


def charges(directory, require_resolved):
    calls, receipts = [], []
    for path in sorted(directory.glob('rank-*/*.json')):
        data = json.loads(path.read_text())
        calls.extend(data['calls'])
        receipts.append(dict(path=str(path), sha256=sha(path)))
    assert calls
    unresolved = sum(not call['resolved'] for call in calls)
    if require_resolved:
        assert unresolved == 0
    total = sum(call.get('executed_or_conservative', call['requested']) for call in calls)
    return dict(total=total, unresolved_intents=unresolved, receipts=receipts)


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    directories = dict(baseline=root / 'comparison-r1/baseline', fixed=root / 'comparison-fixed-r2')
    evaluator = root / 'baseline-endpoint-r2/gpu'
    legacy_clock = json.loads((root / 'comparison-r1/clock.json').read_text())
    fixed_clock = json.loads((directories['fixed'] / 'clock.json').read_text())
    evaluation_clock = json.loads((evaluator / 'clock.json').read_text())
    assert legacy_clock['exit_code'] == 255, 'The original failed attempt must remain failed'
    assert fixed_clock['exit_code'] == evaluation_clock['exit_code'] == 0
    assert fixed_clock['allocated_gpus'] == 8 and evaluation_clock['allocated_gpus'] == 1
    assert fixed_clock['observed_gpu_seconds'] <= 9600 and evaluation_clock['observed_gpu_seconds'] <= 1080
    admission = json.loads((root / 'RECOVERY-ADMISSION.json').read_text())
    assert admission['completed'] and admission['prior_conservative_gpu_seconds'] == 9265
    assert sha(root / 'comparison-input-r2/RESTORED.json') == admission['canonical_restore_sha256']
    restored = json.loads((root / 'comparison-input-r2/RESTORED.json').read_text())
    assert restored['completed'] and len(restored['files']) == 24
    for row in restored['files']:
        assert sha(row['target']) == row['sha256']
    endpoint = json.loads((evaluator / 'RESULT.json').read_text())
    endpoint_admission = json.loads((root / 'BASELINE-ENDPOINT-ADMISSION.json').read_text())
    assert endpoint['completed'] and endpoint['cases'] == 64 and endpoint['step'] == 3
    assert endpoint['SGD'] == 0 and endpoint['all_controls'] == endpoint['policy_controls'] == 38400
    assert endpoint['model_sha256'] == endpoint_admission['model_sha256']
    assert endpoint['fixed_eight_lane_batches'] and endpoint['original_rank_rng_restored']
    assert endpoint['original_env_samplers_restored']
    eval_probe = json.loads((evaluator / 'gpu-probe.json').read_text())
    assert eval_probe['status'] == 'passed' and len(eval_probe['devices']) == 1
    eval_charge = charges(evaluator.parent / 'accounting', True)
    assert eval_charge['total'] == 38400
    arms, panels, normalized_configs, summary = {}, {}, {}, []
    for arm, directory in directories.items():
        progress = json.loads((directory / 'progress.json').read_text())
        assert progress['completed_iterations'] == progress['target_iterations'] == 3
        cfg = json.loads((directory / 'resolved-config.json').read_text())
        reference = json.loads(Path(cfg['native_pilot']['before_reference']).read_text())
        assert reference['completed'] and reference['cases'] == 64 and reference['metrics']['success_once'] == .5
        assert cfg['runner']['resume_dir'] == str(root / 'comparison-input-r1/global_step_0')
        assert reference['checkpoint'] == cfg['runner']['resume_dir']
        assert cfg['actor']['global_batch_size'] == 256 and cfg['algorithm']['update_epoch'] == 4
        assert sha(cfg['native_pilot']['before_reference']) == cfg['native_pilot']['before_reference_sha256']

        def normalize(value, path=()):
            if path in [('native_pilot', key) for key in ['attempt', 'deadline_unix', 'before_reference', 'before_reference_sha256']]:
                return '<OPERATIONAL_FIELD>'
            if isinstance(value, dict):
                return {key: normalize(item, path + (key,)) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize(item, path) for item in value]
            if isinstance(value, str):
                value = value.replace(str(directory), '<ARM_OUTPUT>')
                value = value.replace(str(root / 'comparison-r1/accounting-baseline'), '<ARM_ACCOUNTING>')
                value = value.replace(str(root / 'accounting-fixed-r2'), '<ARM_ACCOUNTING>')
            return value

        normalized_configs[arm] = normalize(cfg)
        probe = json.loads((directory / 'gpu-probe.json').read_text())
        assert probe['status'] == 'passed' and len(probe['devices']) == 8
        assert all(device['cuda_name'] == 'NVIDIA RPBZZZ6' for device in probe['devices'])
        updates = []
        for iteration in [1, 2, 3]:
            checks = []
            for rank in [0, 1]:
                path = directory / f'updates/iteration-{iteration:03d}-rank-{rank:02d}.json'
                row = json.loads(path.read_text())
                assert row['completed'] and row['frozen_vlm_unchanged']
                assert row['optimizer_steps_before'] == 24 * (iteration - 1)
                assert row['optimizer_steps_after'] == 24 * iteration
                assert row['producing_policy_iteration'] == iteration - 1
                assert all(math.isfinite(value) for value in row['metrics'].values())
                logprob = json.loads((directory / f'logprob-parity/iteration-{iteration:03d}-rank-{rank:02d}.json').read_text())
                value = json.loads((directory / f'value-parity/iteration-{iteration:03d}-rank-{rank:02d}.json').read_text())
                assert logprob['passed'] and logprob['before_backward'] and value['before_backward']
                if arm == 'fixed':
                    assert value['passed'] and value['max_abs_value_delta'] <= 1e-3
                checks.append(dict(rank=rank, update_sha256=sha(path), logprob=logprob, value=value))
            updates.append(dict(iteration=iteration, metrics=row['metrics'], checks=checks))
        checkpoint = directory / 'PickPlaceDrawerToCounter/checkpoints/global_step_3'
        boundary = json.loads((checkpoint / 'pilot-boundary.json').read_text())
        assert boundary['completed'] and boundary['global_step'] == 3 and len(boundary['workers']) == 18
        files = []
        for row in boundary['actor_checkpoint_files'] + boundary['workers']:
            path = Path(row['path'])
            assert path.is_relative_to(checkpoint) and sha(path) == row['sha256']
            files.append(dict(path=str(path.relative_to(checkpoint)), sha256=row['sha256'], bytes=path.stat().st_size))
        for rank in [0, 1]:
            assert tensor_load(checkpoint / f'pilot/actor-{rank:02d}.pt')['optimizer_steps'] == 72
        if arm == 'baseline':
            model = next(row for row in boundary['actor_checkpoint_files'] if row['path'].endswith('/full_weights.pt'))
            assert model['sha256'] == endpoint['model_sha256']
        train, panel = {}, {}
        for path in sorted((directory / 'environment').glob('rank-*/*-episodes.json')):
            data = json.loads(path.read_text())
            assert data['completed'] and all(row['native_counts']['controls'] == 600 for row in data['rows'])
            if data['phase'] == 'training':
                train.setdefault(data['iteration'], []).extend(data['rows'])
            elif data['phase'] == 'after':
                assert arm == 'fixed' and data['iteration'] == 3
                for item in data['rows']:
                    assert item['seed'] not in panel
                    panel[item['seed']] = item
        assert set(train) == {0, 1, 2}
        for iteration, items in sorted(train.items()):
            assert len(items) == 128 and {item['seed'] for item in items} == set(range(
                2300000 + 128 * iteration, 2300000 + 128 * (iteration + 1)))
            count = sum(item['success_once'] for item in items)
            updates[iteration]['train_successes'] = count
            metrics = updates[iteration]['metrics']
            summary.append(dict(arm=arm, iteration=iteration + 1, train_successes=count, train_cases=128,
                critic_loss=metrics['critic/value_loss'], critic_clip_ratio=metrics['critic/value_clip_ratio'],
                policy_loss=metrics['actor/policy_loss'], policy_clip_fraction=metrics['actor/clip_fraction']))
        if arm == 'baseline':
            assert len(endpoint['rows']) == 8
            for bank in endpoint['rows']:
                assert sha(bank['episodes']) == bank['episodes_sha256']
                data = json.loads(Path(bank['episodes']).read_text())
                assert data['completed'] and len(data['rows']) == 8
                for item in data['rows']:
                    assert item['seed'] not in panel and item['native_counts']['controls'] == 600
                    panel[item['seed']] = item
            charge = charges(root / 'comparison-r1/accounting-baseline', False)
            charge['total_with_recovered_endpoint'] = charge['total'] + eval_charge['total']
        else:
            after = json.loads((directory / 'after-evaluation.json').read_text())
            assert after['completed'] and after['global_step'] == 3
            charge = charges(root / 'accounting-fixed-r2', True)
            charge['total_with_recovered_endpoint'] = charge['total']
            assert abs(after['metrics']['success_once'] - sum(item['success_once'] for item in panel.values()) / 64) <= 1e-8
        assert set(panel) == set(range(630100, 630164))
        assert charge['total_with_recovered_endpoint'] <= 372736
        panels[arm] = panel
        arms[arm] = dict(completed_updates=3, optimizer_steps=72, updates=updates,
            endpoint_successes=sum(item['success_once'] for item in panel.values()), endpoint_cases=64,
            checkpoint=str(checkpoint), checkpoint_files=files, boundary_sha256=sha(checkpoint / 'pilot-boundary.json'),
            controls=charge, gpu_uuids=[device['uuid'] for device in probe['devices']])
    assert normalized_configs['baseline'] == normalized_configs['fixed'], 'Scientific configuration differs'
    coupling = []
    for rank in range(8):
        first = [tensor_load(directory / f'sampling/rank-{rank:02d}/00000.pt.gz') for directory in directories.values()]
        a, b = first
        assert a['mode'] == b['mode'] == 'train' and not a['terminal_value_only'] and not b['terminal_value_only']
        assert tree_sha(a['rng_before']) == tree_sha(b['rng_before']), 'Initial saved sampler RNG differs'
        fa, fb = a['forward_sampling_fields'], b['forward_sampling_fields']
        assert torch.equal(fa['denoise_inds'], fb['denoise_inds'])
        initial = [tensor_load(directory / f'environment/rank-{rank:02d}/training-i000-round00-initial.pt')
            for directory in directories.values()]
        coupling.append(dict(rank=rank, initial_rng_equal=True, selected_index_equal=True,
            initial_noise_equal=torch.equal(fa['chains'][:, 0], fb['chains'][:, 0]),
            initial_noise_max_abs_delta=float((fa['chains'][:, 0] - fb['chains'][:, 0]).abs().max()),
            initial_public_observation_equal=a['public_observation_sha256'] == b['public_observation_sha256'],
            initial_states_equal=tree_sha(initial[0]['states']) == tree_sha(initial[1]['states']),
            initial_physical_equal=tree_sha(initial[0]['physical']) == tree_sha(initial[1]['physical']),
            initial_images_equal=tree_sha(initial[0]['images']) == tree_sha(initial[1]['images']),
            first_action_equal=torch.equal(fa['action'], fb['action']),
            first_action_max_abs_delta=float((fa['action'] - fb['action']).abs().max())))
    paired = [dict(seed=seed, baseline=panels['baseline'][seed]['success_once'], fixed=panels['fixed'][seed]['success_once'])
        for seed in range(630100, 630164)]
    fixed_only = sum(row['fixed'] and not row['baseline'] for row in paired)
    baseline_only = sum(row['baseline'] and not row['fixed'] for row in paired)
    n = fixed_only + baseline_only
    pvalue = min(1., 2 * sum(math.comb(n, k) for k in range(min(fixed_only, baseline_only) + 1)) / 2 ** n) if n else 1.
    total_charge = sum(row['controls']['total_with_recovered_endpoint'] for row in arms.values())
    gpu_seconds = 9265 + fixed_clock['observed_gpu_seconds'] + evaluation_clock['observed_gpu_seconds']
    assert total_charge <= 745472 and gpu_seconds <= 21600
    result = dict(completed=True, script_sha256=sha(__file__), original_attempt_remains_failed=True,
        original_failed_clock=legacy_clock, fixed_clock=fixed_clock, baseline_endpoint_clock=evaluation_clock,
        prior_conservative_gpu_seconds=9265, process_plus_prior_gpu_seconds=gpu_seconds,
        gpu_cap_seconds=21600, arms=arms, scientific_configs_equal=True, initial_coupling=coupling,
        paired_endpoint=dict(cases=64, fixed_only=fixed_only, baseline_only=baseline_only,
            exact_mcnemar_two_sided_p=pvalue, rows=paired),
        policy_controls=537600, charged_controls=total_charge, control_cap=745472,
        evaluation_migration='Retained baseline policy3 evaluated in original rank-local B8 panels serialized on1 GPU; fixed native8-rank panel.',
        limits='Three updates, one seed, known simulator variability; action/image equality recorded, no bitwise GPU kernel rule. '
               'This does not establish convergence or explain the historical teacher-gate failures.')
    (root / 'RECOVERY-AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    with (root / 'comparison-summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(summary)
    print(json.dumps(dict(completed=True, endpoints={arm: data['endpoint_successes'] for arm, data in arms.items()},
        fixed_only=fixed_only, baseline_only=baseline_only, paired_p=pvalue,
        charged_controls=total_charge, process_plus_prior_gpu_seconds=gpu_seconds)))


if __name__ == '__main__':
    main()
