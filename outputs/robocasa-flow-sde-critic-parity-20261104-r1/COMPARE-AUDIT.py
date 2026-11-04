"""Audit the completed two-arm diagnostic on Linux CPU, with no model calls."""
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
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    if isinstance(value, torch.Tensor):
        return dict(dtype=str(value.dtype), shape=list(value.shape),
            sha256=hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape),
            sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def tree_sha(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True).encode()).hexdigest()


def load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rb') as stream:
            return torch.load(stream, map_location='cpu', weights_only=False)
    return torch.load(path, map_location='cpu', weights_only=False)


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    job = root / 'comparison-r1'
    clock = json.loads((job / 'clock.json').read_text())
    assert clock['exit_code'] == 0 and clock['allocated_gpus'] == 8
    assert clock['observed_gpu_seconds'] <= 19200
    prepared_path = root / 'comparison-input-r1/PREPARED.json'
    prepared = json.loads(prepared_path.read_text())
    admission = json.loads((root / 'COMPARISON-ADMISSION.json').read_text())
    assert prepared['completed'] and sha(prepared_path) == admission['prepared_sha256']
    assert sha(root / 'comparison-input-r1/DEPLOYED.json') == admission['deployed_sha256']
    common = Path(prepared['checkpoint'])
    common_receipts = []
    for row in prepared['files']:
        assert sha(common / row['path']) == row['sha256']
        common_receipts.append(row)
    arms, episodes, configs, csv_rows = {}, {}, {}, []
    for arm in ['baseline', 'fixed']:
        directory = job / arm
        own_clock = json.loads((directory / 'clock.json').read_text())
        progress = json.loads((directory / 'progress.json').read_text())
        after = json.loads((directory / 'after-evaluation.json').read_text())
        before = json.loads((directory / 'before-evaluation.json').read_text())
        assert own_clock['exit_code'] == 0 and own_clock['observed_gpu_seconds'] <= 8 * 1140
        assert progress['completed_iterations'] == progress['target_iterations'] == 3
        assert after['completed'] and after['global_step'] == 3
        assert before['completed'] and before['new_evaluation_controls'] == 0
        assert before['reused_reference']['sha256'] == prepared['before_reference_sha256']
        probe = json.loads((directory / 'gpu-probe.json').read_text())
        assert probe['status'] == 'passed' and len(probe['devices']) == 8
        assert probe['hostname'].startswith('job-eigq1w5m3ba9')
        config = json.loads((directory / 'resolved-config.json').read_text())
        assert config['runner']['resume_dir'] == str(common)

        def normalize(value, path=()):
            if path in [('native_pilot', 'attempt'), ('native_pilot', 'deadline_unix')]:
                return '<ARM_OPERATIONAL_FIELD>'
            if isinstance(value, dict):
                return {key: normalize(item, path + (key,)) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize(item, path) for item in value]
            if isinstance(value, str):
                value = value.replace(str(directory), '<ARM_OUTPUT>')
                value = value.replace(str(job / ('accounting-' + arm)), '<ARM_ACCOUNTING>')
            return value

        configs[arm] = normalize(config)
        updates = []
        for iteration in [1, 2, 3]:
            rank_rows = []
            for rank in [0, 1]:
                path = directory / 'updates' / f'iteration-{iteration:03d}-rank-{rank:02d}.json'
                row = json.loads(path.read_text())
                assert row['completed'] and row['frozen_vlm_unchanged']
                assert row['producing_policy_iteration'] == iteration - 1
                assert row['optimizer_steps_before'] == 24 * (iteration - 1)
                assert row['optimizer_steps_after'] == 24 * iteration
                assert all(math.isfinite(value) for value in row['metrics'].values())
                parity = json.loads((directory / 'logprob-parity' /
                    f'iteration-{iteration:03d}-rank-{rank:02d}.json').read_text())
                value = json.loads((directory / 'value-parity' /
                    f'iteration-{iteration:03d}-rank-{rank:02d}.json').read_text())
                assert parity['passed'] and parity['before_backward'] and value['before_backward']
                if arm == 'fixed':
                    assert value['passed'] and value['max_abs_value_delta'] <= 1e-3
                rank_rows.append(dict(rank=rank, update_sha256=sha(path), value_parity=value,
                    logprob_parity=parity))
            assert rank_rows[0]['value_parity']['optimizer_steps'] == 24 * (iteration - 1)
            updates.append(dict(iteration=iteration, metrics=row['metrics'], ranks=rank_rows))
        retained = []
        checkpoint = directory / 'PickPlaceDrawerToCounter/checkpoints/global_step_3'
        boundary = json.loads((checkpoint / 'pilot-boundary.json').read_text())
        assert boundary['completed'] and boundary['global_step'] == 3 and len(boundary['workers']) == 18
        for row in boundary['actor_checkpoint_files'] + boundary['workers']:
            path = Path(row['path'])
            assert path.is_relative_to(checkpoint) and sha(path) == row['sha256']
            retained.append(dict(path=str(path.relative_to(checkpoint)), sha256=row['sha256'], bytes=path.stat().st_size))
        train, panel = {}, {}
        for path in sorted((directory / 'environment').glob('rank-*/*-episodes.json')):
            data = json.loads(path.read_text())
            assert data['completed']
            rows = data['rows']
            assert all(row['native_counts']['controls'] == 600 for row in rows)
            if data['phase'] == 'training':
                index = data['iteration']
                assert index in [0, 1, 2]
                train.setdefault(index, []).extend(rows)
            elif data['phase'] == 'after':
                assert data['iteration'] == 3
                for row in rows:
                    assert row['seed'] not in panel
                    panel[row['seed']] = row
        assert set(panel) == set(range(630100, 630164))
        for iteration, rows in sorted(train.items()):
            assert len(rows) == 128 and {r['seed'] for r in rows} == set(range(
                2300000 + 128 * iteration, 2300000 + 128 * (iteration + 1)))
            successes = sum(row['success_once'] for row in rows)
            update = updates[iteration]
            update['train_successes'] = successes
            update['train_cases'] = 128
            csv_rows.append(dict(arm=arm, iteration=iteration + 1, train_successes=successes,
                train_cases=128, value_loss=update['metrics']['critic/value_loss'],
                value_clip_ratio=update['metrics']['critic/value_clip_ratio'],
                policy_loss=update['metrics']['actor/policy_loss'],
                policy_clip_fraction=update['metrics']['actor/clip_fraction']))
        assert set(train) == {0, 1, 2}
        successes = sum(row['success_once'] for row in panel.values())
        assert abs(successes / 64 - after['metrics']['success_once']) <= 1e-8
        charges = []
        for path in sorted((job / ('accounting-' + arm)).glob('rank-*/*.json')):
            data = json.loads(path.read_text())
            assert all(call['resolved'] for call in data['calls'])
            charges.extend(data['calls'])
        charge = sum(call.get('executed_or_conservative', call['requested']) for call in charges)
        assert charge <= 372736
        arms[arm] = dict(completed=True, updates=updates, endpoint_successes=successes, endpoint_cases=64,
            charged_controls=charge, fresh_policy_controls=268800, clock=own_clock,
            final_checkpoint=str(checkpoint), final_boundary_sha256=sha(checkpoint / 'pilot-boundary.json'),
            final_checkpoint_files=retained,
            gpu_uuids=[device['uuid'] for device in probe['devices']])
        episodes[arm] = panel
    assert configs['baseline'] == configs['fixed'], 'A scientific recipe field differs across arms'
    assert arms['baseline']['gpu_uuids'] == arms['fixed']['gpu_uuids']
    paired, coupling = [], []
    for seed in range(630100, 630164):
        paired.append(dict(seed=seed, baseline=episodes['baseline'][seed]['success_once'],
            fixed=episodes['fixed'][seed]['success_once']))
    fixed_only = sum(row['fixed'] and not row['baseline'] for row in paired)
    baseline_only = sum(row['baseline'] and not row['fixed'] for row in paired)
    discordant = fixed_only + baseline_only
    pvalue = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(fixed_only, baseline_only) + 1)) / 2 ** discordant) if discordant else 1.
    for rank in range(8):
        a, b = [load(job / arm / f'sampling/rank-{rank:02d}/00000.pt.gz') for arm in ['baseline', 'fixed']]
        assert a['mode'] == b['mode'] == 'train' and not a['terminal_value_only'] and not b['terminal_value_only']
        rng_equal = tree_sha(a['rng_before']) == tree_sha(b['rng_before'])
        fields_a, fields_b = a['forward_sampling_fields'], b['forward_sampling_fields']
        initial_noise_equal = torch.equal(fields_a['chains'][:, 0], fields_b['chains'][:, 0])
        index_equal = torch.equal(fields_a['denoise_inds'], fields_b['denoise_inds'])
        assert rng_equal and initial_noise_equal and index_equal, 'Initial sampler coupling failed'
        initial_a, initial_b = [load(job / arm / f'environment/rank-{rank:02d}/training-i000-round00-initial.pt')
            for arm in ['baseline', 'fixed']]
        coupling.append(dict(rank=rank, sampler_rng_equal=rng_equal, initial_noise_equal=initial_noise_equal,
            selected_index_equal=index_equal, public_observation_hash_equal=a['public_observation_sha256'] == b['public_observation_sha256'],
            initial_state_equal=tree_sha(initial_a['states']) == tree_sha(initial_b['states']),
            initial_physical_equal=tree_sha(initial_a['physical']) == tree_sha(initial_b['physical']),
            initial_images_equal=tree_sha(initial_a['images']) == tree_sha(initial_b['images']),
            first_actions_equal=torch.equal(fields_a['action'], fields_b['action']),
            first_actions_max_abs_delta=float((fields_a['action'] - fields_b['action']).abs().max()),
            full_first_chain_equal=torch.equal(fields_a['chains'], fields_b['chains'])))
    output = dict(completed=True, script_sha256=sha(__file__), common_checkpoint_sha256=prepared['marker_sha256'],
        common_checkpoint_files=common_receipts, arms=arms, normalized_scientific_configs_equal=True,
        initial_coupling=coupling, paired_endpoint=dict(cases=64, fixed_only=fixed_only,
            baseline_only=baseline_only, exact_mcnemar_two_sided_p=pvalue, rows=paired),
        new_policy_controls=537600, charged_controls=sum(a['charged_controls'] for a in arms.values()),
        controls_cap=745472, clock=clock, prior_conservative_gpu_seconds=393,
        process_plus_prior_gpu_seconds=clock['observed_gpu_seconds'] + 393, cap_gpu_seconds=21600,
        scope='Three native on-policy updates per arm from common policy0, same recipe and saved RNG. '
              'One seed and64-case endpoint; simulator variability and short horizon limit causal and convergence claims.')
    assert output['process_plus_prior_gpu_seconds'] <= 21600
    (root / 'COMPARISON-AUDIT.json').write_text(json.dumps(output, indent=2) + '\n')
    with (root / 'comparison-summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps(dict(completed=True, endpoints={arm: row['endpoint_successes'] for arm, row in arms.items()},
        fixed_only=fixed_only, baseline_only=baseline_only, paired_p=pvalue,
        charged_controls=output['charged_controls'], process_plus_prior_gpu_seconds=output['process_plus_prior_gpu_seconds'])))


if __name__ == '__main__':
    main()
