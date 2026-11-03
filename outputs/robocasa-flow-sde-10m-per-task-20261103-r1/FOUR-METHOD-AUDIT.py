"""Audit actual saved source payloads on allocated Linux CPU, without model calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import torch
import numpy as np


def load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rb') as stream:
            return torch.load(stream, map_location='cpu', weights_only=False)
    return torch.load(path, map_location='cpu', weights_only=False)


def row_hashes(value):
    value = torch.as_tensor(value)
    return [hashlib.sha256(str(tuple(row.shape)).encode() + str(row.dtype).encode() +
        row.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        for row in value]


def observation_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif torch.is_tensor(item) or isinstance(item, np.ndarray):
            array = item.detach().cpu().numpy() if torch.is_tensor(item) else item
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def observation(value):
    assert value['states'].shape == (16, 25)
    assert torch.isfinite(value['states']).all()
    assert len(value['task_descriptions']) == 16 and all(value['task_descriptions'])
    for key in ['main_images', 'wrist_images', 'extra_view_images']:
        image = value[key]
        assert image.dtype == torch.uint8 and image.shape[0] == 16
        assert image.shape[-3:] == (256, 256, 3)


def audit(root, collection, tag):
    producing = collection - 1
    contract = json.loads((root / 'flow-contract.json').read_text())
    assert contract['horizon'] == 600
    expected_seed = contract['train_seed'] + producing * 128
    env_index, identities, successes = {}, set(), {}
    reward_totals = {}
    paths = sorted((root / 'environment').glob(f'*/training-i{producing:03d}-*-chunk*.pt.gz'))
    assert len(paths) == 96
    for path in paths:
        value = load(path)
        chunk = int(path.name.split('-chunk')[1].split('.')[0])
        assert value['iteration'] == producing and value['phase'] == 'training'
        observation(value['current_observation'])
        assert len(value['episode_ids']) == len(value['seeds']) == 16
        actions = torch.as_tensor(value['decoded_actions'])
        assert actions.shape == (16, 50, 12) and torch.isfinite(actions).all()
        assert value['rewards'].shape == value['terminations'].shape == value['truncations'].shape == (16, 50)
        assert torch.isfinite(value['rewards']).all()
        assert set(value['rewards'].unique().tolist()) <= {0., 1.}
        assert observation_hash(value['current_observation']) == value['current_observation_sha256']
        # Native environment routing removes diagnostic task_metadata before
        # sending observations to rollout. Images/state/instruction are intact.
        key = observation_hash({k: v for k, v in value['current_observation'].items()
                                if k != 'task_metadata'})
        assert key not in env_index, 'Ambiguous environment observation batch'
        env_index[key] = (path, value['episode_ids'], chunk)
        for lane, (identity, seed) in enumerate(zip(value['episode_ids'], value['seeds'], strict=True)):
            assert identity == f"{contract['task']}/{seed}"
            assert expected_seed <= seed < expected_seed + 128
            assert (identity, chunk) not in identities
            identities.add((identity, chunk))
            reward_totals[identity] = reward_totals.get(identity, 0.) + float(value['rewards'][lane].sum())
        if chunk == 12:
            assert value['terminal_observation'] is not None
            observation(value['terminal_observation'])
    episode_paths = sorted((root / 'environment').glob(f'*/training-i{producing:03d}-*-episodes.json'))
    assert len(episode_paths) == 8
    for path in episode_paths:
        value = json.loads(path.read_text())
        assert value['completed'] and value['iteration'] == producing
        for row in value['rows']:
            counts = row['native_counts']
            assert counts['controls'] - counts['reset_controls'] == 600
            successes[row['episode_id']] = row['success_once']
    assert len(identities) == 1536 and len(successes) == len(reward_totals) == 128
    assert all(reward_totals[key] == float(successes[key]) for key in successes)
    sampling_index, sampling_records, sampled_env = {}, 0, set()
    for path in sorted((root / 'sampling').glob('*/*.pt.gz')):
        value = load(path)
        if (value['mode'] != 'train' or value['terminal_value_only'] or
                value['collection_iteration'] != collection):
            continue
        assert value['producing_policy_iteration'] == producing
        assert value['sampling_record'] == 'full'
        assert value['rng_before'] is not None and value['rng_after'] is not None
        assert value['processed_model_state'].shape == (16, 32)
        assert value['model_actions'].shape == (16, 50, 32)
        assert torch.isfinite(value['model_actions']).all()
        fields = value['forward_sampling_fields']
        assert fields['chains'].shape == (16, 11, 50, 32)
        assert torch.isfinite(fields['chains']).all()
        assert len(value['actual_gaussian_parameters']) == 10
        environment = value['public_observation_sha256']
        assert environment in env_index and environment not in sampled_env
        sampled_env.add(environment)
        env_path, episode_ids, chunk = env_index[environment]
        model_hashes = row_hashes(fields['model_action'])
        logprob_hashes = row_hashes(value['old_logprobs'])
        for lane, signature in enumerate(zip(model_hashes, logprob_hashes, strict=True)):
            assert signature not in sampling_index, 'Ambiguous native target/logprob signature'
            sampling_index[signature] = dict(episode_id=episode_ids[lane], chunk=chunk,
                environment_record=str(env_path), sampling_record=str(path), sampling_lane=lane)
        sampling_records += 1
    assert sampled_env == set(env_index) and sampling_records == 96 and len(sampling_index) == 1536
    joins = []
    actor_paths = sorted((root / 'training-rollouts').glob(f'iteration-{collection:04d}-*.pt'))
    assert len(actor_paths) == 2
    for path in actor_paths:
        saved = load(path)
        assert saved['producing_policy_iteration'] == producing and not saved['evaluation_data']
        batch = saved['rollout_batch']
        assert batch['prev_logprobs'].shape == (12, 64, 50, 12)
        fields = batch['forward_inputs']
        model_hashes = row_hashes(fields['model_action'].flatten(0, 1))
        logprob_hashes = row_hashes(batch['prev_logprobs'].flatten(0, 1))
        for index, signature in enumerate(zip(model_hashes, logprob_hashes, strict=True)):
            assert signature in sampling_index, 'Retained actor row lacks an exact sampling/environment join'
            joins.append(dict(actor_record=str(path), time_index=index // 64,
                actor_lane=index % 64, **sampling_index[signature]))
        assert batch['rewards'].shape == (12, 64, 50)
        for name in ['advantages', 'returns', 'loss_mask', 'prev_values', 'dones']:
            assert name in batch
    assert len(joins) == 1536 and {(row['episode_id'], row['chunk']) for row in joins} == identities
    join_path = root.parent / f'{tag}-JOIN-{root.name}-i{collection:03d}.json'
    assert not join_path.exists()
    join_path.write_text(json.dumps(dict(completed=True, rows=joins), indent=2) + '\n')
    return dict(task=contract['task'], root=str(root), collection_iteration=collection,
        episodes=128, chunk_samples=1536, post_success_suffix_retained=True,
        raw_three_view_RGB_state_instruction=True, primitive_actions_rewards_boundaries=True,
        horizon_terminal_observations=128, native_continuous_model_targets=True,
        full_sampling_chain_and_RNG=True, producing_policy_version=True,
        actor_sampling_environment_join=str(join_path),
        observation_join='Exact native payload hash; only environment-only task_metadata excluded',
        join_sha256=hashlib.sha256(join_path.read_bytes()).hexdigest())


def audit_task(arguments):
    root, task, tag = arguments
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stage = 'q1-r2' if task == 'OpenDrawer' else 'q1-r1'
    rows = [audit(root / task / stage, 1, tag)]
    production = root / task / 'production-r1'
    if (production / 'progress.json').exists() and json.loads((production / 'progress.json').read_text())['completed_iterations'] >= 2:
        rows.append(audit(production, 2, tag))
    print(json.dumps(dict(task=task, completed=True, segments=len(rows))), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    tasks = ['OpenDrawer', 'CloseToasterOvenDoor', 'PickPlaceDrawerToCounter', 'CoffeeSetupMug', 'TurnOffStove']
    # Five independent audits, two CPU threads each, within the allocated
    # development container's thirteen CPUs; no GPU visibility/model calls.
    with ProcessPoolExecutor(max_workers=5, mp_context=multiprocessing.get_context('spawn')) as pool:
        groups = list(pool.map(audit_task, [(args.root, task, args.output.stem) for task in tasks]))
    rows = [row for group in groups for row in group]
    result = dict(completed=True, checked_unix=time.time(), audits=rows, new_controls=0, SGD=0,
        ready_methods=['AD', 'DPT', 'DIT', 'in-context IQL'],
        remaining=['Finish and audit all 130 source versions per task',
            'Retain and qualify each final teacher; query DPT labels offline',
            'Adapt corpus export/fitting to this five-task 130-update contract; legacy exporter is 18-update-specific',
            'Fit and qualify reactive interface before history fits; preserve native action encoding/mask'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
