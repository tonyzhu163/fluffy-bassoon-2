"""Audit actual saved source payloads on allocated Linux CPU, without model calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import time

import torch
import numpy as np


FILE_HASHES = {}


def payload(path):
    data = Path(path).read_bytes()
    FILE_HASHES[str(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return data


def load(path):
    data = io.BytesIO(payload(path))
    if str(path).endswith('.gz'):
        with gzip.GzipFile(fileobj=data, mode='rb') as stream:
            return torch.load(stream, map_location='cpu', weights_only=False)
    return torch.load(data, map_location='cpu', weights_only=False)


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



def index_sampling(root, collections, previous_audit=None):
    """One ordered pass; success-dependent terminal-value calls vary in count."""
    completed = json.loads((root / 'progress.json').read_text())['completed_iterations']
    closed = (root / 'clock.json').exists() and json.loads((root / 'clock.json').read_text())['exit_code'] == 0
    assert completed >= max(collections)
    assert closed or completed > max(collections), 'Require a following complete update to avoid reading live files'
    indexed = {collection: [] for collection in collections}
    anchors = {}
    if previous_audit is not None:
        prior = json.loads(previous_audit.read_text())
        assert prior['completed']
        candidates = [row for row in prior['audits'] if row['root'] == str(root)
                      and row['collection_iteration'] == min(collections) - 1]
        assert len(candidates) == 1, 'Require the immediately preceding audited collection'
        reference = candidates[0]
        manifest_bytes = Path(reference['raw_file_manifest']).read_bytes()
        assert hashlib.sha256(manifest_bytes).hexdigest() == reference['raw_file_manifest_sha256']
        manifest = json.loads(manifest_bytes)
        assert manifest['completed']
        for rank in range(8):
            directory = root / 'sampling' / f'rank-{rank:02d}'
            files = [(Path(name), record) for name, record in manifest['files'].items()
                     if Path(name).parent == directory]
            assert files, 'No sampling boundary in previous raw-file manifest'
            path, record = max(files, key=lambda item: int(item[0].name.split('.')[0]))
            value = load(path)
            assert FILE_HASHES[str(path)] == record, 'Previous sampling anchor changed'
            ordinal = int(path.name.split('.')[0])
            assert value['rank'] == rank and value['prediction_ordinal'] == ordinal
            assert value['mode'] == 'train' and value['collection_iteration'] == min(collections) - 1
            assert value['producing_policy_iteration'] == min(collections) - 2
            anchors[rank] = ordinal
    for rank in range(8):
        previous = min(collections) - 1 if anchors else -1
        for path in sorted((root / 'sampling' / f'rank-{rank:02d}').glob('*.pt.gz')):
            if rank in anchors and int(path.name.split('.')[0]) <= anchors[rank]:
                continue
            value = load(path)
            assert value['rank'] == rank and value['prediction_ordinal'] == int(path.name.split('.')[0])
            if value['mode'] != 'train':
                if previous >= max(collections):
                    break
                continue
            collection = value['collection_iteration']
            assert collection >= previous
            previous = collection
            if collection > max(collections):
                break
            assert value['producing_policy_iteration'] == collection - 1
            if collection not in indexed:
                continue
            row = dict(path=str(path), file=FILE_HASHES[str(path)].copy(), rank=rank,
                       prediction_ordinal=value['prediction_ordinal'],
                       terminal_value_only=bool(value['terminal_value_only']))
            if not row['terminal_value_only']:
                assert value['sampling_record'] == 'full'
                assert value['rng_before'] is not None and value['rng_after'] is not None
                assert value['processed_model_state'].shape == (16, 32)
                assert value['model_actions'].shape == (16, 50, 32)
                assert torch.isfinite(value['model_actions']).all()
                fields = value['forward_sampling_fields']
                assert fields['chains'].shape == (16, 11, 50, 32)
                assert torch.isfinite(fields['chains']).all()
                assert len(value['actual_gaussian_parameters']) == 10
                row.update(environment=value['public_observation_sha256'],
                    model_hashes=row_hashes(fields['model_action']),
                    logprob_hashes=row_hashes(value['old_logprobs']))
            indexed[collection].append(row)
    for collection, rows in indexed.items():
        assert all(sum(not row['terminal_value_only'] and row['rank'] == rank for row in rows) == 12
                   for rank in range(8)), 'Every target batch needs twelve actual decisions per rank'
    return indexed


def audit(root, collection, tag, sampling):
    FILE_HASHES.clear()
    producing = collection - 1
    contract = json.loads(payload(root / 'flow-contract.json'))
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
        value = json.loads(payload(path))
        assert value['completed'] and value['iteration'] == producing
        for row in value['rows']:
            counts = row['native_counts']
            assert counts['controls'] - counts['reset_controls'] == 600
            successes[row['episode_id']] = row['success_once']
    assert len(identities) == 1536 and len(successes) == len(reward_totals) == 128
    assert all(reward_totals[key] == float(successes[key]) for key in successes)
    initial_paths = sorted((root / 'environment').glob(f'*/training-i{producing:03d}-*-initial.pt'))
    assert len(initial_paths) == 8
    for path in initial_paths:
        payload(path)
    sampling_index, sampling_records, sampled_env = {}, 0, set()
    terminal_records = 0
    rank_records = {}
    for record in sampling:
        FILE_HASHES[record['path']] = record['file']
        terminal_records += int(record['terminal_value_only'])
        if record['terminal_value_only']:
            continue
        path = Path(record['path'])
        environment = record['environment']
        assert environment in env_index and environment not in sampled_env
        sampled_env.add(environment)
        env_path, episode_ids, chunk = env_index[environment]
        for lane, signature in enumerate(zip(record['model_hashes'], record['logprob_hashes'], strict=True)):
            assert signature not in sampling_index, 'Ambiguous native target/logprob signature'
            sampling_index[signature] = dict(episode_id=episode_ids[lane], chunk=chunk,
                environment_record=str(env_path), sampling_record=str(path), sampling_lane=lane)
        sampling_records += 1
        rank_records[path.parent] = rank_records.get(path.parent, 0) + 1
    assert len(rank_records) == 8 and set(rank_records.values()) == {12}
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
    join_digest = hashlib.sha256(join_path.read_bytes()).hexdigest()
    reference = None
    if collection == 2:
        candidates = []
        for reference_path in [root.parents[1] / 'FOUR-METHOD-AUDIT-R3.json',
                               root.parents[1] / 'FOUR-METHOD-AUDIT-R4.json']:
            if reference_path.exists():
                candidates.extend((reference_path, row) for row in json.loads(reference_path.read_text())['audits']
                                  if row['root'] == str(root) and row['collection_iteration'] == 2)
        assert len(candidates) == 1, 'Require the already qualified first production join reference'
        reference_path, reference_row = candidates[0]
        assert join_digest == reference_row['join_sha256'], 'Indexed join differs from full-scan reference'
        reference = dict(path=str(reference_path), join_sha256=join_digest, byte_identical_join=True)
    manifest_path = root.parent / f'{tag}-FILES-{root.name}-i{collection:03d}.json'
    assert not manifest_path.exists()
    manifest_path.write_text(json.dumps(dict(completed=True, files=dict(sorted(FILE_HASHES.items()))), indent=2) + '\n')
    return dict(task=contract['task'], root=str(root), collection_iteration=collection,
        episodes=128, chunk_samples=1536, post_success_suffix_retained=True,
        raw_three_view_RGB_state_instruction=True, primitive_actions_rewards_boundaries=True,
        horizon_terminal_observations=128, native_continuous_model_targets=True,
        full_sampling_chain_and_RNG=True, producing_policy_version=True,
        actor_sampling_environment_join=str(join_path),
        observation_join='Exact native payload hash; only environment-only task_metadata excluded',
        join_sha256=join_digest, first_batch_reference=reference,
        raw_file_manifest=str(manifest_path), raw_files=len(FILE_HASHES),
        raw_file_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        raw_bytes=sum(row['bytes'] for row in FILE_HASHES.values()),
        terminal_value_sampling_records=terminal_records,
        sampling_index='Single monotonic metadata pass; variable terminal-value predictions retained')


def audit_task(arguments):
    root, task, tag, production_only, collections, previous_audit, reference_audit, threads = arguments
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    stage = 'q1-r2' if task == 'OpenDrawer' else 'q1-r1'
    rows = []
    if not production_only:
        q1 = root / task / stage
        rows.append(audit(q1, 1, tag, index_sampling(q1, [1])[1]))
    production = root / task / 'production-r1'
    if (production / 'progress.json').exists():
        indexed = index_sampling(production, collections, previous_audit)
        for collection in collections:
            rows.append(audit(production, collection, tag, indexed[collection]))
    assert rows, 'Requested production batch is not complete yet'
    if reference_audit is not None:
        reference = json.loads(reference_audit.read_text())
        assert reference['completed']
        for row in rows:
            matches = [item for item in reference['audits'] if item['root'] == row['root']
                       and item['collection_iteration'] == row['collection_iteration']]
            assert len(matches) == 1
            for key in ['join_sha256', 'raw_file_manifest_sha256', 'raw_files', 'raw_bytes',
                        'episodes', 'chunk_samples', 'terminal_value_sampling_records']:
                assert row[key] == matches[0][key], f'Reference mismatch: {task} {key}'
            row['qualification_reference'] = str(reference_audit)
    print(json.dumps(dict(task=task, completed=True, segments=len(rows))), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    tasks = ['OpenDrawer', 'CloseToasterOvenDoor', 'PickPlaceDrawerToCounter', 'CoffeeSetupMug', 'TurnOffStove']
    parser.add_argument('--tasks', choices=tasks, nargs='+', default=tasks)
    parser.add_argument('--production-only', action='store_true')
    parser.add_argument('--collections', type=int, nargs='+', default=[2])
    parser.add_argument('--previous-audit', type=Path,
                        help='Reuse an audited prefix; verify its last sampling record on each rank')
    parser.add_argument('--reference-audit', type=Path,
                        help='Qualification only: require identical joins and raw-file manifests')
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    assert not args.output.exists()
    assert len(set(args.tasks)) == len(args.tasks)
    assert len(set(args.collections)) == len(args.collections) and all(2 <= x <= 130 for x in args.collections)
    assert 1 <= args.workers <= 5 and 1 <= args.threads <= 2
    assert args.workers * args.threads <= 10
    # Q1 may be audited alongside any disjoint completed production block.
    # Five independent audits, two CPU threads each, within the allocated
    # development container's thirteen CPUs; no GPU visibility/model calls.
    with ProcessPoolExecutor(max_workers=min(args.workers, len(args.tasks)), mp_context=multiprocessing.get_context('spawn')) as pool:
        groups = list(pool.map(audit_task, [(args.root, task, args.output.stem, args.production_only,
            sorted(args.collections), args.previous_audit, args.reference_audit, args.threads) for task in args.tasks]))
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
