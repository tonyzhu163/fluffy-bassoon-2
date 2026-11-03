"""Read-only Q1 audit on allocated Linux CPU; no policy or simulator calls."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import torch


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def load(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rb') as stream:
            return torch.load(stream, map_location='cpu', weights_only=False)
    return torch.load(path, map_location='cpu', weights_only=False)


def hashes(tensor):
    tensor = torch.as_tensor(tensor).contiguous()
    return [hashlib.sha256(str(tuple(row.shape)).encode() + str(row.dtype).encode() +
        row.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        for row in tensor]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('q1', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.q1
    contract = read(root / 'flow-contract.json')
    assert read(root / 'clock.json')['exit_code'] == 0
    assert read(root / 'progress.json')['completed_iterations'] == 1
    assert read(root / 'operator-checkpoint-exit.json')['global_step'] == 1
    before = read(root / 'before-evaluation.json')
    assert before['completed'] and .05 <= before['metrics']['success_once'] <= .70
    before_files = sorted((root / 'environment').glob('*/before-*-episodes.json'))
    if not before_files and before.get('reused_reference'):
        reused = read(before['reused_reference']['path'])
        before_files = [Path(row['path']) for row in reused['episode_refs']]
    before_rows = [row for path in before_files for row in read(path)['rows']]
    train_files = sorted((root / 'environment').glob('*/training-i000-*-episodes.json'))
    train_rows = [row for path in train_files for row in read(path)['rows']]
    assert len(before_rows) == 64 and len(train_rows) == 128
    for rows, seed in [(before_rows, contract['eval_seed']), (train_rows, contract['train_seed'])]:
        assert sorted(row['seed'] for row in rows) == list(range(seed, seed + len(rows)))
        assert all(row['native_counts']['controls'] - row['native_counts']['reset_controls'] == 600
                   for row in rows)
    updates = [read(root / f'updates/iteration-001-rank-{rank:02d}.json') for rank in range(2)]
    assert all(row['completed'] and row['optimizer_steps_after'] == 24 and
        row['optimizer_steps_before'] == 0 and row['frozen_vlm_unchanged'] and
        all(math.isfinite(value) for value in row['losses'].values()) for row in updates)
    parity = [read(root / f'logprob-parity/iteration-001-rank-{rank:02d}.json') for rank in range(2)]
    assert all(row['passed'] and row['optimizer_steps'] == 0 and row['before_backward'] for row in parity)
    fields = {'observation/image': 'main_images', 'observation/wrist_image': 'wrist_images',
              'observation/extra_view_image': 'extra_view_images'}
    env_hashes = {name: Counter() for name in fields}
    actor_hashes = {name: Counter() for name in fields}
    sampling_chains, actor_chains = Counter(), Counter()
    env_chunks = sorted((root / 'environment').glob('*/training-i000-*-chunk*.pt.gz'))
    assert len(env_chunks) == 8 * 12
    for path in env_chunks:
        value = load(path)
        assert len(value['episode_ids']) == len(value['seeds']) == 16
        for name, env_name in fields.items():
            env_hashes[name].update(hashes(value['current_observation'][env_name]))
        if path.name.endswith('chunk12.pt.gz'):
            assert value['terminal_observation'] is not None
    for path in sorted((root / 'sampling').glob('*/*.pt.gz')):
        value = load(path)
        if value['mode'] == 'train' and not value['terminal_value_only']:
            assert value['sampling_record'] == 'full'
            sampling_chains.update(hashes(value['forward_sampling_fields']['chains']))
    actor_paths = sorted((root / 'training-rollouts').glob('iteration-0001-*.pt'))
    assert len(actor_paths) == 2
    dropped_uncompressed_bytes = 0
    for path in actor_paths:
        value = load(path)
        batch = value['rollout_batch']
        inputs = batch['forward_inputs']
        time_count, lanes = batch['prev_logprobs'].shape[:2]
        assert time_count == 12 and lanes == 64
        for name in fields:
            tensor = inputs[name]
            actor_hashes[name].update(hashes(tensor.reshape(time_count * lanes, *tensor.shape[2:])))
            dropped_uncompressed_bytes += tensor.numel() * tensor.element_size()
        tensor = inputs['chains']
        actor_chains.update(hashes(tensor.reshape(time_count * lanes, *tensor.shape[2:])))
        dropped_uncompressed_bytes += tensor.numel() * tensor.element_size()
    assert all(actor_hashes[name] == env_hashes[name] and sum(actor_hashes[name].values()) == 1536
               for name in fields), 'Actor images are not byte-for-byte covered by the environment corpus'
    assert actor_chains == sampling_chains and sum(actor_chains.values()) == 1536
    checkpoint = root / contract['task'] / 'checkpoints/global_step_1'
    boundary = read(checkpoint / 'pilot-boundary.json')
    assert boundary['completed'] and boundary['global_step'] == 1 and len(boundary['workers']) == 18
    for row in boundary['workers'] + boundary['actor_checkpoint_files']:
        assert digest(row['path']) == row['sha256']
    reference = dict(completed=True, cases=64, checkpoint=str(checkpoint),
        checkpoint_boundary_sha256=digest(checkpoint / 'pilot-boundary.json'),
        canonical_directory=str(root / 'canonical-eval'), metrics=before['metrics'],
        episode_refs=[dict(path=str(path), sha256=digest(path)) for path in before_files],
        original_output=str(root), original_source=read(root / 'manifest.json')['source'])
    reference_path = args.output.with_name('BEFORE-RESUME.json')
    reference_path.write_text(json.dumps(reference, indent=2) + '\n')
    components = {name: sum(path.stat().st_size for path in (root / name).rglob('*') if path.is_file())
                  for name in ['environment', 'sampling', 'training-rollouts', contract['task']]}
    result = dict(completed=True, checked_unix=time.time(), task=contract['task'],
        source=read(root / 'manifest.json')['source'], before=before['metrics'],
        train_episodes=128, train_chunk_samples=1536, updates=updates, parity=parity,
        actor_images_equal_environment=True, actor_chains_equal_sampling=True,
        actor_duplicate_uncompressed_bytes=dropped_uncompressed_bytes,
        component_bytes=components, full_boundary_verified=True,
        before_reference=dict(path=str(reference_path), sha256=digest(reference_path)),
        new_controls=0, SGD=0)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
