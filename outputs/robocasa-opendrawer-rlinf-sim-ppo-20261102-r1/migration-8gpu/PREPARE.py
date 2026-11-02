"""Allocated-CPU artifact conversion; never runs a policy or simulator."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from fluffy_bassoon.meta_rl.rlinf_native_ppo_runtime import save_tensor
from fluffy_bassoon.meta_rl.robocasa_pairing import canonical_case, compare_native_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boundary', required=True)
    parser.add_argument('--before', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, out = Path(args.boundary), Path(args.output)
    assert not out.exists(), 'Never overwrite a prepared migration'
    b = json.loads((source / 'pilot-boundary.json').read_text())
    assert b['completed'] and b['global_step'] >= 1
    for row in b['workers'] + b['actor_checkpoint_files']:
        assert sha(row['path']) == row['sha256'], row['path']
    before = json.loads(Path(args.before).read_text())
    assert before['completed'] and before['cases'] == 64
    out.mkdir(parents=True)
    checkpoint = out / f"global_step_{b['global_step']}"
    checkpoint.mkdir()
    (checkpoint / 'actor').symlink_to(source / 'actor', target_is_directory=True)
    pilot = checkpoint / 'pilot'
    pilot.mkdir()
    # Learner world size, Adam, LR and all learner RNG are unchanged.
    for rank in range(2):
        shutil.copyfile(source / 'pilot' / f'actor-{rank:02d}.pt', pilot / f'actor-{rank:02d}.pt')
    mapping = []
    for rank in range(8):
        old, child = divmod(rank, 4)
        for role in ('env', 'rollout'):
            f = source / 'pilot' / f'{role}-{old:02d}.pt'
            state = torch.load(f, map_location='cpu', weights_only=False)
            state['rank'] = rank
            seed = int(hashlib.sha256(f"{sha(source / 'pilot-boundary.json')}:{role}:{rank}".encode()).hexdigest()[:8], 16)
            if role == 'rollout':
                state['migration_rng_seed'] = seed
                # Prediction files start in a fresh attempt; old files remain immutable.
            else:
                for row in state['samplers']:
                    size = 4 if row['mode'] == 'train' else 8
                    row['task_ids'] = row['task_ids'][child * size:(child + 1) * size].copy()
                    assert len(row['task_ids']) == size
                    row['seed'] = (210100 if row['mode'] == 'train' else 410100) + rank
                    row['generator'] = np.random.default_rng(seed).bit_generator.state
                state['recovery'] = 'Global iteration/case cursor retained; deterministic new-world sampler mapping, not bitwise stochastic continuation'
            receipt = save_tensor(pilot / f'{role}-{rank:02d}.pt', state)
            mapping.append(dict(role=role, old_rank=old, new_rank=rank,
                new_world_seed=seed, source_sha256=sha(f), target=receipt))
    # Repartition the exact original native public/physical snapshots by seed.
    originals = [torch.load(Path(before['canonical_directory']) / f'rank-{rank:02d}/initial.pt',
                            map_location='cpu', weights_only=False) for rank in range(2)]
    for rank in range(8):
        old, child = divmod(rank, 4)
        original = originals[old]
        selection = slice(child * 8, (child + 1) * 8)
        snapshot = deepcopy(original)
        snapshot['case']['seeds'] = original['case']['seeds'][selection]
        snapshot['physical'] = original['physical'][selection]
        snapshot['states'] = original['states'][selection].copy()
        snapshot['task_descriptions'] = original['task_descriptions'][selection]
        snapshot['images'] = {key: value[selection].copy() for key, value in original['images'].items()}
        compare_native_pair(snapshot, snapshot)
        canonical_case(out / 'canonical-eval' / f'rank-{rank:02d}', snapshot,
            save=save_tensor, load=lambda p: torch.load(p, map_location='cpu', weights_only=False),
            source=dict(original_before_reference=args.before, migration='2-to-8 environment ranks; same64 native cases'))
    # Keep the before metrics/episode hashes; bind only the new resume/canonical layout.
    reference = deepcopy(before)
    reference.update(checkpoint=str(checkpoint), canonical_directory=str(out / 'canonical-eval'),
        migration_source_boundary=str(source), original_before_reference=args.before,
        learner_world_size=2, rollout_environment_world_size=8,
        stochastic_continuation='Deterministic newly seeded rollout streams at the complete update boundary; original RNG sidecars preserved in source')
    write(out / 'BEFORE-RECOVERY-8GPU.json', reference)
    write(out / 'MIGRATION.json', dict(completed=True, original_boundary=str(source),
        original_boundary_sha256=sha(source / 'pilot-boundary.json'), global_step=b['global_step'],
        actor_DCP_Adam_LR='Unchanged2-rank native DCP; no optimizer reshard or cold initialization',
        actor_sidecars='Byte-identical2-rank copies, including gradient scaler, optimizer-step count and RNG',
        rollout_environment_mapping=mapping, source_policy_actions_replayed=0,
        next_global_train_seed=210100 + b['global_step'] * 128,
        before_cases=64, new_evaluation_controls=0,
        before_reference_sha256=sha(out / 'BEFORE-RECOVERY-8GPU.json')))
    print((out / 'MIGRATION.json').read_text())


if __name__ == '__main__':
    main()
