"""First on-policy collection check, run before the first optimizer step."""
import json
from pathlib import Path

import torch
from fluffy_bassoon.meta_rl.robocasa_pairing import compare_native_pair

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-ppo-grpo-meta-stage1-20261104-r1/ppo/attempt-01')


def verify(output, rank, contrast):
    def load(path):
        return torch.load(path, map_location='cpu', weights_only=False)

    def batch(root):
        p = root / 'training-rollouts' / f'iteration-0001-policy-0000-rank-{rank:02d}.pt'
        return load(p)['rollout_batch']

    old, new = batch(BASE), batch(output)
    a, b = old['forward_inputs'], new['forward_inputs']
    initial_noise_equal = torch.equal(a['chains'][0, :, 0], b['chains'][0, :, 0])
    noises_equal = torch.equal(a['chains'][:, :, 0], b['chains'][:, :, 0])
    indices_equal = torch.equal(a['denoise_inds'], b['denoise_inds'])
    first_actions = (new['actions'][0] - old['actions'][0]).abs()
    same_rgb = torch.ones(first_actions.shape[0], dtype=torch.bool)
    for key in ('observation/image', 'observation/wrist_image', 'observation/extra_view_image'):
        same_rgb &= (a[key][0] == b[key][0]).flatten(1).all(1)
    matched_first_max = float(first_actions[same_rgb].max()) if same_rgb.any() else None
    physical = []
    for env_rank in range(rank * 4, (rank + 1) * 4):
        rel = Path('environment') / f'rank-{env_rank:02d}' / 'training-i000-round00-initial.pt'
        physical.append(compare_native_pair(load(BASE / rel), load(output / rel)))
    passed = initial_noise_equal and torch.equal(a['observation/state'][0], b['observation/state'][0])
    if contrast != 'D':
        passed = passed and noises_equal and indices_equal and matched_first_max == 0
    row = dict(passed=bool(passed), rank=rank, contrast=contrast, before_optimizer_steps=0,
               initial_noise_equal=initial_noise_equal, all_chunk_initial_noises_equal=noises_equal,
               all_selected_indices_equal=indices_equal, unchanged_rgb_first_lanes=int(same_rgb.sum()),
               max_first_action_gap_equal_rgb=matched_first_max, max_first_action_gap=float(first_actions.max()),
               selected_indices=sorted(set(b['denoise_inds'][:, :, 0].flatten().tolist())),
               old_reward_sum=float(old['rewards'].sum()), new_reward_sum=float(new['rewards'].sum()),
               physical_public_resets=physical)
    (output / f'first-collection-match-rank-{rank:02d}.json').write_text(json.dumps(row, indent=2)+'\n')
    if not passed:
        raise ValueError(f'First collection differs from the common starting state on rank {rank}')
