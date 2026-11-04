"""Check native scene identity and initial policy replay before the first SGD."""
import hashlib
import json
from pathlib import Path
import time

import torch
from fluffy_bassoon.meta_rl.robocasa_pairing import compare_native_pair

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-ppo-grpo-meta-stage1-20261104-r1/ppo/attempt-01')
CAMERAS = ('observation/image', 'observation/wrist_image', 'observation/extra_view_image')


def collection(root):
    rows = {}
    for rank in range(2):
        p = root / 'training-rollouts' / f'iteration-0001-policy-0000-rank-{rank:02d}.pt'
        deadline = time.monotonic() + 90
        while not p.with_suffix('.json').is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(f'First collection was not published: {p}')
            time.sleep(0.25)
        batch = torch.load(p, mmap=True, map_location='cpu', weights_only=False)['rollout_batch']
        inputs = batch['forward_inputs']
        assert inputs['chains'].shape == (12, 64, 11, 50, 32)
        for lane in range(64):
            noise = inputs['chains'][:, lane, 0]
            key = hashlib.sha256(noise[0].contiguous().numpy().tobytes()).hexdigest()
            if key in rows:
                raise ValueError(f'Duplicate sampled lane in first collection of {root}')
            rows[key] = dict(rank=rank, lane=lane, noise=noise, indices=inputs['denoise_inds'][:,lane,0],
                state=inputs['observation/state'][0,lane], rgb=[inputs[k][0,lane] for k in CAMERAS],
                actions=batch['actions'][0,lane], reward=float(batch['rewards'][:,lane].sum()))
    return rows


def verify(output, rank, contrast):
    old, new = collection(BASE), collection(output)
    if len(old) != 128 or set(new) != set(old):
        raise ValueError('First collection failed the one-to-one 128-lane initial-noise join')
    pairs = [(old[k], b) for k,b in new.items() if b['rank'] == rank]
    assert len(pairs) == 64
    physical = []
    for env_rank in range(rank * 4, (rank + 1) * 4):
        rel = Path('environment') / f'rank-{env_rank:02d}' / 'training-i000-round00-initial.pt'
        snapshots = [torch.load(root/rel, map_location='cpu', weights_only=False) for root in (BASE,output)]
        physical.append(compare_native_pair(*snapshots))
    rgb_exact = [all(torch.equal(u,v) for u,v in zip(a['rgb'],b['rgb'],strict=True)) for a,b in pairs]
    action_gap = [float((b['actions']-a['actions']).abs().max()) for a,b in pairs]
    same_image_gaps = [gap for same,gap in zip(rgb_exact,action_gap,strict=True) if same]
    states_equal = all(torch.equal(a['state'],b['state']) for a,b in pairs)
    noise_equal = all(torch.equal(a['noise'],b['noise']) for a,b in pairs)
    indices_equal = all(torch.equal(a['indices'],b['indices']) for a,b in pairs)
    selected = sorted({i for r in new.values() for i in r['indices'].tolist()})
    passed = states_equal
    if contrast == 'D':
        passed = passed and 9 in selected and set(selected) <= set(range(10))
    else:
        passed = passed and noise_equal and indices_equal and bool(same_image_gaps) and max(same_image_gaps)==0
    result = dict(passed=passed, contrast=contrast, rank=rank, before_optimizer_steps=0, globally_joined=128,
        initial_public_states_equal=states_equal, all_chunk_initial_noises_equal=noise_equal,
        all_selected_indices_equal=indices_equal, unchanged_rgb_first_lanes=sum(rgb_exact),
        max_first_action_gap_equal_rgb=max(same_image_gaps,default=None), max_first_action_gap=max(action_gap),
        matched_reward_lanes=sum(a['reward']==b['reward'] for a,b in pairs), selected_indices=selected,
        stage1_rank_lanes_in_received_order=[[a['rank'],a['lane']] for a,b in pairs],
        physical_public_resets=physical)
    (output/f'first-collection-match-rank-{rank:02d}.json').write_text(json.dumps(result,indent=2)+'\n')
    if not passed:
        raise ValueError(f'First on-policy collection failed the matched {contrast} check on actor {rank}')
