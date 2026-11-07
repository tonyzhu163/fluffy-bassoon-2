"""GRPO numerical/signal audit on recorded PPO-policy0 simulator groups.

Counterfactual reduction only: no GRPO policy, model inference, optimizer or
simulator runs. Raw pre-bootstrap simulator rewards come from immutable traces,
not the PPO actor buffers that include critic timeout terms.
"""
import gc
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import rlinf.algorithms
from rlinf.algorithms.registry import calculate_adv_and_returns
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from motus_learner_reference import audit

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
SOURCE = BASE/'motusoptimized03/state'


def main():
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    native = Path(os.environ['REPO_PATH'])
    assert subprocess.check_output(['git', '-C', str(native), 'rev-parse', 'HEAD'], text=True).strip() == 'c605b9d55ed93a180119b099e6f836abff293b08'
    started = time.monotonic()
    output = BASE/'grpo-rawgroupaudit02'; output.mkdir()
    report = dict(status='checking', source_commit=os.environ['TASK_SOURCE'],
        gpu_seconds=0, new_simulator_actions=0, model_calls=0, optimizer_updates=0,
        source_policy='optimized03 fresh Motus policy0 collected for PPO',
        source_native_batch=32, group_size=8, grouping='same reset case, eight replicas',
        critic_bootstrap_included=False, grpo_training_run=False, policy_improvement_claim=False,
        numerical_protocol='Counterfactual native GRPO reduction of recorded raw simulator rewards; no PPO actor rewards reused',
        trace_loading='Read compressed bytes once; decompress once into seekable CPU memory before torch.load',
        ranks=[], files={}, groups=[])
    write(output/'RESULT.json', report)
    fixtures = []
    for rank in range(8):
        seeds, rewards, endings = {}, {}, {}
        for round_index in range(4):
            path = SOURCE/f'group-resets/rank-{rank:02d}-reset-{round_index:04d}.json'
            receipt = json.loads(path.read_bytes()); report['files'][str(path)] = sha(path)
            assert receipt['passed'] and receipt['rank'] == rank and receipt['reset_ordinal'] == round_index
            assert len(receipt['groups']) == 4
            values = sorted(receipt['groups'], key=lambda row: row['local_group'])
            assert all(row['passed'] and row['replica_count'] == 8 and row['policy_iteration'] == 0 for row in values)
            seeds[round_index] = [row['scene_seed'] for row in values for _ in range(8)]
        count = 0
        for path in sorted((SOURCE/f'traces/rank-{rank:02d}').glob('*.pt.gz')):
            # torch.load seeks inside its ZIP stream. Seeking a GzipFile can
            # repeatedly decompress the same large image archive.
            compressed = path.read_bytes()
            digest = hashlib.sha256(compressed).hexdigest()
            with io.BytesIO(gzip.decompress(compressed)) as handle:
                trace = torch.load(handle, map_location='cpu', weights_only=False)
            del compressed
            if trace['evaluation_data']:
                del trace
                continue
            assert trace['rank'] == rank and trace['producing_policy_iteration'] == 0
            assert trace['collection_iteration'] == 1
            round_index, chunk = trace['round_index'], trace['chunk_index']
            assert (round_index, chunk) == divmod(count, 6)
            assert trace['reset_state_ids'].tolist() == seeds[round_index]
            raw = torch.as_tensor(trace['outputs'][1]).float().clone()
            ended = (torch.as_tensor(trace['outputs'][2]).bool() | torch.as_tensor(trace['outputs'][3]).bool()).clone()
            assert raw.shape == ended.shape == (32, 32) and torch.isfinite(raw).all()
            rewards[round_index, chunk] = raw
            endings[round_index, chunk] = ended
            report['files'][str(path)] = digest
            count += 1
            del trace; gc.collect()
            if count % 6 == 0: print('RAW_GROUP_ROUND_READ', rank, count // 6, flush=True)
            if count == 24: break  # Exclude later DEV and the interrupted policy1 collection.
        assert count == 24
        reward = torch.stack([torch.cat([rewards[r, c] for r in range(4)], dim=0) for c in range(6)])
        done = torch.cat([torch.zeros((1, 128, 32), dtype=torch.bool),
            torch.stack([torch.cat([endings[r, c] for r in range(4)], dim=0) for c in range(6)])])
        assert done[-1].any(-1).all(), 'Every source episode must be complete'
        # Chunk-level validity: the decision that ends an episode participates;
        # only decisions after a prior terminal/timeout are masked out.
        ended_before = torch.cumsum(done[:-1].any(-1).to(torch.int64), dim=0) > 0
        mask = (~ended_before)[..., None].expand_as(reward).clone()
        batch = dict(rewards=reward, dones=done, loss_mask=mask)
        result = calculate_adv_and_returns(task_type='embodied', adv_type='grpo',
            rewards=reward.clone(), dones=done.clone(), values=None, loss_mask=mask.clone(),
            loss_mask_sum=None, reward_type='chunk_level', num_action_chunks=32,
            gamma=.99, gae_lambda=.95, group_size=8)
        batch.update(result)
        reference = audit(batch, 'grpo')
        reference['rewards_already_contain_native_timeout_bootstrap'] = False
        assert reference['passed'], reference
        assert reference['all_tied_advantages_exact_zero']
        score = np.zeros(128, np.float64)
        reward_np = reward.numpy().astype(np.float64).sum(-1)
        done_np = done.numpy().any(-1)
        for chunk in range(5, -1, -1): score = score*(~done_np[chunk+1])+reward_np[chunk]
        for group in range(16):
            values = score[group*8:(group+1)*8]
            round_index, local_group = divmod(group, 4)
            report['groups'].append(dict(rank=rank, round_index=round_index, local_group=local_group,
                scene_seed=seeds[round_index][local_group*8], raw_scores=values.tolist(),
                tied=bool(np.std(values, ddof=1) == 0),
                positive_score_replicas=int((values > 0).sum())))
        fixtures.append(dict(rank=rank, **batch))
        report['ranks'].append(dict(rank=rank, trace_chunks=count, native_reference=reference))
        write(output/'RESULT.json', report)
        print('RAW_GROUP_RANK_COMPLETE', rank, flush=True)
    assert len(report['groups']) == 128 and len({row['scene_seed'] for row in report['groups']}) == 128
    # Small reward/mask fixture only; no images, model tensors or cached features.
    torch.save(fixtures, output/'raw-reward-fixtures.pt')
    report.update(status='passed_counterfactual_raw_reward_grpo_reduction',
        total_episode_groups=128, total_episodes=1024, recorded_decision_calls=6144,
        valid_chunk_decisions=sum(row['native_reference']['valid_native_decisions'] for row in report['ranks']),
        tied_groups=sum(row['tied'] for row in report['groups']),
        mixed_score_groups=sum(not row['tied'] for row in report['groups']),
        positive_score_episodes=sum(row['positive_score_replicas'] for row in report['groups']),
        worst_advantage_error=max(row['native_reference']['max_abs_error']['advantages'] for row in report['ranks']),
        fixture_sha256=sha(output/'raw-reward-fixtures.pt'), seconds=time.monotonic()-started)
    assert not torch.cuda.is_initialized()
    write(output/'RESULT.json', report)


if __name__ == '__main__':
    main()
