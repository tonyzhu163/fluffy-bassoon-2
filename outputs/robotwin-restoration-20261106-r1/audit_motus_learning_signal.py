"""CPU-only diagnostic of the actual first PPO batch; no new policy updates."""
import gc
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from motus_learner_reference import audit

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def describe(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    assert x.size and np.isfinite(x).all()
    return dict(n=int(x.size), mean=float(x.mean()), std=float(x.std()),
                minimum=float(x.min()), median=float(np.median(x)), maximum=float(x.max()))


def gae(reward, value, done):
    carry = np.zeros(reward.shape[1], np.float64)
    advantages = np.empty_like(reward)
    for t in range(len(reward)-1, -1, -1):
        continuation = ~done[t+1]
        carry = reward[t] + .99*value[t+1]*continuation - value[t] + .99*.95*continuation*carry
        advantages[t] = carry
    return advantages


def main():
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    started = time.monotonic()
    output = BASE/'motus-learningaudit02'; output.mkdir()
    raw_path = BASE/'grpo-rawgroupaudit02/raw-reward-fixtures.pt'
    raw_report = json.loads((raw_path.parent/'RESULT.json').read_bytes())
    assert raw_report['status'] == 'passed_counterfactual_raw_reward_grpo_reduction'
    assert sha(raw_path) == raw_report['fixture_sha256']
    raw_rows = torch.load(raw_path, map_location='cpu', weights_only=False)
    report = dict(status='checking', source_commit=os.environ['TASK_SOURCE'], gpu_seconds=0,
        optimizer_updates=0, simulator_actions=0, model_calls=0,
        data_policy='optimized03 policy0 actual pre-SGD PPO batches',
        counterfactual_raw_gae_is_diagnostic_only=True, ranks=[], files={}, logged_metrics={})
    log = BASE/'motusoptimized03-signals/train.log'
    report['files'][str(log)] = sha(log)
    pattern = r'((?:actor|critic)/[A-Za-z0-9_/]+)\s*=\s*(-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)'
    for key, value in re.findall(pattern, log.read_text(errors='replace'), re.I):
        report['logged_metrics'][key] = float(value)
    report['logged_metrics_scope'] = 'Native aggregate over first collection and five epochs; not a per-update time series or final-checkpoint critic evaluation'
    fixtures = []
    used_environment_ranks = set()
    write(output/'RESULT.json', report)
    for rank in range(8):
        receipt_path = BASE/f'motusoptimized03/state/training-rollouts/iteration-0001-policy-0000-rank-{rank:02d}.json'
        receipt = json.loads(receipt_path.read_bytes())
        report['files'][str(receipt_path)] = sha(receipt_path)
        path = Path(receipt['path']); compressed = path.read_bytes()
        digest = hashlib.sha256(compressed).hexdigest()
        assert digest == receipt['sha256'] and len(compressed) == receipt['bytes']
        report['files'][str(path)] = digest
        with io.BytesIO(gzip.decompress(compressed)) as handle:
            saved = torch.load(handle, map_location='cpu', weights_only=False)
        del compressed
        assert saved['rank'] == rank and saved['optimizer_steps_before'] == 0
        batch = saved['rollout_batch']
        reference = audit(batch, 'ppo'); assert reference['passed'], reference
        done = batch['dones'].bool().any(-1).numpy()
        mask = batch['loss_mask'].bool().any(-1).numpy()
        # Native actors pop whole trajectories from a shared channel; actor
        # and environment rank numbers are not identity. Use boundaries only
        # to shortlist, then verify every command and all public observations.
        candidates = [row for row in raw_rows
            if np.array_equal(done, row['dones'].bool().any(-1).numpy())
            and np.array_equal(mask, row['loss_mask'].bool().any(-1).numpy())]
        assert len(candidates) == 1, [row['rank'] for row in candidates]
        raw = candidates[0]; environment_rank = raw['rank']
        assert environment_rank not in used_environment_ranks
        used_environment_ranks.add(environment_rank)
        trace_paths = sorted(Path(name) for name in raw_report['files']
            if '/traces/' in name and f'/rank-{environment_rank:02d}/' in name)
        assert len(trace_paths) == 24
        for ordinal, trace_path in enumerate(trace_paths):
            compressed = trace_path.read_bytes()
            assert hashlib.sha256(compressed).hexdigest() == raw_report['files'][str(trace_path)]
            with io.BytesIO(gzip.decompress(compressed)) as handle:
                trace = torch.load(handle, map_location='cpu', weights_only=False)
            del compressed
            round_index, chunk = divmod(ordinal, 6)
            assert (trace['round_index'], trace['chunk_index']) == (round_index, chunk)
            lanes = slice(round_index*32, (round_index+1)*32)
            assert np.array_equal(batch['actions'][chunk, lanes].numpy().reshape(32,32,14),
                                  np.asarray(trace['proposed_motor_actions']))
            for actor_key, trace_key in (('observation/state','states'),
                    ('observation/image','main_images'), ('observation/wrist_image','wrist_images')):
                assert np.array_equal(batch['forward_inputs'][actor_key][chunk, lanes].numpy(),
                                      np.asarray(trace['current_observation'][trace_key])), actor_key
            del trace
        def arr(key): return batch[key].numpy().astype(np.float64)
        rewards = arr('rewards').sum(-1)
        raw_rewards = raw['rewards'].numpy().astype(np.float64).sum(-1)
        values = arr('prev_values')[..., 0]
        returns = arr('returns')[..., 0]
        advantages = arr('advantages')[..., 0]
        bootstrap = rewards - raw_rewards
        trunc = batch['truncations'].bool().any(-1).numpy()[1:]
        term = batch['terminations'].bool().any(-1).numpy()[1:]
        raw_gae = gae(raw_rewards, values, done)
        normalized_raw = (raw_gae - raw_gae[mask].mean())/(raw_gae[mask].std(ddof=1)+1e-5)
        raw_mc = np.empty_like(raw_rewards); carry = np.zeros(raw_rewards.shape[1])
        for t in range(len(rewards)-1, -1, -1):
            carry = raw_rewards[t]+.99*(~done[t+1])*carry
            raw_mc[t] = carry
        score = np.zeros(rewards.shape[1])
        for t in range(len(rewards)-1, -1, -1): score = score*(~done[t+1])+raw_rewards[t]
        positive = np.broadcast_to(score[None, :] > 0, mask.shape) & mask
        zero = np.broadcast_to(score[None, :] == 0, mask.shape) & mask
        var = np.var(raw_mc[mask])
        row = dict(rank=rank, environment_rank=environment_rank,
            all24_command_and_public_observation_batches_exact=True,
            native_gae_reference=reference,
            raw_episode_scores=describe(score), positive_score_episodes=int((score>0).sum()),
            raw_rewards=describe(raw_rewards[mask]), native_rewards_with_bootstrap=describe(rewards[mask]),
            critic_values_before_sgd=describe(values[:-1][mask]), native_gae_return_targets=describe(returns[mask]),
            native_normalized_advantages=describe(advantages[mask]),
            timeout_bootstrap_added=describe(bootstrap[mask]),
            nonzero_bootstrap_decisions=int((np.abs(bootstrap[mask])>1e-7).sum()),
            bootstrap_outside_any_timeout_max_abs=float(np.abs(bootstrap[~trunc]).max(initial=0)),
            bootstrap_minus_gamma_next_value_at_timeout_max_abs=float(np.abs(bootstrap-.99*values[1:]*trunc).max()),
            normalized_advantage_sign_changes_without_bootstrap=int(((advantages>0)!=(normalized_raw>0))[mask].sum()),
            valid_decisions=int(mask.sum()), terminated_episodes=int(term.any(0).sum()),
            timeout_episodes=int(trunc.any(0).sum()),
            initial_value_explained_variance_vs_raw_mc=float(1-np.var(raw_mc[mask]-values[:-1][mask])/var) if var>0 else None,
            positive_episode_advantages=describe(advantages[positive]) if positive.any() else None,
            zero_score_episode_advantages=describe(advantages[zero]) if zero.any() else None)
        fixtures.append(dict(rank=rank, rewards=batch['rewards'], raw_rewards=raw['rewards'],
            dones=batch['dones'], loss_mask=batch['loss_mask'], prev_values=batch['prev_values'],
            returns=batch['returns'], advantages=batch['advantages'],
            terminations=batch['terminations'], truncations=batch['truncations']))
        report['ranks'].append(row); write(output/'RESULT.json', report)
        print('LEARNING_SIGNAL_RANK_COMPLETE', rank, flush=True)
        del batch, saved; gc.collect()
    assert used_environment_ranks == set(range(8))
    torch.save(fixtures, output/'learner-signal-fixtures.pt')
    report.update(status='completed_actual_batch_learning_signal_audit', seconds=time.monotonic()-started,
        fixture_sha256=sha(output/'learner-signal-fixtures.pt'),
        total_positive_score_episodes=sum(r['positive_score_episodes'] for r in report['ranks']),
        total_valid_decisions=sum(r['valid_decisions'] for r in report['ranks']),
        total_advantage_sign_changes_without_bootstrap=sum(r['normalized_advantage_sign_changes_without_bootstrap'] for r in report['ranks']),
        root_cause_established=False, policy_improvement_claim=False)
    assert not torch.cuda.is_initialized()
    write(output/'RESULT.json', report)


if __name__ == '__main__': main()
