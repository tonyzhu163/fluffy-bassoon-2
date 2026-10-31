"""Allocated-Linux qualification of ragged complete-trial PPO batches."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.ppo import generalized_advantages, update_ppo
from fluffy_bassoon.meta_rl.vla_rl2 import RL2Config, collect_rl2_trial, make_rl2_policy
from fluffy_bassoon.meta_rl import vla_rl2_batch as batch


def policy(seed=19):
    return make_rl2_policy(RL2Config(observation_dim=3, noise_dim=2, width=8,
        layers=1, heads=2, context_length=4), seed=seed)


def trial(current, length=1, state=1):
    def episode(sim, motor, choose, *, initial_state, scene_seed, chunk, bounded, feedback):
        rows = []
        for index in range(length):
            action = choose(dict(public=np.asarray([initial_state / 40, .2, index / 3], np.float32)))
            done = index == length - 1
            feedback(action, float(done), done)
            rows.append(dict(done=done, primitive_steps=1))
        return rows, dict(success=True, primitive_steps=length, initial_observation_sha256=f'{initial_state}:{scene_seed}')
    return collect_rl2_trial(current, SimpleNamespace(metadata=dict(settle_steps=10)), None,
        lambda observation: observation['public'], initial_state=state, scene_seed=123,
        attempts=4, chunk=1, episode_runner=episode)


def test_ragged_exact_replay_resets_every_memory_leaf_and_feedback():
    current = policy()
    rollouts = [trial(current, length)[0] for length in (1, 3, 2)]
    original_masks = [row['resets'].clone() for row in rollouts]
    merged, metrics = batch.replay_complete_batch(current, rollouts)
    assert [row['decisions'] for row in metrics['trial_boundaries']] == [4, 12, 8]
    assert all(metrics[key] == 0. for key in metrics if key.endswith('max_error'))
    starts = [row['start'] for row in metrics['trial_boundaries']]
    assert merged['resets'][:, 0].nonzero().flatten().tolist() == starts
    assert all(torch.equal(row['resets'], original) for row, original in zip(rollouts, original_masks))
    assert merged['action_mask'].all() and len(merged['rewards']) == 24
    calls, recorded = 0, []
    forward = current.core.forward
    def checked_forward(observation, state=None):
        nonlocal calls
        if calls in starts:
            if state is not None:
                initial = current.core.initial_state(1)
                assert state.keys() == initial.keys() == {'memory', 'length'}
                assert all(torch.equal(state[key], initial[key]) for key in state)
            assert not observation[:, 3:-1].any() and observation[0, -1] == 1
            recorded.append(calls)
        calls += 1
        return forward(observation, state)
    current.core.forward = checked_forward
    current.evaluate_sequence(merged['observations'], merged['raw_actions'], merged['resets'])
    assert recorded == starts


def test_gae_and_gradient_never_cross_a_trial_but_cross_episode_boundaries():
    current = policy()
    first, _ = trial(current, 2)
    second, _ = trial(current, 1)
    merged, _ = batch.concatenate_trials([first, second], observation_dim=3, noise_dim=2)
    actual = generalized_advantages(merged['rewards'], merged['values'], merged['trial_ends'])
    expected = [generalized_advantages(row['rewards'], row['values'], row['trial_ends']) for row in (first, second)]
    for index in range(2):
        torch.testing.assert_close(actual[index], torch.cat([row[index] for row in expected]), rtol=0, atol=0)
    tokens = merged['observations'].clone().requires_grad_()
    values = current.evaluate_sequence(tokens, merged['raw_actions'], merged['resets'])[2]
    values[-1].sum().backward()
    assert not tokens.grad[:len(first['rewards'])].any()
    assert tokens.grad[len(first['rewards'])].abs().sum() > 0
    # First trial's second episode still depends on earlier episode tokens.
    tokens = merged['observations'].clone().requires_grad_()
    current.evaluate_sequence(tokens, merged['raw_actions'], merged['resets'])[2][3].sum().backward()
    assert tokens.grad[0].abs().sum() > 0


def test_valid_token_loss_and_gradient_equal_sum_of_ragged_trial_losses():
    actual_policy = policy()
    trials = [trial(actual_policy, length)[0] for length in (1, 3, 2)]
    manual = copy.deepcopy(actual_policy)
    merged, _ = batch.replay_complete_batch(actual_policy, trials)
    advantages, returns = generalized_advantages(merged['rewards'], merged['values'], merged['trial_ends'])
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    torch.manual_seed(631)
    metrics = update_ppo(actual_policy, torch.optim.SGD(actual_policy.parameters(), lr=0), merged,
        epochs=1, minibatch_trials=1, bounded_entropy=True, max_grad_norm=1e12)
    torch.manual_seed(631)
    torch.randperm(1)  # Same existing optimizer's one-lane minibatch RNG operation.
    losses, offset = [], 0
    for row in trials:
        count = len(row['rewards'])
        log, entropy, value = manual.evaluate_sequence(row['observations'], row['raw_actions'], row['resets'],
            bounded_entropy=True)
        ratio = (log - row['log_probs']).exp()
        advantage = advantages[offset:offset + count]
        actor = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(.8, 1.2))
        target = returns[offset:offset + count]
        clipped = row['values'] + (value - row['values']).clamp(-.2, .2)
        critic = .5 * torch.maximum((value - target).square(), (clipped - target).square())
        losses.append((actor + .5 * critic - .01 * entropy).sum())
        offset += count
    loss = sum(losses) / offset
    loss.backward()
    assert metrics['optimizer_steps'] == 1 and metrics['active_action_samples'] == offset
    assert metrics['loss'] == pytest.approx(float(loss), rel=1e-6, abs=1e-7)
    for actual, expected in zip(actual_policy.parameters(), manual.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize('change', ['feedback', 'end', 'cut', 'padding', 'likelihood'])
def test_bad_or_modified_complete_trial_is_rejected(change):
    current = policy()
    data, _ = trial(current, 2)
    if change == 'feedback':
        data['observations'][0, 0, -2] = 1
    elif change == 'end':
        data['trial_ends'][1] = True
    elif change == 'cut':
        data = {key: value[:-1] for key, value in data.items()}
    elif change == 'padding':
        data['action_mask'] = torch.ones_like(data['resets'])
        data['action_mask'][0] = False
    else:
        data['log_probs'][0] += .01
    with pytest.raises((ValueError, RuntimeError)):
        batch.replay_complete_batch(current, [data])


class Runtime:
    def __init__(self):
        self.cursor, self.physical, self.calls, self.descriptors = 0, 0, 0, []
        self.fail_at, self.wall_after, self.wall = None, None, 0.

    def snapshot(self):
        return dict(cursor=self.cursor, sampler_adapter_rng=dict(saved='motor RNG stand-in'))

    def restore(self, saved):
        self.cursor = saved['cursor']
        assert saved['sampler_adapter_rng'] == dict(saved='motor RNG stand-in')

    def plan(self, sampler):
        return [dict(task_cursor=self.cursor + i, task_key=f'source:{(self.cursor + i) % 40:02d}',
            initial_state=int(sampler.integers(40)), scene_seed=int(sampler.integers(2**31)),
            length=1 + i % 2, maximum_actions=44 + 4 * (i % 2)) for i in range(14)]

    def collect(self, current, descriptor, trial_id, update, behavior_hash):
        assert descriptor['task_cursor'] == self.cursor
        assert trial_id == self.cursor + 1 and update == self.cursor // 14
        assert len(behavior_hash) == 64
        self.calls += 1
        if self.fail_at == self.cursor:
            self.physical += 1  # Actual failed RPC remains charged outside restored source RNG/cursor.
            torch.rand(3)
            raise RuntimeError('injected interrupted current trial')
        data, receipt = trial(current, descriptor['length'], descriptor['initial_state'])
        self.cursor += 1
        self.physical += receipt['primitive_steps'] + receipt['settling_steps']
        self.descriptors.append(descriptor)
        if self.wall_after == self.cursor:
            self.wall = 920.
        return data, dict(receipt, descriptor=descriptor, behavior_update=update,
            behavior_policy_sha256=behavior_hash)


def train(current, runtime, output, **kwargs):
    return batch.train_complete_batches(current, planner=runtime.plan, collector=runtime.collect,
        runtime_state=runtime.snapshot, restore_runtime=runtime.restore, physical_cost=lambda: runtime.physical,
        output=output, source_contract=dict(source_commit='f' * 40, source_only=True, source_data_from_targets=False),
        updates=1, max_total_actions=10_000, stop_unix=1e12, **kwargs)


def test_fourteen_complete_trials_use_two_global_optimizer_steps(tmp_path):
    runtime = Runtime()
    summary = train(policy(), runtime, tmp_path / 'complete')
    assert summary['updates'] == 1 and summary['complete_source_trials'] == 14
    assert summary['counters']['optimizer_steps'] == 2
    assert summary['counters']['update_forward_samples'] == summary['counters']['rollout_forward_samples'] * 2
    assert summary['batch_statistics'][0]['all_success_trials'] == 14
    assert summary['batch_statistics'][0]['fail_then_success_trials'] == 0
    assert summary['actual_tasks_with_complete_trials'] == 14


def test_partial_batch_wall_stop_recovers_without_recollecting_complete_trials(tmp_path, monkeypatch):
    expected, baseline = policy(), Runtime()
    train(expected, baseline, tmp_path / 'expected')
    interrupted = Runtime()
    interrupted.wall_after = 3
    monkeypatch.setattr(batch.time, 'time', lambda: interrupted.wall)
    summary = batch.train_complete_batches(policy(), planner=interrupted.plan, collector=interrupted.collect,
        runtime_state=interrupted.snapshot, restore_runtime=interrupted.restore,
        physical_cost=lambda: interrupted.physical, output=tmp_path / 'actual',
        source_contract=dict(source_commit='f' * 40, source_only=True, source_data_from_targets=False),
        updates=1, max_total_actions=10_000, stop_unix=1000.)
    assert summary['updates'] == 0 and summary['pending_complete_trials'] == 3
    assert not (tmp_path / 'actual/candidate.pt').exists()
    interrupted.wall, interrupted.wall_after = 0., None
    actual = policy()
    summary = train(actual, interrupted, tmp_path / 'actual', resume=True)
    assert interrupted.calls == baseline.calls == 14
    assert summary['physical_action_charge'] == baseline.physical
    assert interrupted.descriptors == baseline.descriptors
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)


def test_interrupted_current_trial_restores_rng_but_charges_failed_actions(tmp_path):
    expected, baseline = policy(), Runtime()
    train(expected, baseline, tmp_path / 'expected')
    interrupted = Runtime()
    interrupted.fail_at = 3
    with pytest.raises(RuntimeError, match='interrupted current trial'):
        train(policy(), interrupted, tmp_path / 'actual')
    interrupted.fail_at = None
    actual = policy()
    summary = train(actual, interrupted, tmp_path / 'actual', resume=True)
    assert interrupted.calls == baseline.calls + 1 and summary['physical_action_charge'] == baseline.physical + 1
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)


def test_optimizer_failure_keeps_ready_batch_recovery_before_any_sgd(tmp_path, monkeypatch):
    expected, baseline = policy(), Runtime()
    train(expected, baseline, tmp_path / 'expected')
    actual_runtime = Runtime()
    original = batch.update_ppo
    def failure(current, *args, **kwargs):
        with torch.no_grad():
            next(current.parameters()).add_(1.)
        raise RuntimeError('injected interrupted optimizer burst')
    monkeypatch.setattr(batch, 'update_ppo', failure)
    with pytest.raises(RuntimeError, match='optimizer burst'):
        train(policy(), actual_runtime, tmp_path / 'actual')
    assert list((tmp_path / 'actual').glob('unexpected-update-*.json'))
    saved = torch.load(tmp_path / 'actual/latest.pt', weights_only=False)
    assert saved['progress']['phase'] == 'ready_update'
    assert len(saved['progress']['pending_rollouts']) == 14 and saved['progress']['iteration'] == 0
    monkeypatch.setattr(batch, 'update_ppo', original)
    actual = policy()
    train(actual, actual_runtime, tmp_path / 'actual', resume=True)
    assert actual_runtime.calls == 14  # No repeated physical collection for a repeated optimizer burst.
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)


def test_no_collection_if_full_batch_worst_case_cannot_fit(tmp_path):
    runtime = Runtime()
    summary = batch.train_complete_batches(policy(), planner=runtime.plan, collector=runtime.collect,
        runtime_state=runtime.snapshot, restore_runtime=runtime.restore,
        physical_cost=lambda: runtime.physical, output=tmp_path / 'too-small',
        source_contract=dict(source_commit='f' * 40, source_only=True, source_data_from_targets=False),
        updates=1, max_total_actions=643, stop_unix=1e12)
    assert runtime.calls == 0 and runtime.physical == 0
    assert summary['stop_reason'] == 'insufficient_actions_for_another_complete_batch'


def test_retry_cost_keeps_partial_batch_instead_of_cutting_to_fit(tmp_path, monkeypatch):
    runtime = Runtime()
    runtime.wall_after = 3
    monkeypatch.setattr(batch.time, 'time', lambda: runtime.wall)
    kwargs = dict(planner=runtime.plan, collector=runtime.collect,
        runtime_state=runtime.snapshot, restore_runtime=runtime.restore,
        physical_cost=lambda: runtime.physical, output=tmp_path / 'partial',
        source_contract=dict(source_commit='f' * 40, source_only=True, source_data_from_targets=False),
        updates=1, max_total_actions=644, stop_unix=1000.)
    summary = batch.train_complete_batches(policy(), **kwargs)
    assert summary['pending_complete_trials'] == 3 and runtime.calls == 3
    runtime.wall, runtime.wall_after = 0., None
    runtime.physical += 1  # A requalification or failed RPC prefix consumed the reserved slack.
    summary = batch.train_complete_batches(policy(), resume=True, **kwargs)
    assert summary['stop_reason'] == 'retry_cost_prevents_completion_of_preserved_pending_batch'
    assert summary['pending_complete_trials'] == 3 and runtime.calls == 3
    saved = torch.load(tmp_path / 'partial/latest.pt', weights_only=False)
    assert len(saved['progress']['pending_rollouts']) == 3


def test_changed_behavior_during_collection_stops_before_optimization(tmp_path):
    runtime = Runtime()
    original = runtime.collect
    def invalid(current, *args):
        result = original(current, *args)
        with torch.no_grad():
            next(current.parameters()).add_(.1)
        return result
    runtime.collect = invalid
    with pytest.raises(RuntimeError, match='changed while collecting'):
        train(policy(), runtime, tmp_path / 'changed')
    saved = torch.load(tmp_path / 'changed/latest.pt', weights_only=False)
    assert saved['progress']['iteration'] == 0 and saved['progress']['completed_trials'] == 0
    assert list((tmp_path / 'changed').glob('unexpected-collection-*.json'))


def test_target_data_is_rejected_before_any_collection(tmp_path):
    runtime = Runtime()
    with pytest.raises(ValueError, match='source trials'):
        batch.train_complete_batches(policy(), planner=runtime.plan, collector=runtime.collect,
            runtime_state=runtime.snapshot, restore_runtime=runtime.restore,
            physical_cost=lambda: runtime.physical, output=tmp_path / 'leak',
            source_contract=dict(source_commit='f' * 40, source_only=True, source_data_from_targets=True),
            updates=1, max_total_actions=10_000, stop_unix=1e12)
    assert runtime.calls == 0 and not (tmp_path / 'leak').exists()


def test_success_failure_counts_are_observed_not_fabricated():
    receipts = [dict(attempts=[dict(success=char == 'S') for char in pattern])
        for pattern in ('SSSS', 'FSSS', 'SSSF', 'FFFF')]
    counts = batch.receipt_counts(receipts)
    assert counts['successes'] == 10 and counts['failures'] == 6
    assert counts['all_success_trials'] == 1 and counts['fail_then_success_trials'] == 1
