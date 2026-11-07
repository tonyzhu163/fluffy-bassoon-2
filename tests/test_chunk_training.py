"""Complete physical windows and independent decoded-student recovery."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.chunk_history import ChunkHistoryLayout
from fluffy_bassoon.meta_rl.chunk_policy import ChunkModelConfig
from fluffy_bassoon.meta_rl.chunk_training import (
    ChunkFitConfig, ChunkTrainer, load_chunk_policy, sample_chunk_windows)
from fluffy_bassoon.meta_rl.chunk_value import chunk_batch


def trial():
    return dict(features=np.arange(12, dtype=np.float32).reshape(6, 2)/10,
        bases=np.zeros((6, 2, 1), np.float32), issued=np.full((6, 2, 1), .2, np.float32),
        delivered_masks=np.tile(np.array([True, False]), (6, 1)),
        rewards=np.array([0, 1, 0, 0, 0, 1], np.float32),
        terminated=np.array([False, True, False, True, False, False]),
        truncated=np.array([False, False, False, False, False, True]),
        physics_seconds=np.full(6, .004, np.float32),
        episode_starts=np.array([True, False, True, False, True, False]))


def trainer(method='dit', *, value_updates=2, actor_updates=2):
    model = ChunkModelConfig(ChunkHistoryLayout(2, 2, 1), width=8, layers=1, heads=2, context_length=8)
    config = ChunkFitConfig(method=method, mode='static' if method == 'static_behavior_cloning' else 'full',
        seed=27, batch_size=2, max_episodes=2, value_updates=value_updates, actor_updates=actor_updates)
    statistics = dict(feature_mean=torch.tensor([.1, .2]), feature_std=torch.tensor([.5, .7]),
                      residual_radius=torch.ones(2, 1))
    return ChunkTrainer(model, config, statistics, dict(eligible_methods=[method], discount=.99,
        teacher_policy_versions=[0], chronological_learning_histories=False))


def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_tree_equal(left, right)
    else:
        assert a == b


def test_windows_are_complete_physical_episodes_and_start_with_empty_feedback():
    data = trial()
    sampled = sample_chunk_windows([data], np.random.default_rng(7), batch_size=30,
        max_episodes=3, context_length=4)
    model = trainer().learner.actor
    for window in sampled:
        assert len(window['features']) in (2, 4)
        assert window['episode_starts'][0] and (window['terminated'][-1] or window['truncated'][-1])
        batch = chunk_batch([window], model)
        begin = model.config.layout.current_dim
        assert not batch['tokens'][0, 0, begin:-1].any()
        assert (batch['actions'][..., -1, :] == .2).all()
    with pytest.raises(ValueError, match='exceeds'):
        sample_chunk_windows([data], np.random.default_rng(7), batch_size=1, max_episodes=2, context_length=1)


@pytest.mark.parametrize('completed', [1, 2, 3])
def test_independent_resume_reproduces_next_update_in_both_stages(tmp_path, completed):
    original = trainer()
    data = [trial()]
    for _ in range(completed):
        original.step(data, execution_batch_size=1)
    path = tmp_path/'latest.pt'
    checksum = original.save(path)
    contract = deepcopy(original.contract)
    draws = random.random(), np.random.random(), torch.rand(3)
    expected = original.step(data, execution_batch_size=1)
    restored = ChunkTrainer.restore(path, checksum, expected_contract=contract)
    actual_draws = random.random(), np.random.random(), torch.rand(3)
    assert_tree_equal(draws, actual_draws)
    actual = restored.step(data, execution_batch_size=1)
    assert_tree_equal(actual, expected)
    assert_tree_equal(restored.learner.state_dict(), original.learner.state_dict())
    for phase in ('value', 'actor'):
        assert_tree_equal(restored.optimizers[phase].state_dict(), original.optimizers[phase].state_dict())
        assert_tree_equal(restored.samplers[phase].bit_generator.state, original.samplers[phase].bit_generator.state)
    assert restored.valid_positions == original.valid_positions


def test_actor_sampling_is_matched_across_value_objectives_and_static_control():
    value = trainer('ic_iql')
    control = trainer('static_behavior_cloning', value_updates=0)
    before = deepcopy(value.samplers['actor'].bit_generator.state)
    value.step([trial()]);value.step([trial()])
    assert_tree_equal(before, value.samplers['actor'].bit_generator.state)
    assert_tree_equal(value.samplers['actor'].bit_generator.state, control.samplers['actor'].bit_generator.state)


def test_actor_export_preserves_density_statistics_and_contains_no_critic(tmp_path):
    fitted = trainer('static_behavior_cloning', value_updates=0, actor_updates=1)
    fitted.step([trial()])
    path = tmp_path/'policy.pt'
    checksum = fitted.export_policy(path)
    policy, exported = load_chunk_policy(path, checksum, expected_contract=fitted.contract)
    assert set(exported) == {'format', 'contract', 'updates', 'policy'}
    assert_tree_equal(policy.state_dict(), fitted.learner.actor.state_dict())
    tokens = chunk_batch([trial()], policy, mode='static')['tokens'][0]
    left, _, _ = policy.step(tokens, deterministic=True)
    right, _, _ = fitted.learner.actor.step(tokens, deterministic=True)
    assert torch.equal(left, right)
    assert not policy.training and all(not p.requires_grad for p in policy.parameters())
    with pytest.raises(RuntimeError, match='budget'):
        fitted.step([trial()])


def test_failed_update_cannot_replace_last_complete_checkpoint(tmp_path):
    fitted = trainer('static_behavior_cloning', value_updates=0)
    path = tmp_path/'latest.pt'
    checksum = fitted.save(path)
    bad = trial();bad['issued'][:] = 2.
    with pytest.raises(ValueError, match='bounds'):
        fitted.step([bad])
    with pytest.raises(RuntimeError, match='failed update'):
        fitted.save(path)
    restored = ChunkTrainer.restore(path, checksum, expected_contract=fitted.contract)
    restored.step([trial()])


def test_validation_keeps_training_streams_and_fixed_statistics():
    fitted = trainer()
    states = {key:deepcopy(value.bit_generator.state) for key,value in fitted.samplers.items()}
    before = {key:value.clone() for key,value in fitted.learner.state_dict().items()}
    a = fitted.validate([trial()], phase='value', batches=2)
    b = fitted.validate([trial()], phase='value', batches=2)
    assert a == b
    assert_tree_equal(before, fitted.learner.state_dict())
    for key, sampler in fitted.samplers.items():
        assert_tree_equal(states[key], sampler.bit_generator.state)
    with torch.no_grad():
        fitted.learner.v.feature_mean.add_(1)
    with pytest.raises(RuntimeError, match='statistics changed'):
        fitted.step([trial()])
