"""Validate forensic measurements against the actual PPO update and known cache errors."""

from copy import deepcopy

import pytest
import torch

from fluffy_bassoon.meta_rl.environments import HiddenGainReach
from fluffy_bassoon.meta_rl.metaworld_ppo_diagnose import (
    critic_diagnostics, gradient_components, replay_difference, replay_drift)
from fluffy_bassoon.meta_rl.ppo import GRUCore, RecurrentActorCritic, collect_trials, update_ppo
from gymnasium.spaces.utils import flatdim


def setup_batch():
    torch.set_num_threads(1)
    torch.manual_seed(82)
    envs = [HiddenGainReach(attempts=2, horizon=3) for _ in range(2)]
    policy = RecurrentActorCritic(GRUCore(flatdim(envs[0].observation_space), width=8), envs[0].action_space).double()
    rollout = collect_trials(policy, envs, [11, 12], replay_chunk_length=3)
    optimization = dict(epochs=1, minibatch_trials=2, clip=.2, gamma=.99, gae_lambda=.95,
                        entropy_coef=0., value_coef=.5, max_grad_norm=.5,
                        normalize_advantages=True, replay_chunk_length=3)
    for env in envs:
        env.close()
    return policy, rollout, optimization


def test_component_gradients_sum_to_production_ppo_gradient():
    policy, rollout, optimization = setup_batch()
    report = gradient_components(policy, rollout, optimization, all_windows=True)
    actual = update_ppo(policy, torch.optim.Adam(policy.parameters(), lr=0.), rollout, **optimization)
    assert actual['optimizer_steps'] == 1
    assert report['groups']['all']['joint_norm'] == pytest.approx(actual['gradient_norm'], rel=1e-8)
    assert report['actor_loss'] + report['weighted_critic_loss'] == pytest.approx(actual['loss'], abs=1e-10)
    assert report['groups']['actor_head']['weighted_critic_norm'] == 0
    assert report['groups']['critic_head']['actor_norm'] == 0
    assert report['groups']['core']['actor_norm'] > 0
    assert report['groups']['core']['weighted_critic_norm'] > 0


def test_frozen_replay_drift_detects_a_corrupted_boundary_without_weight_changes():
    policy, rollout, _ = setup_batch()
    frozen = replay_drift(policy, rollout, 3)
    assert frozen['log_probability']['maximum_absolute'] < 1e-10
    assert frozen['value']['maximum_absolute'] < 1e-10
    changed = deepcopy(rollout)
    changed['replay_states'][1] += 2
    corrupted = replay_drift(policy, changed, 3)
    assert corrupted['log_probability']['maximum_absolute'] > 1e-3
    assert corrupted['value']['maximum_absolute'] > 1e-3


def test_gradient_diagnostic_preserves_learner_rng_weights_and_grad_buffers():
    policy, rollout, optimization = setup_batch()
    weights = deepcopy(policy.state_dict())
    rng = torch.get_rng_state().clone()
    gradient_components(policy, rollout, optimization)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    assert all(p.grad is None for p in policy.parameters())


def test_value_clip_measurement_identifies_a_flat_loss_despite_better_predictions():
    old, target = torch.zeros(3), torch.full((3,), 10.)
    value = torch.full((3,), .3, requires_grad=True)
    report = critic_diagnostics(value, old, target, .2)
    assert report['new_value_mse'] < report['old_value_mse']
    assert report['flat_clipped_loss_fraction'] == 1.
    clipped = old + (value-old).clamp(-.2, .2)
    torch.maximum((value-target).square(), (clipped-target).square()).mean().backward()
    torch.testing.assert_close(value.grad, torch.zeros(3))


def test_mixed_replay_tolerance_accepts_small_fp32_roundoff_and_rejects_real_drift():
    expected = torch.tensor([100., 1., .1])
    ulp = torch.nextafter(expected, torch.full_like(expected, float('inf')))-expected
    assert replay_difference(expected+8*ulp, expected)['maximum_scaled_error'] < 1
    assert replay_difference(expected+.01, expected)['maximum_scaled_error'] > 1
    assert replay_difference(torch.tensor([3e-5]), torch.zeros(1))['maximum_scaled_error'] > 1
