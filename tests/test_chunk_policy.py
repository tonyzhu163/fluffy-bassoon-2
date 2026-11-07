"""Decoded-action model density, complete-command inputs and reset isolation."""
from copy import deepcopy
import math

import pytest
import torch
from torch.distributions import Normal

from fluffy_bassoon.meta_rl.chunk_history import ChunkHistoryLayout
from fluffy_bassoon.meta_rl.chunk_policy import ChunkModelConfig, ChunkResidualPolicy, ChunkHistoryCritic


def model():
    cfg = ChunkModelConfig(ChunkHistoryLayout(3, 2, 2), width=8, layers=1, heads=2, context_length=4)
    return ChunkResidualPolicy(cfg, torch.tensor([[.2, .3], [.4, .5]]))


def test_decoded_density_matches_transformed_gaussian_reference():
    policy = model().double()
    for parameter in policy.actor.parameters():
        parameter.data.zero_()
    tokens = torch.zeros(2, policy.core.input_dim, dtype=torch.float64)
    target = torch.tensor([[[.02, -.03], [.08, -.05]]]*2, dtype=torch.float64)
    scaled = (target/policy.residual_radius).flatten(1)
    z = torch.atanh(scaled)
    expected = (Normal(0., policy.log_std.exp()).log_prob(z)
                -policy.residual_radius.flatten().log()-torch.log1p(-scaled.square())).sum(-1)
    actual, _ = policy.log_prob(tokens, target)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    (-actual.mean()).backward()
    assert policy.actor.weight.grad.isfinite().all() and policy.log_std.grad.isfinite().all()
    with pytest.raises(ValueError, match='audit coverage'):
        policy.log_prob(tokens, policy.residual_radius.expand_as(target))


def test_sampling_uses_owned_generator_and_fixed_decoded_bounds():
    policy = model()
    radius = policy.residual_radius.clone()
    tokens = torch.randn(3, policy.core.input_dim)
    a, logp, _ = policy.step(tokens, generator=torch.Generator().manual_seed(19))
    b, _, _ = policy.step(tokens, generator=torch.Generator().manual_seed(19))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert (a.abs() <= radius).all() and logp.isfinite().all()
    score, _ = policy.log_prob(tokens, a)
    torch.testing.assert_close(score, logp, rtol=1e-5, atol=1e-5)


def test_lane_reset_clears_both_cache_and_embedded_feedback():
    policy = model()
    tokens = torch.randn(2, policy.core.input_dim)
    _, _, state = policy.step(tokens, deterministic=True)
    reset = torch.tensor([True, False])
    actual, _, after = policy.step(tokens, state, reset=reset, deterministic=True)
    fresh = tokens[:1].clone(); fresh[:, policy.config.layout.current_dim:-1] = 0
    expected, _, empty = policy.step(fresh, deterministic=True)
    torch.testing.assert_close(actual[:1], expected, rtol=1e-6, atol=1e-6)
    assert after['length'].tolist() == [1, 2] and empty['length'].tolist() == [1]
    changed = tokens.clone(); changed[0, policy.config.layout.current_dim:-1] = 1e6
    repeated, _, _ = policy.step(changed, state, reset=reset, deterministic=True)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


def test_q_consumes_undelivered_tail_and_v_rejects_action_labels():
    policy = model()
    q = ChunkHistoryCritic(policy.config, action_value=True)
    v = ChunkHistoryCritic(policy.config, action_value=False)
    tokens = torch.zeros(1, policy.core.input_dim)
    action = torch.zeros(1, 2, 2, requires_grad=True)
    value, _ = q.step(tokens, action=action); value.sum().backward()
    assert action.grad[0, -1].abs().sum() > 0
    with pytest.raises(ValueError, match='must not consume'):
        v.step(tokens, action=action)


def test_features_normalize_without_reinterpreting_chunk_feedback():
    policy = model()
    policy.fit_feature_normalization(torch.tensor([[1., 2., 3.], [3., 6., 9.]]))
    torch.testing.assert_close(policy.feature_mean, torch.tensor([2., 4., 6.]))
    torch.testing.assert_close(policy.feature_std, torch.tensor([1., 2., 3.]))
    with pytest.raises(ValueError, match='decoded-chunk tokens'):
        policy.step(torch.zeros(1, 3+4+3))  # Legacy noise layout cannot pass.
    restored = model(); restored.load_state_dict(deepcopy(policy.state_dict()), strict=True)
    token = torch.randn(1, policy.core.input_dim)
    a, _, _ = policy.step(token, deterministic=True)
    b, _, _ = restored.step(token, deterministic=True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
