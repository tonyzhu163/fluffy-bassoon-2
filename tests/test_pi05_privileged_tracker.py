"""Privileged targets must traverse the real bounded four-Euler sampler."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fluffy_bassoon.meta_rl.bridge_distillation import corrected_action_label
from fluffy_bassoon.meta_rl.pi05_privileged_tracker import PrivilegedTargetTracker
from fluffy_bassoon.rl.pi05_bridge_policy import StochasticPi05BridgePolicy
from fluffy_bassoon.rl.pi05_feature_bridge import Pi05FeatureBridge


class ToyMotor(nn.Module):
    action_horizon, action_dim = 8, 8

    def __init__(self):
        super().__init__()
        self.prefixes, self.suffixes = 0, 0

    def build_prefix_cache(self, observation):
        self.prefixes += 1
        key = observation.state[:, None, None].expand(-1, 3, 1, -1)
        return None, torch.ones(key.shape[:2], dtype=torch.bool), ((key, key + .1),)

    def run_suffix(self, observation, actions, time, cache, mask):
        self.suffixes += 1
        return actions

    def velocity_from_suffix(self, suffix):
        return .2 * suffix

    def decode_actions(self, actions, state):
        # The target must rotate decoded XY, rather than normalized XY.
        return actions[:, :5, :7] * actions.new_tensor([.2, .7, 1., 1., 1., 1., 1.]) + .1

    def sample_actions(self, *args, **kwargs):
        raise AssertionError("Prepared tracker must not call a preprocessing sampler")


def tracker(gain=1.):
    bridge = Pi05FeatureBridge(core="transformer_xl", kv_dim=8, suffix_dim=8,
        action_dim=8, width=8, memory_length=4, residual_scale=1.)
    return PrivilegedTargetTracker(bridge, gain=gain)


def batch():
    return ToyMotor().eval(), SimpleNamespace(state=torch.zeros(2, 4)), torch.zeros(2, 11)


def native_endpoint(noise):
    result = noise.clone()
    for _ in range(4):
        result = result - .25 * (.2 * result)
    return result


def test_tracker_uses_super_path_same_noise_correct_target_and_bounded_means(monkeypatch):
    policy, (motor, observation, feedback) = tracker(), batch()
    policy.select_theta([-45., 45.])
    noise = torch.full((2, 8, 8), 5.)
    original = StochasticPi05BridgePolicy.sample_chunk
    calls = []
    def observed_super(self, model, prepared, public_feedback, memory=None, **kwargs):
        calls.append(dict(kwargs))
        return original(self, model, prepared, public_feedback, memory, **kwargs)
    monkeypatch.setattr(StochasticPi05BridgePolicy, "sample_chunk", observed_super)
    actions, memory, record = policy.sample_chunk(motor, observation, feedback, noise=noise)
    assert len(calls) == 1 and torch.equal(calls[0]["noise"], noise)
    assert calls[0]["num_steps"] == 4 and calls[0]["deterministic"] is True
    assert motor.prefixes == 2 and motor.suffixes == 8
    native = native_endpoint(noise)
    target = corrected_action_label(motor, observation.state, native, [-45., 45.])
    assert torch.equal(target[:, 5:], native[:, 5:])
    for index in range(4):
        x, velocity, time = (record[key][index] for key in ("noisy_actions", "native_velocity", "timesteps"))
        expected = (((x - target) / time[:, None, None] - velocity)[..., :7]).clamp(-1, 1)
        torch.testing.assert_close(record["behavior_means"][index], expected, rtol=0, atol=0)
    assert record["behavior_means"].abs().max() == 1
    assert not torch.equal(actions, target)  # Saturation prevents direct target execution.
    torch.testing.assert_close(actions[..., 7:], native[..., 7:], rtol=0, atol=0)
    assert memory and torch.equal(record["feedback"], feedback)
    assert record["privileged_tracker"]["extra_native_prefix_queries"] == 2
    assert record["privileged_tracker"]["extra_native_suffix_queries"] == 8
    assert record["privileged_tracker"]["actual_executed_action_steps"] == 5
    assert not record["privileged_tracker"]["training_eligible"]
    assert policy._target is None and policy._zero_theta is None
    assert torch.equal(noise, torch.full_like(noise, 5.))
    assert not any(parameter.requires_grad for parameter in policy.parameters())


def test_zero_theta_exactly_preserves_nonlinear_native_path_without_extra_oracle_queries():
    policy, (motor, observation, feedback) = tracker(), batch()
    policy.select_theta([0., 0.])
    noise = torch.randn(2, 8, 8)
    actions, _, record = policy.sample_chunk(motor, observation, feedback, noise=noise)
    assert torch.equal(actions, native_endpoint(noise))
    assert record["behavior_means"].eq(0).all()
    assert motor.prefixes == 1 and motor.suffixes == 4
    assert record["privileged_tracker"]["extra_native_prefix_queries"] == 0
    assert record["privileged_tracker"]["extra_native_suffix_queries"] == 0


def test_mixed_theta_preserves_zero_lane_and_counts_full_extra_batch():
    policy, (motor, observation, feedback) = tracker(gain=.25), batch()
    policy.select_theta([0., 45.])
    noise = torch.ones(2, 8, 8)
    actions, _, record = policy.sample_chunk(motor, observation, feedback, noise=noise)
    assert torch.equal(actions[0], native_endpoint(noise)[0])
    assert record["behavior_means"][:, 0].eq(0).all()
    assert record["privileged_tracker"]["gain"] == .25
    assert record["privileged_tracker"]["extra_native_prefix_queries"] == 2
    assert record["privileged_tracker"]["extra_native_suffix_queries"] == 8


def test_tracker_rejects_training_missing_theta_and_implicit_noise():
    policy, (motor, observation, feedback) = tracker(), batch()
    with pytest.raises(ValueError, match="theta selection"):
        policy.sample_chunk(motor, observation, feedback, noise=torch.ones(2, 8, 8))
    policy.select_theta([45., 45.])
    with pytest.raises(ValueError, match="checkpointed generator"):
        policy.sample_chunk(motor, observation, feedback)
    with pytest.raises(ValueError, match="deterministic"):
        policy.sample_chunk(motor, observation, feedback, deterministic=False)
    with pytest.raises(ValueError, match="not student PPO"):
        policy.replay_chunk({})
    for gain in (0., -1., 1.01, float("nan"), True):
        with pytest.raises(ValueError):
            tracker(gain)
    with pytest.raises(ValueError, match="finite private"):
        policy.select_theta([float("inf")])


def test_explicit_generator_is_consumed_once_and_gain_contract_is_guarded():
    policy, (motor, observation, feedback) = tracker(), batch()
    policy.select_theta([45., 45.])
    generator = torch.Generator().manual_seed(49)
    expected = torch.Generator().set_state(generator.get_state())
    noise = torch.randn(2, 8, 8, generator=expected)
    first, _, record = policy.sample_chunk(motor, observation, feedback, rng=generator)
    assert torch.equal(record["noisy_actions"][0], noise)
    assert torch.equal(generator.get_state(), expected.get_state())
    second, _, _ = policy.sample_chunk(motor, observation, feedback, noise=noise)
    assert torch.equal(first, second)
    with pytest.raises(RuntimeError, match="gain differs"):
        tracker(gain=.5).load_state_dict(policy.state_dict())
