"""Native KV isolation, recurrent lifetime and exact zero-residual motor parity."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fluffy_bassoon.rl.pi05_feature_bridge import (
    Pi05FeatureBridge, clone_generator, native_kv_tokens, public_chunk_feedback,
    sample_prepared_with_bridge,
)


def inputs():
    key = torch.randn(2, 5, 1, 8, requires_grad=True)
    value = torch.randn(2, 5, 1, 8, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    return ((key, value),), mask, torch.randn(2, 5)


def bridge(kind):
    return Pi05FeatureBridge(core=kind, feedback_dim=5, kv_dim=16, suffix_dim=12,
                             action_dim=4, width=16, memory_length=4).eval()


@pytest.mark.parametrize("kind", ["persistent_ctm", "transformer_xl"])
def test_masked_frozen_kv_isolation_and_lane_reset(kind):
    module = bridge(kind)
    cache, mask, feedback = inputs()
    tokens = native_kv_tokens(cache, mask)
    assert tokens.shape == (2, 5, 16) and not tokens.requires_grad
    first, state = module.read_context(cache, mask, feedback)
    changed = tuple(tuple(value.detach().masked_fill(~mask[..., None, None], 10000) for value in pair)
                    for pair in cache)
    equivalent, _ = module.read_context(changed, mask, feedback)
    torch.testing.assert_close(first, equivalent)
    copied = {key: value.clone() for key, value in state.items()}
    reset, mixed = module.read_context(cache, mask, feedback, state,
                                       reset=torch.tensor([True, False]))
    fresh, fresh_state = module.read_context(cache, mask, feedback)
    carried, carried_state = module.read_context(cache, mask, feedback, state)
    torch.testing.assert_close(reset[0], fresh[0])
    torch.testing.assert_close(reset[1], carried[1])
    for key in mixed:
        torch.testing.assert_close(mixed[key][0], fresh_state[key][0])
        torch.testing.assert_close(mixed[key][1], carried_state[key][1])
        torch.testing.assert_close(state[key], copied[key])


@pytest.mark.parametrize("kind", ["persistent_ctm", "transformer_xl"])
def test_residual_zero_initialization_and_gradients_freeze_native_features(kind):
    module = bridge(kind)
    cache, mask, feedback = inputs()
    features, _ = module.read_context(cache, mask, feedback)
    velocity = torch.randn(2, 3, 4, requires_grad=True)
    suffix = torch.randn(2, 3, 12, requires_grad=True)
    actions = torch.randn(2, 3, 4, requires_grad=True)
    time = torch.ones(2)
    corrected = module.correct_velocity(velocity, suffix, actions, time, features)
    assert torch.equal(corrected, velocity)
    corrected.square().mean().backward()
    assert module.residual_output.weight.grad.abs().sum() > 0
    assert all(value.grad is None for pair in cache for value in pair)
    assert velocity.grad is None and suffix.grad is None and actions.grad is None
    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.residual_output.weight.normal_(std=.1)
    features, _ = module.read_context(cache, mask, feedback)
    corrected = module.correct_velocity(velocity, suffix, actions, time, features)
    assert (corrected - velocity).abs().max() <= .1 + 1e-7
    corrected.square().mean().backward()
    assert module.feedback_projection[0].weight.grad.abs().sum() > 0
    assert all(value.grad is None for pair in cache for value in pair)


class FakeMotor(nn.Module):
    action_horizon, action_dim = 3, 4

    def build_prefix_cache(self, observation):
        key = observation.state[:, None, None, :].expand(-1, 5, 1, -1)
        mask = torch.ones(2, 5, dtype=torch.bool)
        return None, mask, ((key, key + .25),)

    def run_suffix(self, observation, actions, time, cache, mask):
        return actions.repeat(1, 1, 3)

    def velocity_from_suffix(self, suffix):
        return suffix[..., :4] * .25 + .1


@pytest.mark.parametrize("kind", ["persistent_ctm", "transformer_xl"])
def test_zero_bridge_matches_every_native_euler_step_and_advances_memory_once(kind):
    motor, module = FakeMotor().eval(), bridge(kind)
    observation = SimpleNamespace(state=torch.randn(2, 8))
    noise, feedback = torch.randn(2, 3, 4), torch.randn(2, 5)
    reference, time, dt = noise, 1., -.25
    _, mask, cache = motor.build_prefix_cache(observation)
    while time >= -dt / 2:
        timestep = torch.full((2,), time)
        suffix = motor.run_suffix(observation, reference, timestep, cache, mask)
        reference = reference + dt * motor.velocity_from_suffix(suffix)
        time += dt
    audit = {}
    actions, state = sample_prepared_with_bridge(motor, module, observation, feedback, noise=noise, audit=audit)
    assert torch.equal(actions, reference)
    assert audit["kv_key_shape"] == [2, 5, 1, 8]
    assert audit["suffix_shapes"] == [[2, 3, 12]] * 4
    assert audit["recurrence_updates"] == 1
    _, once = module.read_context(cache, mask, feedback)
    for key in state:
        torch.testing.assert_close(state[key], once[key])
    if kind == "transformer_xl":
        assert state["length"].tolist() == [1, 1]
    with pytest.raises(ValueError, match="eval-mode"):
        sample_prepared_with_bridge(motor.train(), module, observation, feedback, noise=noise)


def test_cloned_noise_streams_advance_independently_and_reproduce_samples():
    original = torch.Generator().manual_seed(52)
    cloned = clone_generator(original)
    before = original.get_state().clone()
    expected = torch.randn(2, 3, generator=cloned)
    assert torch.equal(original.get_state(), before)
    actual = torch.randn(2, 3, generator=original)
    assert torch.equal(actual, expected)
    assert torch.equal(original.get_state(), cloned.get_state())


def test_public_feedback_has_only_declared_history_fields():
    previous = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    feedback = public_chunk_feedback(previous, previous_reward=1., attempt_end=True,
                                     attempt_index=2, chunk_index=8)
    assert feedback.shape == (2, 11)
    torch.testing.assert_close(feedback[:, :7], previous)
    torch.testing.assert_close(feedback[:, 7:], torch.tensor([[1., 1., .25, .5]]).expand(2, -1))
    assert torch.equal(public_chunk_feedback(torch.zeros(1, 7)), torch.zeros(1, 11))
    with pytest.raises(ValueError, match="public clocks"):
        public_chunk_feedback(previous, chunk_index=16)
    with pytest.raises(ValueError, match="previous actions"):
        public_chunk_feedback(torch.zeros(2, 8))
