"""Exact joint flow likelihood, active-channel support and recurrent PPO replay."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from fluffy_bassoon.rl.pi05_bridge_policy import StochasticPi05BridgePolicy, record_bytes
from fluffy_bassoon.rl.pi05_feature_bridge import Pi05FeatureBridge, sample_prepared_with_bridge


class FakeMotor(nn.Module):
    action_horizon, action_dim = 3, 10

    def build_prefix_cache(self, observation):
        key = observation.state[:, None, None, :].expand(-1, 5, 1, -1)
        mask = torch.ones(observation.state.shape[0], 5, dtype=torch.bool)
        return None, mask, ((key, key + .25),)

    def run_suffix(self, observation, actions, time, cache, mask):
        return torch.cat((actions, actions[..., :2]), dim=-1)

    def velocity_from_suffix(self, suffix):
        return suffix[..., :10] * .25 + .1


def make_policy(kind="transformer_xl", sigma=.05, scale=1.):
    bridge = Pi05FeatureBridge(core=kind, feedback_dim=5, kv_dim=16, suffix_dim=12,
                               action_dim=10, width=16, memory_length=4, residual_scale=scale)
    return StochasticPi05BridgePolicy(bridge, noise_scale=sigma).eval()


@pytest.mark.parametrize("kind", ["persistent_ctm", "transformer_xl"])
def test_joint_gaussian_density_is_summed_and_replays_exactly(kind):
    torch.manual_seed(2)
    policy, motor = make_policy(kind), FakeMotor().eval()
    observation = SimpleNamespace(state=torch.randn(2, 8))
    feedback = torch.randn(2, 5)
    actions, state, record = policy.sample_chunk(motor, observation, feedback,
                                                rng=torch.Generator().manual_seed(91))
    direct = Normal(record["behavior_means"], .05).log_prob(record["sampled_residuals"]).sum((-1, -2)).sum(0)
    assert torch.equal(direct, record["old_log_prob"])
    log_prob, replay_state, diagnostics = policy.replay_chunk(record)
    torch.testing.assert_close(log_prob, direct, rtol=0, atol=0)
    torch.testing.assert_close(diagnostics["value"], record["old_value"], rtol=0, atol=0)
    for key in state:
        torch.testing.assert_close(replay_state[key], state[key], rtol=0, atol=0)
    assert diagnostics["recurrence_updates"] == 1
    assert record["gaussian_scalars_per_lane"] == diagnostics["gaussian_scalars_per_lane"] == 4 * 3 * 7
    assert record["storage_bytes"] == record_bytes(record)
    assert all(value.device.type == "cpu" and not value.requires_grad for value in record.values()
               if isinstance(value, torch.Tensor))
    # Reconstruct every behavior transition, including deterministic native padded channels.
    for step in range(4):
        velocity = record["native_velocity"][step].clone()
        velocity[..., :7] += record["sampled_residuals"][step]
        expected = record["noisy_actions"][step] - .25 * velocity
        actual = record["noisy_actions"][step + 1] if step < 3 else actions
        assert torch.equal(actual, expected)
        assert torch.equal(velocity[..., 7:], record["native_velocity"][step, ..., 7:])


@pytest.mark.parametrize("kind", ["persistent_ctm", "transformer_xl"])
def test_deterministic_zero_mean_is_native_and_cannot_be_used_as_ppo_behavior(kind):
    policy, motor = make_policy(kind), FakeMotor().eval()
    observation = SimpleNamespace(state=torch.randn(2, 8))
    noise, feedback = torch.randn(2, 3, 10), torch.randn(2, 5)
    native, _ = sample_prepared_with_bridge(motor, policy.bridge, observation, feedback, noise=noise)
    deterministic, _, record = policy.sample_chunk(motor, observation, feedback, noise=noise, deterministic=True)
    assert torch.equal(native, deterministic)
    assert record["sampled_residuals"].abs().max() == 0
    with pytest.raises(ValueError, match="stochastic behavior"):
        policy.replay_chunk(record)


@pytest.mark.parametrize("kind", ["persistent_ctm", "transformer_xl"])
def test_chronological_replay_has_gradients_and_update_changes_the_policy(kind):
    torch.manual_seed(18)
    policy, motor = make_policy(kind), FakeMotor().eval()
    with torch.no_grad():
        policy.bridge.residual_output.weight.normal_(std=.03)
        policy.critic.weight.normal_(std=.03)
    observation = SimpleNamespace(state=torch.randn(2, 8, requires_grad=True))
    memory, records = None, []
    generator = torch.Generator().manual_seed(83)
    for index in range(3):
        feedback = torch.randn(2, 5)
        _, memory, record = policy.sample_chunk(motor, observation, feedback, memory, rng=generator)
        records.append(record)
    memory, replayed = None, []
    features = []
    for record in records:
        log_prob, memory, diagnostics = policy.replay_chunk(record, memory)
        torch.testing.assert_close(log_prob, record["old_log_prob"], rtol=0, atol=0)
        diagnostics["features"].retain_grad()
        features.append(diagnostics["features"])
        replayed.append((log_prob, diagnostics["value"]))
    old_log_probs = torch.stack([record["old_log_prob"] for record in records])
    current = torch.stack([pair[0] for pair in replayed])
    values = torch.stack([pair[1] for pair in replayed])
    advantages = torch.tensor([[1., -.5], [.2, -.8], [.7, -.1]])
    ratio = (current - old_log_probs).exp()
    loss = -(ratio * advantages).mean() + .5 * (values - advantages).square().mean()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    loss.backward()
    assert policy.bridge.residual_output.weight.grad.abs().sum() > 0
    assert policy.bridge.feedback_projection[0].weight.grad.abs().sum() > 0
    assert policy.critic.weight.grad.abs().sum() > 0
    assert observation.state.grad is None
    before = policy.bridge.residual_output.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, policy.bridge.residual_output.weight)
    after, _, _ = policy.replay_chunk(records[0])
    assert not torch.equal(after, records[0]["old_log_prob"])


def test_saved_policy_config_rejects_changed_sigma_and_support():
    original = make_policy()
    restored = make_policy()
    restored.load_state_dict(original.state_dict())
    assert restored.config() == original.config()
    with pytest.raises(RuntimeError, match="noise, active dimensions or residual scale"):
        make_policy(sigma=.1).load_state_dict(original.state_dict())
    with pytest.raises(RuntimeError, match="noise, active dimensions or residual scale"):
        make_policy(scale=.5).load_state_dict(original.state_dict())


@pytest.mark.parametrize("kind", ["persistent_ctm", "transformer_xl"])
def test_later_chunk_likelihood_backpropagates_through_earlier_context(kind):
    policy, motor = make_policy(kind), FakeMotor().eval()
    with torch.no_grad():
        policy.bridge.residual_output.weight.normal_(std=.05)
    observation = SimpleNamespace(state=torch.randn(2, 8))
    records, memory = [], None
    for _ in range(3):
        _, memory, record = policy.sample_chunk(motor, observation, torch.randn(2, 5), memory)
        records.append(record)
    projected_contexts = []
    def retain_context(module, arguments, output):
        output.retain_grad()
        projected_contexts.append(output)
    hook = policy.bridge.feedback_projection.register_forward_hook(retain_context)
    try:
        memory = None
        for record in records:
            log_prob, memory, _ = policy.replay_chunk(record, memory)
        log_prob.mean().backward()
    finally:
        hook.remove()
    assert projected_contexts[0].grad is not None
    assert projected_contexts[0].grad.abs().sum() > 0
