"""Independent equation oracles for CMM §3.2 and Appendix A; run on Linux."""
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from lerobot_policy_ctm.architectures.continuous_memory_machine import CMMConfig, ContinuousMemoryMachine
from lerobot_policy_ctm.architectures.continuous_memory_machine.core import JointMemoryBlock


def small(**options):
    return ContinuousMemoryMachine(3, 2, CMMConfig(width=8, stm_length=3, ltm_slots=2, heads=2, **options))


def test_paper_initialization_and_unique_saved_pairs():
    model = small()
    state = model.initial_state(2)
    assert not model.initial_activated.requires_grad and state["activated"].count_nonzero() == 0
    assert not model.initial_trace.requires_grad and state["trace"].count_nonzero() == 0
    assert model.initial_ltm.requires_grad and model.sink.count_nonzero() == 0
    assert model.positions.shape == (6, 8)
    assert len(set(zip(model.pair_left.tolist(), model.pair_right.tolist()))) == 8
    restored = small(pair_seed=123)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored.pair_left, model.pair_left)
    assert torch.equal(restored.pair_right, model.pair_right)


def test_joint_attention_matches_explicit_equations_with_zero_sink_values():
    torch.manual_seed(1)
    config = CMMConfig(width=8, heads=2)
    block = JointMemoryBlock(8, 2, norm_eps=config.norm_eps).eval()
    tokens, positions = torch.randn(2, 6, 8), torch.randn(6, 8)
    actual = block(tokens, positions)
    normalized = F.layer_norm(tokens, (8,), block.attention_norm.weight, block.attention_norm.bias, config.norm_eps)
    def split(x):
        return x.reshape(2, 6, 2, 4).transpose(1, 2)
    q = split(F.linear(normalized + positions, block.query.weight))
    k = split(F.linear(normalized + positions, block.key.weight))
    # Explicit content-only values; position must not be included here.
    value = F.linear(normalized, block.value.weight)
    value = torch.cat((torch.zeros_like(value[:, :1]), value[:, 1:]), 1)
    attention = ((q @ k.transpose(-1, -2)) / math.sqrt(4)).softmax(-1)
    attended = (attention @ split(value)).transpose(1, 2).reshape(2, 6, 8)
    intermediate = tokens + F.linear(attended, block.output.weight)
    norm = F.layer_norm(intermediate, (8,), block.ffn_norm.weight, block.ffn_norm.bias, config.norm_eps)
    expected = intermediate + F.linear(F.silu(F.linear(norm, block.ffn[0].weight)), block.ffn[2].weight)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert all(module.bias is None for module in block.modules() if isinstance(module, nn.Linear))


def test_zero_learned_gate_is_identity_and_both_branches_share_it():
    block = JointMemoryBlock(8, 2, gated=True)
    x = torch.randn(2, 6, 8)
    y = block(x, torch.randn(6, 8))
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert isinstance(block.gate, nn.Parameter)
    y.square().sum().backward()
    assert block.gate.grad is not None and block.gate.grad.abs() > 0


def test_raw_stm_is_carried_while_transformed_stm_only_feeds_neurons():
    class AddToMemories(nn.Module):
        def forward(self, tokens, positions):
            return tokens + 7

    model = small()
    model.blocks = nn.ModuleList([AddToMemories()])
    seen = []
    handle = model.neurons.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    observation = torch.randn(1, 3)
    original = model.initial_state(1)
    snapshot = {k: v.detach().clone() for k, v in original.items()}
    expected_pre = model.synapses(torch.cat((original["activated"], observation), -1))
    _, updated = model.step(observation, original)
    expected_raw = torch.cat((original["trace"][:, :, 1:], expected_pre[:, :, None]), -1)
    torch.testing.assert_close(updated["trace"], expected_raw, atol=0, rtol=0)
    torch.testing.assert_close(seen[0], expected_raw + 7, atol=0, rtol=0)
    for name in original:
        torch.testing.assert_close(original[name], snapshot[name], atol=0, rtol=0)
    handle.remove()


def test_linear_sync_normalizer_matches_direct_weighted_history():
    model = small().double()
    with torch.no_grad():
        model.decay_params.uniform_(0, 2)
    observations = torch.randn(1, 5, 3, dtype=torch.double)
    state, activated = None, []
    for t in range(5):
        _, state = model.step(observations[:, t], state)
        activated.append(state["activated"])
    products = torch.stack([z[:, model.pair_left] * z[:, model.pair_right] for z in activated], 1)
    rates = model.decay_params.clamp(0, 15)
    weights = torch.exp(-torch.arange(4, -1, -1, dtype=torch.double)[:, None] * rates)
    expected = (products * weights).sum(1) / weights.sum(0) + 1e-8
    expected = expected * (expected.square().mean(-1, keepdim=True) + model.config.rms_eps).rsqrt()
    expected = expected * model.synchronization_norm.weight
    torch.testing.assert_close(model.synchronization(state), expected, atol=1e-12, rtol=1e-12)


def test_sequence_reset_padding_and_streaming_match_independent_episodes():
    # Different active batch sizes can change FP32 reduction order. Use a
    # seeded FP64 equation oracle here; allocated CUDA tests retain FP32 replay.
    torch.manual_seed(83)
    model = small().double().eval()
    observations = torch.randn(2, 6, 3, dtype=torch.float64)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    reset = torch.zeros_like(valid)
    reset[1, 3] = True
    output, state = model(observations, valid_mask=valid, reset_mask=reset)
    first, first_state = model(observations[:1, :4])
    second, second_state = model(observations[1:, 3:])
    torch.testing.assert_close(output[:1, :4], first, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(output[1:, 3:], second, rtol=1e-10, atol=1e-12)
    assert output[0, 4:].count_nonzero() == 0
    for name in state:
        torch.testing.assert_close(state[name][:1], first_state[name], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(state[name][1:], second_state[name], rtol=1e-10, atol=1e-12)
    # Splitting a sequence does not reset memory or synchronization.
    prefix, memory = model(observations[:, :2])
    suffix, _ = model(observations[:, 2:], memory)
    whole, _ = model(observations)
    torch.testing.assert_close(torch.cat((prefix, suffix), 1), whole, rtol=0, atol=0)


def test_gradients_cross_ltm_and_early_observations_without_implicit_detach():
    model = small()
    observations = torch.randn(1, 7, 3, requires_grad=True)
    output, state = model(observations)
    output[:, -1].square().sum().backward()
    assert observations.grad[:, 0].abs().sum() > 0
    assert model.initial_ltm.grad is not None and model.initial_ltm.grad.abs().sum() > 0
    assert model.neurons.weight.grad.abs().sum() > 0
    assert model.blocks[0].query.weight.grad.abs().sum() > 0
    assert state["long_term"].grad_fn is not None


def test_zero_ltm_is_explicit_cmm_stm_ablation_and_invalid_state_fails():
    model = ContinuousMemoryMachine(3, 2, CMMConfig(width=8, heads=2, ltm_slots=0))
    output, state = model(torch.randn(2, 3, 3))
    assert output.shape == (2, 3, 2) and state["long_term"].shape == (2, 0, 8)
    bad = {name: value.detach().clone() for name, value in state.items()}
    bad.pop("trace")
    with pytest.raises(ValueError, match="complete CMM state"):
        model.step(torch.randn(2, 3), bad)


def test_paper_alignment_bias_free_neurons_zero_decay_and_seeded_pairs():
    model = small()
    assert [name for name, _ in model.neurons.named_parameters()] == [
        "weight", "normalization.weight", "normalization.bias"]
    assert model.decay_params.count_nonzero() == 0
    again = small()
    assert torch.equal(again.pair_left, model.pair_left) and torch.equal(again.pair_right, model.pair_right)
    other = small(pair_seed=7)
    assert not (torch.equal(other.pair_left, model.pair_left) and torch.equal(other.pair_right, model.pair_right))
    with torch.no_grad():
        model.decay_params.fill_(-1)
    model.constrain_parameters()
    assert model.decay_params.min() == 0


def test_meta_rl_wrapper_is_the_same_recurrence():
    from fluffy_bassoon.meta_rl.cmm import CMMCore
    wrapper = CMMCore(3, width=8, stm_length=3, ltm_tokens=2, heads=2, synapse_depth=2, ticks=2).double()
    base = ContinuousMemoryMachine(3, None, CMMConfig(width=8, stm_length=3, ltm_slots=2, heads=2,
                                                      synapse_depth=2, ticks=2)).double()
    base.load_state_dict(wrapper.state_dict(), strict=True)
    observation = torch.randn(2, 3, dtype=torch.double)
    expected, expected_state = base.step(observation)
    actual, state = wrapper(observation)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name in expected_state:
        torch.testing.assert_close(state[name], expected_state[name], rtol=0, atol=0)
    assert actual.shape == (2, 8) and wrapper.readout is None


def test_ctm_control_has_no_memory_parameters_and_rejects_ltm():
    model = ContinuousMemoryMachine(3, None, CMMConfig(width=8, stm_length=3, ltm_slots=0, memory_layers=0))
    assert not list(model.blocks.parameters()) and model.sink.numel() == model.positions.numel() == 0
    features, state = model.step(torch.randn(2, 3))
    assert features.shape == (2, 8) and state["long_term"].shape == (2, 0, 8)
    with pytest.raises(ValueError):
        CMMConfig(memory_layers=0, ltm_slots=2)
