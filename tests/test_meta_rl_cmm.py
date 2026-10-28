"""Numerical requirements of the paper-based CMM, independent of reward learning."""

import copy

import pytest
import torch
from torch.nn import functional as F

from fluffy_bassoon.meta_rl.cmm import CMMCore, MemoryBlock


def core(**kwargs):
    return CMMCore(3, width=8, stm_length=3, ltm_tokens=2, heads=2,
                   synapse_depth=2, **kwargs).double().eval()


def test_memory_block_matches_explicit_attention_equations():
    torch.manual_seed(4)
    block = MemoryBlock(8, 2).double().eval()
    tokens, positions = torch.randn(2, 6, 8, dtype=torch.float64), torch.randn(6, 8, dtype=torch.float64)
    normalized = block.attention_norm(tokens)
    query, key, value = block.query(normalized + positions), block.key(normalized + positions), block.value(normalized)
    value = torch.cat((torch.zeros_like(value[:, :1]), value[:, 1:]), 1)
    heads = []
    for head in range(2):
        q, k, v = (x[:, :, head * 4:(head + 1) * 4] for x in (query, key, value))
        heads.append(((q @ k.transpose(-1, -2)) / 2).softmax(-1) @ v)
    expected = tokens + block.output(torch.cat(heads, -1))
    expected = expected + block.ffn(block.ffn_norm(expected))
    torch.testing.assert_close(block(tokens, positions), expected, rtol=1e-12, atol=1e-12)


def test_zero_residual_gate_is_identity_but_can_learn():
    block = MemoryBlock(8, 2, gated=True).double()
    tokens, positions = torch.randn(2, 5, 8, dtype=torch.float64), torch.randn(5, 8, dtype=torch.float64)
    output = block(tokens, positions)
    torch.testing.assert_close(output, tokens, rtol=0, atol=0)
    output.sum().backward()
    assert block.gate.grad is not None and block.gate.grad.abs() > 0


def test_raw_sliding_history_is_carried_instead_of_transformed_stm():
    model = core()
    observation = torch.randn(2, 3, dtype=torch.float64)
    initial = model.initial_state(2)
    preactivation = model.synapses(torch.cat((initial["activated"], observation), -1))
    expected_trace = torch.cat((initial["trace"][:, :, 1:], preactivation.unsqueeze(-1)), -1)
    _, state = model(observation, initial)
    torch.testing.assert_close(state["trace"], expected_trace)
    tokens = torch.cat((model.sink.expand(2, -1, -1), initial["long_term"], expected_trace.transpose(1, 2)), 1)
    transformed = model.blocks[0](tokens, model.positions)
    assert not torch.allclose(state["trace"], transformed[:, 3:].transpose(1, 2))
    torch.testing.assert_close(state["long_term"], model.ltm_norm(transformed[:, 1:3]))
    assert torch.count_nonzero(initial["trace"]) == 0


def test_synchronization_uses_linear_weight_sum_and_fixed_unique_pairs():
    model = core()
    with torch.no_grad():
        model.decay_params.fill_(.3)
    first, state = model(torch.randn(2, 3, dtype=torch.float64))
    before = {k: v.clone() for k, v in state.items()}
    features, state = model(torch.randn(2, 3, dtype=torch.float64), state)
    product = state["activated"][:, model.pair_left] * state["activated"][:, model.pair_right]
    decay = torch.exp(torch.tensor(-.3, dtype=torch.float64))
    alpha, beta = before["alpha"] * decay + product, before["beta"] * decay + 1
    torch.testing.assert_close(state["alpha"], alpha)
    torch.testing.assert_close(state["beta"], beta)
    torch.testing.assert_close(features, model.synchronization_norm(alpha / beta + 1e-8))
    assert torch.unique(model.pair_left * model.width + model.pair_right).numel() == model.width
    assert first.isfinite().all()


def test_cross_observation_gradients_and_incoming_state_immutability():
    model = core()
    observation = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    _, state = model(observation)
    before = {k: v.detach().clone() for k, v in state.items()}
    features, _ = model(torch.randn(2, 3, dtype=torch.float64), state)
    (features * torch.arange(1, 9, dtype=torch.float64)).sum().backward()
    assert observation.grad is not None and observation.grad.isfinite().all() and observation.grad.abs().sum() > 0
    assert model.initial_ltm.grad is not None and model.initial_ltm.grad.abs().sum() > 0
    for key in state:
        torch.testing.assert_close(state[key].detach(), before[key], rtol=0, atol=0)


def test_batched_lanes_match_independent_execution_and_saved_model():
    model = core(ticks=2)
    restored = core(ticks=2)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    observations = torch.randn(3, 3, dtype=torch.float64)
    features, state = model(observations)
    for lane in range(3):
        separate, memory = restored(observations[lane:lane + 1])
        torch.testing.assert_close(features[lane:lane + 1], separate)
        for key in state:
            torch.testing.assert_close(state[key][lane:lane + 1], memory[key])
    assert not model.initial_activated.requires_grad and not model.initial_trace.requires_grad
    torch.testing.assert_close(model.initial_activated, torch.zeros_like(model.initial_activated))


def test_no_ltm_ablation_and_complete_state_validation():
    model = CMMCore(3, width=8, stm_length=3, ltm_tokens=0, heads=2, synapse_depth=2).double()
    features, state = model(torch.randn(2, 3, dtype=torch.float64))
    assert state["long_term"].shape == (2, 0, 8) and features.shape == (2, 8)
    with pytest.raises(ValueError, match="complete"):
        model(torch.randn(2, 3, dtype=torch.float64), {"activated": state["activated"]})
    with pytest.raises(ValueError, match="observation"):
        model(torch.randn(2, 3))


def test_inference_does_not_modify_parameters_and_projection_is_explicit():
    model = core()
    with torch.no_grad():
        model.decay_params.fill_(-.5)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model(torch.randn(2, 3, dtype=torch.float64))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    model.constrain_parameters()
    assert model.decay_params.min() == 0


def test_ctm_control_matches_zero_gated_cmm_without_unused_memory_parameters():
    ctm = CMMCore(3, width=8, stm_length=3, ltm_tokens=0, layers=0,
                  heads=2, synapse_depth=2).double().eval()
    cmm = core(gated=True)
    shared = {key: value for key, value in ctm.state_dict().items()
              if key in cmm.state_dict() and value.shape == cmm.state_dict()[key].shape}
    cmm.load_state_dict({**cmm.state_dict(), **shared})
    ctm_state = cmm_state = None
    for observation in torch.randn(3, 2, 3, dtype=torch.float64):
        expected, ctm_state = ctm(observation, ctm_state)
        actual, cmm_state = cmm(observation, cmm_state)
        torch.testing.assert_close(actual, expected)
        for key in ("activated", "trace", "alpha", "beta"):
            torch.testing.assert_close(cmm_state[key], ctm_state[key])
    assert not list(ctm.blocks.parameters())
    assert ctm.initial_ltm.numel() == ctm.sink.numel() == ctm.positions.numel() == 0
    assert not ctm.initial_ltm.requires_grad
    with pytest.raises(ValueError, match="no LTM"):
        CMMCore(3, layers=0, ltm_tokens=2)
