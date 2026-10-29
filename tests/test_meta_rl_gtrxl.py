"""GTrXL equations, independent causal reference and recurrent PPO contracts.

Run only on allocated Linux resources. Dense tests do not call the streaming
block or gate forwards, so cache indexing, pre-normalization, gate algebra and
all parameter gradients are checked against a separate sequence computation.
"""

from copy import deepcopy
from io import BytesIO

import gymnasium as gym
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fluffy_bassoon.meta_rl.gtrxl import GRUResidualGate, GatedTransformerXLCore
from fluffy_bassoon.meta_rl.ppo import RecurrentActorCritic


def reference_gate(gate, residual, update):
    reset = torch.sigmoid(F.linear(update, gate.reset_update.weight)
                          + F.linear(residual, gate.reset_input.weight))
    mixture = torch.sigmoid(F.linear(update, gate.update_update.weight)
                            + F.linear(residual, gate.update_input.weight) - gate.update_bias)
    candidate = torch.tanh(F.linear(update, gate.candidate_update.weight)
                           + F.linear(reset * residual, gate.candidate_input.weight))
    return residual * (1 - mixture) + candidate * mixture


def dense_reference(core, observations, resets=None, *, return_layer_inputs=False):
    """Independent [batch, time] causal graph; no recurrence or cached tensors."""
    current = F.linear(observations, core.input_projection.weight, core.input_projection.bias)
    batch, length, width = current.shape
    times = torch.arange(length, device=current.device)
    distance = times[:, None] - times[None]
    allowed = ((distance >= 0) & (distance <= core.memory_length))[None].expand(batch, -1, -1)
    if resets is not None:
        segments = resets.to(torch.long).cumsum(dim=1)
        allowed = allowed & (segments[:, :, None] == segments[:, None, :])
    positions = core.relative_positions[(core.memory_length - distance).clamp(0, core.memory_length)]
    inputs = []
    for layer in core.layers:
        inputs.append(current)
        normalized = F.layer_norm(current, (width,), layer.norm_attention.weight,
                                  layer.norm_attention.bias, layer.norm_attention.eps)
        heads, dim = layer.heads, layer.head_dim
        query = F.linear(normalized, layer.query.weight).reshape(batch, length, heads, dim)
        key, value = F.linear(normalized, layer.key_value.weight).chunk(2, dim=-1)
        key, value = (entry.reshape(batch, length, heads, dim) for entry in (key, value))
        relative = F.linear(positions, layer.relative.weight).reshape(length, length, heads, dim)
        content_scores = torch.einsum("bthd,bshd->bhts", query + layer.content_bias, key)
        position_scores = torch.einsum("bthd,tshd->bhts", query + layer.position_bias, relative)
        attention = ((content_scores + position_scores) * dim ** -.5).masked_fill(
            ~allowed[:, None], -torch.inf).softmax(dim=-1)
        attended = torch.einsum("bhts,bshd->bthd", attention, value).reshape(batch, length, width)
        attention_update = F.relu(F.linear(attended, layer.output.weight))
        current = reference_gate(layer.attention_gate, current, attention_update)
        normalized = F.layer_norm(current, (width,), layer.norm_ffn.weight,
                                  layer.norm_ffn.bias, layer.norm_ffn.eps)
        hidden = F.relu(F.linear(normalized, layer.ffn[0].weight, layer.ffn[0].bias))
        ffn_update = F.relu(F.linear(hidden, layer.ffn[2].weight, layer.ffn[2].bias))
        current = reference_gate(layer.ffn_gate, current, ffn_update)
    return (current, inputs) if return_layer_inputs else current


def stream(core, observations):
    state, values = None, []
    for time in range(observations.shape[1]):
        value, state = core(observations[:, time], state)
        values.append(value)
    return torch.stack(values, dim=1), state


def make_policy(*, memory=8):
    return RecurrentActorCritic(
        GatedTransformerXLCore(3, width=8, memory_length=memory, heads=2),
        gym.spaces.Discrete(3),
    )


def test_gate_matches_paper_formula_and_identity_bias_sign():
    torch.manual_seed(101)
    gate = GRUResidualGate(5).double()
    residual, update = torch.randn(2, 5, dtype=torch.float64), torch.randn(2, 5, dtype=torch.float64)
    torch.testing.assert_close(gate(residual, update), reference_gate(gate, residual, update), rtol=0, atol=0)
    torch.testing.assert_close(gate.update_bias, torch.full((5,), 2., dtype=torch.float64), rtol=0, atol=0)
    assert all(module.bias is None for module in gate.modules() if isinstance(module, nn.Linear))
    # With zero transforms, h=0 and z=sigmoid(-b). A positive *subtractive*
    # update bias retains more of x; reversing the sign would retain less.
    with torch.no_grad():
        for module in gate.modules():
            if isinstance(module, nn.Linear):
                module.weight.zero_()
    biased = gate(residual, update)
    expected = torch.sigmoid(gate.update_bias) * residual
    torch.testing.assert_close(biased, expected, rtol=1e-14, atol=1e-14)
    with torch.no_grad():
        gate.update_bias.zero_()
    unbiased = gate(residual, update)
    assert (biased - residual).abs().sum() < (unbiased - residual).abs().sum()
    assert not torch.equal(biased, residual)  # Identity-favoring is not exact identity.


def test_pre_norm_keeps_unnormalized_identity_path_and_independent_gates():
    torch.manual_seed(102)
    core = GatedTransformerXLCore(3, width=8, memory_length=4, heads=2).double()
    observation = torch.randn(2, 3, dtype=torch.float64)
    embedded = core.input_projection(observation)
    with torch.no_grad():
        for layer in core.layers:
            for parameter in layer.parameters():
                parameter.zero_()
            layer.attention_gate.update_bias.fill_(2.)
            layer.ffn_gate.update_bias.fill_(2.)
    actual, _ = core(observation)
    factor = torch.sigmoid(torch.tensor(2., dtype=torch.float64)) ** (2 * len(core.layers))
    torch.testing.assert_close(actual, embedded * factor, rtol=1e-13, atol=1e-14)
    first, second = core.layers
    assert first.query.weight.data_ptr() != second.query.weight.data_ptr()
    assert first.content_bias.data_ptr() != second.content_bias.data_ptr()
    assert first.attention_gate.update_bias.data_ptr() != first.ffn_gate.update_bias.data_ptr()
    assert not any(isinstance(module, nn.Dropout) for module in core.modules())


@pytest.mark.parametrize("length,memory", [(6, 9), (9, 3)])
def test_streaming_matches_dense_causal_outputs_and_all_parameter_gradients(length, memory):
    torch.manual_seed(103)
    core = GatedTransformerXLCore(3, width=8, memory_length=memory, heads=2).double()
    observations = torch.randn(2, length, 3, dtype=torch.float64, requires_grad=True)
    actual, state = stream(core, observations)
    expected = dense_reference(core, observations)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    weights = torch.randn_like(actual)
    variables = [observations, *core.parameters()]
    actual_gradients = torch.autograd.grad((actual * weights).sum(), variables, retain_graph=True)
    expected_gradients = torch.autograd.grad((expected * weights).sum(), variables)
    for observed, reference in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference, rtol=3e-8, atol=3e-9)
    assert state["length"].tolist() == [min(length, memory)] * 2


def test_cache_contains_each_layers_inputs_and_earliest_trial_gradient_survives():
    torch.manual_seed(104)
    core = GatedTransformerXLCore(3, width=8, memory_length=7, heads=2).double()
    observations = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    actual, state = stream(core, observations)
    _, layer_inputs = dense_reference(core, observations, return_layer_inputs=True)
    for index, inputs in enumerate(layer_inputs):
        torch.testing.assert_close(state["memory"][:, index, -5:], inputs, rtol=2e-10, atol=2e-11)
    actual[:, -1].square().sum().backward()
    assert observations.grad[:, 0].abs().sum() > 0
    assert torch.isfinite(observations.grad).all()
    assert state["memory"].grad_fn is not None


def test_padding_is_ignored_even_for_poisoned_slots_and_mixed_lengths():
    torch.manual_seed(105)
    core = GatedTransformerXLCore(3, width=8, memory_length=5, heads=2).double()
    _, state = stream(core, torch.randn(2, 2, 3, dtype=torch.float64))
    state = {name: value.detach().clone() for name, value in state.items()}
    state["length"][0] = 0
    valid = torch.arange(core.memory_length)[None] >= core.memory_length - state["length"][:, None]
    clean = dict(memory=torch.where(valid[:, None, :, None], state["memory"], 0.), length=state["length"])
    dirty = dict(memory=torch.where(valid[:, None, :, None], state["memory"], torch.nan), length=state["length"])
    observation = torch.randn(2, 3, dtype=torch.float64)
    expected, _ = core(observation, clean)
    actual, _ = core(observation, dirty)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    fresh, _ = core(observation[:1])
    torch.testing.assert_close(actual[:1], fresh, rtol=1e-12, atol=1e-13)


def test_state_is_functional_no_cross_trial_or_cross_lane_state_and_no_future_leak():
    torch.manual_seed(106)
    core = GatedTransformerXLCore(3, width=8, memory_length=5, heads=2).double()
    observations = torch.randn(2, 6, 3, dtype=torch.float64, requires_grad=True)
    initial = core.initial_state(2)
    before = deepcopy(initial)
    core(observations[:, 0], initial)
    for name in initial:
        torch.testing.assert_close(initial[name], before[name], rtol=0, atol=0)
    first, _ = stream(core, observations)
    stream(core, torch.randn_like(observations))
    repeated, _ = stream(core, observations)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    for lane in range(2):
        independent, _ = stream(core, observations[lane:lane + 1])
        torch.testing.assert_close(first[lane:lane + 1], independent, rtol=1e-11, atol=1e-12)
    changed = observations.detach().clone()
    changed[:, 3:] += 100
    changed_output = dense_reference(core, changed)
    torch.testing.assert_close(first[:, :3], changed_output[:, :3], rtol=2e-10, atol=2e-11)
    gradient = torch.autograd.grad(first[:, 2].sum(), observations)[0]
    assert not gradient[:, 3:].any()


def test_policy_lane_reset_matches_dense_segment_mask_and_cuts_history_gradient():
    torch.manual_seed(107)
    policy = make_policy().double()
    observations = torch.randn(6, 2, 3, dtype=torch.float64, requires_grad=True)
    actions = torch.zeros(6, 2, dtype=torch.long)
    resets = torch.zeros(6, 2, dtype=torch.bool)
    resets[3, 0] = True
    _, _, values = policy.evaluate_sequence(observations, actions, resets)
    features = dense_reference(policy.core, observations.transpose(0, 1), resets.transpose(0, 1))
    expected = F.linear(features, policy.critic.weight, policy.critic.bias).squeeze(-1).transpose(0, 1)
    torch.testing.assert_close(values, expected, rtol=2e-10, atol=2e-11)
    values[-1].sum().backward()
    assert not observations.grad[:3, 0].any()
    assert observations.grad[:3, 1].abs().sum() > 0
    assert observations.grad[3:, 0].abs().sum() > 0


def test_policy_likelihood_replay_is_exact_without_dropout_in_training_mode():
    torch.manual_seed(108)
    policy = make_policy()
    observations = torch.randn(7, 2, 3)
    resets = torch.zeros(7, 2, dtype=torch.bool)
    resets[3, 0], resets[5, 1] = True, True
    state, actions, likelihoods, values = None, [], [], []
    policy.eval()
    with torch.no_grad():
        for time in range(len(observations)):
            _, raw, likelihood, _, value, state = policy.step(observations[time], state, reset=resets[time])
            actions.append(raw)
            likelihoods.append(likelihood)
            values.append(value)
    policy.train()
    replayed, _, replayed_values = policy.evaluate_sequence(observations, torch.stack(actions), resets)
    torch.testing.assert_close(replayed, torch.stack(likelihoods), rtol=0, atol=0)
    torch.testing.assert_close(replayed_values, torch.stack(values), rtol=0, atol=0)


def test_policy_memory_and_optimizer_recovery_preserves_next_sample_and_update():
    torch.manual_seed(109)
    policy = make_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    observation = torch.randn(2, 3)
    # Populate optimizer state before taking the complete recovery snapshot.
    _, _, log_prob, _, value, _ = policy.step(observation)
    (log_prob.sum() + value.square().mean()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        state = None
        for _ in range(3):
            *_, state = policy.step(torch.randn(2, 3), state)
    payload = BytesIO()
    torch.save(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(), state=state,
                    rng=torch.get_rng_state()), payload)
    payload.seek(0)
    saved = torch.load(payload, weights_only=False)
    restored = make_policy()
    restored.load_state_dict(saved["policy"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=3e-4)
    restored_optimizer.load_state_dict(saved["optimizer"])
    state_before = {key: entry.clone() for key, entry in state.items()}
    torch.set_rng_state(saved["rng"])
    expected = policy.step(observation, state)
    torch.set_rng_state(saved["rng"])
    actual = restored.step(observation, saved["state"])
    for expected_tensor, actual_tensor in zip(expected[:-1], actual[:-1], strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    for key in state:
        torch.testing.assert_close(actual[-1][key], expected[-1][key], rtol=0, atol=0)
        torch.testing.assert_close(state[key], state_before[key], rtol=0, atol=0)
    (expected[2].sum() + expected[4].square().mean()).backward()
    (actual[2].sum() + actual[4].square().mean()).backward()
    optimizer.step()
    restored_optimizer.step()
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    for original_group, recovered_group in zip(optimizer.param_groups, restored_optimizer.param_groups, strict=True):
        for original_parameter, recovered_parameter in zip(original_group["params"], recovered_group["params"], strict=True):
            original, recovered = optimizer.state[original_parameter], restored_optimizer.state[recovered_parameter]
            assert original.keys() == recovered.keys()
            for key in original:
                torch.testing.assert_close(recovered[key], original[key], rtol=0, atol=0)


def test_default_memory_covers_complete_four_attempt_can_trial():
    core = GatedTransformerXLCore(55)
    assert core.output_dim == 128 and len(core.layers) == 2 and core.layers[0].heads == 4
    assert core.memory_length == 300 and core.memory_length >= 4 * (300 // 4)
    state = core.initial_state(2)
    assert state["memory"].shape == (2, 2, 300, 128)
    assert state["length"].dtype == torch.long and not state["length"].any()


@pytest.mark.parametrize("configuration", [dict(width=7), dict(heads=0), dict(heads=3), dict(ticks=2),
                                           dict(memory_length=0), dict(layers=0), dict(gate_bias=-1),
                                           dict(gate_bias=float("nan"))])
def test_invalid_architecture_is_rejected(configuration):
    with pytest.raises(ValueError):
        GatedTransformerXLCore(3, **configuration)


@pytest.mark.parametrize("corruption", ["negative_length", "long_length", "wrong_shape", "wrong_dtype", "extra_key"])
def test_invalid_state_is_rejected(corruption):
    core = GatedTransformerXLCore(3, width=8, memory_length=3, heads=2)
    state = core.initial_state(2)
    if corruption == "negative_length":
        state["length"][0] = -1
    elif corruption == "long_length":
        state["length"][0] = 4
    elif corruption == "wrong_shape":
        state["memory"] = state["memory"][:, :-1]
    elif corruption == "wrong_dtype":
        state["memory"] = state["memory"].double()
    else:
        state["surprise"] = torch.zeros(2)
    with pytest.raises(ValueError, match="complete valid GTrXL"):
        core(torch.zeros(2, 3), state)


@pytest.mark.parametrize("observation", [torch.zeros(3), torch.zeros(0, 3), torch.zeros(2, 4),
                                        torch.zeros(2, 3, dtype=torch.float64)])
def test_invalid_observation_is_rejected(observation):
    with pytest.raises(ValueError, match="observation"):
        GatedTransformerXLCore(3, width=8, memory_length=3, heads=2)(observation)
