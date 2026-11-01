from dataclasses import asdict

import pytest
import torch

from fluffy_bassoon.meta_rl.causal_decoder import CausalDecoderCore, alibi_slopes
from fluffy_bassoon.meta_rl.gtrxl import GatedTransformerXLCore
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy


def full_causal_attention(core, tokens):
    """Unbatched [time, input_dim] reference without any cache."""
    current = core.input_projection(tokens)
    time = len(tokens)
    distances = (torch.arange(time)[:, None] - torch.arange(time)[None]).to(current)
    for layer in core.layers:
        normalized = layer.norm_attention(current)
        query = layer.query(normalized).reshape(time, layer.heads, layer.head_dim).transpose(0, 1)
        key, value = layer.key_value(normalized).chunk(2, dim=-1)
        key, value = (entry.reshape(time, layer.heads, layer.head_dim).transpose(0, 1) for entry in (key, value))
        scores = (query @ key.transpose(1, 2) * layer.head_dim ** -.5
                  - alibi_slopes(layer.heads).to(current)[:, None, None] * distances)
        attended = scores.masked_fill(distances < 0, -torch.inf).softmax(-1) @ value
        current = current + layer.output(attended.transpose(0, 1).reshape(time, -1))
        current = current + layer.ffn(layer.norm_ffn(current))
    return core.final_norm(current)


def step_through(core, tokens):
    state, outputs = None, []
    for token in tokens:
        output, state = core(token[None], state)
        outputs.append(output[0])
    return torch.stack(outputs), state


def test_cached_steps_equal_full_causal_attention():
    torch.manual_seed(3)
    core = CausalDecoderCore(5, width=8, memory_length=7, layers=2, heads=2).double()
    tokens = torch.randn(6, 5, dtype=torch.float64)
    stepped, state = step_through(core, tokens)
    torch.testing.assert_close(stepped, full_causal_attention(core, tokens))
    assert state["length"].tolist() == [6]


def test_future_tokens_cannot_change_earlier_outputs():
    torch.manual_seed(4)
    core = CausalDecoderCore(5, width=8, memory_length=7, heads=2).double()
    tokens = torch.randn(5, 5, dtype=torch.float64)
    changed = tokens.clone()
    changed[3:] = torch.randn(2, 5, dtype=torch.float64)
    torch.testing.assert_close(step_through(core, tokens)[0][:3], step_through(core, changed)[0][:3])


def test_window_drops_tokens_older_than_memory():
    torch.manual_seed(5)
    core = CausalDecoderCore(3, width=8, memory_length=2, layers=1, heads=2).double()
    tokens = torch.randn(6, 3, dtype=torch.float64)
    changed = tokens.clone()
    changed[0] = torch.randn(3, dtype=torch.float64)
    # One layer, window 2: output 5 attends to tokens 3..5 only.
    torch.testing.assert_close(step_through(core, tokens)[0][5], step_through(core, changed)[0][5])
    assert step_through(core, tokens)[1]["length"].tolist() == [2]


def test_padding_with_nan_does_not_leak():
    torch.manual_seed(6)
    core = CausalDecoderCore(3, width=8, memory_length=4, heads=2).double()
    token = torch.randn(1, 3, dtype=torch.float64)
    state = core.initial_state(1)
    state["memory"].fill_(torch.nan)
    output, _ = core(token, state)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, core(token)[0])


def test_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        CausalDecoderCore(3, width=6, heads=4)
    core = CausalDecoderCore(3, width=8, memory_length=3, heads=2)
    with pytest.raises(ValueError):
        core(torch.zeros(1, 4))


def test_history_policy_defaults_to_decoder_and_loads_legacy_gtrxl():
    config = HistoryConfig(3, 2, width=8, heads=2, layers=1, context_length=8)
    assert isinstance(HistoryNoisePolicy(config).core, CausalDecoderCore)
    legacy = {key: value for key, value in asdict(config).items() if key != "backbone"}
    restored = HistoryConfig.from_saved(legacy)
    assert restored.backbone == "gtrxl"
    assert isinstance(HistoryNoisePolicy(restored).core, GatedTransformerXLCore)
    assert HistoryConfig.from_saved(asdict(config)) == config
    with pytest.raises(ValueError):
        HistoryConfig(3, 2, backbone="ctm")
