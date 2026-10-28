"""Streaming relative attention must match a complete causal sequence and gradients."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from fluffy_bassoon.meta_rl.transformer_xl import TransformerXLCore


def dense_reference(core, observations):
    current = core.input_projection(observations)
    batch, length, width = current.shape
    times = torch.arange(length, device=current.device)
    distance = times[:, None] - times[None]
    allowed = (distance >= 0) & (distance <= core.memory_length)
    positions = core.relative_positions[(core.memory_length - distance).clamp(0, core.memory_length)]
    for layer in core.layers:
        heads, dim = layer.heads, layer.head_dim
        query = F.linear(current, layer.query.weight).reshape(batch, length, heads, dim)
        key, value = F.linear(current, layer.key_value.weight).chunk(2, dim=-1)
        key, value = (x.reshape(batch, length, heads, dim) for x in (key, value))
        relative = F.linear(positions, layer.relative.weight).reshape(length, length, heads, dim)
        scores = (torch.einsum('bthd,bshd->bhts', query + core.content_bias, key)
                  + torch.einsum('bthd,tshd->bhts', query + core.position_bias, relative)) * dim ** -.5
        attention = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        attended = torch.einsum('bhts,bshd->bthd', attention, value).reshape(batch, length, width)
        current = layer.norm_attention(current + F.linear(attended, layer.output.weight))
        current = layer.norm_ffn(current + layer.ffn(current))
    return current


@pytest.mark.parametrize('length,memory', [(5,8),(9,3)])
def test_streaming_matches_full_causal_outputs_and_all_parameter_gradients(length,memory):
    torch.manual_seed(18)
    core = TransformerXLCore(3,width=8,memory_length=memory,heads=2).double()
    observations = torch.randn(2,length,3,dtype=torch.float64,requires_grad=True)
    state, rows = None, []
    for index in range(length):
        value,state=core(observations[:,index],state);rows.append(value)
    streamed=torch.stack(rows,dim=1)
    expected=dense_reference(core,observations)
    torch.testing.assert_close(streamed,expected,rtol=1e-9,atol=1e-10)
    weights=torch.randn_like(streamed)
    variables=[observations,*core.parameters()]
    actual_gradients=torch.autograd.grad((streamed*weights).sum(),variables,retain_graph=True)
    expected_gradients=torch.autograd.grad((expected*weights).sum(),variables)
    for actual,expected in zip(actual_gradients,expected_gradients,strict=True):
        torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-9)
    assert state['length'].tolist()==[min(length,memory)]*2
    assert core.layers[0].query.weight.data_ptr()!=core.layers[1].query.weight.data_ptr()


def test_padding_is_masked_and_full_trial_gradient_reaches_earliest_observation():
    torch.manual_seed(23)
    core=TransformerXLCore(3,width=8,memory_length=8,heads=2).double()
    observations=torch.randn(2,5,3,dtype=torch.float64,requires_grad=True)
    state=core.initial_state(2)
    state['memory'].normal_(mean=100,std=10)
    first,_=core(observations[:,0],state)
    clean,_=core(observations[:,0])
    torch.testing.assert_close(first,clean,rtol=0,atol=0)
    state=None
    for index in range(5):last,state=core(observations[:,index],state)
    last[:,0].sum().backward()
    assert observations.grad[:,0].abs().sum()>0
    assert torch.isfinite(observations.grad).all()


def test_streaming_state_is_functional_and_restores_exactly():
    torch.manual_seed(29)
    core=TransformerXLCore(3,width=8,memory_length=3,heads=2)
    state=core.initial_state(2);before=deepcopy(state)
    _,next_state=core(torch.randn(2,3),state)
    for name in state:torch.testing.assert_close(state[name],before[name],rtol=0,atol=0)
    observation=torch.randn(2,3)
    output,expected=core(observation,next_state)
    snapshot={key:value.detach().clone() for key,value in next_state.items()}
    restored=deepcopy(core);actual,actual_state=restored(observation,snapshot)
    torch.testing.assert_close(actual,output,rtol=0,atol=0)
    for name in expected:torch.testing.assert_close(actual_state[name],expected[name],rtol=0,atol=0)
    bad={key:value.detach().clone() for key,value in next_state.items()};bad['length'][0]=-1
    with pytest.raises(ValueError,match='complete valid'):core(observation,bad)


@pytest.mark.parametrize('configuration', [dict(width=7),dict(heads=0),dict(ticks=2),dict(memory_length=0),dict(layers=0)])
def test_invalid_architecture_is_rejected(configuration):
    with pytest.raises(ValueError):TransformerXLCore(3,**configuration)
