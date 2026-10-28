"""Meaningful interface and likelihood checks; run on allocated Linux."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fluffy_bassoon.rl.pi05_context_policy import NativeContextPi05, PublicHistory, attention_mask


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(2)]))


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.configs = [SimpleNamespace(width=8)]
        self.layers = nn.ModuleList([nn.ModuleDict({"attn": TinyAttention()}) for _ in range(2)])
        # ModuleDict supports registration but native code uses layer.attn.
        for layer in self.layers:
            object.__setattr__(layer, "attn", layer["attn"])

    def forward(self, xs, *, positions, mask, prefix_requires_grad=True):
        x = xs[0]
        caches = []
        for layer in self.layers:
            a = layer.attn
            q, k, v = (getattr(a, n)[0](x) for n in ("q_proj", "k_proj", "v_proj"))
            weights = ((q @ k.transpose(-1,-2))/8**.5).masked_fill(~mask, -1e6).softmax(-1)
            x = x + a.o_proj[0](weights @ v)
            caches.append((k, v))
        return [x, None], tuple(caches)


class TinyMotor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pi05, self.action_horizon, self.action_dim = True, 8, 8
        self.llm = TinyLLM()
        self.head = nn.Linear(8, 8, bias=False)

    def run_suffix(self, prepared, x, time, cache, valid):
        k, v = cache[-1]
        weights = (x @ k.transpose(-1,-2)/8**.5).masked_fill(~valid[:,None], -1e6).softmax(-1)
        return x + weights @ v

    def velocity_from_suffix(self, suffix):
        return self.head(suffix)


def fixture():
    torch.manual_seed(4)
    policy = NativeContextPi05(TinyMotor(), rank=2)
    inputs = (torch.randn(2, 3, 8), torch.ones(2,3,dtype=torch.bool), torch.zeros(3,dtype=torch.bool))
    history = torch.zeros(2, 3, 11)
    history[:, 1, 7] = 1
    mask = torch.ones(2,3,dtype=torch.bool)
    return policy, inputs, history, mask


def test_history_reset_and_overflow():
    h = PublicHistory(2, 2)
    feedback = torch.zeros(2,11)
    feedback[0,7] = 1
    h.append(feedback)
    snapshot, valid = h.snapshot()
    snapshot.zero_()
    assert h.snapshot()[0][0,0,7] == 1
    h.reset(torch.tensor([True,False]))
    h.append(feedback)
    values, valid = h.snapshot()
    assert valid.tolist() == [[True,False],[True,True]]
    with pytest.raises(ValueError, match="capacity"):
        h.append(feedback)
    h.reset()
    assert h.snapshot()[0].shape == (2,0,11)
    bad = feedback.clone()
    bad[0,7] = 2
    with pytest.raises(ValueError, match="binary"):
        h.append(bad)


def test_native_mask_visibility_and_padding():
    valid = torch.tensor([[True,True,True,True,False]])
    mask = attention_mask(valid, torch.tensor([True,True,True,False,False]))
    assert mask[0,0].tolist() == [True,False,False,False,False]
    assert mask[0,1].tolist() == [True,True,False,False,False]
    assert mask[0,2].tolist() == [True,True,True,True,False]
    assert not mask[0,4].any()


def test_empty_history_has_exact_native_prefix_parity():
    policy, inputs, history, mask = fixture()
    native, native_cache = policy.motor.llm([inputs[0], None], positions=torch.arange(3)[None].expand(2,-1),
        mask=attention_mask(inputs[1],inputs[2]))
    valid, cache, _ = policy.build_prefix(inputs, history[:,:0], mask[:,:0])
    assert torch.equal(valid, inputs[1])
    assert all(torch.equal(x,y) for c,d in zip(cache,native_cache) for x,y in zip(c,d))


def test_history_causality_and_masked_values():
    policy, inputs, history, mask = fixture()
    _, before, _ = policy.build_prefix(inputs,history,mask)
    altered = history.clone()
    altered[:,2,7] = 1
    _, after, _ = policy.build_prefix(inputs,altered,mask)
    assert torch.equal(before[-1][0][:,:2],after[-1][0][:,:2])
    assert not torch.equal(before[-1][0][:,3:],after[-1][0][:,3:])
    mask[:,2] = False
    _, first, _ = policy.build_prefix(inputs,history,mask)
    altered[:,2] = float('nan')
    _, second, _ = policy.build_prefix(inputs,altered,mask)
    assert torch.equal(first[-1][0],second[-1][0])


def test_flow_gradients_reach_internal_adapters_and_preserve_base():
    policy, inputs, history, mask = fixture()
    frozen = policy.base_fingerprint()
    target, noise = torch.randn(2,8,8), torch.randn(2,8,8)
    time = torch.tensor([.3,.7])
    parameters = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters,lr=.01,weight_decay=0)
    initial = policy.flow_loss(None,inputs,history,mask,target,noise,time)
    initial.backward()
    assert policy.feedback.weight.grad.abs().sum() > 0
    assert policy.motor.llm.layers[0].attn.q_proj[0].up.grad.abs().sum() > 0
    opt.step()
    changed = policy.flow_loss(None,inputs,history,mask,target,noise,time)
    assert not torch.equal(initial,changed)
    assert policy.base_fingerprint() == frozen


def test_joint_velocity_replay_and_trainable_roundtrip():
    policy, inputs, history, mask = fixture()
    generator = torch.Generator().manual_seed(6)
    _, record = policy.sample_chunk(None,inputs,history,mask,generator=generator)
    lp, value = policy.replay_chunk(None,inputs,history,mask,record)
    torch.testing.assert_close(lp,record['old_log_prob'],atol=1e-5,rtol=0)
    torch.testing.assert_close(value,record['old_value'],atol=0,rtol=0)
    assert record['velocities'].shape == (4,2,8,8)
    saved = policy.trainable_state()
    with torch.no_grad():
        policy.feedback.bias.add_(.1)
    changed, _ = policy.replay_chunk(None,inputs,history,mask,record)
    assert not torch.allclose(lp,changed)
    policy.load_trainable_state(saved)
    restored, _ = policy.replay_chunk(None,inputs,history,mask,record)
    torch.testing.assert_close(lp,restored,atol=0,rtol=0)
    (-restored.mean()).backward()
    assert policy.feedback.weight.grad.abs().sum() > 0
    record['deterministic'] = True
    with pytest.raises(ValueError,match='stochastic'):
        policy.replay_chunk(None,inputs,history,mask,record)
