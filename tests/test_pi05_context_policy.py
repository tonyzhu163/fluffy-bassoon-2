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
    def __init__(self, *, depth=2, normalize=False):
        super().__init__()
        self.configs = [SimpleNamespace(width=8)]
        self.normalize = normalize
        self.layers = nn.ModuleList([nn.ModuleDict({"attn": TinyAttention()}) for _ in range(depth)])
        # ModuleDict supports registration but native code uses layer.attn.
        for layer in self.layers:
            object.__setattr__(layer, "attn", layer["attn"])

    def forward(self, xs, *, positions, mask, prefix_requires_grad=True):
        x = xs[0]
        caches = []
        for layer in self.layers:
            a = layer.attn
            normalized = x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6) if self.normalize else x
            q, k, v = (getattr(a, n)[0](normalized) for n in ("q_proj", "k_proj", "v_proj"))
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


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_collector_returns_cpu_decoded_commands_to_policy(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires allocated CUDA')
    from fluffy_bassoon.meta_rl.pi05_context_pilot import collect_trial

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.reference = nn.Parameter(torch.zeros((),device=device))
            self.motor = SimpleNamespace(
                env_obs_to_observation=lambda obs: SimpleNamespace(state=torch.zeros(1,32,device=device)),
                decode_actions=lambda actions,state: actions.cpu().double())
            self.histories = []

        def embed_observation(self, prepared):
            return (prepared.state[:,None],)

        def sample_chunk(self,prepared,inputs,history,mask,**kwargs):
            assert history.device == self.reference.device
            assert history.dtype == self.reference.dtype
            self.histories.append(history.cpu().clone())
            return torch.ones(1,8,7,device=device), {}

    class Env:
        device = 'cpu'
        env = SimpleNamespace(unwrapped=SimpleNamespace(select_carrot_ids=torch.tensor([0])))

        def reset(self, **kwargs):
            return {}, {}

        def chunk_step(self, commands):
            assert commands.device.type == 'cpu'
            return [{}]*5,None,None,None,[dict(success=torch.tensor([True]))]*5

    policy = Policy()
    plan = dict(scenes=[[dict(episode_id=0,object=0)]]*2,reset_seeds=[1,2],theta_degrees=[45.])
    trial = collect_trial(policy,Env(),plan,torch.Generator(device=device),behavior='mean',
                          preprocess=lambda obs:obs,prepare_actions=lambda actions:actions[:,:5])
    assert len(policy.histories) == len(trial['records']) == 32
    boundary = policy.histories[16][0,-1]
    torch.testing.assert_close(boundary[:7],torch.ones(7))
    assert boundary[7:9].tolist() == [1.,1.]
    assert trial['rewards'].sum() == 2
    assert all(record['feedback'].dtype == torch.float32 for record in trial['records'])


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


def test_zero_public_record_has_finite_deep_rmsnorm_gradients():
    torch.manual_seed(9)
    motor=TinyMotor()
    motor.llm=TinyLLM(depth=18,normalize=True)
    policy=NativeContextPi05(motor,rank=2)
    inputs=(torch.randn(1,3,8),torch.ones(1,3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
    history=torch.zeros(1,1,11)
    mask=torch.ones(1,1,dtype=torch.bool)
    loss=policy.flow_loss(None,inputs,history,mask,torch.randn(1,8,8),torch.randn(1,8,8),torch.tensor([.5]))
    loss.backward()
    gradients=[p.grad for p in policy.parameters() if p.grad is not None]
    assert all(g.isfinite().all() for g in gradients)
    assert policy.feedback.bias.grad.norm()>0
    assert torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True).isfinite()
