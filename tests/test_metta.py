"""Independent inner-gradient, causal-memory, PPO and recovery qualifications."""

import copy
import math
import os
from pathlib import Path
import json
from time import perf_counter

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fluffy_bassoon.meta_rl.checkpoint import restore_checkpoint, save_checkpoint
from fluffy_bassoon.meta_rl.metta import MettaMemory, TTTMLP
from fluffy_bassoon.meta_rl.metaworld_metta import DEFAULT_SETTINGS, MettaActorCritic, update_metta_ppo
from fluffy_bassoon.meta_rl.ppo import collect_trials
from test_metaworld_pilot import fake_env


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def oracle_memory(u, weights, layer):
    y = torch.bmm(u[:, None], weights['w1']).squeeze(1) + weights['b1']
    y = torch.bmm(F.gelu(y)[:, None], weights['w2']).squeeze(1) + weights['b2']
    return u + F.layer_norm(y, (layer.width,), layer.norm.weight, layer.norm.bias, layer.eps)


def oracle_sequence(layer, inputs):
    current = {name:getattr(layer, 'initial_' + name)[None].expand(inputs.shape[1], *getattr(layer, 'initial_' + name).shape)
               for name in ('w1','b1','w2','b2')}
    outputs = []
    for i, x in enumerate(inputs):
        if i % layer.chunk_length == 0:
            base = dict(current)
        key, target, query = layer.key(x), layer.value(x), layer.query(x)
        loss = .5 * (oracle_memory(key, base, layer) - target).square().sum()
        gradients = torch.autograd.grad(loss, tuple(base.values()), create_graph=True)
        rate = layer.base_lr / layer.width * torch.sigmoid(F.linear(x, layer.learning_rate.weight, layer.learning_rate.bias))
        current = {name: weight - rate.reshape(-1, *([1] * (weight.ndim-1))) * g
                   for (name,weight),g in zip(current.items(),gradients,strict=True)}
        outputs.append(oracle_memory(query, current, layer))
    return torch.stack(outputs), current


@pytest.mark.parametrize('chunk', [1,2,4])
def test_ttt_outputs_fast_states_and_outer_gradients_match_autograd_oracle(chunk):
    torch.manual_seed(14)
    layer = TTTMLP(3, expansion=2, chunk_length=chunk, base_lr=.003).double()
    reference = copy.deepcopy(layer)
    x = torch.randn(5,2,3,dtype=torch.float64,requires_grad=True)
    other_x = x.detach().clone().requires_grad_()
    state = None; actual=[]
    for observation in x:
        h,state=layer(observation,state);actual.append(h)
    actual=torch.stack(actual)
    expected,weights=oracle_sequence(reference,other_x)
    torch.testing.assert_close(actual,expected,rtol=2e-9,atol=2e-10)
    for name in weights:torch.testing.assert_close(state[name],weights[name],rtol=2e-9,atol=2e-10)
    a=torch.autograd.grad(actual.square().sum(),(*layer.parameters(),x))
    b=torch.autograd.grad(expected.square().sum(),(*reference.parameters(),other_x))
    for left,right in zip(a,b,strict=True):torch.testing.assert_close(left,right,rtol=2e-7,atol=2e-8)


@pytest.mark.parametrize('context',[torch.no_grad,torch.inference_mode])
def test_fast_weights_update_with_frozen_slow_parameters_without_autograd(context):
    torch.manual_seed(5);layer=TTTMLP(4,chunk_length=3,base_lr=.01)
    original={k:v.detach().clone() for k,v in layer.state_dict().items()}
    with context():
        h,state=layer(torch.randn(2,4));_,state2=layer(torch.randn(2,4),state)
        assert not torch.equal(state2['w1'],state['w1'])
        assert not state2['w1'].requires_grad and torch.isfinite(h).all()
    for k,v in layer.state_dict().items():assert torch.equal(v,original[k])


def test_chunk_gradients_use_the_chunk_start_and_write_before_read():
    torch.manual_seed(6);layer=TTTMLP(4,chunk_length=2,base_lr=.001).double()
    state=layer.initial_state(1);x=torch.randn(1,4,dtype=torch.float64)
    base={n:state[n] for n in ('w1','b1','w2','b2')}
    h,one=layer(x,state)
    for n in base:torch.testing.assert_close(one['base_'+n],base[n],rtol=0,atol=0)
    assert not torch.allclose(h,oracle_memory(layer.query(x),base,layer))
    _,two=layer(x,one)
    for n in base:torch.testing.assert_close(two['base_'+n],two[n],rtol=0,atol=0)
    assert two['position'].item()==0


def small_memory(**kwargs):
    return MettaMemory(5,width=4,layers=2,expansion=2,chunk_length=2,max_episodes=4,base_lr=.001,**kwargs).double()


def test_episode_boundary_resets_fast_weights_but_keeps_previous_summary():
    torch.manual_seed(10);memory=small_memory()
    x=torch.randn(2,5,dtype=torch.float64)
    _,old=memory(x);snapshot={k:v.clone() for k,v in old.items()}
    _,new=memory(x,old,episode_start=torch.tensor([True,False]))
    _,fresh=memory(x)
    assert new['summaries'].tolist()==[1,0]
    torch.testing.assert_close(new['history'][0,0],old['last'][0])
    for key in new:
        if key.startswith('layer'):torch.testing.assert_close(new[key][0],fresh[key][0])
    for key in old:torch.testing.assert_close(old[key],snapshot[key],rtol=0,atol=0)


@pytest.mark.parametrize('aggregation',['attention','ema','none'])
def test_empty_history_is_finite_and_exactly_current_representation(aggregation):
    memory=small_memory(aggregation=aggregation)
    x=torch.randn(2,5,dtype=torch.float64,requires_grad=True)
    h,state=memory(x);torch.testing.assert_close(h,state['last'],rtol=0,atol=0)
    h.square().sum().backward();assert torch.isfinite(x.grad).all()


def test_attention_ignores_padding_and_is_invariant_to_episode_permutation():
    torch.manual_seed(21);memory=small_memory()
    h=torch.randn(2,4,dtype=torch.float64);history=torch.randn(2,4,4,dtype=torch.float64)
    counts=torch.tensor([3,1]);ema=torch.zeros_like(h)
    expected=memory.retrieve(h,history,counts,ema)
    changed=history.clone();changed[0,3]=1e8;changed[1,1:]=-1e8
    torch.testing.assert_close(memory.retrieve(h,changed,counts,ema),expected)
    changed=history.clone();changed[0,:3]=changed[0,[2,0,1]]
    torch.testing.assert_close(memory.retrieve(h,changed,counts,ema),expected)


@pytest.mark.parametrize('aggregation',['attention','none'])
def test_gradients_reach_prior_episode_only_when_cross_episode_retrieval_is_enabled(aggregation):
    torch.manual_seed(17);memory=small_memory(aggregation=aggregation)
    first=torch.randn(1,5,dtype=torch.float64,requires_grad=True)
    _,state=memory(first)
    h,_=memory(torch.randn(1,5,dtype=torch.float64),state,episode_start=torch.ones(1,dtype=torch.bool))
    grad=torch.autograd.grad((h*torch.arange(4,dtype=h.dtype)).sum(),first)[0]
    assert torch.isfinite(grad).all()
    assert (grad.abs().max()>1e-12) if aggregation=='attention' else torch.equal(grad,torch.zeros_like(grad))


def test_no_online_update_ablation_keeps_all_fast_weights_unchanged():
    torch.manual_seed(33);layer=TTTMLP(4,chunk_length=2)
    state=layer.initial_state(2)
    for _ in range(5):
        _,state=layer(torch.randn(2,4),state,online_updates=False)
    for name in ('w1','b1','w2','b2'):
        torch.testing.assert_close(state[name],getattr(layer,'initial_'+name)[None].expand_as(state[name]),rtol=0,atol=0)


def test_summary_capacity_overflow_fails_instead_of_discarding_history():
    memory=MettaMemory(2,width=4,layers=1,max_episodes=1,base_lr=.001)
    x=torch.randn(1,2);_,state=memory(x)
    _,state=memory(x,state,episode_start=torch.tensor([True]))
    with pytest.raises(ValueError,match='capacity'):
        memory(x,state,episode_start=torch.tensor([True]))


def settings(**kwargs):
    return dict(DEFAULT_SETTINGS, width=4,layers=1,expansion=2,chunk_length=2,max_episodes=4,base_lr=.001,**kwargs)


def rollout_policy(device='cpu',**kwargs):
    env=fake_env(goal_visible=True)
    return MettaActorCritic(env,settings(**kwargs)).to(device)


def test_collected_likelihoods_replay_and_public_clocks_do_not_enter_network():
    torch.manual_seed(38);envs=[fake_env(goal_visible=True) for _ in range(2)]
    policy=rollout_policy()
    rollout=collect_trials(policy,envs,[21,22]);obs=rollout['observations']
    with torch.no_grad():
        log,_,value=policy.evaluate_sequence(obs,rollout['raw_actions'],rollout['resets'])
        torch.testing.assert_close(log,rollout['log_probs'],rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(value,rollout['values'],rtol=1e-6,atol=1e-6)
        changed=obs.clone();clock=policy.core.memory.fields['clock'];changed[:,:,clock]+=1000
        other=policy.evaluate_sequence(changed,rollout['raw_actions'],rollout['resets'])
        torch.testing.assert_close(log,other[0],rtol=0,atol=0)


def test_episode_initial_feedback_is_cleared_without_clearing_summary():
    torch.manual_seed(40);policy=rollout_policy();adapter=policy.core.memory
    x=torch.randn(1,adapter.input_dim);x[:,adapter.fields['attempt_end']]=0
    _,state=adapter(x)
    x[:,adapter.fields['attempt_end']]=1;changed=x.clone()
    for key in ('previous_action','previous_reward'):changed[:,adapter.fields[key]]+=123
    h,a=adapter(x,state);other,b=adapter(changed,state)
    torch.testing.assert_close(h,other,rtol=0,atol=0);assert a['summaries'].item()==1


def test_activation_checkpoint_preserves_outputs_and_all_outer_gradients():
    torch.manual_seed(44);left=rollout_policy(checkpoint_length=2).double();right=copy.deepcopy(left)
    right.checkpoint_length=0
    envs=[fake_env(goal_visible=True) for _ in range(2)]
    r=collect_trials(left,envs,[21,22])
    a=left.evaluate_sequence(r['observations'],r['raw_actions'],r['resets'])
    b=right.evaluate_sequence(r['observations'],r['raw_actions'],r['resets'])
    for x,y in zip(a,b,strict=True):torch.testing.assert_close(x,y,rtol=0,atol=0)
    sum(x.square().sum() for x in a).backward();sum(x.square().sum() for x in b).backward()
    for (name,x),(other,y) in zip(left.named_parameters(),right.named_parameters(),strict=True):
        assert name==other and (x.grad is None)==(y.grad is None)
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=1e-10,atol=1e-10)


def test_segment_ppo_reconstructs_fresh_prefixes_and_counts_work():
    torch.manual_seed(45);policy=rollout_policy();r=collect_trials(policy,[fake_env(goal_visible=True)]*1,[11])
    optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    calls=[];original=policy.evaluate_sequence
    def record(obs,*args,**kwargs):
        calls.append((len(obs),next(policy.parameters()).detach().clone(),kwargs.get('initial_state')))
        return original(obs,*args,**kwargs)
    policy.evaluate_sequence=record
    result=update_metta_ppo(policy,optimizer,r,epochs=2,minibatch_trials=1)
    assert [x[0] for x in calls]==[3,6,3,6]
    assert all(x[2] is None for x in calls)
    assert not torch.equal(calls[0][1],calls[1][1])
    assert result['optimizer_steps']==4 and result['forward_samples']==12
    assert result['reconstructed_prefix_samples']==18 and all(math.isfinite(v) for v in result.values())
    with pytest.raises(ValueError,match='cached'):
        update_metta_ppo(policy,optimizer,r,replay_chunk_length=2)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_exact_checkpoint_next_update_recovery(tmp_path,device):
    if device=='cuda' and os.environ.get('FLUFFY_META_RL_CUDA_QUALIFICATION')!='1':pytest.skip('Allocated CUDA qualification only')
    torch.manual_seed(51);policy=rollout_policy(device);optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    sampler=np.random.default_rng(71);envs=[fake_env(goal_visible=True) for _ in range(2)]
    r=collect_trials(policy,envs,sampler.integers(0,10000,2));update_metta_ppo(policy,optimizer,r,epochs=1,minibatch_trials=1)
    policy.core.observe_after_update(r['observations'])
    path=tmp_path/'complete.pt';contract={'method':'metta-test'}
    save_checkpoint(path,policy=policy,optimizer=optimizer,environment_rng=sampler,iteration=1,counters={'steps':12},contract=contract)
    seeds=sampler.integers(0,10000,2);r=collect_trials(policy,envs,seeds);a=update_metta_ppo(policy,optimizer,r,epochs=1,minibatch_trials=1)
    expected={k:v.detach().clone() for k,v in policy.state_dict().items()}
    other=rollout_policy(device);opt=torch.optim.Adam(other.parameters(),lr=3e-4);rng=np.random.default_rng(0)
    assert restore_checkpoint(path,policy=other,optimizer=opt,environment_rng=rng,contract=contract)==(1,{'steps':12})
    np.testing.assert_array_equal(rng.integers(0,10000,2),seeds)
    replay=collect_trials(other,envs,seeds)
    for key in r:torch.testing.assert_close(r[key],replay[key],rtol=0,atol=0)
    b=update_metta_ppo(other,opt,replay,epochs=1,minibatch_trials=1);assert a==b
    for key,value in other.state_dict().items():torch.testing.assert_close(value,expected[key],rtol=0,atol=0)


@pytest.mark.skipif(os.environ.get('FLUFFY_METAWORLD_REAL_QUALIFICATION')!='1',reason='Allocated real simulator only')
def test_real_ml10_episode_summary_and_likelihood_replay():
    from fluffy_bassoon.meta_rl.metaworld_env import benchmark_pools,make_lanes
    envs=make_lanes(benchmark_pools('ml10')['train'],2,benchmark='ml10',attempts=2,horizon=500)
    try:
        policy=MettaActorCritic(envs[0]).cuda()
        r=collect_trials(policy,envs,[101,102],record_states=True)
        with torch.no_grad():log,_,value=policy.evaluate_sequence(r['observations'],r['raw_actions'],r['resets'])
        torch.testing.assert_close(log,r['log_probs'],rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(value,r['values'],rtol=2e-5,atol=2e-5)
        assert r['attempt_ends'].sum().item()==4 and all(len(e.outcomes)==2 for e in envs)
        assert r['attempt_states']['summaries'][:,0].tolist()==[0,1]
        assert r['attempt_states']['layer0_position'][:,0].tolist()==[116,116]
    finally:
        for env in envs:env.close()


@pytest.mark.skipif(os.environ.get('FLUFFY_METTA_DEFAULT_CAPACITY')!='1',reason='Allocated default network capacity gate only')
def test_default_width128_two_layers_chunk128_forward_backward_capacity():
    torch.manual_seed(61);env=fake_env(goal_visible=True);policy=MettaActorCritic(env).cuda()
    obs=torch.randn(128,10,policy.core.input_dim,device='cuda');obs[:,:,policy.core.memory.fields['attempt_end']]=0
    actions=torch.zeros(128,10,4,device='cuda')
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=perf_counter()
    log,_,value=policy.evaluate_sequence(obs,actions)
    (log.mean()+value.square().mean()).backward();torch.cuda.synchronize()
    assert torch.isfinite(log).all() and torch.isfinite(value).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    receipt=dict(default_parameters=sum(p.numel() for p in policy.parameters()),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 elapsed_seconds=perf_counter()-started,batch=10,time=128,width=128,layers=2,
                 chunk_length=128,activation_checkpoint_length=32,device=torch.cuda.get_device_name())
    print(receipt)
    if os.environ.get('FLUFFY_METTA_CAPACITY_OUTPUT'):
        Path(os.environ['FLUFFY_METTA_CAPACITY_OUTPUT']).write_text(json.dumps(receipt,indent=2)+'\n')
