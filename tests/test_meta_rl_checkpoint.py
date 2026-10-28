"""A between-trial resume must reproduce the next sampled trial and PPO update."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from gymnasium.spaces.utils import flatdim

from fluffy_bassoon.meta_rl.checkpoint import restore_checkpoint, save_checkpoint
from fluffy_bassoon.meta_rl.environments import HiddenRewardBandit
from fluffy_bassoon.meta_rl.transformer_xl import TransformerXLCore
from fluffy_bassoon.meta_rl.ppo import GRUCore, LoopedTransformerCore, RecurrentActorCritic, collect_trials, update_ppo


def learner(env, kind="gru"):
    factory = {"gru": GRUCore, "looped_transformer": LoopedTransformerCore, "transformer_xl": TransformerXLCore}[kind]
    policy = RecurrentActorCritic(factory(int(flatdim(env.observation_space)), 8), env.action_space)
    return policy, torch.optim.AdamW(policy.parameters(), lr=1e-3)


@pytest.mark.parametrize("kind", ["gru", "looped_transformer", "transformer_xl"])
def test_resume_reproduces_next_trial_actions_optimizer_and_parameters(tmp_path, kind):
    torch.manual_seed(21)
    envs = [HiddenRewardBandit() for _ in range(4)]
    policy, optimizer = learner(envs[0], kind)
    environment_rng = np.random.default_rng(22)
    contract = {"source_commit": "a" * 40, "width": 8, "mode": "meta"}
    seeds = environment_rng.integers(0, 1_000_000_000, size=4)
    update_ppo(policy, optimizer, collect_trials(policy, envs, seeds), epochs=1, minibatch_trials=4)
    checkpoint = tmp_path / "latest.pt"
    digest = save_checkpoint(checkpoint, policy=policy, optimizer=optimizer,
                             environment_rng=environment_rng, iteration=1,
                             counters={"env_steps": 16}, contract=contract)
    assert len(digest) == 64 and not checkpoint.with_suffix(".partial").exists()
    expected_seeds = environment_rng.integers(0, 1_000_000_000, size=4)
    expected_rollout = collect_trials(policy, envs, expected_seeds)
    expected_metrics = update_ppo(policy, optimizer, expected_rollout, epochs=1, minibatch_trials=4)
    expected_model, expected_optimizer = deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict())
    restored, restored_optimizer = learner(envs[0], kind)
    restored_rng = np.random.default_rng(999)
    iteration, counters = restore_checkpoint(checkpoint, policy=restored, optimizer=restored_optimizer,
                                              environment_rng=restored_rng, contract=contract)
    assert iteration == 1 and counters == {"env_steps": 16}
    seeds = restored_rng.integers(0, 1_000_000_000, size=4)
    np.testing.assert_array_equal(seeds, expected_seeds)
    rollout = collect_trials(restored, envs, seeds)
    for key in rollout:
        torch.testing.assert_close(rollout[key], expected_rollout[key], rtol=0, atol=0)
    metrics = update_ppo(restored, restored_optimizer, rollout, epochs=1, minibatch_trials=4)
    assert metrics == expected_metrics
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected_model[key], rtol=0, atol=0)
    for parameter, state in restored_optimizer.state_dict()["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, expected_optimizer["state"][parameter][key], rtol=0, atol=0)


def test_incompatible_contract_is_rejected_before_loading_model(tmp_path):
    env = HiddenRewardBandit()
    policy, optimizer = learner(env)
    rng = np.random.default_rng(0)
    contract = {"source_commit": "b" * 40}
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(checkpoint, policy=policy, optimizer=optimizer, environment_rng=rng,
                    iteration=0, counters={}, contract=contract)
    before = deepcopy(policy.state_dict())
    with pytest.raises(ValueError, match="contract"):
        restore_checkpoint(checkpoint, policy=policy, optimizer=optimizer, environment_rng=rng,
                           contract={"source_commit": "c" * 40})
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize('bad',[None,'checksum','mode','horizon','tensor','tensor_dtype','counters','missing_counters','chain'])
def test_motor_initialization_preserves_actor_and_rng_but_resets_value_baseline(tmp_path,bad):
    from fluffy_bassoon.meta_rl.checkpoint import initialize_from_motor_checkpoint
    from fluffy_bassoon.evaluation_stages import sha256
    from fluffy_bassoon.meta_rl.environments import HiddenGainReach
    env=HiddenGainReach(attempts=4,horizon=4)
    parent=RecurrentActorCritic(GRUCore(int(flatdim(env.observation_space)),8),env.action_space)
    optimizer=torch.optim.AdamW(parent.parameters(),lr=.001)
    contract=dict(source_commit='a'*40,mode='episode',task='reach',attempts=4,horizon=4,
        observation_encoding='clock_by_declared_max_v1',arms=None,core='gru',
        core_config=dict(width=8,ticks=1),dtype='float32')
    target_contract=dict(contract,source_commit='b'*40,mode='meta')
    path=tmp_path/'motor.pt'
    save_checkpoint(path,policy=parent,optimizer=optimizer,environment_rng=np.random.default_rng(2),
        iteration=2,counters=dict(environment_steps=1024,rollout_forward_samples=1024,update_forward_samples=4096,optimizer_steps=16),contract=contract)
    if bad not in (None,'checksum'):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if bad=='mode':payload['contract']['mode']='meta'
        elif bad=='horizon':payload['contract']['horizon']=8
        elif bad=='tensor':payload['policy']['actor.weight'].fill_(float('nan'))
        elif bad=='tensor_dtype':payload['policy']['actor.weight']=payload['policy']['actor.weight'].double()
        elif bad=='counters':payload['counters']['environment_steps']=-1
        elif bad=='missing_counters':del payload['counters']['update_forward_samples']
        else:payload['contract']['initialization']={'old_stage':True}
        torch.save(payload,path)
    digest='0'*64 if bad=='checksum' else sha256(path)
    policy=deepcopy(parent)
    with torch.no_grad():policy.actor.weight.zero_()
    before=deepcopy(policy.state_dict());rng=torch.random.get_rng_state().clone()
    if bad:
        with pytest.raises(ValueError):
            initialize_from_motor_checkpoint(path,digest,policy=policy,contract=target_contract)
        for key,value in before.items():torch.testing.assert_close(policy.state_dict()[key],value,rtol=0,atol=0)
        return
    result=initialize_from_motor_checkpoint(path,digest,policy=policy,contract=target_contract)
    assert result['pretraining_counters']==dict(environment_steps=1024,rollout_forward_samples=1024,update_forward_samples=4096,optimizer_steps=16)
    assert result['source_iteration']==2 and result['checkpoint_sha256']==sha256(path)==digest
    assert torch.equal(torch.random.get_rng_state(),rng)
    for key,value in parent.state_dict().items():
        if key.startswith('critic.'):
            assert not policy.state_dict()[key].any()
        else:torch.testing.assert_close(policy.state_dict()[key],value,rtol=0,atol=0)
