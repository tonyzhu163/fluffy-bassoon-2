"""Causal, budget and recovery contracts; run on allocated Linux only."""
from copy import deepcopy

import numpy as np
import torch

from fluffy_bassoon.meta_rl.vla_noise_ppo import PPOConfig, RecurrentNoisePPO
from fluffy_bassoon.meta_rl.ppo import generalized_advantages


class FakeNoiseEnv:
    observation_dim=2
    noise_shape=(1,2)
    native_horizon=2

    def reset(self,task_name,seed):
        self.seed=seed; self.steps=0
        return np.array([seed%7,0],np.float32)

    def step(self,noise,*,max_primitive_steps=None):
        count=min(2,5-self.steps)
        if max_primitive_steps is not None: count=min(count,max_primitive_steps)
        self.steps+=count
        end=self.steps==5
        cut=max_primitive_steps is not None and count>=max_primitive_steps and not end
        reward=float(end)
        return (np.array([self.seed%7,self.steps],np.float32),reward,end,cut,
                dict(primitive_steps=count,success=end,budget_truncated=cut))


def learner(reset=False,seed=4):
    return RecurrentNoisePPO(2,(1,2),device="cpu",seed=seed,config=PPOConfig(
        width=8,layers=1,heads=2,memory_length=8,epochs=1,minibatch_trials=1,
        train_episodes=2,episode_limit=5,reset_each_episode=reset))


def test_initialization_seed_is_private_and_reset_control_matches_weights():
    torch.manual_seed(137)
    before=torch.get_rng_state().clone()
    left=learner(False,9)
    torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
    right=learner(True,9)
    for key,value in left.policy.state_dict().items():
        torch.testing.assert_close(value,right.policy.state_dict()[key],rtol=0,atol=0)


def test_trial_return_crosses_episode_boundary_and_old_likelihood_replays():
    model=learner()
    rollout,result=model.collect_trial(FakeNoiseEnv(),"scan_object",[1,2])
    assert result["primitive_steps"]==10 and len(result["episodes"])==2
    assert rollout["trial_ends"].flatten().tolist()==[False]*5+[True]
    assert rollout["attempt_ends"].flatten().tolist()==[False,False,True]*2
    errors=model.replay_error(rollout)
    assert errors["log_prob_max_error"]==errors["value_max_error"]==0
    rewards=torch.tensor([[0.],[0.],[0.],[1.]])
    ends=torch.tensor([[False],[False],[False],[True]])
    advantages,_=generalized_advantages(rewards,torch.zeros_like(rewards),ends,gamma=1,gae_lambda=1)
    torch.testing.assert_close(advantages,torch.ones_like(advantages))
    before=deepcopy(model.policy.state_dict())
    metrics=model.update([rollout])
    assert metrics["optimizer_steps"]==1 and np.isfinite(list(metrics.values())).all()
    assert any(not torch.equal(value,model.policy.state_dict()[key]) for key,value in before.items())


def test_trained_reset_removes_previous_episode_feedback():
    model=learner(True)
    rollout,_=model.collect_trial(FakeNoiseEnv(),"scan_object",[1,1])
    first,second=rollout["observations"][0,0],rollout["observations"][3,0]
    # Public trial phase is retained in both policies; previous-outcome feedback
    # and the recurrent cache are cleared by this trained reset control.
    torch.testing.assert_close(torch.cat((first[:6],first[7:])),
                               torch.cat((second[:6],second[7:])),rtol=0,atol=0)
    assert first[6]==0 and second[6]==1
    assert rollout["resets"].flatten().tolist()==[True,False,False]*2


def test_cache_only_lesion_is_distinct_from_trained_feedback_reset():
    model=learner(False)
    rollout,result=model.collect_trial(FakeNoiseEnv(),"scan_object",[1,1],reset_memory=True)
    assert float(rollout["observations"][3,0,4])==1  # prior reward remains public
    assert not result["episodes"][1]["reset_episode_feedback"]
    assert rollout["resets"][3,0]


def test_exact_external_budget_cut_bootstraps_and_is_not_terminal_task():
    model=learner()
    rollout,result=model.collect_trial(FakeNoiseEnv(),"scan_object",[1,2],primitive_budget=7)
    assert result["primitive_steps"]==7 and result["budget_truncated"]
    assert result["episodes"][-1]["primitive_steps"]==2
    assert not rollout["trial_ends"].any()
    assert rollout["bootstrap"].shape==(1,)
    assert np.isfinite(rollout["bootstrap"]).all()
    model.update([rollout])


def test_full_recovery_reproduces_next_trial_and_optimizer_update():
    model=learner()
    first,_=model.collect_trial(FakeNoiseEnv(),"scan_object",[1,2])
    model.update([first])
    saved=deepcopy(model.state_dict())
    expected,expected_result=model.collect_trial(FakeNoiseEnv(),"scan_object",[3,4])
    expected_metrics=model.update([expected])
    expected_weights=deepcopy(model.policy.state_dict())
    restored=learner()
    restored.load_state_dict(saved)
    actual,actual_result=restored.collect_trial(FakeNoiseEnv(),"scan_object",[3,4])
    assert actual_result==expected_result
    for key in expected: torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
    assert restored.update([actual])==expected_metrics
    for key,value in expected_weights.items():
        torch.testing.assert_close(value,restored.policy.state_dict()[key],rtol=0,atol=0)
