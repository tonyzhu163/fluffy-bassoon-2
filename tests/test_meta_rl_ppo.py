"""Full-trial credit and deterministic recurrent likelihood replay for PPO."""

from copy import deepcopy
import os

import numpy as np
import pytest
import torch
from gymnasium.spaces.utils import flatdim, unflatten
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from fluffy_bassoon.meta_rl.cmm import CMMCore
from fluffy_bassoon.meta_rl.transformer_xl import TransformerXLCore
from fluffy_bassoon.meta_rl.environments import BernoulliBandit, HiddenGainReach, HiddenRewardBandit
from fluffy_bassoon.meta_rl.ppo import (
    GRUCore, LoopedTransformerCore, PersistentCTMVectorCore, RecurrentActorCritic, collect_trials,
    generalized_advantages, update_ppo,
)


def actor(env, kind="gru"):
    inputs = int(flatdim(env.observation_space))
    core = (GRUCore(inputs, 8) if kind == "gru" else
            TransformerXLCore(inputs, width=8, memory_length=4, heads=2) if kind == "transformer_xl" else
            LoopedTransformerCore(inputs, width=8, ticks=2, memory_tokens=2) if kind == "looped_transformer" else
            PersistentCTMVectorCore(inputs, width=8, ticks=2, history_length=3, synapse_depth=2, heads=2,
                                    bounded_synchronisation=True) if kind == "persistent_ctm" else
            CMMCore(inputs, width=8, stm_length=3, ltm_tokens=0 if kind == "ctm" else 2,
                    layers=0 if kind == "ctm" else 1, heads=2, synapse_depth=2))
    return RecurrentActorCritic(core, env.action_space)


def test_later_attempt_rewards_reach_earlier_attempt_actions():
    rewards = torch.tensor([[0.], [1.], [0.], [1.]])
    values = torch.zeros_like(rewards)
    trial_ends = torch.tensor([[False], [False], [False], [True]])
    attempt_ends = torch.tensor([[False], [True], [False], [True]])
    advantages, returns = generalized_advantages(rewards, values, trial_ends)
    torch.testing.assert_close(advantages[:, 0], torch.tensor([2., 2., 1., 1.]))
    torch.testing.assert_close(returns, advantages)
    ordinary, _ = generalized_advantages(rewards, values, attempt_ends)
    torch.testing.assert_close(ordinary[:, 0], torch.ones(4))


def test_nonterminal_returns_require_and_use_bootstrap():
    rewards, values = torch.zeros(2, 1), torch.zeros(2, 1)
    ends = torch.zeros(2, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="bootstrap"):
        generalized_advantages(rewards, values, ends)
    _, returns = generalized_advantages(rewards, values, ends, gamma=.5, bootstrap=torch.tensor([3.]))
    torch.testing.assert_close(returns[:, 0], torch.tensor([.75, 1.5]))


@pytest.mark.parametrize("kind", ["gru", "ctm", "persistent_ctm", "cmm", "looped_transformer", "transformer_xl"])
@pytest.mark.parametrize("factory", [HiddenRewardBandit, lambda: BernoulliBandit(attempts=8, arms=3),
                                     lambda: HiddenGainReach(attempts=2, horizon=2)])
def test_collected_likelihoods_and_values_replay_without_updates(kind, factory):
    torch.manual_seed(8)
    envs = [factory() for _ in range(3)]
    policy = actor(envs[0], kind).eval()
    rollout = collect_trials(policy, envs, [10, 11, 12])
    log_probs, _, values = policy.evaluate_sequence(
        rollout["observations"], rollout["raw_actions"], rollout["resets"])
    torch.testing.assert_close(log_probs, rollout["log_probs"])
    torch.testing.assert_close(values, rollout["values"])
    assert rollout["trial_ends"][-1].all()
    assert not rollout["trial_ends"][:-1].any()
    assert not rollout["observations"].requires_grad
    if not policy.discrete:
        assert rollout["raw_actions"].is_floating_point()


def test_gaussian_likelihood_includes_tanh_change_of_variables():
    env = HiddenGainReach()
    policy = actor(env).double().eval()
    with torch.no_grad():
        policy.actor.weight.zero_()
        policy.actor.bias.copy_(torch.tensor([.2, -.3], dtype=torch.float64))
    observation = torch.zeros(2, policy.core.input_dim, dtype=torch.float64)
    raw = torch.tensor([[-2., .4], [.7, 1.3]], dtype=torch.float64)
    action, _, log_prob, _, _, _ = policy.step(observation, raw_action=raw)
    base = Normal(policy.actor.bias, policy.log_std.exp())
    transformed = TransformedDistribution(base, [TanhTransform()])
    torch.testing.assert_close(log_prob, transformed.log_prob(action).sum(-1))
    _, _, extreme, *_ = policy.step(observation, raw_action=torch.full_like(raw, 50.))
    assert extreme.isfinite().all()
    assert action.abs().max() <= 1


@pytest.mark.parametrize("kind", ["gru", "ctm", "cmm", "looped_transformer", "transformer_xl"])
def test_attempt_reset_cuts_only_the_selected_lane_gradient(kind):
    torch.manual_seed(7)
    policy = actor(HiddenRewardBandit(), kind).double().eval()
    observations = torch.randn(3, 2, policy.core.input_dim, dtype=torch.float64, requires_grad=True)
    actions = torch.zeros(3, 2, dtype=torch.long)
    resets = torch.zeros(3, 2, dtype=torch.bool)
    resets[1, 0] = True
    _, _, values = policy.evaluate_sequence(observations, actions, resets)
    values[-1].sum().backward()
    assert observations.grad[0, 0].abs().sum() == 0
    assert observations.grad[0, 1].abs().sum() > 0


@pytest.mark.parametrize("kind", ["gru", "ctm", "cmm", "looped_transformer", "transformer_xl"])
@pytest.mark.parametrize("offset", [0., .5])
def test_ppo_updates_whole_sequences_and_reports_actual_work(kind, offset):
    torch.manual_seed(9)
    envs = [HiddenRewardBandit() for _ in range(4)]
    policy = actor(envs[0], kind)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    rollout = collect_trials(policy, envs, [1, 2, 3, 4])
    raw_rewards = rollout["rewards"].clone()
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    metrics = update_ppo(policy, optimizer, rollout, epochs=2, minibatch_trials=2, reward_offset=offset)
    torch.testing.assert_close(rollout["rewards"], raw_rewards, rtol=0, atol=0)
    assert metrics["optimizer_steps"] == 4
    assert metrics["forward_samples"] == 2 * rollout["rewards"].numel()
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(value, before[key]) for key, value in policy.state_dict().items())


def test_episode_mode_preserves_the_recorded_attempt_reset_mask():
    envs = [HiddenRewardBandit() for _ in range(2)]
    policy = actor(envs[0])
    rollout = collect_trials(policy, envs, [1, 2], reset_each_attempt=True)
    assert not rollout["resets"][0].any() and rollout["resets"][1:].all()
    log_probs, _, values = policy.evaluate_sequence(
        rollout["observations"], rollout["raw_actions"], rollout["resets"])
    torch.testing.assert_close(log_probs, rollout["log_probs"])
    torch.testing.assert_close(values, rollout["values"])


def test_reward_blind_collector_zeros_only_feedback_and_keeps_true_scoring():
    envs = [HiddenRewardBandit(attempts=6) for _ in range(3)]
    envs[0].reset(seed=3)
    rewarded_action = envs[0]._target
    policy = actor(envs[0])
    with torch.no_grad():
        policy.actor.weight.zero_()
        policy.actor.bias.zero_()
        policy.actor.bias[rewarded_action] = 1.
    normal = collect_trials(policy, envs, [3, 4, 5], deterministic=True)
    blind = collect_trials(policy, envs, [3, 4, 5], deterministic=True, reward_blind=True)
    torch.testing.assert_close(normal["raw_actions"], blind["raw_actions"])
    torch.testing.assert_close(normal["rewards"], blind["rewards"])
    assert normal["rewards"].sum() > 0
    changed = False
    for step in range(6):
        for lane in range(3):
            a = unflatten(envs[0].observation_space, normal["observations"][step, lane].numpy())
            b = unflatten(envs[0].observation_space, blind["observations"][step, lane].numpy())
            assert b["previous_reward"].item() == 0
            changed |= a["previous_reward"].item() > 0
            for key in a.keys() - {"previous_reward"}:
                np.testing.assert_array_equal(a[key], b[key])
    assert changed


@pytest.mark.parametrize("attempts", [1, 4])
def test_collector_scales_clock_by_public_bounds_without_changing_environment(attempts):
    env = HiddenGainReach(attempts=attempts, horizon=2)
    rollout = collect_trials(actor(env), [env], [12])
    for step, vector in enumerate(rollout["observations"][:, 0].numpy()):
        clock = unflatten(env.observation_space, vector)["clock"]
        np.testing.assert_allclose(clock, [(step // 2) / max(attempts - 1, 1), (step % 2) / 2])
    np.testing.assert_array_equal(env.observation_space["clock"].high, [attempts - 1, 2])


def test_constant_reward_centering_is_equivalent_to_a_known_finite_horizon_value_offset():
    rewards = torch.tensor([[0., 1.], [1., 0.], [1., 1.], [0., 0.]])
    values = torch.tensor([[.2, .3], [.4, .2], [.1, .7], [.8, .4]])
    ends = torch.tensor([[False, False], [False, True], [False, False], [True, True]])
    potential = torch.tensor([[2., 1.], [1.5, .5], [1., 1.], [.5, .5]])
    raw_advantages, raw_returns = generalized_advantages(rewards, values, ends)
    shifted_advantages, shifted_returns = generalized_advantages(rewards - .5, values - potential, ends)
    torch.testing.assert_close(shifted_advantages, raw_advantages)
    torch.testing.assert_close(shifted_returns, raw_returns - potential)


def test_legacy_raw_clock_evaluation_preserves_the_original_observation_encoding():
    env = HiddenRewardBandit(attempts=8)
    rollout = collect_trials(actor(env), [env], [12], normalize_clock=False)
    for step, vector in enumerate(rollout["observations"][:, 0].numpy()):
        np.testing.assert_array_equal(unflatten(env.observation_space, vector)["clock"], [step, 0])


@pytest.mark.parametrize("offset", [float("nan"), -.1, 1.1])
def test_reward_centering_rejects_invalid_offsets(offset):
    with pytest.raises(ValueError, match="optimization"):
        update_ppo(None, None, {}, reward_offset=offset)


def test_an_action_mask_cannot_remove_every_action_of_a_trial():
    rollout = dict(rewards=torch.zeros(1, 1), values=torch.zeros(1, 1),
                   trial_ends=torch.ones(1, 1, dtype=torch.bool), action_mask=torch.zeros(1, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="action mask"):
        update_ppo(None, None, rollout)


def test_zero_value_coefficient_keeps_the_critic_head_fixed_while_the_actor_learns():
    torch.manual_seed(9)
    envs = [HiddenRewardBandit() for _ in range(4)]
    policy = actor(envs[0])
    before_critic = {key: value.clone() for key, value in policy.critic.state_dict().items()}
    before_actor = {key: value.clone() for key, value in policy.actor.state_dict().items()}
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=0.)
    update_ppo(policy, optimizer, collect_trials(policy, envs, [1, 2, 3, 4]),
                epochs=1, minibatch_trials=4, value_coef=0., entropy_coef=0.)
    for key, value in policy.critic.state_dict().items():
        torch.testing.assert_close(value, before_critic[key], rtol=0, atol=0)
    assert any(not torch.equal(value, before_actor[key]) for key, value in policy.actor.state_dict().items())


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    os.environ.get('FLUFFY_META_RL_CUDA_QUALIFICATION')!='1', reason='Requires probed Slurm CUDA step'))])
def test_single_trial_raw_advantages_match_terminal_reward_policy_gradient(device):
    torch.manual_seed(41)
    env=HiddenGainReach(attempts=1,horizon=3)
    policy=actor(env).to(device)
    with torch.no_grad():
        policy.critic.weight.zero_();policy.critic.bias.zero_()
    rollout=collect_trials(policy,[env],[17])
    # Analytic one-trajectory terminal-return case, independent of simulator success.
    rollout['rewards'].zero_();rollout['rewards'][-1]=1
    manual,centered=deepcopy(policy),deepcopy(policy)
    before=policy.actor.weight.detach().clone()
    optimizer=torch.optim.SGD(policy.parameters(),lr=.001)
    manual_optimizer=torch.optim.SGD(manual.parameters(),lr=.001)
    log_prob,_,_=manual.evaluate_sequence(rollout['observations'],rollout['raw_actions'],rollout['resets'])
    # At the behavior weights, the PPO ratio has the same score-function gradient.
    expected=-log_prob.mean()
    manual_optimizer.zero_grad();expected.backward();manual_optimizer.step()
    metrics=update_ppo(policy,optimizer,rollout,epochs=1,minibatch_trials=1,
                       value_coef=0.,entropy_coef=0.,max_grad_norm=1e6,normalize_advantages=False)
    assert metrics['loss']==pytest.approx(-1.,abs=1e-6)
    assert not torch.equal(policy.actor.weight,before)
    for key,value in policy.state_dict().items():
        torch.testing.assert_close(value,manual.state_dict()[key],rtol=1e-6,atol=1e-7)
    update_ppo(centered,torch.optim.SGD(centered.parameters(),lr=.001),rollout,
               epochs=1,minibatch_trials=1,value_coef=0.,entropy_coef=0.)
    torch.testing.assert_close(centered.actor.weight,before,rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    os.environ.get('FLUFFY_META_RL_CUDA_QUALIFICATION')!='1',reason='Requires probed allocated CUDA'))])
def test_bounded_entropy_restores_mean_gradient_at_saturation_and_preserves_likelihood(device):
    if device=='cuda':
        from fluffy_bassoon.gpu_execution import validate_execution
        validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION','slurm'),os.environ)
        assert torch.cuda.device_count()==1
    policy=actor(HiddenGainReach()).to(device).double().eval()
    with torch.no_grad():
        policy.actor.weight.zero_();policy.actor.bias.fill_(4.);policy.log_std.fill_(np.log(.5))
    observations=torch.zeros(64,policy.core.input_dim,device=device,dtype=torch.float64)
    actions=torch.zeros(64,2,device=device,dtype=torch.float64)
    original=policy.step(observations,raw_action=actions)
    torch.manual_seed(83)
    bounded=policy.step(observations,raw_action=actions,bounded_entropy=True)
    for index in (0,1,2,4):
        torch.testing.assert_close(original[index],bounded[index],rtol=0,atol=0)
    assert bounded[3].mean() < original[3].mean()-10
    bounded[3].mean().backward()
    assert (policy.actor.bias.grad < -1.9).all()
    assert policy.log_std.grad.isfinite().all()
    envs=[HiddenGainReach(attempts=2,horizon=2) for _ in range(2)]
    rollout=collect_trials(policy,envs,[1,2])
    metrics=update_ppo(policy,torch.optim.AdamW(policy.parameters(),lr=1e-3),rollout,
        epochs=1,minibatch_trials=2,bounded_entropy=True)
    assert metrics['forward_samples']==8 and all(np.isfinite(v) for v in metrics.values())


def test_bounded_entropy_uses_fresh_reparameterized_policy_sample():
    policy=actor(HiddenGainReach()).double().eval()
    with torch.no_grad():
        policy.actor.weight.zero_();policy.actor.bias.copy_(torch.tensor([.2,-.3]))
    observations=torch.zeros(3,policy.core.input_dim,dtype=torch.float64)
    torch.manual_seed(91)
    result=policy.step(observations,raw_action=torch.zeros(3,2,dtype=torch.float64),bounded_entropy=True)
    base=Normal(policy.actor.bias.expand(3,-1),policy.log_std.exp())
    torch.manual_seed(91)
    sample=base.rsample()
    expected=base.entropy().sum(-1)+TanhTransform().log_abs_det_jacobian(sample,sample.tanh()).sum(-1)
    torch.testing.assert_close(result[3],expected,rtol=0,atol=0)
    with pytest.raises(ValueError,match='continuous'):
        actor(HiddenRewardBandit()).step(torch.zeros(1,int(flatdim(HiddenRewardBandit().observation_space))),bounded_entropy=True)
