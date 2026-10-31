"""Independent causal-trial regressions; execute on allocated Linux only."""
import numpy as np
import torch

from fluffy_bassoon.meta_rl.ppo import generalized_advantages
from fluffy_bassoon.meta_rl.vla_noise_ppo import PPOConfig, RecurrentNoisePPO
from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv


def policy(*, reset=False, episode_limit=1):
    config = PPOConfig(width=8, layers=1, heads=2, memory_length=8,
                       train_episodes=4, episode_limit=episode_limit,
                       epochs=1, minibatch_trials=2,
                       gamma=1., gae_lambda=1., reset_each_episode=reset)
    return RecurrentNoisePPO(3, (1, 2), config=config, device="cpu", seed=17)


class SingleStepEpisodes:
    native_horizon = 1

    def __init__(self, rewards):
        self.rewards = rewards
        self.episode = -1

    def reset(self, task, seed):
        self.episode += 1
        return np.zeros(3, np.float32)

    def step(self, noise):
        reward = self.rewards[self.episode]
        return np.ones(3, np.float32), reward, bool(reward), not bool(reward), dict(
            primitive_steps=1, success=bool(reward), budget_truncated=False)


def test_same_constructor_seed_fixes_initial_weights_despite_caller_rng():
    torch.manual_seed(99)
    first = policy()
    torch.manual_seed(1001)
    second = policy()
    for name, value in first.policy.state_dict().items():
        torch.testing.assert_close(value, second.policy.state_dict()[name], rtol=0, atol=0)


def test_last_attempt_reward_credits_earlier_attempts():
    learner = policy()
    rollout, _ = learner.collect_trial(SingleStepEpisodes([0., 0., 0., 1.]),
                                       "handover_mic", [1, 2, 3, 4], deterministic=True)
    assert rollout["attempt_ends"].all()
    assert not rollout["trial_ends"][:-1].any()
    assert rollout["trial_ends"][-1].all()
    _, returns = generalized_advantages(rollout["rewards"], rollout["values"],
                                        rollout["trial_ends"], gamma=1., gae_lambda=1.)
    torch.testing.assert_close(returns, torch.ones_like(returns), rtol=0, atol=2e-6)


def test_trained_episode_reset_removes_direct_previous_reward_feedback():
    learner = policy(reset=True)
    positive, _ = learner.collect_trial(SingleStepEpisodes([1., 0., 0., 0.]),
                                        "handover_mic", [1, 2, 3, 4], deterministic=True)
    negative, _ = learner.collect_trial(SingleStepEpisodes([0., 0., 0., 0.]),
                                        "handover_mic", [1, 2, 3, 4], deterministic=True)
    # A reset control must remove explicit previous action/reward channels,
    # not merely empty attention memories before a feedback-bearing first token.
    torch.testing.assert_close(positive["observations"][1], negative["observations"][1],
                               rtol=0, atol=0)
    torch.testing.assert_close(positive["observations"][1, 0, 3:6], torch.zeros(3),
                               rtol=0, atol=0)


class NaturalHorizonBackend:
    observation_dim = 3
    noise_shape = (2, 2)

    def __init__(self):
        self.reset_seeds = []

    def reset(self, task, seed):
        self.reset_seeds.append(seed)
        return np.asarray([seed, 0., 0.], np.float32), dict(requested_seed=seed)

    def step(self, noise, action_count):
        return np.full(3, 99., np.float32), 0., False, False, dict(
            primitive_steps=action_count, success=False)

    def close(self):
        pass


def test_budget_coincident_with_episode_end_bootstraps_from_next_episode():
    backend = NaturalHorizonBackend()
    env = RobotwinNoiseEnv(backend=backend, episode_limit=2)
    config = PPOConfig(width=8, layers=1, heads=2, memory_length=8,
                       train_episodes=4, episode_limit=2)
    learner = RecurrentNoisePPO(3, (2, 2), config=config, device="cpu", seed=17)
    rollout, result = learner.collect_trial(env, "handover_mic", [1, 2, 3, 4],
                                             deterministic=True, primitive_budget=2)
    assert result["primitive_steps"] == 2 and result["budget_truncated"]
    assert backend.reset_seeds == [1, 2]
    assert not rollout["trial_ends"].any() and "bootstrap" in rollout
