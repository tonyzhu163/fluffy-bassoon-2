"""Credit assignment across boundaries and nonterminal bootstrap values."""
import pytest
import torch

from fluffy_bassoon.meta_rl.ppo import generalized_advantages


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
