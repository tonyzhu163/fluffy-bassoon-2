"""Generalized advantage estimates for the residual-policy baselines."""
import torch


def generalized_advantages(rewards, values, ends, *, gamma=1., gae_lambda=1., bootstrap=None):
    if (rewards.ndim != 2 or not all(rewards.shape) or rewards.shape != values.shape
            or ends.shape != rewards.shape or ends.dtype != torch.bool
            or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1):
        raise ValueError("Require matching [time, batch] rewards/values/boolean ends and discounts in [0, 1]")
    if bootstrap is None:
        if not ends[-1].all():
            raise ValueError("Nonterminal rollout endings require bootstrap values")
        bootstrap = torch.zeros_like(values[-1])
    if bootstrap.shape != values.shape[1:]:
        raise ValueError("Bootstrap values must have shape [batch]")
    advantages = torch.zeros_like(rewards)
    carry, next_value = torch.zeros_like(bootstrap), bootstrap
    for step in range(rewards.shape[0] - 1, -1, -1):
        alive = (~ends[step]).to(rewards.dtype)
        delta = rewards[step] + gamma * next_value * alive - values[step]
        carry = delta + gamma * gae_lambda * alive * carry
        advantages[step] = carry
        next_value = values[step]
    return advantages, advantages + values
