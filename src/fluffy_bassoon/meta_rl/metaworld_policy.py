"""Common state bypass, normalization and actor/critic heads for all cores."""

import torch
from torch import nn

from .cores import build_policy, policy_size


class StateContextCore(nn.Module):
    def __init__(self, memory, state_dim=39):
        super().__init__()
        self.memory, self.state_dim = memory, state_dim
        self.input_dim, self.output_dim = memory.input_dim, memory.output_dim + state_dim
        self.ticks = memory.ticks
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_var", torch.ones(state_dim))
        self.register_buffer("state_count", torch.tensor(1e-4, dtype=torch.float64))

    def initial_state(self, batch):
        return self.memory.initial_state(batch)

    def forward(self, observation, state=None):
        # Gymnasium Dict flattening sorts keys; 'state' is the final field.
        current = ((observation[..., -self.state_dim:] - self.state_mean)
                   / torch.sqrt(self.state_var + 1e-8)).clamp(-10, 10)
        normalized = torch.cat((observation[..., :-self.state_dim], current), -1)
        context, state = self.memory(normalized, state)
        return torch.cat((current, context), -1), state

    def constrain_parameters(self):
        if hasattr(self.memory, "constrain_parameters"):
            self.memory.constrain_parameters()

    @torch.no_grad()
    def observe_after_update(self, observations, mask=None):
        # Hold statistics fixed throughout collection AND its PPO replay.
        values = observations[..., -self.state_dim:].reshape(-1, self.state_dim).double()
        if mask is not None:
            values = values[mask.reshape(-1)]
            if not len(values):
                raise ValueError("State normalization needs real transitions")
        n = values.shape[0]
        mean, var = values.mean(0), values.var(0, unbiased=False)
        old_n = self.state_count
        total = old_n + n
        delta = mean - self.state_mean.double()
        combined = (self.state_var.double() * old_n + var * n
                    + delta.square() * old_n * n / total) / total
        self.state_mean.copy_(self.state_mean.double() + delta * n / total)
        self.state_var.copy_(combined)
        self.state_count.copy_(total)


def make_policy(core, config, env):
    policy = build_policy(core, config, env)
    policy.core = StateContextCore(policy.core, env.observation_space["state"].shape[0])
    width = policy.core.output_dim
    def head(outputs):
        return nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                             nn.Linear(64, outputs))
    policy.actor, policy.critic = head(env.action_space.shape[0]), head(1)
    nn.init.orthogonal_(policy.actor[-1].weight, gain=.01)
    nn.init.zeros_(policy.actor[-1].bias)
    policy.log_std.data.zero_()
    return policy


def match_policy(core, settings, env, budget=250_000):
    selected = None
    for width in range(16, 513, 4):
        config = dict(settings, width=width)
        policy = make_policy(core, config, env)
        size = policy_size(policy)
        if size["parameters"] > budget:
            break
        selected = config, size
    if selected is None or selected[1]["parameters"] < .85 * budget:
        raise ValueError("Cannot match the declared parameter budget")
    return selected
