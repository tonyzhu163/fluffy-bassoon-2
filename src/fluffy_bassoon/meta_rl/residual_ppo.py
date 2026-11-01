"""Memoryless PPO over a bounded action residual on a frozen VLA.

This is the ordinary RL baseline and the history source for the in-context
meta-learner: one decision per frozen-policy action chunk, a Gaussian over
the unbounded control ``u``, and ``ResidualBound`` mapping ``u`` to the
executed residual. Rewards are terminal only (simulator success or a learned
rewarder's score), placed on each episode's last chunk.

The rollout loop is written against two small interfaces so it carries no
simulator or VLA code. A motor has ``infer(observation) -> (chunk [H, A],
features [F], noise_seed)`` and runs the frozen VLA without gradients. An
environment has ``reset() -> observation`` and ``step_chunk(chunk [H, A]) ->
(observation, done, info)``. ``terminal_reward(info) -> float`` scores a
finished episode. Real computation and tests belong in allocated Linux jobs.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .ppo import generalized_advantages


@dataclass(frozen=True)
class ResidualPPOConfig:
    feature_dim: int
    residual_dim: int
    width: int = 256
    layers: int = 2
    # Pre-tanh std; tanh(0.5) keeps initial residuals near half the bound at one std.
    initial_std: float = .5
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 64
    clip: float = .2
    gamma: float = 1.
    gae_lambda: float = .95
    entropy_coef: float = 0.
    value_coef: float = .5
    max_grad_norm: float = 1.

    def __post_init__(self):
        counts = (self.feature_dim, self.residual_dim, self.width, self.layers, self.epochs, self.minibatch_size)
        if (any(type(value) is not int or value < 1 for value in counts)
                or not math.isfinite(self.initial_std) or self.initial_std <= 0
                or not 0 < self.clip < 1 or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1
                or self.learning_rate <= 0 or self.max_grad_norm <= 0
                or self.entropy_coef < 0 or self.value_coef < 0):
            raise ValueError("Invalid residual PPO configuration")


def _mlp(input_dim, width, layers, output_dim):
    modules, size = [], input_dim
    for _ in range(layers):
        modules += [nn.Linear(size, width), nn.Tanh()]
        size = width
    return nn.Sequential(*modules, nn.Linear(size, output_dim))


class ResidualActorCritic(nn.Module):
    """Gaussian actor over u and a value critic, on normalized frozen features."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.actor = _mlp(config.feature_dim, config.width, config.layers, config.residual_dim)
        self.critic = _mlp(config.feature_dim, config.width, config.layers, 1)
        # Zero mean at initialization: the deterministic policy reproduces the frozen VLA.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((config.residual_dim,), math.log(config.initial_std)))
        self.register_buffer("feature_mean", torch.zeros(config.feature_dim))
        self.register_buffer("feature_std", torch.ones(config.feature_dim))

    @torch.no_grad()
    def fit_feature_normalization(self, features):
        """Fit once on the first rollout's features; fixed thereafter."""
        features = torch.as_tensor(features, dtype=self.feature_mean.dtype, device=self.feature_mean.device)
        self.feature_mean.copy_(features.reshape(-1, self.config.feature_dim).mean(0))
        self.feature_std.copy_(features.reshape(-1, self.config.feature_dim).std(0, unbiased=False).clamp(min=1e-4))

    def _distribution(self, features):
        normalized = (features - self.feature_mean) / self.feature_std
        mean = self.actor(normalized)
        return Normal(mean, self.log_std.clamp(-5, 2).exp().expand_as(mean)), normalized

    @torch.no_grad()
    def act(self, features, *, deterministic=False, generator=None, action_noise=None):
        distribution, normalized = self._distribution(features)
        mean, std = distribution.mean, distribution.stddev
        if action_noise is not None:
            noise = torch.as_tensor(action_noise, device=mean.device, dtype=mean.dtype)
            if deterministic or noise.shape != mean.shape or not torch.isfinite(noise).all():
                raise ValueError("Explicit rollout noise must match one stochastic actor draw")
        else:
            noise = None if deterministic else torch.randn(
                mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        control = mean if deterministic else mean + std * noise
        return control, distribution.log_prob(control).sum(-1), self.critic(normalized).squeeze(-1)

    def evaluate(self, features, control):
        distribution, normalized = self._distribution(features)
        return (distribution.log_prob(control).sum(-1), distribution.entropy().sum(-1),
                self.critic(normalized).squeeze(-1))


def collect_episodes(policy, bound, motor, environment, terminal_reward, *, episodes,
                     deterministic=False, generator=None):
    """Run complete episodes and return chronological per-chunk records.

    Each record holds the frozen features, base chunk noise seed, control u,
    its log probability and value, the reward (terminal only), done and
    episode start. The base chunk itself is not stored; it is reproducible
    from the motor, observation and noise seed.
    """
    if bound.residual_dim != policy.config.residual_dim or type(episodes) is not int or episodes < 1:
        raise ValueError("Require a bound matching the policy and a positive episode count")
    weight = next(policy.parameters())
    records = []
    for _ in range(episodes):
        observation, done, start = environment.reset(), False, True
        while not done:
            chunk, features, noise_seed = motor.infer(observation)
            features = torch.as_tensor(np.asarray(features, dtype=np.float32), device=weight.device,
                                       dtype=weight.dtype)
            control, log_prob, value = policy.act(features[None], deterministic=deterministic,
                                                  generator=generator)
            control = control[0].cpu().numpy()
            observation, done, info = environment.step_chunk(bound.apply(np.asarray(chunk), control))
            reward = float(terminal_reward(info)) if done else 0.
            if not math.isfinite(reward):
                raise ValueError("Terminal reward must be finite")
            records.append(dict(features=features.cpu().numpy(), noise_seed=noise_seed,
                                control=control.astype(np.float32), log_prob=float(log_prob[0]),
                                value=float(value[0]), reward=reward, done=bool(done), episode_start=start))
            start = False
    return records


def stack_records(records, device=None):
    """Concatenate chronological single-lane records into [T, 1] tensors."""
    if not records or not records[-1]["done"]:
        raise ValueError("Require complete episodes")
    as_tensor = lambda key, dtype: torch.as_tensor(np.asarray([record[key] for record in records]),
                                                  dtype=dtype, device=device)
    return dict(features=as_tensor("features", torch.float32)[:, None],
                controls=as_tensor("control", torch.float32)[:, None],
                log_probs=as_tensor("log_prob", torch.float32)[:, None],
                values=as_tensor("value", torch.float32)[:, None],
                rewards=as_tensor("reward", torch.float32)[:, None],
                ends=as_tensor("done", torch.bool)[:, None])


def update(policy, optimizer, rollout, *, generator=None):
    """Clipped PPO over flattened chunk decisions; returns mean metrics."""
    config = policy.config
    advantages, returns = generalized_advantages(rollout["rewards"], rollout["values"], rollout["ends"],
                                                 gamma=config.gamma, gae_lambda=config.gae_lambda)
    flat = dict(features=rollout["features"].reshape(-1, config.feature_dim),
                controls=rollout["controls"].reshape(-1, config.residual_dim),
                log_probs=rollout["log_probs"].reshape(-1), values=rollout["values"].reshape(-1),
                advantages=advantages.reshape(-1), returns=returns.reshape(-1))
    advantage = flat["advantages"]
    flat["advantages"] = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    size, totals, count = len(advantage), {}, 0
    policy.train()
    for _ in range(config.epochs):
        order = torch.randperm(size, generator=generator).to(advantage.device)
        for begin in range(0, size, config.minibatch_size):
            index = order[begin:begin + config.minibatch_size]
            log_prob, entropy, value = policy.evaluate(flat["features"][index], flat["controls"][index])
            log_ratio = log_prob - flat["log_probs"][index]
            ratio = log_ratio.exp()
            adv = flat["advantages"][index]
            actor_loss = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - config.clip, 1 + config.clip)).mean()
            old_value, target = flat["values"][index], flat["returns"][index]
            clipped = old_value + (value - old_value).clamp(-config.clip, config.clip)
            critic_loss = .5 * torch.maximum((value - target).square(), (clipped - target).square()).mean()
            loss = actor_loss + config.value_coef * critic_loss - config.entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite residual PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            metrics = dict(loss=loss, actor_loss=actor_loss, critic_loss=critic_loss,
                           entropy=entropy.mean(), approximate_kl=((ratio - 1) - log_ratio).mean(),
                           clip_fraction=((ratio - 1).abs() > config.clip).float().mean())
            for key, metric in metrics.items():
                totals[key] = totals.get(key, 0.) + float(metric.detach()) * len(index)
            count += len(index)
    policy.eval()
    return {key: value / count for key, value in totals.items()}


def history_trial(records_by_update):
    """Turn one lane's chronological [(update, records), ...] into a history trial.

    Controls are both executed and target labels (AD). ``teacher_steps`` is
    the PPO update that produced each decision.
    """
    rows = [(update, record) for update, records in records_by_update for record in records]
    if not rows:
        raise ValueError("Require at least one record")
    column = lambda key, dtype: np.asarray([record[key] for _, record in rows], dtype=dtype)
    controls = column("control", np.float32)
    return dict(observations=column("features", np.float32), executed_noise=controls,
                target_noise=controls.copy(), rewards=column("reward", np.float32),
                dones=column("done", np.bool_), episode_starts=column("episode_start", np.bool_),
                teacher_steps=np.asarray([update for update, _ in rows], dtype=np.int64))


class HistoryResidualController:
    """Deploy a frozen history controller whose outputs are residual controls u.

    Wraps ``vla_history.FrozenHistoryController``: the decoder sees the same
    features and records the same u that residual PPO histories contain; the
    bound is applied only when acting on the frozen VLA's chunk.
    """

    def __init__(self, controller, bound):
        if controller.policy.config.noise_dim != bound.residual_dim:
            raise ValueError("History controller output must match the residual dimension")
        self.controller, self.bound = controller, bound

    def reset_trial(self):
        self.controller.reset_trial()

    def act(self, chunk, features, *, deterministic=False):
        control = self.controller.act(features, deterministic=deterministic)
        return self.bound.apply(np.asarray(chunk), control), control

    def observe(self, control, reward, done):
        self.controller.observe(control, reward, done)


def config_record(config, bound):
    """Serializable contract for receipts."""
    return dict(ppo=asdict(config), bound=asdict(bound))
