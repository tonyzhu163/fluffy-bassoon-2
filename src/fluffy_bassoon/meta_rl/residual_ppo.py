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
import time

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
    # Mean sampled KL(old || new), checked over the full rollout after each
    # optimization epoch. None preserves the historical fixed-epoch learner.
    target_kl: float | None = None

    def __post_init__(self):
        counts = (self.feature_dim, self.residual_dim, self.width, self.layers, self.epochs, self.minibatch_size)
        if (any(type(value) is not int or value < 1 for value in counts)
                or not math.isfinite(self.initial_std) or self.initial_std <= 0
                or not 0 < self.clip < 1 or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1
                or self.learning_rate <= 0 or self.max_grad_norm <= 0
                or self.entropy_coef < 0 or self.value_coef < 0):
            raise ValueError("Invalid residual PPO configuration")
        if self.target_kl is not None and (type(self.target_kl) not in (int, float)
                or not math.isfinite(self.target_kl) or self.target_kl <= 0):
            raise ValueError("target_kl must be a finite positive number or None")


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


def update(policy, optimizer, rollout, *, generator=None, epoch_callback=None):
    """PPO with full-rollout epoch diagnostics and optional KL stopping.

    Existing scalar loss/KL metrics remain sample-weighted averages of the
    pre-step optimization minibatches. initial_metrics, epoch_metrics and
    final_metrics instead evaluate a single policy on the entire rollout.
    Stop both actor and critic after an epoch exceeds target_kl; the completed
    epoch is retained, not rolled back. The target is a stopping threshold,
    not a hard trust-region bound. No extra random draws enter diagnostics.
    """
    started = time.perf_counter()
    config = policy.config
    advantages, returns = generalized_advantages(rollout["rewards"], rollout["values"], rollout["ends"],
                                                 gamma=config.gamma, gae_lambda=config.gae_lambda)
    flat = dict(features=rollout["features"].reshape(-1, config.feature_dim),
                controls=rollout["controls"].reshape(-1, config.residual_dim),
                log_probs=rollout["log_probs"].reshape(-1), values=rollout["values"].reshape(-1),
                advantages=advantages.reshape(-1), returns=returns.reshape(-1))
    advantage = flat["advantages"]
    advantage_mean, advantage_std = float(advantage.mean()), float(advantage.std(unbiased=False))
    flat["advantages"] = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    size, totals, count = len(advantage), {}, 0

    def losses(index):
        log_prob, entropy, value = policy.evaluate(flat["features"][index], flat["controls"][index])
        log_ratio = log_prob - flat["log_probs"][index]
        ratio = log_ratio.exp()
        adv = flat["advantages"][index]
        actor_loss = torch.maximum(-adv * ratio, -adv * ratio.clamp(1 - config.clip, 1 + config.clip)).mean()
        old_value, target = flat["values"][index], flat["returns"][index]
        clipped = old_value + (value - old_value).clamp(-config.clip, config.clip)
        critic_loss = .5 * torch.maximum((value - target).square(), (clipped - target).square()).mean()
        loss = actor_loss + config.value_coef * critic_loss - config.entropy_coef * entropy.mean()
        metrics = dict(loss=loss, actor_loss=actor_loss, critic_loss=critic_loss,
                       entropy=entropy.mean(), approximate_kl=((ratio - 1) - log_ratio).mean(),
                       clip_fraction=((ratio - 1).abs() > config.clip).float().mean())
        return loss, metrics, value, log_ratio

    @torch.no_grad()
    def full_rollout_metrics():
        sums, predictions = {}, []
        for begin in range(0, size, config.minibatch_size):
            index = slice(begin, min(begin + config.minibatch_size, size))
            _, metrics, value, log_ratio = losses(index)
            # expm1 avoids cancellation when the policies are almost equal.
            metrics["approximate_kl"] = (torch.expm1(log_ratio) - log_ratio).mean()
            weight = len(value)
            for key, metric in metrics.items():
                scalar = float(metric)
                if not math.isfinite(scalar):
                    raise FloatingPointError(f"Nonfinite PPO diagnostic: {key}")
                sums[key] = sums.get(key, 0.) + scalar * weight
            predictions.append(value)
        result = {key: value / size for key, value in sums.items()}
        variance = flat["returns"].var(unbiased=False)
        result["explained_variance"] = (float(1 - (flat["returns"] - torch.cat(predictions)).var(
            unbiased=False) / variance) if float(variance) > 0 else None)
        return result

    initial_metrics = full_rollout_metrics()
    final_metrics = initial_metrics
    epoch_metrics, optimizer_steps = [], 0
    early_stopped = config.target_kl is not None and initial_metrics["approximate_kl"] > config.target_kl
    policy.train()
    for epoch in range(config.epochs):
        if early_stopped:
            break
        epoch_started = time.perf_counter()
        order = torch.randperm(size, generator=generator).to(advantage.device)
        epoch_steps, gradient_norm_sum, gradient_norm_max = 0, 0., 0.
        for begin in range(0, size, config.minibatch_size):
            index = order[begin:begin + config.minibatch_size]
            loss, metrics, _, _ = losses(index)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite residual PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm,
                                                         error_if_nonfinite=True))
            optimizer.step()
            optimizer_steps += 1
            epoch_steps += 1
            gradient_norm_sum += gradient_norm
            gradient_norm_max = max(gradient_norm_max, gradient_norm)
            for key, metric in metrics.items():
                totals[key] = totals.get(key, 0.) + float(metric.detach()) * len(index)
            count += len(index)
        final_metrics = full_rollout_metrics()
        early_stopped = config.target_kl is not None and final_metrics["approximate_kl"] > config.target_kl
        epoch_metrics.append(dict(epoch=epoch + 1, **final_metrics, optimizer_steps=epoch_steps,
            cumulative_optimizer_steps=optimizer_steps, samples=size,
            gradient_norm_mean=gradient_norm_sum / epoch_steps, gradient_norm_max=gradient_norm_max,
            elapsed_seconds=time.perf_counter() - epoch_started, kl_threshold_exceeded=early_stopped))
        if epoch_callback is not None:
            epoch_callback(dict(epoch_metrics[-1]))
    policy.eval()
    result = ({key: value / count for key, value in totals.items()} if count else
              {key: value for key, value in initial_metrics.items() if key != "explained_variance"})
    result.update(samples=size, samples_processed=count, epochs_requested=config.epochs,
        epochs_completed=len(epoch_metrics), optimizer_steps=optimizer_steps, target_kl=config.target_kl,
        early_stopped=early_stopped,
        stop_reason="target_kl" if early_stopped else "epochs_completed",
        advantage_mean=advantage_mean, advantage_std=advantage_std,
        return_mean=float(flat["returns"].mean()), return_std=float(flat["returns"].std(unbiased=False)),
        initial_metrics=initial_metrics, final_metrics=final_metrics, epoch_metrics=epoch_metrics,
        elapsed_seconds=time.perf_counter() - started)
    return result


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


def ppo_config_record(config):
    """Keep disabled-KL contracts identical to pre-KL checkpoints."""
    result = asdict(config)
    if result["target_kl"] is None:
        del result["target_kl"]
    return result


def config_record(config, bound):
    """Serializable contract for receipts."""
    return dict(ppo=ppo_config_record(config), bound=asdict(bound))
