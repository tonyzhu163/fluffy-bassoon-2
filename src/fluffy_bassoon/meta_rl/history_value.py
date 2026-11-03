"""DIT and in-context IQL on actual frozen-steering trajectories.

Actor inputs contain only shifted real feedback. Returns are critic labels,
never actor tokens. Each method fits independent causal Q/V estimators before
freezing them for advantage-weighted Gaussian policy extraction. IC-IQL is
the decoder/IQL adaptation, without S-ICQL's world-model prompt.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .causal_decoder import CausalDecoderCore
from .vla_history import HistoryNoisePolicy, trial_tokens, validate_trial


VALUE_OBJECTIVES = ("dit", "ic_iql")


@dataclass(frozen=True)
class ValueConfig:
    objective: str
    discount: float = .99
    expectile: float = .7
    temperature: float = 1.
    maximum_weight: float = 100.
    target_tau: float = .005
    bellman_weight: float = .1

    def __post_init__(self):
        values = (self.discount, self.expectile, self.temperature, self.maximum_weight,
                  self.target_tau, self.bellman_weight)
        if (self.objective not in VALUE_OBJECTIVES or not all(math.isfinite(x) for x in values)
                or not 0 <= self.discount <= 1 or not .5 < self.expectile < 1
                or self.temperature <= 0 or self.maximum_weight < 1
                or not 0 < self.target_tau <= 1 or self.bellman_weight < 0):
            raise ValueError("Invalid value objective, discount, expectile or extraction settings")


def discounted_returns(rewards, dones, discount):
    rewards, dones = np.asarray(rewards), np.asarray(dones)
    if (rewards.ndim != 1 or rewards.shape != dones.shape or not len(rewards)
            or dones.dtype != np.bool_ or not dones[-1] or not np.isfinite(rewards).all()
            or not math.isfinite(discount) or not 0 <= discount <= 1):
        raise ValueError("Compute returns from complete finite physical episodes")
    result, following = np.empty(len(rewards), np.float32), 0.
    for t in range(len(rewards) - 1, -1, -1):
        following = float(rewards[t]) + discount * following * (not dones[t])
        result[t] = following
    return result


def expectile_loss(difference, expectile):
    return torch.where(difference >= 0, expectile, 1 - expectile) * difference.square()


def importance_weights(advantage, config):
    # Clamp in log space before exp; large finite advantages cannot overflow.
    return torch.exp((advantage / config.temperature).clamp(
        min=-20., max=math.log(config.maximum_weight))).detach()


def sample_value_batch(trials, policy, rng, *, batch_size, max_episodes, discount, mode="full"):
    """Complete-episode windows; MC labels retain all future rewards in-window.

    The next token is evaluated with the current action/reward already in its
    causal context. Terminal rows have no next query, even if another episode
    follows in the same trial. All sampled windows finish at a real boundary.
    """
    if type(batch_size) is not int or batch_size < 1 or type(max_episodes) is not int or max_episodes < 1:
        raise ValueError("Require positive batch and complete-episode window counts")
    windows = []
    for index in rng.integers(len(trials), size=batch_size):
        trial = trials[int(index)]
        validate_trial(trial, objective="ic_iql")
        boundaries = np.flatnonzero(trial["episode_starts"])
        end_episode = int(rng.integers(1, len(boundaries) + 1))
        count = int(rng.integers(1, min(max_episodes, end_episode) + 1))
        begin = int(boundaries[end_episode - count])
        end = int(boundaries[end_episode]) if end_episode < len(boundaries) else len(trial["rewards"])
        token = trial_tokens(trial, mode=mode)[begin:end].copy()
        token[0, policy.config.observation_dim:-1] = 0
        windows.append(dict(tokens=token, actions=trial["executed_noise"][begin:end],
            rewards=trial["rewards"][begin:end], dones=trial["dones"][begin:end],
            starts=trial["episode_starts"][begin:end],
            returns=discounted_returns(trial["rewards"], trial["dones"], discount)[begin:end]))
    length = max(len(row["tokens"]) for row in windows)
    if length > policy.config.context_length:
        raise ValueError("Complete physical episode windows exceed model context capacity")
    device = next(policy.parameters()).device
    shapes = dict(tokens=(policy.core.input_dim,), actions=(policy.config.noise_dim,),
                  rewards=(), dones=(), starts=(), returns=(), valid=())
    result = {key: torch.zeros((length, batch_size, *shape), device=device,
        dtype=torch.bool if key in ("dones", "starts", "valid") else torch.float32)
        for key, shape in shapes.items()}
    for lane, window in enumerate(windows):
        count = len(window["tokens"])
        result["valid"][:count, lane] = True
        for key, values in window.items():
            result[key][:count, lane] = torch.as_tensor(values, device=device)
    return result


class HistoryCritic(nn.Module):
    def __init__(self, config, *, action_value):
        super().__init__()
        self.config, self.action_value = config, action_value
        self.core = CausalDecoderCore(config.observation_dim + config.noise_dim + 3,
            width=config.width, layers=config.layers, heads=config.heads, memory_length=config.context_length)
        extra = config.noise_dim if action_value else 0
        self.head = nn.Sequential(nn.Linear(config.width + extra, config.width), nn.GELU(), nn.Linear(config.width, 1))
        self.register_buffer("observation_mean", torch.zeros(config.observation_dim))
        self.register_buffer("observation_std", torch.ones(config.observation_dim))

    def sequence(self, batch, *, mode="full"):
        tokens, valid, starts = (batch[key] for key in ("tokens", "valid", "starts"))
        state, values = None, []
        for t in range(len(tokens)):
            reset = torch.ones_like(valid[t]) if mode == "static" else starts[t] if mode == "episode" else None
            if state is not None and reset is not None:
                empty = self.core.initial_state(tokens.shape[1])
                state = {key: torch.where(reset.reshape(-1, *([1] * (value.ndim - 1))), empty[key], value)
                         for key, value in state.items()}
            token = torch.where(valid[t, :, None], tokens[t], 0.)
            d = self.config.observation_dim
            token = torch.cat(((token[:, :d] - self.observation_mean) / self.observation_std, token[:, d:]), -1)
            feature, state = self.core(token, state)
            if self.action_value:
                action = torch.where(valid[t, :, None], batch["actions"][t], 0.)
                feature = torch.cat((feature, action), -1)
            values.append(self.head(feature).squeeze(-1))
        return torch.stack(values)


def next_values(values, dones, valid):
    following = torch.cat((values[1:], torch.zeros_like(values[:1])), 0)
    has_next = torch.cat((valid[1:], torch.zeros_like(valid[:1])), 0)
    if ((~dones & valid) & ~has_next).any():
        raise ValueError("A nonterminal value target has no next query in its episode")
    return torch.where(valid & ~dones, following, 0.)


class HistoryValueLearner(nn.Module):
    def __init__(self, history_config, value_config):
        super().__init__()
        if history_config.backbone != "decoder":
            raise ValueError("The qualified value-history implementation uses the causal decoder")
        self.config = value_config
        self.actor = HistoryNoisePolicy(history_config)
        self.q = HistoryCritic(history_config, action_value=True)
        self.v = HistoryCritic(history_config, action_value=False)
        self.target_q = deepcopy(self.q).requires_grad_(False)

    def fit_normalization(self, trials):
        self.actor.fit_observation_normalization(trials)
        with torch.no_grad():
            for critic in (self.q, self.v, self.target_q):
                critic.observation_mean.copy_(self.actor.observation_mean)
                critic.observation_std.copy_(self.actor.observation_std)

    def value_loss(self, batch, *, mode="full"):
        valid, dones = batch["valid"], batch["dones"]
        rewards = torch.where(valid, batch["rewards"], 0.)
        q, value = self.q.sequence(batch, mode=mode), self.v.sequence(batch, mode=mode)
        with torch.no_grad():
            frozen_q = self.target_q.sequence(batch, mode=mode)
            next_v = next_values(value.detach(), dones, valid)
            next_q = next_values(frozen_q, dones, valid)
            td = rewards + self.config.discount * next_v
        if self.config.objective == "ic_iql":
            q_loss = (q - td).square()[valid].mean()
            v_loss = expectile_loss(frozen_q - value, self.config.expectile)[valid].mean()
        else:
            returns = torch.where(valid, batch["returns"], 0.)
            q_loss = ((q - returns).square() + self.config.bellman_weight *
                      (q - rewards - self.config.discount * next_q).square())[valid].mean()
            v_loss = ((value - returns).square() + self.config.bellman_weight *
                      (value - td).square())[valid].mean()
        return q_loss + v_loss, dict(q_loss=q_loss.detach(), v_loss=v_loss.detach())

    def actor_loss(self, batch, *, mode="full"):
        valid, tokens = batch["valid"], batch["tokens"]
        with torch.no_grad():
            advantage = self.q.sequence(batch, mode=mode) - self.v.sequence(batch, mode=mode)
            weights = importance_weights(advantage, self.config)
        state, losses = None, []
        for t in range(len(tokens)):
            reset = torch.ones_like(valid[t]) if mode == "static" else batch["starts"][t] if mode == "episode" else None
            mean, _, state = self.actor.step(torch.where(valid[t, :, None], tokens[t], 0.), state,
                                            reset=reset, deterministic=True)
            action = torch.where(valid[t, :, None], batch["actions"][t], mean.detach())
            nll = -Normal(mean, self.actor.log_std.clamp(-5, 2).exp()).log_prob(action).sum(-1)
            losses.append(nll)
        nll = torch.stack(losses)
        return (weights * nll)[valid].mean(), dict(nll=nll[valid].mean().detach(),
            weight_mean=weights[valid].mean(), weight_max=weights[valid].max())

    @torch.no_grad()
    def update_target(self):
        for target, parameter in zip(self.target_q.parameters(), self.q.parameters(), strict=True):
            target.lerp_(parameter, self.config.target_tau)


def learning_step(learner, optimizer, batch, *, phase, mode="full", execution_batch_size=None):
    if phase not in ("value", "actor"):
        raise ValueError("Choose value fitting or frozen-value policy extraction")
    width = batch["valid"].shape[1]
    execution = width if execution_batch_size is None else execution_batch_size
    if type(execution) is not int or execution < 1:
        raise ValueError("Execution batch must be positive")
    optimizer.zero_grad(set_to_none=True)
    total, metrics = int(batch["valid"].sum()), {}
    loss_function = learner.value_loss if phase == "value" else learner.actor_loss
    for begin in range(0, width, execution):
        part = {key: values[:, begin:begin + execution] for key, values in batch.items()}
        loss, extra = loss_function(part, mode=mode)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite history value/policy loss")
        weight = int(part["valid"].sum()) / total
        (loss * weight).backward()
        for key, value in dict(loss=loss.detach(), **extra).items():
            metrics[key] = metrics.get(key, 0.) + float(value) * weight
    parameters = list(learner.q.parameters()) + list(learner.v.parameters()) if phase == "value" else list(learner.actor.parameters())
    metrics["gradient_norm"] = float(nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
    optimizer.step()
    if any(not torch.isfinite(value).all() for value in learner.parameters()):
        raise FloatingPointError("Nonfinite history value/policy parameter")
    if phase == "value":
        learner.update_target()
    return metrics
