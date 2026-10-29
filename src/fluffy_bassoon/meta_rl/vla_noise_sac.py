"""Feature-conditioned SAC over a frozen VLA's initial-noise input.

This is a DSRL-SAC-style steering baseline, not the authors' PixelSAC or
noise-aliasing implementation. Both compared controllers use the same frozen
public features; no perception encoder or motor is owned/updated here.

The actor uses tanh-normalized coordinates and scales them to ``action_limit``.
Entropy and its target use normalized coordinates, as in normalized-action
SAC. The Q-functions consume the actual scaled steering coordinate. A time
limit truncation bootstraps; genuine termination does not.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SACConfig:
    hidden_dims: tuple[int, ...] = (128, 128, 128)
    learning_rate: float = 3e-4
    batch_size: int = 256
    warmup_chunks: int = 500
    utd: int = 20
    replay_capacity: int = 100_000
    gamma: float = .99
    tau: float = .005
    initial_temperature: float = 1.
    target_entropy: float | None = None
    layer_norm: bool = True
    max_grad_norm: float | None = None

    def __post_init__(self):
        if (not self.hidden_dims or any(type(n) is not int or n < 1 for n in self.hidden_dims)
                or any(type(n) is not int or n < 1 for n in
                       (self.batch_size, self.utd, self.replay_capacity))
                or type(self.warmup_chunks) is not int or self.warmup_chunks < 0
                or self.replay_capacity < self.batch_size
                or not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.initial_temperature) or self.initial_temperature <= 0
                or self.target_entropy is not None and not math.isfinite(self.target_entropy)
                or self.max_grad_norm is not None and
                   (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0)):
            raise ValueError("Invalid noise SAC configuration")


def _mlp(input_dim, hidden_dims, output_dim, layer_norm):
    layers = []
    for width in hidden_dims:
        layers.append(nn.Linear(input_dim, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(nn.Tanh())
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def squashed_log_prob(mean, log_std, raw_action):
    """Stable log density of tanh(raw_action) in normalized coordinates."""
    normal_log_prob = -.5 * (((raw_action - mean) / log_std.exp()).square()
                            + 2 * log_std + math.log(2 * math.pi))
    log_jacobian = 2 * (math.log(2.) - raw_action - F.softplus(-2 * raw_action))
    return (normal_log_prob - log_jacobian).sum(dim=-1, keepdim=True)


def sac_target(rewards, terminated, next_soft_values, gamma):
    """Bellman target; truncated episodes deliberately remain bootstrapable."""
    return rewards + gamma * (1. - terminated.to(rewards.dtype)) * next_soft_values


class SquashedGaussianActor(nn.Module):
    def __init__(self, observation_dim, action_dim, config, action_limit):
        super().__init__()
        self.network = _mlp(observation_dim, config.hidden_dims, 2 * action_dim,
                            config.layer_norm)
        self.action_limit = action_limit

    def forward(self, observations, *, generator, deterministic=False):
        mean, log_std = self.network(observations).chunk(2, dim=-1)
        log_std = log_std.clamp(-20., 2.)
        raw = mean if deterministic else mean + log_std.exp() * torch.randn(
            mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        return self.action_limit * raw.tanh(), squashed_log_prob(mean, log_std, raw)


class TwinQ(nn.Module):
    def __init__(self, observation_dim, action_dim, config):
        super().__init__()
        self.q1 = _mlp(observation_dim + action_dim, config.hidden_dims, 1, config.layer_norm)
        self.q2 = _mlp(observation_dim + action_dim, config.hidden_dims, 1, config.layer_norm)

    def forward(self, observations, actions):
        inputs = torch.cat((observations, actions), dim=-1)
        return self.q1(inputs), self.q2(inputs)


class NoiseSAC:
    """Reactive noise policy with complete replay, optimizer and private RNG state.

    ``act`` always uses the learned actor, including during frozen evaluation.
    Source collection explicitly calls ``sample_warmup`` before the configured
    threshold. Warmup is bounded uniform sampling, an explicit difference from
    unbounded-Gaussian warmup used by some released DSRL examples.

    Restore the same complete source state independently for every target
    trial, then call ``begin_target_adaptation`` for online evaluation. Its
    default clears source replay while retaining trained weights, temperature,
    optimizer state and source counters. Already-trained target adaptation can
    sample with replacement immediately from its first target transition.
    """
    format = "frozen_vla_feature_noise_sac_v1"

    def __init__(self, observation_dim, action_dim, *, device="cuda", action_limit=1.5,
                 seed=0, config=None):
        self.config = SACConfig() if config is None else config
        if isinstance(self.config, dict):
            values = dict(self.config)
            if "hidden_dims" in values:
                values["hidden_dims"] = tuple(values["hidden_dims"])
            self.config = SACConfig(**values)
        if (not isinstance(self.config, SACConfig)
                or any(type(n) is not int or n < 1 for n in (observation_dim, action_dim))
                or not math.isfinite(action_limit) or action_limit <= 0
                or type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("Require valid feature/action dimensions, bound and seed")
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.action_limit, self.seed = float(action_limit), seed
        self.device = torch.device(device)
        # Initialization is reproducible without changing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.actor = SquashedGaussianActor(observation_dim, action_dim, self.config,
                                               self.action_limit).to(self.device)
            self.critic = TwinQ(observation_dim, action_dim, self.config).to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.log_temperature = torch.tensor(math.log(self.config.initial_temperature),
                                             device=self.device, requires_grad=True)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=self.config.learning_rate)
        self.target_entropy = (-float(action_dim) if self.config.target_entropy is None
                               else self.config.target_entropy)
        self.rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator(device=self.device).manual_seed(seed)
        # Optimizer sampling must not consume the interaction stream: frozen
        # and adapting target copies use paired stochastic action draws.
        self.train_torch_rng = torch.Generator(device=self.device).manual_seed((seed + 1) % 2**32)
        self.replay = []
        self.replay_position = 0
        self.environment_steps = self.primitive_steps = self.updates = 0
        self.pending_chunks = 0
        self.update_seconds = 0.
        self.target_replay_mode = None
        self.target_start = None
        self.source_replay_rows_at_target_start = 0

    @property
    def observed_chunks(self):
        return self.environment_steps

    @property
    def transition_count(self):
        return self.environment_steps

    @property
    def optimizer_steps(self):
        """Total optimizer.step calls: actor + twin-critic + temperature."""
        return 3 * self.updates

    @property
    def ready_to_update(self):
        if self.target_replay_mode is not None:
            return len(self.replay) >= 1
        return (self.environment_steps > self.config.warmup_chunks
                and len(self.replay) >= self.config.batch_size)

    @property
    def warming_up(self):
        return self.target_replay_mode is None and self.environment_steps < self.config.warmup_chunks

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def _features(self, observation):
        value = np.asarray(observation, dtype=np.float32)
        single = value.ndim == 1
        if single:
            value = value[None]
        if (value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != self.observation_dim
                or not np.isfinite(value).all()):
            raise ValueError("Require finite current public feature vectors")
        return value, single

    def act(self, observation, deterministic=False):
        features, single = self._features(observation)
        with torch.no_grad():
            action, _ = self.actor(torch.as_tensor(features, device=self.device),
                                    generator=self.torch_rng, deterministic=deterministic)
        result = action.cpu().numpy().astype(np.float32, copy=True)
        return result[0] if single else result

    def sample_warmup(self, observation=None, *, batch_size=None):
        if observation is not None:
            features, single = self._features(observation)
            shape = (len(features), self.action_dim)
        elif batch_size is None:
            shape, single = (1, self.action_dim), True
        else:
            if type(batch_size) is not int or batch_size < 1:
                raise ValueError("Require a positive warmup batch size")
            shape, single = (batch_size, self.action_dim), False
        result = self.rng.uniform(-self.action_limit, self.action_limit, shape).astype(np.float32)
        return result[0] if single else result

    warmup_action = sample_warmup

    def begin_target_adaptation(self, replay_mode="fresh_target"):
        """Begin an independent target fine-tune from a trained source restore.

        ``fresh_target`` is the primary convention: source replay must not
        dilute <=104 target chunks, nor should a fresh buffer disable updates
        because it has fewer than the nominal 256 batch rows. Sampling is with
        replacement. ``carry_source`` is an explicitly labeled diagnostic.
        This method never resets learned parameters, optimizers or source cost.
        """
        if replay_mode not in ("fresh_target", "carry_source"):
            raise ValueError("Unknown target replay convention")
        if self.target_replay_mode is not None:
            raise ValueError("Restore the source learner before starting another target trial")
        if self.updates < 1 or self.environment_steps <= self.config.warmup_chunks:
            raise ValueError("Target transfer requires a source learner with completed optimization")
        self.target_start = dict(environment_steps=self.environment_steps,
                                 primitive_steps=self.primitive_steps,
                                 updates=self.updates, optimizer_steps=self.optimizer_steps,
                                 update_seconds=self.update_seconds)
        self.source_replay_rows_at_target_start = len(self.replay)
        self.target_replay_mode = replay_mode
        if replay_mode == "fresh_target":
            self.replay = []
            self.replay_position = 0
        self.pending_chunks = 0
        return dict(replay_mode=replay_mode, source_counters=deepcopy(self.target_start),
                    source_replay_rows=self.source_replay_rows_at_target_start,
                    source_replay_excluded=replay_mode == "fresh_target",
                    sample_with_replacement=True, minimum_target_replay_rows=1,
                    additional_warmup_chunks=0)

    def target_counters(self):
        if self.target_start is None:
            return {}
        return dict(environment_steps=self.environment_steps-self.target_start["environment_steps"],
                    primitive_steps=self.primitive_steps-self.target_start["primitive_steps"],
                    updates=self.updates-self.target_start["updates"],
                    optimizer_steps=self.optimizer_steps-self.target_start["optimizer_steps"],
                    update_seconds=self.update_seconds-self.target_start["update_seconds"],
                    replay_rows=len(self.replay), replay_mode=self.target_replay_mode)

    def observe(self, observations, actions, rewards, next_observations, terminated,
                truncated=None, primitive_steps=None):
        observations, _ = self._features(observations)
        next_observations, _ = self._features(next_observations)
        batch = len(observations)
        actions = np.asarray(actions, dtype=np.float32).reshape(batch, self.action_dim)
        def vector(value, dtype, default=None):
            result = np.asarray(default if value is None else value, dtype=dtype)
            return np.broadcast_to(result, (batch,)).copy()
        rewards = vector(rewards, np.float32)
        terminated = vector(terminated, bool)
        truncated = vector(truncated, bool, False)
        steps = vector(primitive_steps, np.int64, 1)
        raw_steps = vector(primitive_steps, np.float64, 1.)
        if (len(next_observations) != batch or not np.isfinite(actions).all()
                or np.any(np.abs(actions) > self.action_limit + 1e-6)
                or not np.isfinite(rewards).all() or np.any(steps < 1)
                or not np.array_equal(steps, raw_steps)):
            raise ValueError("Invalid steering transition or primitive-step count")
        for i in range(batch):
            row = (observations[i].copy(), actions[i].copy(), np.float32(rewards[i]),
                   next_observations[i].copy(), bool(terminated[i]), bool(truncated[i]), int(steps[i]))
            if len(self.replay) < self.config.replay_capacity:
                self.replay.append(row)
            else:
                self.replay[self.replay_position] = row
            self.replay_position = (self.replay_position + 1) % self.config.replay_capacity
        self.environment_steps += batch
        self.primitive_steps += int(steps.sum())
        self.pending_chunks += batch

    def _batch(self):
        indices = self.rng.integers(0, len(self.replay), self.config.batch_size)
        rows = [self.replay[int(i)] for i in indices]
        columns = [np.stack([row[i] for row in rows]) for i in (0, 1, 2, 3, 4)]
        obs, actions, reward, following, terminal = [
            torch.as_tensor(column, device=self.device, dtype=torch.float32) for column in columns]
        return obs, actions, reward[:, None], following, terminal[:, None]

    def _step(self, loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite noise SAC loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(parameters)
        gradient_norm = nn.utils.clip_grad_norm_(
            parameters, self.config.max_grad_norm if self.config.max_grad_norm is not None else math.inf,
            error_if_nonfinite=True)
        optimizer.step()
        return gradient_norm

    def update(self, gradient_steps=None):
        if gradient_steps is not None and (type(gradient_steps) is not int or gradient_steps < 0):
            raise ValueError("Require a nonnegative integral SAC update count")
        if not self.ready_to_update:
            self.pending_chunks = 0  # Warmup has no retroactive optimization debt.
            return {}
        count = self.pending_chunks * self.config.utd if gradient_steps is None else gradient_steps
        if count == 0:
            return {}
        started = time.monotonic()
        metrics = {}
        for index in range(count):
            obs, actions, rewards, following, terminated = self._batch()
            alpha = self.temperature.detach()
            with torch.no_grad():
                next_actions, next_log_prob = self.actor(following, generator=self.train_torch_rng)
                q1_next, q2_next = self.target_critic(following, next_actions)
                next_soft_value = torch.minimum(q1_next, q2_next) - alpha * next_log_prob
                target = sac_target(rewards, terminated, next_soft_value, self.config.gamma)
            q1, q2 = self.critic(obs, actions)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            critic_gradient = self._step(critic_loss, self.critic_optimizer, self.critic.parameters())
            # Gradients through action are required; critic parameter gradients are not.
            self.critic.requires_grad_(False)
            try:
                new_actions, log_prob = self.actor(obs, generator=self.train_torch_rng)
                q1_new, q2_new = self.critic(obs, new_actions)
                actor_loss = (alpha * log_prob - torch.minimum(q1_new, q2_new)).mean()
                actor_gradient = self._step(actor_loss, self.actor_optimizer, self.actor.parameters())
            finally:
                self.critic.requires_grad_(True)
            temperature_loss = -(self.log_temperature * (log_prob + self.target_entropy).detach()).mean()
            temperature_gradient = self._step(temperature_loss, self.temperature_optimizer,
                                               [self.log_temperature])
            with torch.no_grad():
                for target_parameter, parameter in zip(self.target_critic.parameters(),
                                                       self.critic.parameters(), strict=True):
                    target_parameter.lerp_(parameter, self.config.tau)
            self.updates += 1
            if index == count - 1:
                metrics = dict(critic_loss=float(critic_loss.detach()), actor_loss=float(actor_loss.detach()),
                               temperature_loss=float(temperature_loss.detach()),
                               temperature=float(self.temperature.detach()),
                               target_mean=float(target.mean()), entropy=float(-log_prob.detach().mean()),
                               actor_gradient_norm=float(actor_gradient),
                               critic_gradient_norm=float(critic_gradient),
                               temperature_gradient_norm=float(temperature_gradient))
        elapsed = time.monotonic() - started
        self.update_seconds += elapsed
        self.pending_chunks = 0
        metrics.update(updates=self.updates, gradient_steps=count, optimizer_steps=self.optimizer_steps,
                       actor_optimizer_steps=self.updates, critic_optimizer_steps=self.updates,
                       temperature_optimizer_steps=self.updates, update_seconds=elapsed,
                       update_seconds_total=self.update_seconds, replay_rows=len(self.replay))
        return metrics

    def contract(self):
        return dict(format=self.format, observation_dim=self.observation_dim,
                    action_dim=self.action_dim, action_limit=self.action_limit,
                    config=asdict(self.config), seed=self.seed,
                    algorithm="feature-conditioned DSRL-SAC-style; not authors' PixelSAC/DSRL-NA",
                    warmup_distribution="bounded uniform in the shared steering support",
                    entropy_coordinates="tanh-normalized; external action scale excluded",
                    stochastic_streams="separate interaction and optimizer generators; paired frozen/online action draws",
                    truncation="bootstrap on time-limit truncation; stop on genuine termination",
                    target_restore="independent complete source restore, then explicit fresh_target replay for primary online fine-tune; source weights/optimizers/costs retained",
                    target_replay="fresh_target samples with replacement from >=1 new target row; no repeated source warmup; carry_source optional diagnostic",
                    actor_parameters=sum(p.numel() for p in self.actor.parameters()),
                    critic_parameters=sum(p.numel() for p in self.critic.parameters()))

    def state_dict(self, *, include_replay=True):
        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {key: cpu(item) for key, item in value.items()}
            if isinstance(value, list):
                return [cpu(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cpu(item) for item in value)
            return deepcopy(value)
        return dict(contract=self.contract(), actor=cpu(self.actor.state_dict()),
                    critic=cpu(self.critic.state_dict()), target_critic=cpu(self.target_critic.state_dict()),
                    log_temperature=cpu(self.log_temperature),
                    actor_optimizer=cpu(self.actor_optimizer.state_dict()),
                    critic_optimizer=cpu(self.critic_optimizer.state_dict()),
                    temperature_optimizer=cpu(self.temperature_optimizer.state_dict()),
                    replay=deepcopy(self.replay) if include_replay else None,
                    replay_position=self.replay_position, environment_steps=self.environment_steps,
                    primitive_steps=self.primitive_steps, updates=self.updates,
                    pending_chunks=self.pending_chunks, update_seconds=self.update_seconds,
                    target_replay_mode=self.target_replay_mode,
                    target_start=deepcopy(self.target_start),
                    source_replay_rows_at_target_start=self.source_replay_rows_at_target_start,
                    numpy_rng=deepcopy(self.rng.bit_generator.state),
                    torch_rng=self.torch_rng.get_state().cpu().clone(),
                    train_torch_rng=self.train_torch_rng.get_state().cpu().clone())

    def load_state_dict(self, payload, *, require_replay=True):
        if payload["contract"] != self.contract():
            raise ValueError("Noise SAC recovery contract changed")
        if require_replay and payload["replay"] is None:
            raise ValueError("Adaptation recovery requires complete replay")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])
        self.target_critic.requires_grad_(False)
        with torch.no_grad():
            self.log_temperature.copy_(payload["log_temperature"].to(self.device))
        for optimizer, key in ((self.actor_optimizer, "actor_optimizer"),
                               (self.critic_optimizer, "critic_optimizer"),
                               (self.temperature_optimizer, "temperature_optimizer")):
            optimizer.load_state_dict(deepcopy(payload[key]))
        self.replay = deepcopy(payload["replay"]) if payload["replay"] is not None else []
        self.replay_position = int(payload["replay_position"])
        for key in ("environment_steps", "primitive_steps", "updates", "pending_chunks"):
            setattr(self, key, int(payload[key]))
        self.update_seconds = float(payload["update_seconds"])
        self.target_replay_mode = payload.get("target_replay_mode")
        if self.target_replay_mode not in (None, "fresh_target", "carry_source"):
            raise ValueError("Invalid target replay recovery mode")
        self.target_start = deepcopy(payload.get("target_start"))
        self.source_replay_rows_at_target_start = int(payload.get("source_replay_rows_at_target_start", 0))
        self.rng.bit_generator.state = deepcopy(payload["numpy_rng"])
        self.torch_rng.set_state(payload["torch_rng"].cpu())
        self.train_torch_rng.set_state(payload["train_torch_rng"].cpu())

    def save(self, path):
        """Atomic learner snapshot; the driver separately saves task/simulator state."""
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(self.state_dict(), temporary)
        temporary.replace(path)

    def load(self, path):
        self.load_state_dict(torch.load(Path(path), map_location="cpu", weights_only=False))


# An explicit alternative name for callers searching by the method family.
DSRLSAC = NoiseSAC
