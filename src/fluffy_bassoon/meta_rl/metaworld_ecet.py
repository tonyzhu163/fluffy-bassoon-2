"""ECET's two-level episode encoder and transition-minibatch PPO on ML10.

This is an explicit port of the pinned official architecture and learner, with
the environment adaptations and upstream-buffer corrections documented in
docs/metaworld-ecet.md. It is not the recurrent Transformer-XL control.
"""

import argparse
import copy
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.nn import functional as F

from fluffy_bassoon.evaluation_stages import write_json
from .checkpoint import restore_checkpoint, save_checkpoint
from .metaworld_env import benchmark_pools, make_lanes, task_digest
from .metaworld_workers import batch_reset, batch_step, make_worker_lanes
from .metaworld_train import preserve_random_streams, synchronize
from .ppo import generalized_advantages
from .screen import allocation, clean_source


UPSTREAM = dict(repository="https://github.com/machinelearningnuremberg/ECET",
                commit="aff709ae1b35406b16abdf62a607697fb0f19a8c",
                policy="src/garage/torch/policies/ecet_policy.py",
                runner="examples/torch/transformer_ep_s.py", paper_context_episodes=25)


@dataclass(frozen=True)
class ECETConfig:
    observation_dim: int = 45  # state, previous executed action, reward, boundary; no task ID
    action_dim: int = 4
    width: int = 32
    heads: int = 4
    intra_layers: int = 2
    cross_layers: int = 2
    feedforward: int = 128
    transitions: int = 5
    episodes: int = 5  # common K5 comparison; paper Table 1 uses E25
    tfixup: bool = True
    remove_intra_layer_norm: bool = True
    learn_std: bool = True

    def __post_init__(self):
        sizes = (self.observation_dim, self.action_dim, self.width, self.heads,
                 self.intra_layers, self.cross_layers, self.feedforward,
                 self.transitions, self.episodes)
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValueError("ECET sizes must be positive integers")
        if self.width % self.heads or self.width % 2:
            raise ValueError("ECET width must be even and divisible by attention heads")


class IntraEncoderLayer(nn.Module):
    """Causal attention/residual/ReLU FFN; optional norms, zero dropout.

    Matches official TransformerEncoderLayerNoLN's equations and parameter
    names. A local stack avoids its incompatible modern TransformerEncoder
    forward signature and retains PyTorch's attention implementation.
    """

    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
        self.linear1 = nn.Linear(config.width, config.feedforward)
        self.linear2 = nn.Linear(config.feedforward, config.width)
        self.norm1 = nn.Identity() if config.remove_intra_layer_norm else nn.LayerNorm(config.width)
        self.norm2 = nn.Identity() if config.remove_intra_layer_norm else nn.LayerNorm(config.width)

    def forward(self, value, mask):
        attended = self.self_attn(value, value, value, attn_mask=mask, need_weights=False)[0]
        value = self.norm1(value + attended)
        return self.norm2(value + self.linear2(F.relu(self.linear1(value))))


def mlp(inputs, outputs):
    result = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64),
                           nn.Tanh(), nn.Linear(64, outputs))
    for layer in result:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return result


class ECETPolicy(nn.Module):
    """IET -> last token per episode -> unpositioned CET -> current token.

    The critic is the official default independent 64,64 tanh Gaussian MLP on
    the current augmented observation; its objective is Gaussian NLL, not a
    clipped value-error head sharing the actor's context encoder.
    """

    def __init__(self, config=ECETConfig()):
        super().__init__()
        self.config = config
        self.phi = nn.Linear(config.observation_dim, config.width, bias=False)
        self.phi_current = nn.Linear(config.observation_dim, config.width, bias=False)
        layer = IntraEncoderLayer(config)
        self.intra = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.intra_layers)])
        cross = nn.TransformerEncoderLayer(config.width, config.heads, config.feedforward,
                                           dropout=0., activation="relu", batch_first=True)
        self.cross = nn.ModuleList([copy.deepcopy(cross) for _ in range(config.cross_layers)])
        positions = torch.arange(config.transitions, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, config.width, 2).float()
                                * (-math.log(10000.) / config.width))
        encoding = torch.zeros(config.transitions, config.width)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("position_encoding", encoding)
        self.register_buffer("causal_mask", torch.ones(config.transitions, config.transitions,
                                                       dtype=torch.bool).triu(1))
        for stack in (self.intra, self.cross):
            for parameter in stack.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        if config.tfixup:
            nn.init.normal_(self.phi.weight, std=config.width ** -.5)
            nn.init.normal_(self.phi_current.weight, std=config.width ** -.5)
            with torch.no_grad():
                self.phi.weight.mul_((9 * config.intra_layers) ** -.25)
                for stack, depth in ((self.intra, config.intra_layers),
                                     (self.cross, config.cross_layers)):
                    scale = .67 * depth ** -.25
                    for name, parameter in stack.named_parameters():
                        if any(part in name for part in ("linear1.weight", "linear2.weight",
                                                         "self_attn.out_proj.weight")):
                            parameter.mul_(scale)
                        elif "self_attn.in_proj_weight" in name:
                            parameter.mul_(scale * math.sqrt(2.))
        self.actor = mlp(2 * config.width, config.action_dim)
        self.critic = mlp(config.observation_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(config.action_dim), requires_grad=config.learn_std)
        self.critic_log_std = nn.Parameter(torch.zeros(1), requires_grad=config.learn_std)

    def encode_episodes(self, contexts):
        if contexts.ndim != 4 or contexts.shape[1:] != (self.config.episodes,
                self.config.transitions, self.config.observation_dim):
            raise ValueError("Require [batch, episodes, transitions, observation] ECET contexts")
        batch = contexts.shape[0]
        value = self.phi(contexts.reshape(-1, self.config.transitions, self.config.observation_dim))
        value = value + self.position_encoding
        for layer in self.intra:
            value = layer(value, self.causal_mask)
        return value[:, -1].reshape(batch, self.config.episodes, self.config.width)

    def cross_context(self, episodes):
        # No episode position encoding and no CET causal mask: the last query
        # represents the current episode and all other inputs are past episodes.
        for layer in self.cross:
            episodes = layer(episodes)
        return episodes[:, -1]

    def distributions(self, contexts, current):
        task = self.cross_context(self.encode_episodes(contexts))
        features = torch.cat((task, self.phi_current(current)), -1)
        actor = Independent(Normal(self.actor(features), self.actor_log_std.clamp_min(math.log(1e-6)).exp()), 1)
        critic = Independent(Normal(self.critic(current), self.critic_log_std.exp()), 1)
        return actor, critic

    def actor_parameters(self):
        return [parameter for name, parameter in self.named_parameters()
                if not name.startswith("critic")]

    def critic_parameters(self):
        return list(self.critic.parameters()) + [self.critic_log_std]


def context_indices(step, batch, *, horizon, episodes, transitions, memory_reset=False, rng=None):
    """Source-row indices for genuine previous episodes and a current prefix.

    Sampling occurs once during behavior collection, then the exact indices
    are replayed by PPO. -1 denotes an empty padded token. The source is the
    time-major [trial_steps, lanes, observation] array, flattened over time/lane.
    """
    if (step < 0 or batch < 1 or horizon < transitions or transitions < 1 or episodes < 1):
        raise ValueError("Invalid ECET context sampling dimensions")
    episode, offset = divmod(step, horizon)
    result = np.full((batch, episodes, transitions), -1, dtype=np.int64)
    lanes = np.arange(batch)[:, None]
    if not memory_reset:
        previous = min(episode, episodes - 1)
        for slot in range(previous):
            episode_index = episode - previous + slot
            integers = np.random.randint if rng is None else rng.integers
            starts = integers(0, horizon - transitions + 1, size=(batch, 1))
            source = episode_index * horizon + starts + np.arange(transitions)
            result[:, episodes - 1 - previous + slot] = source * batch + lanes
    count = min(offset + 1, transitions)
    result[:, -1, -count:] = (np.arange(step - count + 1, step + 1)[None, :] * batch + lanes)
    return result


def materialize_context(observations, indices):
    flat = observations.reshape(-1, observations.shape[-1])
    selected = flat[indices.clamp_min(0)]
    return selected.masked_fill((indices < 0)[..., None], 0.)


def augmented_observation(observations, *, reward_blind=False):
    """Official RL2 input order; clocks and environment/task identifiers omitted."""
    return np.stack([np.concatenate((observation["state"], observation["previous_action"],
                    np.zeros(1, np.float32) if reward_blind else observation["previous_reward"],
                    observation["attempt_end"])) for observation in observations])


@torch.no_grad()
def collect_ecet(policy, envs, seeds, *, deterministic=False, memory_reset=False,
                 reward_blind=False, record=True):
    if not envs or len(envs) != len(seeds):
        raise ValueError("Require one seed per ECET environment lane")
    first, config = envs[0], policy.config
    if any((env.attempts, env.horizon) != (first.attempts, first.horizon) for env in envs):
        raise ValueError("ECET lanes must have the same trial horizon")
    if first.horizon < config.transitions:
        raise ValueError("ECET transition context cannot exceed the episode horizon")
    policy.eval()
    reference = next(policy.parameters())
    length, batch = first.attempts * first.horizon, len(envs)
    # Serial simulators and worker simulators must not perturb the policy's
    # context sampling differently. Trial seeds already belong to checkpoints;
    # this ephemeral stream is completely reconstructible at trial boundaries.
    context_rng = np.random.default_rng(np.random.SeedSequence([int(seed) for seed in seeds]))
    history = reference.new_zeros(length, batch, config.observation_dim)
    records = {key: [] for key in ("context_indices", "raw_actions", "log_probs", "values",
                                    "rewards", "trial_ends", "attempt_ends")}
    observations = [row[0] for row in batch_reset(envs, seeds)]
    for step in range(length):
        current = torch.as_tensor(augmented_observation(observations, reward_blind=reward_blind),
                                   dtype=reference.dtype, device=reference.device)
        history[step].copy_(current)
        indices = torch.as_tensor(context_indices(step, batch, horizon=first.horizon,
            episodes=config.episodes, transitions=config.transitions, memory_reset=memory_reset,
            rng=context_rng),
            device=reference.device)
        contexts = materialize_context(history, indices)
        distribution, critic = policy.distributions(contexts, current)
        raw = distribution.mean if deterministic else distribution.sample()
        # Official ECET uses an unsquashed Gaussian; clipping is the deterministic
        # action map. PPO scores the sampled latent, not its clipped replacement.
        actions = raw.clamp(-1., 1.).cpu().numpy()
        returned = batch_step(envs, actions)
        observations, rewards, terminated, truncated, info = map(list, zip(*returned, strict=True))
        if any(truncated) or any(end != (step == length - 1) for end in terminated):
            raise RuntimeError("ECET requires complete fixed-length meta-trials")
        if record:
            values = dict(context_indices=indices, raw_actions=raw, log_probs=distribution.log_prob(raw),
                values=critic.mean[:, 0], rewards=reference.new_tensor(rewards),
                trial_ends=torch.tensor(terminated, dtype=torch.bool, device=reference.device),
                attempt_ends=torch.tensor([item["attempt_end"] for item in info],
                                          dtype=torch.bool, device=reference.device))
            for key, value in values.items():
                records[key].append(value)
    return dict(observations=history, **{key: torch.stack(values) for key, values in records.items()}) if record else None


@torch.no_grad()
def discounted_returns(rewards, ends, gamma):
    result, carry = torch.zeros_like(rewards), torch.zeros_like(rewards[0])
    for step in range(len(rewards) - 1, -1, -1):
        carry = rewards[step] + gamma * carry * (~ends[step])
        result[step] = carry
    return result


@torch.no_grad()
def replay_error(policy, rollout, batch_size=256):
    history = rollout["observations"]
    current = history.flatten(0, 1)
    indices = rollout["context_indices"].flatten(0, 1)
    actions = rollout["raw_actions"].flatten(0, 1)
    errors = dict(log_probability=0., value=0.)
    for start in range(0, len(current), batch_size):
        stop = start + batch_size
        actor, critic = policy.distributions(materialize_context(history, indices[start:stop]), current[start:stop])
        errors["log_probability"] = max(errors["log_probability"], float((actor.log_prob(actions[start:stop])
            - rollout["log_probs"].flatten()[start:stop]).abs().max()))
        errors["value"] = max(errors["value"], float((critic.mean[:, 0]
            - rollout["values"].flatten()[start:stop]).abs().max()))
    return errors


def update_ecet(policy, optimizer, rollout, *, epochs=10, minibatch_size=32, gamma=.9,
                gae_lambda=.95, clip=.2, entropy_coef=.02, max_grad_norm=5.,
                normalize_advantages=False):
    """Official separate actor PPO / Gaussian-NLL critic transition minibatches.

    Returns use full-trial discounted Monte Carlo targets; GAE supplies actor
    advantages. Each cached input context is re-encoded with current weights.
    """
    if (epochs < 1 or minibatch_size < 1 or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1
            or not 0 < clip < 1 or entropy_coef < 0 or max_grad_norm <= 0):
        raise ValueError("Invalid ECET PPO configuration")
    policy.eval()
    advantages, _ = generalized_advantages(rollout["rewards"], rollout["values"],
        rollout["trial_ends"], gamma=gamma, gae_lambda=gae_lambda)
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    returns = discounted_returns(rollout["rewards"], rollout["trial_ends"], gamma)
    current = rollout["observations"].flatten(0, 1)
    indices = rollout["context_indices"].flatten(0, 1)
    actions = rollout["raw_actions"].flatten(0, 1)
    old_log_prob = rollout["log_probs"].flatten()
    totals = dict(actor_loss=0., critic_loss=0., exploration_entropy=0., approximate_kl=0., clip_fraction=0.)
    samples, actor_steps, critic_steps = len(current) * epochs, 0, 0
    # Keep the official order: all actor epochs, then all independent critic epochs.
    for _ in range(epochs):
        order = torch.randperm(len(current), device=current.device)
        for rows in order.split(minibatch_size):
            distribution, _ = policy.distributions(materialize_context(rollout["observations"], indices[rows]), current[rows])
            log_ratio = distribution.log_prob(actions[rows]) - old_log_prob[rows]
            ratio, advantage = log_ratio.exp(), advantages.flatten()[rows]
            surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
            entropy = distribution.entropy()
            loss = -surrogate.mean() - entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite ECET actor loss")
            optimizer.actor.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor_parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.actor.step()
            for key, value in dict(actor_loss=loss, exploration_entropy=entropy.mean(),
                    approximate_kl=((ratio - 1) - log_ratio).mean(),
                    clip_fraction=((ratio - 1).abs() > clip).float().mean()).items():
                totals[key] += float(value.detach()) * len(rows)
            actor_steps += 1
    for _ in range(epochs):
        order = torch.randperm(len(current), device=current.device)
        for rows in order.split(minibatch_size):
            distribution = Independent(Normal(policy.critic(current[rows]), policy.critic_log_std.exp()), 1)
            loss = -distribution.log_prob(returns.flatten()[rows, None]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite ECET critic NLL")
            optimizer.critic.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.critic_parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.critic.step()
            totals["critic_loss"] += float(loss.detach()) * len(rows)
            critic_steps += 1
    return dict(**{key: value / samples for key, value in totals.items()},
                optimizer_steps=actor_steps + critic_steps, actor_steps=actor_steps,
                critic_steps=critic_steps, forward_samples=2 * samples)


class ECETOptimizers:
    """Checkpoint-compatible pair; actor and independent critic never share Adam moments."""

    def __init__(self, policy, policy_lr=5e-5, value_lr=5e-5):
        self.actor = torch.optim.AdamW(policy.actor_parameters(), lr=policy_lr, weight_decay=.01)
        self.critic = torch.optim.AdamW(policy.critic_parameters(), lr=value_lr, weight_decay=.01)

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict())

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])


@torch.no_grad()
def evaluate_ecet(policy, pools, *, attempts, horizon, lane_count, conditions=("normal",),
                  seed_start=1_000_000_000, env_workers=0, allocated_cpus=16):
    records = {}
    with preserve_random_streams():
        for split in ("train_eval", "test"):
            factory = make_worker_lanes if env_workers else make_lanes
            worker_options = dict(workers=env_workers, allocated_cpus=allocated_cpus) if env_workers else {}
            envs = factory(pools[split], lane_count, benchmark="ml10", attempts=attempts,
                           horizon=horizon, **worker_options)
            try:
                records[split] = {}
                for condition in conditions:
                    # Pair the stochastic context-window selection across interventions.
                    np.random.seed(seed_start)
                    collect_ecet(policy, envs, np.arange(seed_start, seed_start + len(envs)),
                        deterministic=True, memory_reset=condition == "memory_reset",
                        reward_blind=condition == "reward_blind", record=False)
                    outcomes = np.asarray([env.outcomes for env in envs], dtype=bool)
                    families = sorted({env.family for env in envs})
                    by_family = {name: outcomes[[env.family == name for env in envs]].mean(0).tolist()
                                 for name in families}
                    curve = np.asarray(list(by_family.values())).mean(0)
                    records[split][condition] = dict(success=float(curve.mean()),
                        success_by_episode=curve.tolist(), per_family=by_family,
                        outcomes=outcomes.tolist(), native_returns=[env.native_returns for env in envs],
                        family_by_lane=[env.family for env in envs], trials=len(envs),
                        environment_steps=len(envs) * attempts * horizon)
            finally:
                for env in envs:
                    env.close()
    return records


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=1000)
    result.add_argument("--benchmark-seed", type=int, default=0)
    result.add_argument("--num-envs", type=int, default=50)
    result.add_argument("--attempts", type=int, default=5)
    result.add_argument("--horizon", type=int, default=500)
    result.add_argument("--updates", type=int, default=80)
    result.add_argument("--width", type=int, default=32)
    result.add_argument("--context-episodes", type=int, default=5)
    result.add_argument("--context-transitions", type=int, default=5)
    result.add_argument("--ppo-epochs", type=int, default=10)
    result.add_argument("--minibatch-size", type=int, default=32)
    result.add_argument("--policy-lr", type=float, default=5e-5)
    result.add_argument("--value-lr", type=float, default=5e-5)
    result.add_argument("--gamma", type=float, default=.9)
    result.add_argument("--entropy-coef", type=float, default=.02)
    result.add_argument("--normalize-advantages", action="store_true")
    result.add_argument("--eval-every", type=int, default=5)
    result.add_argument("--eval-attempts", type=int, default=25)
    result.add_argument("--eval-lanes", type=int, default=10)
    result.add_argument("--max-elapsed-seconds", type=float, default=3600.)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--env-workers", type=int, default=0,
                        help="Zero: serial simulator reference; positive: CPU worker groups.")
    result.add_argument("--allocated-cpus", type=int, default=16)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--resume", type=Path)
    result.add_argument("--qualify-one-update", action="store_true",
                        help="With --updates 1, time one complete native update without evaluation; no learning curve.")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    evidence, source = allocation(), clean_source(args.source_commit)
    if (min(args.num_envs, args.attempts, args.horizon, args.updates, args.ppo_epochs,
            args.minibatch_size, args.eval_every, args.eval_attempts, args.eval_lanes,
            args.threads, args.allocated_cpus) < 1 or args.env_workers < 0
            or args.horizon > 500 or args.context_transitions > args.horizon
            or not 0 <= args.seed < 1_000_000_000 or not 0 <= args.benchmark_seed < 1_000_000
            or not math.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0
            or not all(math.isfinite(value) and value > 0 for value in (args.policy_lr, args.value_lr))
            or not 0 <= args.gamma <= 1 or not math.isfinite(args.entropy_coef) or args.entropy_coef < 0):
        raise ValueError("Invalid bounded ECET benchmark settings")
    if args.qualify_one_update and (args.updates != 1 or args.resume):
        raise ValueError("One-update qualification requires --updates 1 and a fresh learner")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("ECET GPU-hour accounting requires exactly one visible allocated GPU; run the launcher probe")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = ECETConfig(width=args.width, feedforward=4 * args.width,
                        episodes=args.context_episodes, transitions=args.context_transitions)
    policy = ECETPolicy(config).to(args.device)
    optimizer = ECETOptimizers(policy, args.policy_lr, args.value_lr)
    sampler = np.random.default_rng(args.seed)
    pools = benchmark_pools("ml10", args.benchmark_seed)
    factory = make_worker_lanes if args.env_workers else make_lanes
    worker_options = dict(workers=args.env_workers, allocated_cpus=args.allocated_cpus) if args.env_workers else {}
    envs = factory(pools["train"], args.num_envs, benchmark="ml10", attempts=args.attempts,
                   horizon=args.horizon, **worker_options)
    optimization = dict(epochs=args.ppo_epochs, minibatch_size=args.minibatch_size,
        gamma=args.gamma, gae_lambda=.95, clip=.2, entropy_coef=args.entropy_coef,
        max_grad_norm=5., normalize_advantages=args.normalize_advantages)
    transitions_per_update = args.num_envs * args.attempts * args.horizon
    optimizer_steps_per_network = args.ppo_epochs * math.ceil(transitions_per_update / args.minibatch_size)
    data_reuse = dict(transitions_per_update=transitions_per_update,
        actor_passes=args.ppo_epochs, critic_passes=args.ppo_epochs,
        actor_samples_per_update=transitions_per_update * args.ppo_epochs,
        critic_samples_per_update=transitions_per_update * args.ppo_epochs,
        actor_optimizer_steps_per_update=optimizer_steps_per_network,
        critic_optimizer_steps_per_update=optimizer_steps_per_network,
        total_optimizer_steps_per_transition=2 * optimizer_steps_per_network / transitions_per_update)
    contract = dict(protocol="metaworld_ecet_transition_ppo_v1", source_commit=source,
        mode="one_update_throughput" if args.qualify_one_update else "training",
        upstream=UPSTREAM, core="ecet", core_config=asdict(config), benchmark="ml10",
        benchmark_seed=args.benchmark_seed, seed=args.seed,
        pool_hashes={key: task_digest(pool[1]) for key, pool in pools.items()},
        goal_visible=True, resample_instance=True, attempts=args.attempts, horizon=args.horizon,
        num_envs=args.num_envs, observation="39D state + previous executed action/reward/boundary; no clocks",
        normalization="none", reward="native dense reward multiplied by 0.1; success separate",
        empty_history="official unmasked zero tokens with IET sinusoidal positions; not padding neutral",
        context_rng="NumPy Generator derived from trial reset seed vector; independent of simulator/global RNG",
        actor_critic="ECET two-level context encoder; independent current-observation Gaussian NLL critic",
        action="unsquashed Gaussian latent, executed clip[-1,1]; latent likelihood and entropy",
        optimization=optimization, data_reuse=data_reuse,
        optimizer=dict(name="separate AdamW", policy_lr=args.policy_lr,
            value_lr=args.value_lr, weight_decay=.01), dtype="float32", device=args.device,
        evaluation=dict(attempts=args.eval_attempts, lanes=args.eval_lanes, seed_start=1_000_000_000,
            action_selection="deterministic clipped Gaussian mean; sampled context windows",
            success="any success in episode; equal episode and family average"),
        versions={name: importlib.metadata.version(name) for name in
                  ("torch", "numpy", "gymnasium", "mujoco", "metaworld")})
    counters = dict(environment_steps=0, optimizer_steps=0, evaluation_steps=0,
                    training_seconds=0., evaluation_seconds=0., cumulative_elapsed_seconds=0.)
    iteration = 0
    if args.resume:
        iteration, counters = restore_checkpoint(args.resume, policy=policy, optimizer=optimizer,
                                                environment_rng=sampler, contract=contract)
    parent_elapsed = counters["cumulative_elapsed_seconds"]
    if iteration >= args.updates:
        raise ValueError("Target updates must exceed the restored iteration")
    write_json(args.output_dir / "config.json", dict(contract=contract, allocation=evidence,
        execution=dict(env_workers=args.env_workers, allocated_cpus=args.allocated_cpus,
                       worker_backend=envs[0]._worker_batch.manifest if args.env_workers else "serial"),
        parameters=sum(parameter.numel() for parameter in policy.parameters()),
        actor_parameters=sum(parameter.numel() for parameter in policy.actor_parameters()),
        critic_parameters=sum(parameter.numel() for parameter in policy.critic_parameters()),
        target_updates=args.updates, eval_every=args.eval_every, threads=args.threads,
        max_elapsed_seconds=args.max_elapsed_seconds, resume=str(args.resume) if args.resume else None,
        accounting="one allocated GPU; all active wall time includes CPU simulation and evaluation"))
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    checkpoint_iteration = None

    def snapshot():
        nonlocal checkpoint_iteration
        synchronize(args.device)
        counters["cumulative_elapsed_seconds"] = parent_elapsed + time.monotonic() - started
        digest = save_checkpoint(args.output_dir / "latest.pt", policy=policy, optimizer=optimizer,
            environment_rng=sampler, iteration=iteration, counters=counters, contract=contract)
        checkpoint_iteration = iteration
        return digest

    def evaluation(conditions=("normal",)):
        before = time.monotonic()
        result = evaluate_ecet(policy, pools, attempts=args.eval_attempts, horizon=args.horizon,
                               lane_count=args.eval_lanes, conditions=conditions,
                               env_workers=args.env_workers, allocated_cpus=args.allocated_cpus)
        synchronize(args.device)
        counters["evaluation_seconds"] += time.monotonic() - before
        counters["evaluation_steps"] += sum(value["environment_steps"] for split in result.values()
                                             for value in split.values())
        return result

    def record(stream, **values):
        synchronize(args.device)
        elapsed = parent_elapsed + time.monotonic() - started
        row = dict(iteration=iteration, environment_steps=counters["environment_steps"],
            elapsed_seconds=elapsed, gpu_hours=elapsed / 3600 if args.device == "cuda" else None,
            training_gpu_hours=counters["training_seconds"] / 3600 if args.device == "cuda" else None,
            **values)
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        compact = {key: value for key, value in row.items() if key != "evaluation"}
        if "evaluation" in row:
            compact["success"] = {key: {condition: value["success"] for condition, value in split.items()}
                                  for key, split in row["evaluation"].items()}
        print(json.dumps(compact, allow_nan=False), flush=True)

    try:
        snapshot()
        with (args.output_dir / "metrics.jsonl").open("x", buffering=1) as stream:
            if not args.qualify_one_update:
                record(stream, evaluation=evaluation())
            while (iteration < args.updates and not stop
                   and time.monotonic() - started < args.max_elapsed_seconds):
                before = time.monotonic()
                seeds = sampler.integers(0, 1_000_000_000, size=args.num_envs)
                rollout = collect_ecet(policy, envs, seeds)
                synchronize(args.device)
                collection = time.monotonic() - before
                error = replay_error(policy, rollout) if iteration == 0 else None
                if error is not None and max(error.values()) > 2e-5:
                    raise RuntimeError(f"ECET behavior replay differs: {error}")
                update_started = time.monotonic()
                metrics = update_ecet(policy, optimizer, rollout, **optimization)
                synchronize(args.device)
                update_seconds = time.monotonic() - update_started
                counters["training_seconds"] += time.monotonic() - before
                counters["environment_steps"] += rollout["rewards"].numel()
                counters["optimizer_steps"] += metrics["optimizer_steps"]
                iteration += 1
                snapshot()
                result = dict(training=metrics, behavior_replay_error=error,
                    rollout_success=float(np.mean([env.outcomes for env in envs])),
                    timing=dict(rollout_seconds=collection, update_seconds=update_seconds))
                if not args.qualify_one_update and (iteration % args.eval_every == 0 or iteration == args.updates):
                    result["evaluation"] = evaluation()
                record(stream, **result)
            last_evaluation = None if args.qualify_one_update else evaluation(("normal", "memory_reset", "reward_blind"))
            record(stream, final=True, **({"qualification_only": True} if args.qualify_one_update
                                          else {"evaluation": last_evaluation}))
        digest = snapshot()
        write_json(args.output_dir / "summary.json", dict(completed=iteration == args.updates,
            iteration=iteration, source_commit=source, counters=counters,
            qualification_only=args.qualify_one_update,
            checkpoint_sha256=digest, last_evaluation=last_evaluation,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0))
    except BaseException as exc:
        write_json(args.output_dir / "failure.json", dict(error=type(exc).__name__, message=str(exc),
            last_complete_iteration=iteration, last_checkpoint_iteration=checkpoint_iteration,
            checkpoint=str(args.output_dir / "latest.pt") if checkpoint_iteration is not None else None))
        raise
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
