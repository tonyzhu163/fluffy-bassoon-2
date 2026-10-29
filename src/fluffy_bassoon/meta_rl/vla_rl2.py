"""Direct RL²/PPO steering of a frozen VLA, with complete scene-trial credit.

Reuse the repository's GTrXL, squashed-Gaussian actor/critic, full-trial PPO and
atomic optimizer/sampler/RNG recovery. One unpadded variable-length scene trial
is one PPO batch. There is no online weight update in frozen deployment.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .checkpoint import restore_checkpoint, save_checkpoint
from .gtrxl import GatedTransformerXLCore
from .ppo import RecurrentActorCritic, update_ppo
from .vla_history import FrozenHistoryController, file_sha256
from .warmstart import atomic_save, checked_load, validate_weights


MODEL_FORMAT = "frozen_vla_rl2_policy_v1"
MODES = ("full", "episode")


@dataclass(frozen=True)
class RL2Config:
    observation_dim: int = 58
    noise_dim: int = 32
    width: int = 128
    layers: int = 2
    heads: int = 4
    context_length: int = 300
    initial_std: float = .5

    def __post_init__(self):
        if (any(type(value) is not int or value < 1 for value in
                (self.observation_dim, self.noise_dim, self.width, self.layers, self.heads, self.context_length))
                or self.width % self.heads or self.width % 2
                or not math.isfinite(self.initial_std) or self.initial_std <= 0):
            raise ValueError("Invalid RL² controller dimensions or initial Gaussian scale")


class RL2ActorCritic(RecurrentActorCritic):
    """The existing actor/critic, with optional fixed public-feature normalization."""
    def __init__(self, config):
        core = GatedTransformerXLCore(config.observation_dim + config.noise_dim + 3,
            width=config.width, layers=config.layers, heads=config.heads, memory_length=config.context_length)
        super().__init__(core, gym.spaces.Box(-1., 1., (config.noise_dim,), np.float32))
        self.config = config
        self.register_buffer("observation_mean", torch.zeros(config.observation_dim))
        self.register_buffer("observation_std", torch.ones(config.observation_dim))
        with torch.no_grad():
            self.log_std.fill_(math.log(config.initial_std))

    @torch.no_grad()
    def set_feature_normalization(self, mean, std):
        mean = torch.as_tensor(mean, device=self.observation_mean.device, dtype=self.observation_mean.dtype)
        std = torch.as_tensor(std, device=self.observation_std.device, dtype=self.observation_std.dtype)
        if (mean.shape != self.observation_mean.shape or std.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("Require finite fixed feature normalization with positive scales")
        self.observation_mean.copy_(mean)
        self.observation_std.copy_(std)

    def step(self, observation, state=None, **kwargs):
        d = self.config.observation_dim
        normalized = torch.cat(((observation[:, :d] - self.observation_mean) / self.observation_std,
                                observation[:, d:]), dim=-1)
        return super().step(normalized, state, **kwargs)


def make_rl2_policy(config=None, *, device="cpu", seed=0):
    torch.manual_seed(seed)
    return RL2ActorCritic(config or RL2Config()).to(device)


@torch.no_grad()
def collect_rl2_trial(policy, sim, motor, feature_encoder, *, initial_state, scene_seed,
        attempts, chunk, mode="full", deterministic=False, episode_runner=None):
    """One real K-attempt scene trial; terminal feedback is consumed next episode.

    `feature_encoder` consumes the current full public observation and returns
    the same fixed representation used by the distillation students. Scene and
    reset identities stay in receipts. Policy input t contains only observation
    t and transition t-1. The simulator's actual noise map must preserve the
    sampled bounded action, otherwise PPO likelihoods would be invalid.
    """
    if mode not in MODES or attempts < 1 or chunk < 1:
        raise ValueError("Require a positive complete scene trial and valid history mode")
    if episode_runner is None:
        from .vla_libero import run_episode
        episode_runner = run_episode
    policy.eval()
    reference = next(policy.parameters())
    state, previous_noise = None, np.zeros(policy.config.noise_dim, np.float32)
    previous_reward, previous_done, episode_start = 0., False, True
    records = {key: [] for key in ("observations", "raw_actions", "log_probs", "values", "rewards",
                                   "trial_ends", "attempt_ends", "resets")}
    summaries, initial_hash = [], None
    pending = None
    for attempt in range(attempts):
        records_before = len(records["observations"])
        def choose(observation):
            nonlocal state, pending, previous_noise, previous_reward, previous_done
            if pending is not None:
                raise RuntimeError("The simulator must report an outcome before another RL² query")
            features = np.asarray(feature_encoder(observation), np.float32)
            if features.shape != (policy.config.observation_dim,) or not np.isfinite(features).all():
                raise ValueError("RL² current public features differ from the fixed representation")
            reset = mode == "episode" and episode_start
            if reset:
                previous_noise = np.zeros_like(previous_noise)
                previous_reward, previous_done = 0., False
            token = np.concatenate((features, previous_noise,
                np.asarray([previous_reward, previous_done, episode_start], np.float32)))
            inputs = torch.as_tensor(token[None], device=reference.device, dtype=reference.dtype)
            resets = torch.tensor([reset], device=reference.device, dtype=torch.bool)
            action, raw, log_prob, _, value, state = policy.step(inputs, state, reset=resets,
                                                               deterministic=deterministic)
            pending = dict(observations=inputs, raw_actions=raw, log_probs=log_prob,
                           values=value, resets=resets, sampled_noise=action[0].cpu().numpy())
            return pending["sampled_noise"].copy()

        def feedback(executed_noise, reward, done):
            nonlocal pending, previous_noise, previous_reward, previous_done, episode_start
            actual = np.asarray(executed_noise, np.float32)
            if (pending is None or actual.shape != previous_noise.shape or not np.isfinite(actual).all()
                    or not np.allclose(actual, pending["sampled_noise"], rtol=0, atol=1e-6)
                    or not math.isfinite(float(reward)) or type(done) not in (bool, np.bool_)):
                raise ValueError("PPO requires the actual unchanged sampled bounded noise and real outcome")
            values = dict((key, pending[key]) for key in records if key in pending)
            values.update(rewards=reference.new_tensor([float(reward)]),
                trial_ends=torch.tensor([bool(done) and attempt == attempts - 1], device=reference.device,
                                       dtype=torch.bool),
                attempt_ends=torch.tensor([bool(done)], device=reference.device, dtype=torch.bool))
            for key in records:
                records[key].append(values[key].detach())
            previous_noise, previous_reward, previous_done = actual.copy(), float(reward), bool(done)
            episode_start, pending = bool(done), None

        rows, summary = episode_runner(sim, motor, choose, initial_state=int(initial_state),
            scene_seed=int(scene_seed), chunk=chunk, bounded=True, feedback=feedback)
        if (pending is not None or not rows or not rows[-1]["done"]
                or any(row["done"] for row in rows[:-1]) or not episode_start
                or len(records["observations"]) - records_before != len(rows)
                or int(summary["primitive_steps"]) != sum(int(row["primitive_steps"]) for row in rows)):
            raise RuntimeError("RL² collection requires complete episodes and every actual feedback callback")
        if initial_hash is None:
            initial_hash = summary["initial_observation_sha256"]
        elif summary["initial_observation_sha256"] != initial_hash:
            raise RuntimeError("Same-scene RL² retry changed its initial observation")
        summaries.append(dict(attempt=attempt + 1, **summary))
    rollout = {key: torch.stack(values) for key, values in records.items()}
    if not rollout["trial_ends"][-1].all() or int(rollout["attempt_ends"].sum()) != attempts:
        raise RuntimeError("RL² trial terminal masks do not match the actual completed episodes")
    return rollout, dict(initial_state=int(initial_state), scene_seed=int(scene_seed), attempts=summaries,
        primitive_steps=sum(int(summary["primitive_steps"]) for summary in summaries),
        settling_steps=int(getattr(sim, "metadata", {}).get("settle_steps", 0)) * attempts,
        mode=mode, decisions=len(rollout["observations"]))


class _DeploymentAdapter(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.base = policy
        self.config = policy.config

    def step(self, token, state=None, *, deterministic=False, generator=None):
        if generator is not None:
            raise ValueError("RL² uses the checkpointed global Torch RNG; seed it per paired trial")
        action, _, log_prob, _, _, state = self.base.step(token, state, deterministic=deterministic)
        return action, log_prob, state


class FrozenRL2Controller(FrozenHistoryController):
    """The same act/observe/reset_trial protocol as frozen AD/DPT deployment."""
    def __init__(self, policy, *, mode="full"):
        super().__init__(_DeploymentAdapter(policy), mode=mode)


def export_rl2_policy(path, policy, *, manifest):
    if not manifest or any(not torch.isfinite(value).all() for value in policy.state_dict().values()):
        raise ValueError("Require finite RL² weights and explicit training provenance")
    return atomic_save(path, dict(format=MODEL_FORMAT, config=asdict(policy.config),
        manifest=json.loads(json.dumps(manifest, allow_nan=False)),
        policy={key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}))


def load_rl2_policy(path, *, expected_sha256, device="cpu"):
    saved = checked_load(path, expected_sha256)
    if saved.get("format") != MODEL_FORMAT or not saved.get("manifest"):
        raise ValueError("Require the frozen RL² policy format and provenance")
    policy = RL2ActorCritic(RL2Config(**saved["config"]))
    validate_weights(policy, saved["policy"])
    policy.load_state_dict(saved["policy"], strict=True)
    return policy.to(device).eval().requires_grad_(False), saved["manifest"]


def train_rl2(policy, sim, motor, feature_encoder, *, train_states, attempts, chunk,
        episode_horizon, max_primitive_steps, updates, output_dir, source_contract,
        mode="full", seed=0, learning_rate=3e-4, epochs=4, gamma=1., gae_lambda=1.,
        entropy_coef=.01, value_coef=.5, clip=.2, max_grad_norm=1.,
        max_elapsed_seconds=None, resume=None, episode_runner=None, save_rollouts=True, heldout_states=()):
    """Direct meta-RL from real simulator trials, with one full-trial PPO batch.

    The caller owns allocation/probe and clean-source qualification. The budget
    counts new training primitives; any shared feature-prior cost stays in
    source_contract. Each iteration and recovery boundary is a whole scene trial.
    All parameters are fixed when the returned candidate is evaluated. No TBPTT,
    reward/return labels from other policies, or held-out feedback enter updates.
    """
    if sys.platform != "linux":
        raise ValueError("RL² model computation requires allocated Linux resources")
    if (mode not in MODES or not train_states or len(set(train_states)) != len(train_states)
            or any(type(value) is not int or value < 0 for value in train_states)
            or set(train_states) & set(heldout_states)
            or attempts < 1 or chunk < 1 or episode_horizon < 1 or updates < 1 or epochs < 1
            or max_primitive_steps < attempts * episode_horizon
            or not 0 <= seed < 2**32
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1 or not 0 < clip < 1
            or not math.isfinite(entropy_coef) or entropy_coef < 0
            or not math.isfinite(value_coef) or value_coef < 0
            or not math.isfinite(max_grad_norm) or max_grad_norm <= 0
            or (max_elapsed_seconds is not None and (not math.isfinite(max_elapsed_seconds) or max_elapsed_seconds <= 0))
            or not re.fullmatch(r"[0-9a-f]{40}", source_contract.get("source_commit", ""))):
        raise ValueError("Require explicit source, complete-trial resource budget and valid RL² settings")
    contract = dict(format="frozen_vla_rl2_training_v1", source=source_contract,
        config=asdict(policy.config), train_states=list(train_states), heldout_states=list(heldout_states),
        attempts=attempts,
        resource_budget=dict(max_primitive_steps=max_primitive_steps, updates=updates,
                             max_elapsed_seconds=max_elapsed_seconds),
        chunk=chunk, episode_horizon=episode_horizon, mode=mode, seed=seed,
        optimizer=dict(name="Adam", learning_rate=learning_rate),
        ppo=dict(epochs=epochs, minibatch_trials=1, clip=clip, gamma=gamma, gae_lambda=gae_lambda,
            entropy_coef=entropy_coef, value_coef=value_coef, max_grad_norm=max_grad_norm,
            bounded_entropy=True, episode_returns=False),
        credit="complete K-attempt trial; episode done does not terminate RL² return",
        backpropagation="complete unpadded scene trial; no TBPTT",
        feature_normalization="fixed buffers; identity unless explicitly set before training")
    output_dir = Path(output_dir)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sampler = np.random.default_rng(seed)
    policy.requires_grad_(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    iteration = 0
    counters = dict(environment_steps=0, settling_steps=0, rollout_forward_samples=0,
                    update_forward_samples=0, optimizer_steps=0)
    if resume is not None:
        resume = Path(resume)
        if output_dir.resolve() != resume.parent.resolve():
            raise ValueError("Resume must preserve the original output identity")
        iteration, counters = restore_checkpoint(resume, policy=policy, optimizer=optimizer,
            environment_rng=sampler, contract=contract)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "traces").mkdir()
        (output_dir / "contract.json").write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    started, last_metrics = time.monotonic(), None
    while iteration < updates and counters["environment_steps"] + attempts * episode_horizon <= max_primitive_steps:
        if max_elapsed_seconds is not None and time.monotonic() - started >= max_elapsed_seconds:
            break
        state = int(sampler.choice(train_states))
        scene_seed = int(sampler.integers(0, 2**31 - 1))
        before = time.monotonic()
        rollout, receipt = collect_rl2_trial(policy, sim, motor, feature_encoder,
            initial_state=state, scene_seed=scene_seed, attempts=attempts, chunk=chunk,
            mode=mode, episode_runner=episode_runner)
        with torch.no_grad():
            replay, _, _ = policy.evaluate_sequence(rollout["observations"], rollout["raw_actions"], rollout["resets"])
            replay_error = float((replay - rollout["log_probs"]).abs().max())
        if replay_error > 1e-4:
            raise RuntimeError("RL² behavior-policy likelihood replay failed before its optimizer update")
        if save_rollouts:
            atomic_save(output_dir / "traces" / f"trial-{iteration + 1:06d}.pt",
                dict(rollout={key: value.cpu() for key, value in rollout.items()}, receipt=receipt,
                     behavior_update=iteration, source_commit=source_contract["source_commit"]))
        metrics = update_ppo(policy, optimizer, rollout, minibatch_trials=1, epochs=epochs, clip=clip,
            gamma=gamma, gae_lambda=gae_lambda, entropy_coef=entropy_coef, value_coef=value_coef,
            max_grad_norm=max_grad_norm, bounded_entropy=True, episode_returns=False)
        if any(not torch.isfinite(value).all() for value in policy.parameters()):
            raise FloatingPointError("RL² update produced non-finite parameters")
        iteration += 1
        counters["environment_steps"] += receipt["primitive_steps"]
        counters["settling_steps"] += receipt["settling_steps"]
        counters["rollout_forward_samples"] += receipt["decisions"]
        counters["update_forward_samples"] += metrics["forward_samples"]
        counters["optimizer_steps"] += metrics["optimizer_steps"]
        last_metrics = dict(update=iteration, seconds=time.monotonic() - before,
            replay_error=replay_error, success_by_attempt=[row["success"] for row in receipt["attempts"]],
            counters=dict(counters), **metrics)
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(last_metrics, allow_nan=False) + "\n")
        for row in receipt["attempts"]:
            with (output_dir / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(method="rl2" if mode == "full" else "rl2_episode",
                    split="train", trial=iteration, cumulative_train_steps=counters["environment_steps"],
                    **row), allow_nan=False) + "\n")
        print(json.dumps(dict(rl2_training=last_metrics), allow_nan=False), flush=True)
        save_checkpoint(output_dir / "latest.pt", policy=policy, optimizer=optimizer,
            environment_rng=sampler, iteration=iteration, counters=counters, contract=contract)
    if iteration < 1:
        raise RuntimeError("RL² resource limits allowed no complete training trial")
    digest = export_rl2_policy(output_dir / "candidate.pt", policy,
        manifest=dict(contract=contract, updates=iteration, counters=counters,
            status="candidate; requires held-out robotic performance evaluation"))
    summary = dict(updates=iteration, counters=counters, candidate_sha256=digest,
        budget_guard=counters["environment_steps"] + attempts * episode_horizon > max_primitive_steps,
        latest_metrics=last_metrics)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return policy.eval().requires_grad_(False), summary
