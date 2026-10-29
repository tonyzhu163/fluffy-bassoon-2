"""Trial-wise GTrXL/PPO over a frozen VLA's initial-noise interface.

This module contains no simulator or VLA implementation. Real computation and
tests belong in allocated Linux jobs. Episode boundaries are feedback tokens;
only a trial boundary ends the meta-return. Complete recurrent sequences are
replayed from empty memory, including episode reset masks for the trained
reset control. Native motor features are computed once during collection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .gtrxl import GatedTransformerXLCore
from .ppo import RecurrentActorCritic, generalized_advantages


@dataclass(frozen=True)
class PPOConfig:
    width: int = 128
    layers: int = 2
    heads: int = 4
    memory_length: int = 300
    gate_bias: float = 2.0
    action_limit: float = 1.5
    learning_rate: float = 3e-4
    epochs: int = 5
    minibatch_trials: int = 16
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    dual_clip: float = 3.0
    value_clip: float = .2
    huber_delta: float = 10.0
    value_weight: float = 1.0
    max_grad_norm: float = 1.0
    train_episodes: int = 4
    episode_limit: int = 400
    reset_each_episode: bool = False

    def __post_init__(self):
        for name in ("width", "layers", "heads", "memory_length", "epochs",
                     "minibatch_trials", "train_episodes", "episode_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads or self.width % 2:
            raise ValueError("Require even width divisible by attention heads")
        for name in ("action_limit", "learning_rate", "value_clip", "huber_delta",
                     "value_weight", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.clip < 1 or self.dual_clip <= 1:
            raise ValueError("Invalid PPO clipping parameters")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Discounts must be in [0, 1]")


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def observation_hash(value):
    array = np.ascontiguousarray(value, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def policy_inputs(features, previous_noise, previous_reward, episode_boundary,
                  episode_index, primitive_step, *, train_episodes, episode_limit,
                  reward_blind=False):
    """Causal public input; clocks clamp to the training range during K8 tests."""
    features = np.asarray(features, dtype=np.float32).reshape(-1)
    noise = np.asarray(previous_noise, dtype=np.float32).reshape(-1)
    if not np.isfinite(features).all() or not np.isfinite(noise).all():
        raise FloatingPointError("Nonfinite frozen-motor features or previous noise")
    scalars = [0. if reward_blind else float(previous_reward), float(episode_boundary),
               min(int(episode_index), train_episodes - 1) / max(train_episodes - 1, 1),
               min(int(primitive_step), episode_limit) / episode_limit]
    result = np.concatenate((features, noise, np.asarray(scalars, np.float32)))
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite public feedback")
    return result


class RecurrentNoisePPO:
    """Shared across source tasks; only memory changes at frozen evaluation."""

    def __init__(self, observation_dim, noise_shape, *, config=None, device="cuda", seed=0):
        self.config = config or PPOConfig()
        self.observation_dim = int(observation_dim)
        self.noise_shape = tuple(int(x) for x in noise_shape)
        self.action_dim = math.prod(self.noise_shape)
        if self.observation_dim < 1 or self.action_dim < 1:
            raise ValueError("Invalid observation/noise dimensions")
        self.device = torch.device(device)
        self.sampler = np.random.default_rng(seed)
        c = self.config
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            core = GatedTransformerXLCore(self.observation_dim + self.action_dim + 4,
                                         width=c.width, layers=c.layers, heads=c.heads,
                                         memory_length=c.memory_length, gate_bias=c.gate_bias)
            self.policy = RecurrentActorCritic(
                core, gym.spaces.Box(-1., 1., (self.action_dim,), np.float32)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=c.learning_rate,
                                          betas=(.9, .95), eps=1e-8, weight_decay=.01)
        self.updates = self.optimizer_steps = self.primitive_steps = self.chunks = 0

    @torch.no_grad()
    def collect_trial(self, env, task_name, episode_seeds, *, deterministic=False,
                      reset_memory=None, reward_blind=False, episode_callback=None,
                      primitive_budget=None, decision_callback=None):
        """Run complete episodes; sequential execution preserves causal adaptation.

        Environment step returns its exact primitive-step increment. This can be
        shorter than the native chunk at termination or the 400-step wrapper cap.
        No optimizer update occurs within this function or the trial.
        """
        if not episode_seeds:
            raise ValueError("A trial requires at least one episode")
        if primitive_budget is not None and (type(primitive_budget) is not int or primitive_budget < 1):
            raise ValueError("Primitive budget must be a positive integer")
        c = self.config
        reset_memory = c.reset_each_episode if reset_memory is None else bool(reset_memory)
        self.policy.eval()
        state = None
        previous_noise = np.zeros(self.action_dim, np.float32)
        previous_reward = 0.
        keys = ("observations", "raw_actions", "log_probs", "values", "rewards",
                "trial_ends", "attempt_ends", "resets")
        records = {key: [] for key in keys}
        episodes = []
        total_primitive = 0
        budget_cut = False
        bootstrap = None
        for episode, seed in enumerate(episode_seeds):
            if c.reset_each_episode:
                previous_noise = np.zeros(self.action_dim,np.float32)
                previous_reward = 0.
            features = np.asarray(env.reset(task_name, int(seed)), np.float32)
            if features.shape != (self.observation_dim,):
                raise ValueError("Runtime feature dimension changed")
            initial_hash = observation_hash(features)
            done = False
            primitive = chunks = 0
            reward_sum = 0.
            success = False
            while not done:
                inputs = policy_inputs(features, previous_noise, previous_reward,
                                       not c.reset_each_episode and episode > 0 and chunks == 0,
                                       episode, primitive,
                                       train_episodes=c.train_episodes,
                                       episode_limit=c.episode_limit, reward_blind=reward_blind)
                tensor = torch.as_tensor(inputs[None], device=self.device)
                reset = torch.tensor([reset_memory and chunks == 0], device=self.device)
                action, raw, log_prob, _, value, state = self.policy.step(
                    tensor, state, reset=reset, deterministic=deterministic)
                noise = action[0].cpu().numpy() * c.action_limit
                if decision_callback is not None:
                    # Publish the behavior-policy quantities before the real
                    # action. This is fresh on-policy evidence, not SAC replay.
                    decision_callback(dict(
                        episode_index=episode, chunk_index=chunks,
                        history_index=len(records["rewards"]),
                        policy_update=int(self.updates),
                        pre_squash_actions=raw[0].detach().cpu().numpy().copy(),
                        old_log_prob=log_prob[0].detach().cpu().numpy().copy(),
                        old_value=value[0].detach().cpu().numpy().copy(),
                        policy_input=inputs.copy(),
                        recurrent_reset_mask=reset[0].detach().cpu().numpy().copy(),
                        previous_noise=previous_noise.copy(),
                        previous_reward=float(previous_reward),
                        episode_boundary=bool(not c.reset_each_episode and episode > 0 and chunks == 0),
                        reset_episode_feedback=bool(c.reset_each_episode),
                        reward_blind=bool(reward_blind)))
                if primitive_budget is None:
                    following, reward, terminated, truncated, info = env.step(noise)
                else:
                    following, reward, terminated, truncated, info = env.step(
                        noise, max_primitive_steps=primitive_budget-total_primitive)
                steps = int(info["primitive_steps"])
                if steps < 1 or steps > int(env.native_horizon):
                    raise ValueError("Require positive exact primitive-step increment")
                primitive += steps
                total_primitive += steps
                if primitive_budget is not None and total_primitive > primitive_budget:
                    raise RuntimeError("Runtime exceeded exact primitive budget")
                if primitive > c.episode_limit:
                    raise RuntimeError("Runtime exceeded the wrapper episode limit")
                done = bool(terminated or truncated)
                exhausted = primitive_budget is not None and total_primitive >= primitive_budget
                natural_trial_end = done and episode == len(episode_seeds)-1 and not info.get("budget_truncated",False)
                budget_cut = bool(exhausted and not natural_trial_end)
                if primitive >= c.episode_limit and not done:
                    raise RuntimeError("Runtime must truncate at its episode limit")
                previous_noise, previous_reward = noise, float(reward)
                features = np.asarray(following, np.float32)
                if features.shape != (self.observation_dim,) or not np.isfinite(features).all():
                    raise FloatingPointError("Invalid next public features")
                values = {"observations": tensor[0], "raw_actions": raw[0],
                          "log_probs": log_prob[0], "values": value[0],
                          "rewards": tensor.new_tensor(float(reward)),
                          "trial_ends": torch.tensor(natural_trial_end,
                                                     device=self.device),
                          "attempt_ends": torch.tensor(done, device=self.device),
                          "resets": reset[0]}
                for key, value in values.items():
                    records[key].append(value.detach().cpu())
                reward_sum += float(reward)
                success |= bool(info.get("success", False))
                chunks += 1
                if budget_cut:
                    # Bootstrap an externally cut collection, retaining recurrent
                    # history and the actual episode-boundary semantics.
                    next_episode = episode
                    next_primitive = primitive
                    boundary = done and not info.get("budget_truncated",False)
                    if boundary:
                        next_episode += 1
                        next_primitive = 0
                        if next_episode < len(episode_seeds):
                            features = np.asarray(env.reset(task_name,int(episode_seeds[next_episode])),np.float32)
                    following_inputs = policy_inputs(
                        features,np.zeros_like(previous_noise) if boundary and c.reset_each_episode else previous_noise,
                        0. if boundary and c.reset_each_episode else previous_reward,
                        boundary and not c.reset_each_episode,next_episode,next_primitive,
                        train_episodes=c.train_episodes,episode_limit=c.episode_limit,
                        reward_blind=reward_blind)
                    _,_,_,_,next_value,_ = self.policy.step(
                        torch.as_tensor(following_inputs[None],device=self.device),state,
                        reset=torch.tensor([reset_memory and boundary],device=self.device),
                        deterministic=True)
                    bootstrap=next_value.detach().cpu()
                    done=True
            row = dict(task=task_name, episode=episode+1, seed=int(seed), success=success,
                       return_=reward_sum, primitive_steps=primitive, chunks=chunks,
                       initial_observation_sha256=initial_hash,
                       reset_memory=reset_memory, reward_blind=bool(reward_blind),
                       reset_episode_feedback=c.reset_each_episode,
                       budget_truncated=budget_cut)
            episodes.append(row)
            if episode_callback is not None:
                episode_callback(row)
            if exhausted:
                break
        rollout = {key: torch.stack(value)[:, None] for key, value in records.items()}
        rollout["action_mask"] = torch.ones_like(rollout["trial_ends"], dtype=torch.bool)
        if bootstrap is not None:
            rollout["bootstrap"]=bootstrap
        return rollout, dict(task=task_name, episodes=episodes,
                             primitive_steps=sum(x["primitive_steps"] for x in episodes),
                             chunks=sum(x["chunks"] for x in episodes),
                             budget_truncated=budget_cut,
                             requested_episodes=len(episode_seeds))

    @torch.no_grad()
    def replay_error(self, rollout):
        self.policy.eval()
        arrays = {k: v.to(self.device) for k, v in rollout.items()}
        log_probs, _, values = self.policy.evaluate_sequence(
            arrays["observations"], arrays["raw_actions"], arrays["resets"])
        return {"log_prob_max_error": float((log_probs-arrays["log_probs"]).abs().max()),
                "value_max_error": float((values-arrays["values"]).abs().max())}

    def update(self, rollouts):
        """PPO over complete variable-length trials; no per-chunk shuffling."""
        if not rollouts:
            raise ValueError("Need at least one complete trial")
        c = self.config
        data = []
        for rollout in rollouts:
            r = {key: value.to(self.device) for key, value in rollout.items()}
            if r["trial_ends"][:-1].any() or (not r["trial_ends"][-1].all() and "bootstrap" not in r):
                raise ValueError("Need a terminal trial or an explicitly bootstrapped budget cut")
            advantages, returns = generalized_advantages(
                r["rewards"], r["values"], r["trial_ends"], gamma=c.gamma,
                gae_lambda=c.gae_lambda,bootstrap=r.get("bootstrap"))
            data.append((r, advantages, returns))
        active = torch.cat([a[r["action_mask"]] for r, a, _ in data])
        mean, std = active.mean(), active.std(unbiased=False)
        data = [(r, (a-mean)/(std+1e-8), ret) for r, a, ret in data]
        totals, weight_sum, optimizer_steps = {}, 0, 0
        self.policy.eval()  # no dropout, matching sampled likelihoods
        for _ in range(c.epochs):
            order = torch.randperm(len(data)).tolist()
            for start in range(0, len(order), c.minibatch_trials):
                batch = [data[i] for i in order[start:start+c.minibatch_trials]]
                count = sum(int(r["action_mask"].sum()) for r, _, _ in batch)
                self.optimizer.zero_grad(set_to_none=True)
                metrics = {key: 0. for key in ("loss", "actor_loss", "critic_loss", "approximate_kl", "clip_fraction")}
                for r, advantage, target in batch:
                    log_prob, _, value = self.policy.evaluate_sequence(
                        r["observations"], r["raw_actions"], r["resets"])
                    mask = r["action_mask"]
                    log_ratio = log_prob-r["log_probs"]
                    ratio = log_ratio.exp()
                    surrogate = torch.minimum(ratio*advantage,
                                              ratio.clamp(1-c.clip, 1+c.clip)*advantage)
                    surrogate = torch.where(advantage < 0,
                                            torch.maximum(surrogate, c.dual_clip*advantage), surrogate)
                    actor = -surrogate[mask].sum()/count
                    clipped = r["values"]+(value-r["values"]).clamp(-c.value_clip,c.value_clip)
                    losses = torch.maximum(F.huber_loss(value,target,delta=c.huber_delta,reduction="none"),
                                           F.huber_loss(clipped,target,delta=c.huber_delta,reduction="none"))
                    critic = losses[mask].sum()/count
                    loss = actor+c.value_weight*critic
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite recurrent PPO loss")
                    loss.backward()
                    metrics["loss"] += float(loss.detach())
                    metrics["actor_loss"] += float(actor.detach())
                    metrics["critic_loss"] += float(critic.detach())
                    metrics["approximate_kl"] += float(((ratio-1)-log_ratio)[mask].sum().detach())/count
                    metrics["clip_fraction"] += float(((ratio-1).abs()>c.clip)[mask].sum())/count
                gradient = nn.utils.clip_grad_norm_(self.policy.parameters(),c.max_grad_norm,error_if_nonfinite=True)
                self.optimizer.step()
                metrics["gradient_norm"] = float(gradient)
                for key,value in metrics.items():
                    totals[key] = totals.get(key,0.)+value*count
                weight_sum += count
                optimizer_steps += 1
        self.updates += 1
        self.optimizer_steps += optimizer_steps
        self.chunks += sum(int(r["action_mask"].sum()) for r,_,_ in data)
        return {**{key:value/weight_sum for key,value in totals.items()},
                "optimizer_steps":optimizer_steps,"chunks":sum(len(r["rewards"]) for r,_,_ in data),
                "ppo_updates":self.updates}

    def state_dict(self):
        return dict(schema_version=1,kind="gtrxl_noise_ppo",config=asdict(self.config),
                    observation_dim=self.observation_dim,noise_shape=self.noise_shape,
                    policy=self.policy.state_dict(),optimizer=self.optimizer.state_dict(),
                    sampler=self.sampler.bit_generator.state,rng=capture_rng(),
                    updates=self.updates,optimizer_steps=self.optimizer_steps,
                    primitive_steps=self.primitive_steps,chunks=self.chunks,
                    boundary="complete trial and optimizer update")

    def load_state_dict(self,payload,*,restore_random=True):
        if (payload["config"]!=asdict(self.config) or payload["observation_dim"]!=self.observation_dim
                or tuple(payload["noise_shape"])!=self.noise_shape):
            raise ValueError("Recovery contract differs")
        self.policy.load_state_dict(payload["policy"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.sampler.bit_generator.state=payload["sampler"]
        for name in ("updates","optimizer_steps","primitive_steps","chunks"):
            setattr(self,name,int(payload[name]))
        if restore_random:
            restore_rng(payload["rng"])

    def save(self,path,**extra):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+".tmp")
        torch.save(dict(learner=self.state_dict(),**extra),temporary)
        temporary.replace(path)
