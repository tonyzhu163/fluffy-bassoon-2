"""Full-trial recurrent PPO for the compact meta-RL environments."""

import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.utils import flatdim, flatten
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from lerobot_policy_ctm.architectures.persistent_transformer_control.action_expert import PersistentTransformerExpert
from lerobot_policy_ctm.shared.persistent_ctm import PersistentCTMCore


class GRUCore(nn.Module):
    def __init__(self, input_dim, width=64, ticks=1):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (input_dim, width, ticks)):
            raise ValueError("Require positive integer input_dim, width and ticks")
        self.input_dim, self.output_dim, self.ticks = input_dim, width, ticks
        self.cell = nn.GRUCell(input_dim, width)

    def initial_state(self, batch):
        return self.cell.weight_hh.new_zeros(batch, self.output_dim)

    def forward(self, observation, state=None):
        if state is None:
            state = self.initial_state(observation.shape[0])
        for _ in range(self.ticks):
            state = self.cell(observation, state)
        return state, state


class LoopedTransformerCore(nn.Module):
    """Vector adapter for the existing persistent, weight-tied Transformer expert."""

    def __init__(self, input_dim, width=64, ticks=1, memory_tokens=8):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (input_dim, width, ticks, memory_tokens)) or width % 4:
            raise ValueError("Require positive counts and width divisible by four")
        self.input_dim, self.output_dim, self.ticks = input_dim, width, ticks
        config = SimpleNamespace(context_dim=input_dim, expert_width=width, expert_depth=ticks,
                                 memory_tokens=memory_tokens, chunk_size=1, expert_heads=4,
                                 action_feature=SimpleNamespace(shape=(width,)))
        self.expert = PersistentTransformerExpert(config)
        self.expert.readout = nn.Identity()

    def initial_state(self, batch):
        return {"tokens": self.expert.initial_memory.expand(batch, -1, -1)}

    def forward(self, observation, state=None):
        valid = torch.ones(observation.shape[0], 1, dtype=torch.bool, device=observation.device)
        projected = self.expert.project_context(observation[:, None], valid)
        features, state = self.expert(projected, valid, memory=state)
        return features[:, 0], state


class PersistentCTMVectorCore(nn.Module):
    """Vector adapter for the robot CTM recurrence: one observation token, one population.

    Unlike the `ctm` control (the CMM port without memory blocks), this runs the robot
    expert's dual synchronization banks, deep neuron-level models and attention."""

    def __init__(self, input_dim, width=64, ticks=1, history_length=10, synapse_depth=8,
                 neuron_hidden=16, heads=4, bounded_synchronisation=False, decay_per_observation=False):
        super().__init__()
        if (any(type(value) is not int or value < 1 for value in
                (input_dim, width, ticks, history_length, neuron_hidden, heads))
                or type(synapse_depth) is not int or synapse_depth < 2 or width % heads or width % 2
                or type(bounded_synchronisation) is not bool or type(decay_per_observation) is not bool):
            raise ValueError("Invalid persistent CTM core dimensions or options")
        self.input_dim, self.output_dim, self.ticks = input_dim, width, ticks
        self.expert = PersistentCTMCore(input_dim, population_count=1, readout_dim=width, width=width,
                                        ticks=ticks, heads=heads, history_length=history_length,
                                        neuron_hidden=neuron_hidden, synapse_depth=synapse_depth,
                                        attention_pairs=2 * width, output_pairs=2 * width)
        self.expert.bounded_synchronisation = bounded_synchronisation
        self.expert.decay_per_observation = decay_per_observation

    def initial_state(self, batch):
        # Zero action accumulators reproduce the episode-start (memory=None) first tick exactly.
        expert = self.expert
        activated = expert.start_activated_state.expand(batch, -1)
        _, out_alpha, out_beta = expert.compute_synchronisation(activated, None, None, None, "out")
        zeros = out_alpha.new_zeros(out_alpha.shape)
        return dict(activated=activated, trace=expert.start_trace.expand(batch, -1, -1),
                    action_alpha=zeros, action_beta=zeros, out_alpha=out_alpha, out_beta=out_beta)

    @torch.no_grad()
    def constrain_parameters(self):
        self.expert.decay_params_action.clamp_(0, 15)
        self.expert.decay_params_out.clamp_(0, 15)

    def forward(self, observation, state=None):
        valid = torch.ones(observation.shape[0], 1, dtype=torch.bool, device=observation.device)
        projected = self.expert.project_context(observation[:, None], valid)
        conditioning = self.expert.conditioning_norm(self.expert.positions.expand(observation.shape[0], -1, -1))
        features, state = self.expert.decode_persistent(valid, projected, conditioning, state)
        return features[:, 0], state


class RecurrentActorCritic(nn.Module):
    def __init__(self, core, action_space):
        super().__init__()
        self.core = core
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.discrete:
            action_dim = int(action_space.n)
        elif (isinstance(action_space, gym.spaces.Box) and len(action_space.shape) == 1
              and np.all(action_space.low == -1) and np.all(action_space.high == 1)):
            action_dim = action_space.shape[0]
            self.log_std = nn.Parameter(torch.full((action_dim,), -.5))
        else:
            raise ValueError("Require Discrete actions or a one-dimensional Box bounded by [-1, 1]")
        self.actor = nn.Linear(core.output_dim, action_dim)
        self.critic = nn.Linear(core.output_dim, 1)

    def step(self, observation, state=None, *, reset=None, raw_action=None, deterministic=False, bounded_entropy=False):
        if reset is not None:
            if reset.shape != observation.shape[:1] or reset.dtype != torch.bool or reset.device != observation.device:
                raise ValueError("Reset mask must be boolean [batch] on the observation device")
            if state is not None:
                initial = self.core.initial_state(observation.shape[0])
                if isinstance(state, dict):
                    state = {key: torch.where(reset.reshape(-1, *([1] * (value.ndim - 1))),
                                              initial[key], value) for key, value in state.items()}
                else:
                    state = torch.where(reset[:, None], initial, state)
        if bounded_entropy and self.discrete:
            raise ValueError("Bounded entropy requires a continuous squashed Gaussian policy")
        features, state = self.core(observation, state)
        parameters, value = self.actor(features), self.critic(features).squeeze(-1)
        if self.discrete:
            distribution = Categorical(logits=parameters)
            if raw_action is None:
                raw_action = parameters.argmax(-1) if deterministic else distribution.sample()
            action = raw_action
            log_prob, entropy = distribution.log_prob(raw_action), distribution.entropy()
        else:
            distribution = Normal(parameters, self.log_std.clamp(-5, 2).exp())
            if raw_action is None:
                raw_action = parameters if deterministic else distribution.sample()
            action = raw_action.tanh()
            correction = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
            log_prob = (distribution.log_prob(raw_action) - correction).sum(-1)
            entropy = distribution.entropy().sum(-1)
            if bounded_entropy:
                # Reparameterize a fresh current-policy sample, not the recorded behavior action.
                sample = distribution.rsample()
                entropy = entropy + (2 * (math.log(2) - sample - F.softplus(-2 * sample))).sum(-1)
        return action, raw_action, log_prob, entropy, value, state

    def evaluate_sequence(self, observations, raw_actions, resets=None, *, bounded_entropy=False):
        state, log_probs, entropies, values = None, [], [], []
        for step in range(observations.shape[0]):
            _, _, log_prob, entropy, value, state = self.step(
                observations[step], state, raw_action=raw_actions[step],
                reset=None if resets is None else resets[step], bounded_entropy=bounded_entropy)
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)


@torch.no_grad()
def collect_trials(policy, envs, seeds, *, reset_each_attempt=False, deterministic=False, normalize_clock=True):
    if not envs or len(envs) != len(seeds):
        raise ValueError("Require one reset seed per environment")
    if type(normalize_clock) is not bool:
        raise ValueError("Clock normalization selection must be boolean")
    first = envs[0]
    if any((env.attempts, env.horizon, env.observation_space, env.action_space) !=
           (first.attempts, first.horizon, first.observation_space, first.action_space) for env in envs):
        raise ValueError("Batched trials must share the same environment contract")
    if policy.core.input_dim != flatdim(first.observation_space):
        raise ValueError("Policy input dimension does not match environment observations")
    policy.eval()
    reference = next(policy.parameters())
    observations = [env.reset(seed=int(seed))[0] for env, seed in zip(envs, seeds, strict=True)]
    state = None
    records = {key: [] for key in ("observations", "raw_actions", "log_probs", "values",
                                   "rewards", "trial_ends", "attempt_ends", "resets", "action_mask", "settling_steps")}
    previous_settling = [0] * len(envs)
    clock_scale = np.maximum(first.observation_space["clock"].high, 1) if normalize_clock else 1
    for step in range(first.attempts * first.horizon):
        inputs = torch.as_tensor(np.stack([flatten(first.observation_space,
                                                   dict(obs, clock=obs["clock"] / clock_scale))
                                          for obs in observations]),
                                 dtype=reference.dtype, device=reference.device)
        resets = torch.tensor([reset_each_attempt and bool(obs["attempt_end"].item()) for obs in observations],
                              dtype=torch.bool, device=reference.device)
        action, raw_action, log_prob, _, value, state = policy.step(
            inputs, state, reset=resets, deterministic=deterministic)
        returned = [env.step(a) for env, a in zip(envs, action.cpu().numpy(), strict=True)]
        observations, rewards, ended, truncated, info = map(list, zip(*returned, strict=True))
        if any(truncated) or any(done != (step == first.attempts * first.horizon - 1) for done in ended):
            raise RuntimeError("Collector requires complete fixed-length trials without external truncation")
        settling = [int(item.get("settling_steps", 0)) for item in info]
        increments = [current - previous for current, previous in zip(settling, previous_settling, strict=True)]
        if any(value < 0 for value in increments):
            raise RuntimeError("Reset-action counters changed within a trial")
        previous_settling = settling
        values = dict(observations=inputs, raw_actions=raw_action, log_probs=log_prob,
                      values=value, rewards=reference.new_tensor(rewards),
                      trial_ends=torch.tensor(ended, dtype=torch.bool, device=reference.device),
                      attempt_ends=torch.tensor([item["attempt_end"] for item in info],
                                               dtype=torch.bool, device=reference.device), resets=resets,
                      action_mask=torch.tensor([item.get("simulator_step", True) for item in info],
                                               dtype=torch.bool, device=reference.device),
                      settling_steps=torch.tensor(increments, dtype=torch.int64, device=reference.device))
        for key in records:
            records[key].append(values[key])
    return {key: torch.stack(value) for key, value in records.items()}


@torch.no_grad()
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


def update_ppo(policy, optimizer, rollout, *, epochs=4, minibatch_trials=16, clip=.2,
               gamma=1., gae_lambda=1., entropy_coef=.01, value_coef=.5,
               max_grad_norm=1., episode_returns=False, reward_offset=0., normalize_advantages=True, bounded_entropy=False):
    if (type(epochs) is not int or epochs < 1 or type(minibatch_trials) is not int or minibatch_trials < 1
            or not 0 < clip < 1 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0
            or not math.isfinite(entropy_coef) or entropy_coef < 0
            or not math.isfinite(value_coef) or value_coef < 0
            or not math.isfinite(reward_offset) or not 0 <= reward_offset <= 1
            or type(normalize_advantages) is not bool or type(bounded_entropy) is not bool):
        raise ValueError("Invalid PPO optimization settings")
    ends = rollout["attempt_ends"] if episode_returns else rollout["trial_ends"]
    advantages, returns = generalized_advantages(rollout["rewards"] - reward_offset, rollout["values"], ends,
                                                gamma=gamma, gae_lambda=gae_lambda)
    action_mask = rollout.get("action_mask", torch.ones_like(advantages, dtype=torch.bool))
    if (action_mask.shape != advantages.shape or action_mask.dtype != torch.bool
            or action_mask.device != advantages.device or not action_mask.any(0).all()):
        raise ValueError("Require a boolean action mask with at least one real action per trial")
    if normalize_advantages:
        active = advantages[action_mask]
        advantages = (advantages - active.mean()) / (active.std(unbiased=False) + 1e-8)
    advantages = torch.where(action_mask, advantages, 0.)
    batch = advantages.shape[1]
    totals, count, optimizer_steps, active_samples = {}, 0, 0, 0
    # Deterministic recurrent transitions are required for behavior-policy likelihood replay.
    policy.eval()
    for _ in range(epochs):
        order = torch.randperm(batch, device=advantages.device)
        for start in range(0, batch, minibatch_trials):
            lanes = order[start:start + minibatch_trials]
            log_prob, entropy, value = policy.evaluate_sequence(
                rollout["observations"][:, lanes], rollout["raw_actions"][:, lanes], rollout["resets"][:, lanes],
                bounded_entropy=bounded_entropy)
            mask = action_mask[:, lanes]
            log_ratio = torch.where(mask, log_prob - rollout["log_probs"][:, lanes], 0.)
            ratio = log_ratio.exp()
            advantage = advantages[:, lanes]
            actor_loss = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1 - clip, 1 + clip)).mean()
            old_value, target = rollout["values"][:, lanes], returns[:, lanes]
            clipped_value = old_value + (value - old_value).clamp(-clip, clip)
            critic_loss = .5 * torch.maximum((value - target).square(), (clipped_value - target).square()).mean()
            entropy = torch.where(mask, entropy, 0.)
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite recurrent PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            if hasattr(policy.core, "constrain_parameters"):
                policy.core.constrain_parameters()
            weight = advantage.numel()
            metrics = dict(loss=loss.detach(), actor_loss=actor_loss.detach(),
                           critic_loss=critic_loss.detach(), exploration_entropy=entropy.detach().mean(),
                           approximate_kl=((ratio - 1) - log_ratio).detach().mean(),
                           clip_fraction=((ratio - 1).abs() > clip).float().mean(), gradient_norm=gradient_norm)
            for key, metric in metrics.items():
                totals[key] = totals.get(key, 0.) + float(metric) * weight
            count += weight
            optimizer_steps += 1
            active_samples += int(mask.sum())
    return {**{key: value / count for key, value in totals.items()},
            "optimizer_steps": optimizer_steps, "forward_samples": count, "active_action_samples": active_samples}
