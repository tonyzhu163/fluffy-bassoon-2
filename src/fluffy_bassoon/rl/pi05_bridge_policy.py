"""Exact augmented-action likelihoods for a stochastic frozen-pi0.5 residual.

The policy samples Gaussian residual *velocities* at each native Euler pass.
Their joint density is tractable even though the marginal density of decoded
environment actions is not. PPO must replay complete chronological chunks.
"""

import math

import torch
from torch import nn
from torch.distributions import Normal


RECORD_PROTOCOL = "pi05_gaussian_flow_residual_v1"


def _cpu(tensor):
    return tensor.detach().to(device="cpu", copy=True)


def record_bytes(record):
    """Logical tensor storage, excluding Python container overhead."""
    return sum(value.numel() * value.element_size() for value in record.values()
               if isinstance(value, torch.Tensor))


class StochasticPi05BridgePolicy(nn.Module):
    """Frozen native dynamics, learned residual means and a shared-feature critic.

    sigma is fixed, so entropy is constant. Residual means are bounded by the
    bridge's tanh; Gaussian samples remain unbounded. Only the first active_dims
    model channels receive residuals. Other channels follow native velocities.
    """

    def __init__(self, bridge, *, noise_scale=.05, active_dims=7):
        super().__init__()
        if (type(active_dims) is not int or not 1 <= active_dims <= bridge.action_dim
                or isinstance(noise_scale, bool) or not math.isfinite(noise_scale) or noise_scale <= 0
                or bridge.kv_layer != -1):
            raise ValueError("Require a final-layer bridge, positive fixed noise and valid active dimensions")
        self.bridge, self.noise_scale, self.active_dims = bridge, float(noise_scale), active_dims
        self.register_buffer("saved_policy_config", torch.tensor(
            [self.noise_scale, self.active_dims, bridge.residual_scale], dtype=torch.float64))
        self.critic = nn.Linear(bridge.width, 1)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def initial_state(self, batch):
        return self.bridge.initial_state(batch)

    def config(self):
        return dict(protocol=RECORD_PROTOCOL, noise_scale=self.noise_scale, active_dims=self.active_dims,
                    residual_scale=self.bridge.residual_scale, kv_layer=self.bridge.kv_layer,
                    likelihood="joint Gaussian residual velocities; sum over passes, horizon and active channels",
                    entropy="fixed sigma; constant entropy", dtype="float32")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "saved_policy_config")
        if saved is not None and not torch.equal(saved.cpu(), self.saved_policy_config.cpu()):
            error_msgs.append("Saved flow policy noise, active dimensions or residual scale differs")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _mean(self, native_velocity, suffix, noisy_actions, timestep, features):
        if native_velocity.dtype != torch.float32:
            raise ValueError("The qualified native flow velocity must be float32")
        corrected = self.bridge.correct_velocity(native_velocity, suffix, noisy_actions, timestep, features)
        # Use this identical float32 computational mean for sampling and replay.
        return (corrected - native_velocity.detach())[..., :self.active_dims]

    def _distribution(self, mean):
        return Normal(mean, mean.new_full((), self.noise_scale))

    @torch.no_grad()
    def sample_chunk(self, model, observation, feedback, memory=None, *, rng=None, deterministic=False,
                     reset=None, num_steps=4, noise=None):
        """Return (model_actions, next_memory, detached_CPU_record).

        observation is already processed with RLinf preprocess_observation.
        Decode model_actions with model.decode_actions and the original
        observation.state. This method never changes motor parameters.
        """
        if (model.training or self.training or self.bridge.training
                or any(parameter.requires_grad for parameter in model.parameters())):
            raise ValueError("Require eval-mode policy and frozen eval-mode native motor")
        if type(num_steps) is not int or num_steps < 1 or type(deterministic) is not bool:
            raise ValueError("Require a positive denoising count and boolean deterministic flag")
        batch, device = observation.state.shape[0], observation.state.device
        shape = (batch, model.action_horizon, model.action_dim)
        if model.action_dim != self.bridge.action_dim:
            raise ValueError("Native action dimension differs from bridge")
        if noise is None:
            noise = torch.randn(shape, device=device, dtype=torch.float32, generator=rng)
        if noise.shape != shape or noise.device != device or noise.dtype != torch.float32:
            raise ValueError("Require float32 native initial noise [B,H,A] on motor device")
        _, prefix_mask, kv_cache = model.build_prefix_cache(observation)
        features, memory = self.bridge.read_context(kv_cache, prefix_mask, feedback, memory, reset=reset)
        value = self.critic(features).squeeze(-1)
        record = dict(protocol=RECORD_PROTOCOL, noise_scale=self.noise_scale, active_dims=self.active_dims,
                      num_steps=num_steps, deterministic=deterministic,
                      kv_key=_cpu(kv_cache[-1][0]), kv_value=_cpu(kv_cache[-1][1]),
                      prefix_mask=_cpu(prefix_mask), feedback=_cpu(feedback),
                      reset_mask=None if reset is None else _cpu(reset),
                      bridge_features=_cpu(features), old_value=_cpu(value), recurrence_updates=1)
        transitions = {key: [] for key in ("noisy_actions", "native_velocity", "suffix_hidden", "timesteps",
                                           "sampled_residuals", "behavior_means", "step_log_probs")}
        x_t, dt, time = noise, -1. / num_steps, 1.
        collected_log_probs, collected_entropies = [], []
        for index in range(num_steps):
            timestep = torch.full((batch,), time, device=device, dtype=torch.float32)
            suffix = model.run_suffix(observation, x_t, timestep, kv_cache, prefix_mask).detach()
            native_velocity = model.velocity_from_suffix(suffix).detach()
            mean = self._mean(native_velocity, suffix, x_t, timestep, features)
            distribution = self._distribution(mean)
            sampled = mean if deterministic else mean + self.noise_scale * torch.randn(
                mean.shape, device=device, dtype=mean.dtype, generator=rng)
            step_log_prob = distribution.log_prob(sampled).sum((-1, -2))
            collected_log_probs.append(step_log_prob)
            collected_entropies.append(distribution.entropy().sum((-1, -2)))
            for key, tensor in dict(noisy_actions=x_t, native_velocity=native_velocity, suffix_hidden=suffix,
                                    timesteps=timestep, sampled_residuals=sampled, behavior_means=mean,
                                    step_log_probs=step_log_prob).items():
                transitions[key].append(_cpu(tensor))
            velocity = native_velocity.clone()
            velocity[..., :self.active_dims] += sampled
            x_t = x_t + dt * velocity
            time += dt
        log_prob = torch.stack(collected_log_probs).sum(0)
        entropy = torch.stack(collected_entropies).sum(0)
        record.update({key: torch.stack(values) for key, values in transitions.items()})
        record.update(old_log_prob=_cpu(log_prob), entropy=_cpu(entropy),
                      gaussian_scalars_per_lane=num_steps * model.action_horizon * self.active_dims,
                      residual_mean_abs_max=float(record["behavior_means"].abs().max()),
                      sampled_residual_abs_max=float(record["sampled_residuals"].abs().max()),
                      storage_bytes=0)
        record["storage_bytes"] = record_bytes(record)
        return x_t, memory, record

    def replay_chunk(self, record, memory=None, *, reset=None):
        """Return (log_prob[B], next_memory, differentiable diagnostics).

        Saved behavior noisy-actions/native outputs are conditioning context.
        Recompute current recurrent features; never reuse saved bridge_features
        for actor or critic fitting. Carry next_memory to the next chunk without
        detaching when replaying a complete trial.
        """
        if (record.get("protocol") != RECORD_PROTOCOL or record.get("noise_scale") != self.noise_scale
                or record.get("active_dims") != self.active_dims or record.get("deterministic") is not False
                or type(record.get("num_steps")) is not int or record["num_steps"] < 1):
            raise ValueError("Require a matching stochastic behavior record")
        if self.training or self.bridge.training:
            raise ValueError("Replay uses eval mode to keep recurrent transitions deterministic")
        reference = next(self.parameters())
        def saved(key):
            return record[key].detach().to(reference.device)
        prefix_mask = saved("prefix_mask")
        feedback = saved("feedback").to(reference.dtype)
        cached_reset = record.get("reset_mask")
        if reset is None and cached_reset is not None:
            reset = cached_reset.to(reference.device)
        features, memory = self.bridge.read_context(((saved("kv_key"), saved("kv_value")),),
                                                    prefix_mask, feedback, memory, reset=reset)
        values = self.critic(features).squeeze(-1)
        noisy_actions, native_velocity = saved("noisy_actions"), saved("native_velocity")
        suffix_hidden, timesteps = saved("suffix_hidden"), saved("timesteps")
        sampled = saved("sampled_residuals")
        expected = (record["num_steps"], features.shape[0])
        if (noisy_actions.shape[:2] != expected or native_velocity.shape != noisy_actions.shape
                or suffix_hidden.shape[:3] != noisy_actions.shape[:3]
                or suffix_hidden.shape[-1] != self.bridge.suffix_dim or timesteps.shape != expected
                or sampled.shape != (*noisy_actions.shape[:3], self.active_dims)):
            raise ValueError("Malformed saved flow trajectory")
        log_probs, entropies, means = [], [], []
        for index in range(record["num_steps"]):
            mean = self._mean(native_velocity[index], suffix_hidden[index], noisy_actions[index],
                              timesteps[index], features)
            distribution = self._distribution(mean)
            log_probs.append(distribution.log_prob(sampled[index]).sum((-1, -2)))
            entropies.append(distribution.entropy().sum((-1, -2)))
            means.append(mean)
        step_log_probs = torch.stack(log_probs)
        log_prob, entropy = step_log_probs.sum(0), torch.stack(entropies).sum(0)
        diagnostics = dict(log_prob=log_prob, value=values, entropy=entropy, features=features,
                           means=torch.stack(means), step_log_probs=step_log_probs,
                           recurrence_updates=1, gaussian_scalars_per_lane=sampled[:, 0].numel())
        return log_prob, memory, diagnostics
