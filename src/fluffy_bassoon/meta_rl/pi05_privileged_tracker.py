"""Privileged target tracking through the actual bounded native flow interface.

This is a diagnostic controller, not a learned CTM, student input interface or
PPO behavior. Theta selects a private decoded-action target. Execution still
uses the original policy's four Euler passes with residuals bounded by one.
"""

import math

import torch

from fluffy_bassoon.rl.pi05_bridge_policy import StochasticPi05BridgePolicy
from .bridge_distillation import corrected_action_label


PROTOCOL = "native_pi05_privileged_bounded_target_tracker_v1"


class PrivilegedTargetTracker(StochasticPi05BridgePolicy):
    """Track an inverse-rotation target using only the existing residual channel.

    Private theta and target never enter bridge.read_context or public feedback.
    At each pass, residual = clip(gain*((x_t-target)/t-native_velocity), -1, 1).
    Exact zero-theta lanes instead use residual zero, preserving the complete
    nonlinear native trajectory. A straight-line tracker would not preserve that
    trajectory merely because its terminal target equals the native endpoint.
    """

    def __init__(self, bridge, *, gain=1., noise_scale=.05, active_dims=7):
        if (isinstance(gain, bool) or not math.isfinite(gain) or not 0 < gain <= 1
                or active_dims != 7 or bridge.residual_scale != 1.):
            raise ValueError("Require gain in (0,1], active7 and the same residual bound1")
        super().__init__(bridge, noise_scale=noise_scale, active_dims=active_dims)
        self.gain = float(gain)
        self.register_buffer("saved_tracker_gain", torch.tensor(self.gain, dtype=torch.float64))
        self._theta_degrees = None
        self._target = None
        self._zero_theta = None
        self.requires_grad_(False)
        self.eval()

    def config(self):
        return dict(super().config(), protocol=PROTOCOL, gain=self.gain,
                    controller="private target tracker through bounded residual; not learned CTM",
                    zero_theta="exact zero residual preserves native flow path",
                    eligibility="deterministic diagnostic only; no PPO replay or training")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "saved_tracker_gain")
        if saved is not None and not torch.equal(saved.cpu(), self.saved_tracker_gain.cpu()):
            error_msgs.append("Privileged tracker gain differs from its saved contract")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def select_theta(self, theta_degrees):
        values = [float(theta) for theta in theta_degrees]
        if not values or any(not math.isfinite(theta) for theta in values):
            raise ValueError("Require one finite private theta per batch lane")
        self._theta_degrees = values
        return self

    @torch.no_grad()
    def set_gain(self, gain):
        """Set a declared calibration candidate and keep its saved contract exact."""
        if isinstance(gain, bool) or not math.isfinite(gain) or not 0 < gain <= 1:
            raise ValueError("Require tracker gain in (0,1]")
        self.gain = float(gain)
        self.saved_tracker_gain.fill_(self.gain)
        return self

    def _mean(self, native_velocity, suffix, noisy_actions, timestep, features):
        if native_velocity.dtype != torch.float32 or self._zero_theta is None:
            raise ValueError("Require float32 native flow and an active private tracker chunk")
        if bool(self._zero_theta.all()):
            return torch.zeros_like(native_velocity[..., :self.active_dims])
        if self._target is None or self._target.shape != noisy_actions.shape or not (timestep > 0).all():
            raise ValueError("Require a finite-time paired native target")
        desired = (noisy_actions - self._target) / timestep[:, None, None]
        mean = (self.gain * (desired - native_velocity))[..., :self.active_dims]
        mean = mean.clamp(-self.bridge.residual_scale, self.bridge.residual_scale)
        return mean.masked_fill(self._zero_theta[:, None, None], 0)

    @torch.no_grad()
    def sample_chunk(self, model, observation, feedback, memory=None, *, rng=None,
                     deterministic=True, reset=None, num_steps=4, noise=None, decode_state=None):
        """Run paired native preparation, then the unchanged super sampler path.

        observation is already preprocessed. No model.sample_actions call or
        second preprocessing occurs. decode_state can supply the original
        public observation.state if a preprocessor modifies that field.
        Only nonzero-theta batches incur the extra paired native query. A mixed
        batch queries all lanes and records their full actual extra cost.
        """
        if (deterministic is not True or num_steps != 4 or model.action_horizon != 8
                or self.training or self.bridge.training or model.training
                or any(parameter.requires_grad for parameter in model.parameters())):
            raise ValueError("Require deterministic eval, frozen native motor and the same four-Euler horizon8 path")
        batch, device = observation.state.shape[0], observation.state.device
        if (self._theta_degrees is None or len(self._theta_degrees) != batch
                or feedback.shape[0] != batch):
            raise ValueError("Private theta selection and public batch lanes must match")
        shape = (batch, model.action_horizon, model.action_dim)
        if model.action_dim != self.bridge.action_dim:
            raise ValueError("Native and bridge action dimensions differ")
        if noise is None:
            if rng is None:
                raise ValueError("Provide explicit initial noise or its checkpointed generator")
            noise = torch.randn(shape, device=device, dtype=torch.float32, generator=rng)
        if (noise.shape != shape or noise.device != device or noise.dtype != torch.float32
                or not torch.isfinite(noise).all()):
            raise ValueError("Require finite float32 paired initial noise on the native device")
        self._zero_theta = torch.tensor([theta == 0. for theta in self._theta_degrees], device=device)
        target, extra_prefix, extra_suffix = None, 0, 0
        try:
            if not bool(self._zero_theta.all()):
                _, mask, cache = model.build_prefix_cache(observation)
                native = noise.clone()
                for step in range(4):
                    timestep = torch.full((batch,), 1. - step / 4., device=device, dtype=torch.float32)
                    suffix = model.run_suffix(observation, native, timestep, cache, mask)
                    velocity = model.velocity_from_suffix(suffix)
                    if velocity.dtype != torch.float32 or not torch.isfinite(velocity).all():
                        raise ValueError("Paired native velocity must be finite float32")
                    native = native - .25 * velocity
                state = observation.state if decode_state is None else decode_state
                target = corrected_action_label(model, state, native, self._theta_degrees).detach()
                self._target = target
                extra_prefix, extra_suffix = batch, 4 * batch
            # This is the real intervention path. Never return target as actions.
            actions, next_memory, record = super().sample_chunk(model, observation, feedback, memory,
                rng=rng, deterministic=True, reset=reset, num_steps=4, noise=noise.clone())
            executed = model.decode_actions(actions, observation.state if decode_state is None else decode_state).shape[1]
            difference = torch.zeros_like(actions) if target is None else actions - target
            metadata = dict(protocol=PROTOCOL, gain=self.gain, theta_degrees=list(self._theta_degrees),
                controller="privileged analytic tracker; not learned CTM competence",
                deterministic=True, actual_sampler="StochasticPi05BridgePolicy.sample_chunk",
                num_steps=4, active_dims=self.active_dims, residual_scale=self.bridge.residual_scale,
                actual_executed_action_steps=executed,
                extra_native_prefix_queries=extra_prefix, extra_native_suffix_queries=extra_suffix,
                residual_mean_abs_max=float(record["behavior_means"].abs().max()),
                residual_bound_fraction=float((record["behavior_means"].abs() >= self.bridge.residual_scale).float().mean()),
                target_active_model_action_mse=float(difference[..., :self.active_dims].square().mean()),
                target_executed_model_action_mse=float(difference[:, :executed, :self.active_dims].square().mean()),
                zero_theta_lanes=int(self._zero_theta.sum()), training_eligible=False)
            if not torch.isfinite(actions).all() or not all(math.isfinite(value) for value in (
                    metadata["target_active_model_action_mse"], metadata["target_executed_model_action_mse"])):
                raise RuntimeError("Non-finite bounded privileged tracker output")
            record["privileged_tracker"] = metadata
            return actions, next_memory, record
        finally:
            # Targets are per-observation scratch data, never recurrent state.
            self._target = None
            self._zero_theta = None

    def replay_chunk(self, *args, **kwargs):
        raise ValueError("Privileged target tracker is not student PPO behavior or a training policy")
