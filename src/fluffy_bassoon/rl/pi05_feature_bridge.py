"""Recurrent native-KV readers with a zero-initialized frozen-pi0.5 residual.

This module does not load or modify a motor. Its sampler uses RLinf's public
prefix/suffix helpers; the caller owns allocated execution, feedback and memory
lifetime. Recurrent memory advances once per observation, not per Euler pass.
"""

import torch
from torch import nn

from lerobot_policy_ctm.shared.persistent_ctm import PersistentCTMCore
from fluffy_bassoon.meta_rl.transformer_xl import TransformerXLCore


def native_kv_tokens(kv_cache, prefix_mask, *, layer=-1):
    """Read a layer's RoPE keys and values as detached [B,P,2*KH*HD] tokens."""
    if (not isinstance(kv_cache, (list, tuple)) or not kv_cache
            or type(layer) is not int or not -len(kv_cache) <= layer < len(kv_cache)):
        raise ValueError("Require a native per-layer KV cache and valid layer index")
    pair = kv_cache[layer]
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError("Each native cache layer must contain keys and values")
    key, value = pair
    if (not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor)
            or key.ndim != 4 or key.shape != value.shape or not all(key.shape)
            or key.device != value.device or key.dtype != value.dtype or not key.is_floating_point()
            or prefix_mask.shape != key.shape[:2] or prefix_mask.dtype != torch.bool
            or prefix_mask.device != key.device or not prefix_mask.any(1).all()):
        raise ValueError("Require native KV [B,P,kv_heads,head_dim] and a nonempty boolean prefix mask")
    # Mask before projection, so padded cache entries cannot affect either learner.
    tokens = torch.cat((key.detach().flatten(2), value.detach().flatten(2)), dim=-1)
    return tokens.masked_fill(~prefix_mask[..., None], 0)


class Pi05FeatureBridge(nn.Module):
    """CTM or Transformer-XL reading the same frozen native KV and public feedback.

    Read the prefix once with read_context(). Reuse those features for every
    denoising pass via correct_velocity(). The output projection starts at zero,
    leaving the complete native action expert and its initial behavior intact.
    """

    def __init__(self, *, core="persistent_ctm", feedback_dim=11, kv_dim=512,
                 suffix_dim=1024, action_dim=32, width=64, ticks=1,
                 memory_length=32, residual_scale=.1, kv_layer=-1):
        super().__init__()
        if (core not in ("persistent_ctm", "transformer_xl")
                or any(type(value) is not int or value < 1
                       for value in (feedback_dim, kv_dim, suffix_dim, action_dim, width, ticks, memory_length))
                or width % 4 or type(kv_layer) is not int
                or not 0 < residual_scale <= 1):
            raise ValueError("Invalid native feature bridge dimensions or configuration")
        self.kind, self.feedback_dim, self.kv_dim = core, feedback_dim, kv_dim
        self.suffix_dim, self.action_dim, self.width = suffix_dim, action_dim, width
        self.residual_scale, self.kv_layer = float(residual_scale), kv_layer
        self.feedback_projection = nn.Sequential(nn.Linear(feedback_dim, width), nn.LayerNorm(width))
        if core == "persistent_ctm":
            self.core = PersistentCTMCore(kv_dim, population_count=1, readout_dim=width,
                                          width=width, ticks=ticks, heads=4, history_length=10,
                                          neuron_hidden=16, synapse_depth=8,
                                          attention_pairs=2 * width, output_pairs=2 * width)
            self.core.bounded_synchronisation = True
            self.core.decay_per_observation = True
        else:
            if ticks != 1:
                raise ValueError("Transformer-XL comparison uses one recurrent transition")
            self.reader = nn.MultiheadAttention(width, 4, kdim=kv_dim, vdim=kv_dim,
                                                dropout=0., batch_first=True)
            self.core = TransformerXLCore(2 * width, width=width, memory_length=memory_length,
                                          layers=2, heads=4)
        self.residual_input = nn.Sequential(
            nn.Linear(width + suffix_dim + action_dim + 1, width), nn.SiLU())
        self.residual_output = nn.Linear(width, action_dim)
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)

    def initial_state(self, batch):
        if type(batch) is not int or batch < 1:
            raise ValueError("Require a positive batch size")
        if self.kind == "transformer_xl":
            return self.core.initial_state(batch)
        core = self.core
        activated = core.start_activated_state.expand(batch, -1)
        _, out_alpha, out_beta = core.compute_synchronisation(activated, None, None, None, "out")
        zeros = out_alpha.new_zeros(out_alpha.shape)
        return dict(activated=activated, trace=core.start_trace.expand(batch, -1, -1),
                    action_alpha=zeros, action_beta=zeros, out_alpha=out_alpha, out_beta=out_beta)

    def _reset_state(self, memory, reset, batch):
        if reset is None:
            return memory
        reference = self.feedback_projection[0].weight
        if reset.shape != (batch,) or reset.dtype != torch.bool or reset.device != reference.device:
            raise ValueError("Reset must be boolean [B] on the bridge device")
        if memory is None:
            return None
        initial = self.initial_state(batch)
        if not isinstance(memory, dict) or memory.keys() != initial.keys():
            raise ValueError("Require the complete recurrent bridge state")
        return {key: torch.where(reset.reshape(batch, *([1] * (value.ndim - 1))), initial[key], value)
                for key, value in memory.items()}

    def read_context(self, kv_cache, prefix_mask, feedback, memory=None, *, reset=None):
        reference = self.feedback_projection[0].weight
        tokens = native_kv_tokens(kv_cache, prefix_mask, layer=self.kv_layer)
        batch = tokens.shape[0]
        if (tokens.shape[-1] != self.kv_dim or tokens.device != reference.device
                or feedback.shape != (batch, self.feedback_dim) or feedback.device != reference.device
                or feedback.dtype != reference.dtype or not torch.isfinite(feedback).all()):
            raise ValueError("KV width/device or public feedback shape/dtype differs from the bridge")
        tokens = tokens.to(reference.dtype)
        conditioning = self.feedback_projection(feedback)
        memory = self._reset_state(memory, reset, batch)
        if self.kind == "persistent_ctm":
            conditioning = self.core.conditioning_norm(conditioning[:, None] + self.core.positions[None])
            features, memory = self.core.decode_persistent(
                prefix_mask, self.core.project_context(tokens, prefix_mask), conditioning, memory)
            return features[:, 0], memory
        attended = self.reader(conditioning[:, None], tokens, tokens,
                               key_padding_mask=~prefix_mask, need_weights=False)[0][:, 0]
        return self.core(torch.cat((conditioning, attended), dim=-1), memory)

    def correct_velocity(self, native_velocity, suffix_hidden, noisy_actions, timestep, features):
        reference = self.feedback_projection[0].weight
        if (native_velocity.ndim != 3 or native_velocity.shape[-1] != self.action_dim
                or noisy_actions.shape != native_velocity.shape
                or suffix_hidden.shape != (*native_velocity.shape[:2], self.suffix_dim)
                or features.shape != (native_velocity.shape[0], self.width)
                or timestep.shape != (native_velocity.shape[0],)
                or any(value.device != reference.device for value in
                       (native_velocity, suffix_hidden, noisy_actions, timestep, features))):
            raise ValueError("Require native velocity/actions [B,H,A], suffix [B,H,D], time [B] and features [B,W]")
        horizon = native_velocity.shape[1]
        inputs = torch.cat((features[:, None].expand(-1, horizon, -1),
                            suffix_hidden.detach().to(reference.dtype),
                            noisy_actions.detach().to(reference.dtype),
                            timestep.detach().to(reference.dtype)[:, None, None].expand(-1, horizon, 1)), dim=-1)
        residual = self.residual_scale * self.residual_output(self.residual_input(inputs)).tanh()
        return native_velocity.detach() + residual.to(native_velocity.dtype)


@torch.no_grad()
def sample_prepared_with_bridge(model, bridge, observation, feedback, memory=None, *, noise=None,
                                rng=None, reset=None, num_steps=4):
    """Mirror native Euler inference on an already preprocessed RLinf Observation.

    This is an inference entry point, not a PPO likelihood implementation. The
    bridge exposes differentiable read/correct methods separately for training.
    """
    if model.training or bridge.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Require an eval-mode frozen motor and eval-mode bridge")
    if type(num_steps) is not int or num_steps < 1:
        raise ValueError("num_steps must be positive")
    batch, device = observation.state.shape[0], observation.state.device
    if noise is None:
        noise = torch.randn(batch, model.action_horizon, model.action_dim, device=device, generator=rng)
    if noise.shape != (batch, model.action_horizon, model.action_dim):
        raise ValueError("Initial noise differs from native motor action dimensions")
    _, prefix_mask, kv_cache = model.build_prefix_cache(observation)
    features, memory = bridge.read_context(kv_cache, prefix_mask, feedback, memory, reset=reset)
    x_t, dt, time = noise, -1. / num_steps, 1.
    while time >= -dt / 2:
        timestep = torch.full((batch,), time, device=device, dtype=torch.float32)
        suffix = model.run_suffix(observation, x_t, timestep, kv_cache, prefix_mask)
        velocity = model.velocity_from_suffix(suffix)
        velocity = bridge.correct_velocity(velocity, suffix, x_t, timestep, features)
        x_t = x_t + dt * velocity
        time += dt
    return x_t, memory


@torch.no_grad()
def predict_action_batch_with_bridge(model, bridge, env_obs, feedback, memory=None, **kwargs):
    """RLinf env transforms -> native KV bridge -> unchanged native action decoder."""
    from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
    observation = model.env_obs_to_observation(env_obs)
    prepared = preprocess_observation(observation, train=False)
    model_actions, memory = sample_prepared_with_bridge(
        model, bridge, prepared, feedback, memory, num_steps=model.num_steps, **kwargs)
    return model.decode_actions(model_actions, observation.state), memory
