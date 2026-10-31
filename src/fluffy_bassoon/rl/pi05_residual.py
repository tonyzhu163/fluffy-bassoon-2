"""Optional reactive KV and decoded-ALOHA residuals, not live-qualified policies.

These hooks neither load a motor nor own a learner, sampler or simulator. Native
cache tensors are post-RoPE [batch, prefix, kv_heads, head_dim]; no token or
position is added. The older Pi05FeatureBridge intervenes on flow velocity.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _features(features, reference, dimension, batch):
    if (not isinstance(features, torch.Tensor) or features.shape != (batch, dimension)
            or not features.is_floating_point() or features.device != reference.device
            or not torch.isfinite(features).all()):
        raise ValueError("Require finite current-observation features [B, public_dim] on the hook device")
    return features.detach().to(reference.dtype)


def _add(base, delta):
    result = base + delta
    # Preserve even signed zero in inference. During fitting retain the ordinary
    # derivative at the zero-initialized output basis.
    return result if torch.is_grad_enabled() else torch.where(delta == 0, base, result)


class ReactiveKVResidual(nn.Module):
    """Token-local rank-r edits, conditioned only on the current observation.

    For each selected layer, delta = scale * tanh(A[K,V] + C features) @ B.
    B starts at zero. Padding rows receive no edit. The rank bound applies to
    the concatenated K/V delta before casting to the native dtype. Incoming
    caches/features are detached, and no incoming tensor is modified in place.
    Scale is in native cache units, not an asserted calibrated safety bound.
    """

    def __init__(self, *, public_dim, kv_heads, head_dim, rank=4, layers=(-1,), scale=.1):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (public_dim, kv_heads, head_dim, rank)):
            raise ValueError("Require positive integer KV adapter dimensions")
        if (not isinstance(layers, (tuple, list)) or not layers
                or any(type(i) is not int for i in layers) or len(set(layers)) != len(layers)):
            raise ValueError("Require distinct selected cache layer indices")
        self.public_dim, self.kv_heads, self.head_dim, self.rank = public_dim, kv_heads, head_dim, rank
        self.layers, self.scale = tuple(layers), _positive(scale, "scale")
        width = 2 * kv_heads * head_dim
        self.token_projections = nn.ModuleList(nn.Linear(width, rank) for _ in layers)
        self.context_projections = nn.ModuleList(nn.Linear(public_dim, rank, bias=False) for _ in layers)
        self.output_bases = nn.ModuleList(nn.Linear(rank, width, bias=False) for _ in layers)
        for basis in self.output_bases:
            nn.init.zeros_(basis.weight)

    def config(self):
        return dict(kind="reactive_native_kv_residual_v1", public_dim=self.public_dim,
                    kv_heads=self.kv_heads, head_dim=self.head_dim, rank=self.rank,
                    layers=list(self.layers), scale=self.scale, layout="B,P,KVH,HD_post_RoPE")

    def forward(self, cache, prefix_mask, public_features):
        if not isinstance(cache, (tuple, list)) or not cache:
            raise ValueError("Require a nonempty native per-layer KV cache")
        if (not isinstance(prefix_mask, torch.Tensor) or prefix_mask.ndim != 2
                or prefix_mask.dtype != torch.bool or not prefix_mask.shape[0]
                or not prefix_mask.any(1).all()):
            raise ValueError("Require a nonempty boolean prefix mask [B,P]")
        resolved = tuple(i if i >= 0 else len(cache) + i for i in self.layers)
        if any(not 0 <= i < len(cache) for i in resolved) or len(set(resolved)) != len(resolved):
            raise ValueError("Selected cache layers are absent or aliases of the same layer")
        reference = self.output_bases[0].weight
        features = _features(public_features, reference, self.public_dim, prefix_mask.shape[0])
        if prefix_mask.device != reference.device:
            raise ValueError("Prefix mask and adapter must share a device")
        expected = (*prefix_mask.shape, self.kv_heads, self.head_dim)
        detached = []
        for pair in cache:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each cache layer must contain exactly K and V")
            key, value = pair
            if (any(not isinstance(t, torch.Tensor) or t.shape != expected
                    or t.device != reference.device or not t.is_floating_point() for t in pair)
                    or key.dtype != value.dtype
                    or any(not torch.isfinite(t[prefix_mask]).all() for t in pair)):
                raise ValueError("Require finite valid native KV tokens with unchanged B,P,KVH,HD layout")
            detached.append((key.detach(), value.detach()))
        for slot, layer in enumerate(resolved):
            key, value = detached[layer]
            tokens = torch.cat((key.flatten(2), value.flatten(2)), -1)
            tokens = tokens.masked_fill(~prefix_mask[..., None], 0).to(reference.dtype)
            coefficients = (self.token_projections[slot](tokens)
                            + self.context_projections[slot](features)[:, None]).tanh()
            delta = self.scale * self.output_bases[slot](coefficients)
            if not torch.isfinite(delta).all():
                raise ValueError("Nonfinite native KV residual")
            dk, dv = delta.chunk(2, -1)
            valid = prefix_mask[..., None, None]
            detached[layer] = tuple(torch.where(valid, _add(base, correction.reshape_as(base).to(base.dtype)), base)
                                    for base, correction in ((key, dk), (value, dv)))
        return tuple(detached)


def velocity_with_kv_residual(model, observation, noisy_actions, timestep, cache,
                             prefix_mask, public_features, adapter):
    """One differentiable frozen suffix pass for future supervised flow matching.

    Do not surround this call with no_grad/inference_mode while fitting. Frozen
    parameters still allow derivatives from the suffix output into edited KV.
    Saved detached suffix/velocity records cannot replace this recomputation.
    """
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise RuntimeError("KV suffix training requires autograd; no_grad/inference_mode disables the adapter gradient")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Require an eval-mode motor with every parameter frozen")
    edited = adapter(cache, prefix_mask, public_features)
    suffix = model.run_suffix(observation, noisy_actions.detach(), timestep.detach(), edited, prefix_mask)
    velocity = model.velocity_from_suffix(suffix)
    if not velocity.requires_grad:
        raise RuntimeError("Frozen suffix detached the KV adapter gradient; this runtime is not trainable through the hook")
    return velocity


@dataclass(frozen=True)
class DecodedActionResidual:
    base_actions: torch.Tensor
    requested_delta: torch.Tensor
    applied_delta: torch.Tensor
    clipped: torch.Tensor
    actions: torch.Tensor


class ReactiveALOHAActionResidual(nn.Module):
    """Bounded correction to decoded [B,H,14] ALOHA joint targets.

    Indices 0:6/7:13 are arm joints, indices 6/13 absolute grippers. Bounds
    must be selected explicitly in those decoded command units from source
    data/conventions; this module provides no guessed robot safety limits.
    The head also reads the detached base proposal and normalized slot index.
    Optional command limits clip proposals after correction; out-of-box base
    commands are rejected so zero-head parity cannot silently be lost.
    """

    def __init__(self, *, public_dim, arm_joint_bound, gripper_bound, width=64,
                 command_lower=None, command_upper=None):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (public_dim, width)):
            raise ValueError("Require positive integer action-head dimensions")
        self.public_dim, self.width = public_dim, width
        arm, gripper = _positive(arm_joint_bound, "arm_joint_bound"), _positive(gripper_bound, "gripper_bound")
        self.register_buffer("bounds", torch.tensor([arm] * 6 + [gripper] + [arm] * 6 + [gripper]))
        if not torch.isfinite(self.bounds).all():
            raise ValueError("Residual bounds must be representable in float32 command units")
        if (command_lower is None) != (command_upper is None):
            raise ValueError("Supply both decoded command limits or neither")
        lower = None if command_lower is None else torch.as_tensor(command_lower, dtype=torch.float32).clone()
        upper = None if command_upper is None else torch.as_tensor(command_upper, dtype=torch.float32).clone()
        if lower is not None and (lower.shape != (14,) or upper.shape != (14,)
                or not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or not (lower < upper).all()):
            raise ValueError("Require finite strictly ordered decoded command limits [14]")
        self.register_buffer("command_lower", lower)
        self.register_buffer("command_upper", upper)
        self.hidden = nn.Linear(public_dim + 15, width)
        self.output = nn.Linear(width, 14)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def config(self):
        return dict(kind="reactive_decoded_aloha_residual_v1", public_dim=self.public_dim, width=self.width,
                    bounds=self.bounds.detach().cpu().tolist(), gripper_indices=[6, 13],
                    arm_joint_bound=float(self.bounds[0].detach().cpu()),
                    gripper_bound=float(self.bounds[6].detach().cpu()),
                    reads_base_proposal=True, units="decoded ALOHA joint/absolute-gripper command units",
                    command_lower=None if self.command_lower is None else self.command_lower.detach().cpu().tolist(),
                    command_upper=None if self.command_upper is None else self.command_upper.detach().cpu().tolist())

    def forward(self, base_actions, public_features):
        reference = self.output.weight
        if (not isinstance(base_actions, torch.Tensor) or base_actions.ndim != 3
                or base_actions.shape[-1] != 14 or not all(base_actions.shape)
                or base_actions.device != reference.device or not base_actions.is_floating_point()
                or not torch.isfinite(base_actions).all()):
            raise ValueError("Require finite decoded ALOHA joint targets [B,H,14] on the hook device")
        base = base_actions.detach()
        features = _features(public_features, reference, self.public_dim, base.shape[0])
        if self.command_lower is not None and ((base < self.command_lower).any() or (base > self.command_upper).any()):
            raise ValueError("Base command exceeds declared limits; zero-residual parity would be lost")
        horizon = base.shape[1]
        slots = torch.arange(horizon, device=base.device, dtype=reference.dtype) / max(horizon - 1, 1)
        inputs = torch.cat((features[:, None].expand(-1, horizon, -1), base.to(reference.dtype),
                            slots[None, :, None].expand(base.shape[0], -1, -1)), -1)
        requested = (self.output(self.hidden(inputs).tanh()).tanh() * self.bounds).to(base.dtype)
        if not torch.isfinite(requested).all():
            raise ValueError("Nonfinite decoded ALOHA residual")
        proposal = _add(base, requested)
        actions = proposal if self.command_lower is None else proposal.clamp(
            min=self.command_lower.to(base.dtype), max=self.command_upper.to(base.dtype))
        return DecodedActionResidual(base, requested, actions - base, actions != proposal, actions)
