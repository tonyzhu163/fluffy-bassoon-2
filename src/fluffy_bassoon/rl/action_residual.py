"""Bounded action residual on a frozen VLA's action chunk.

The learner emits an unbounded control ``u`` per chunk. The executed residual
is ``delta = scale * tanh(u)``, added to every step of the frozen policy's
unnormalized action chunk on the declared dimensions only, before any
environment clipping. ``u`` is what histories record and what PPO's Gaussian
likelihood is defined over; tanh is a bijection, so likelihood ratios are
unchanged by the map. A zero residual reproduces the frozen policy exactly.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ResidualBound:
    """Per-dimension residual bound in environment action units."""

    action_dim: int
    indices: tuple
    scale: tuple

    def __post_init__(self):
        if (type(self.action_dim) is not int or self.action_dim < 1 or not self.indices
                or len(self.indices) != len(self.scale) or len(set(self.indices)) != len(self.indices)
                or any(type(index) is not int or not 0 <= index < self.action_dim for index in self.indices)
                or any(not math.isfinite(value) or value <= 0 for value in self.scale)):
            raise ValueError("Require distinct in-range indices and finite positive scales")

    @property
    def residual_dim(self):
        return len(self.indices)

    @classmethod
    def from_std(cls, std, *, indices, fraction, action_dim):
        """Bound each residual dimension at ``fraction`` of that action dimension's std."""
        std = np.asarray(std, dtype=np.float64)[:action_dim]
        if std.shape != (action_dim,) or not math.isfinite(fraction) or fraction <= 0:
            raise ValueError("Require at least action_dim stds and a positive fraction")
        return cls(action_dim, tuple(indices), tuple(float(fraction * std[index]) for index in indices))

    def delta(self, control):
        """Map unbounded controls [..., k] to executed residuals [..., k]."""
        if isinstance(control, torch.Tensor):
            return torch.as_tensor(self.scale, dtype=control.dtype, device=control.device) * torch.tanh(control)
        control = np.asarray(control)
        return np.asarray(self.scale, dtype=control.dtype) * np.tanh(control)

    def apply(self, chunk, control):
        """Add the bounded residual to every step of a [..., H, A] chunk."""
        if chunk.shape[-1] != self.action_dim or control.shape[-1] != self.residual_dim:
            raise ValueError("Require chunk [..., H, action_dim] and control [..., residual_dim]")
        if control.shape[:-1] != chunk.shape[:-2]:
            raise ValueError("Require one control per chunk")
        delta = self.delta(control)
        if isinstance(chunk, torch.Tensor):
            out = chunk.clone()
            out[..., list(self.indices)] += delta[..., None, :]
            return out
        out = np.array(chunk, copy=True)
        out[..., list(self.indices)] += delta[..., None, :]
        return out


def action_std_from_norm_stats(path):
    """Read the action std from an OpenPI ``norm_stats.json``."""
    payload = json.loads(Path(path).read_text())
    stats = payload.get("norm_stats", payload)
    std = np.asarray(stats["actions"]["std"], dtype=np.float64)
    if std.ndim != 1 or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("Require a finite nonnegative 1-D action std")
    return std
