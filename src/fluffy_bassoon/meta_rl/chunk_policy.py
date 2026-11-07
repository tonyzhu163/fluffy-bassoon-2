"""Causal actor and Q/V models for public decoded-action chunk histories.

The layout comes from chunk_history, never the legacy latent-noise format.
Residual radii are fixed training-split statistics in native decoded units.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .causal_decoder import CausalDecoderCore
from .chunk_history import ChunkHistoryLayout


@dataclass(frozen=True)
class ChunkModelConfig:
    layout: ChunkHistoryLayout
    width: int = 128
    layers: int = 2
    heads: int = 4
    context_length: int = 300
    initial_std: float = .5

    def __post_init__(self):
        if (not isinstance(self.layout, ChunkHistoryLayout)
                or any(type(v) is not int or v < 1 for v in
                       (self.width, self.layers, self.heads, self.context_length))
                or self.width % self.heads
                or not math.isfinite(self.initial_std) or self.initial_std <= 0):
            raise ValueError('Invalid decoded-chunk model layout or dimensions')


class ChunkHistoryBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, ChunkModelConfig):
            raise ValueError('Require an explicit decoded-chunk model configuration')
        self.config = config
        self.core = CausalDecoderCore(config.layout.token_dim, width=config.width,
            layers=config.layers, heads=config.heads, memory_length=config.context_length)
        self.register_buffer('feature_mean', torch.zeros(config.layout.feature_dim))
        self.register_buffer('feature_std', torch.ones(config.layout.feature_dim))

    @torch.no_grad()
    def fit_feature_normalization(self, training_features):
        values = torch.as_tensor(training_features, device=self.feature_mean.device,
                                 dtype=self.feature_mean.dtype)
        if (values.ndim != 2 or values.shape[1] != len(self.feature_mean)
                or not len(values) or not torch.isfinite(values).all()):
            raise ValueError('Require finite source-TRAIN public features [N,F]')
        self.feature_mean.copy_(values.mean(0))
        self.feature_std.copy_(values.std(0, unbiased=False).clamp(min=1e-4))

    def encode(self, tokens, state=None, *, reset=None):
        if (tokens.ndim != 2 or tokens.shape[1] != self.config.layout.token_dim
                or not torch.isfinite(tokens).all()):
            raise ValueError('Require finite public decoded-chunk tokens [B,token_dim]')
        if reset is not None:
            if reset.shape != tokens.shape[:1] or reset.dtype != torch.bool or reset.device != tokens.device:
                raise ValueError('Require boolean per-lane attention/feedback reset')
            if state is not None:
                empty = self.core.initial_state(len(tokens))
                state = {key: torch.where(reset.reshape(-1, *([1]*(value.ndim-1))), empty[key], value)
                         for key, value in state.items()}
            # Reset the embedded feedback as well as the decoder cache. This
            # keeps static/episode controls safe even with precomputed tokens.
            begin = self.config.layout.current_dim
            tokens = torch.cat((tokens[:, :begin],
                torch.where(reset[:, None], 0., tokens[:, begin:-1]), tokens[:, -1:]), -1)
        d = self.config.layout.feature_dim
        normalized = torch.cat(((tokens[:, :d]-self.feature_mean)/self.feature_std, tokens[:, d:]), -1)
        return self.core(normalized, state)

    def encode_sequence(self, tokens, *, reset, valid):
        if (tokens.ndim != 3 or tokens.shape[-1] != self.config.layout.token_dim
                or reset.shape != tokens.shape[:2] or valid.shape != tokens.shape[:2]
                or reset.dtype != torch.bool or valid.dtype != torch.bool
                or reset.device != tokens.device or valid.device != tokens.device):
            raise ValueError('Require decoded sequence tokens and boolean reset/valid masks')
        tokens = torch.where(valid[..., None], tokens, 0.)
        if not torch.isfinite(tokens).all():
            raise ValueError('Require finite valid decoded sequence tokens')
        begin, d = self.config.layout.current_dim, self.config.layout.feature_dim
        tokens = torch.cat((tokens[..., :begin],
            torch.where(reset[..., None], 0., tokens[..., begin:-1]), tokens[..., -1:]), -1)
        normalized = torch.cat(((tokens[..., :d]-self.feature_mean)/self.feature_std, tokens[..., d:]), -1)
        return self.core.sequence(normalized, reset=reset, valid=valid)


class ChunkResidualPolicy(ChunkHistoryBackbone):
    """Diagonal Gaussian transformed to a bounded full decoded residual chunk."""
    def __init__(self, config, residual_radius):
        super().__init__(config)
        radius = torch.as_tensor(residual_radius, dtype=self.feature_mean.dtype)
        if (radius.shape != config.layout.command_shape or not torch.isfinite(radius).all()
                or not (radius > 0).all()):
            raise ValueError('Require positive finite per-coordinate decoded residual radii [H,A]')
        self.register_buffer('residual_radius', radius.detach().clone())
        count = config.layout.chunk_length*config.layout.action_dim
        self.actor = nn.Linear(config.width, count)
        self.log_std = nn.Parameter(torch.full((count,), math.log(config.initial_std)))

    def distribution(self, tokens, state=None, *, reset=None):
        features, state = self.encode(tokens, state, reset=reset)
        mean = self.actor(features)
        return Normal(mean, self.log_std.clamp(-5, 2).exp()), state

    def _decoded_log_prob(self, distribution, latent):
        log_jacobian = (self.residual_radius.flatten().log()
                        + 2*(math.log(2.)-latent-F.softplus(-2*latent)))
        return (distribution.log_prob(latent)-log_jacobian).sum(-1)

    def step(self, tokens, state=None, *, reset=None, deterministic=False, generator=None):
        distribution, state = self.distribution(tokens, state, reset=reset)
        latent = distribution.mean
        if not deterministic:
            latent = latent + distribution.stddev*torch.randn(latent.shape, device=latent.device,
                dtype=latent.dtype, generator=generator)
        residual = latent.tanh().reshape(-1, *self.config.layout.command_shape)*self.residual_radius
        return residual, self._decoded_log_prob(distribution, latent), state

    def log_prob(self, tokens, residual, state=None, *, reset=None):
        expected = (len(tokens), *self.config.layout.command_shape)
        if residual.shape != expected or not torch.isfinite(residual).all():
            raise ValueError('Require finite full decoded residual targets [B,H,A]')
        scaled = residual/self.residual_radius
        if not (scaled.abs() < 1).all():
            raise ValueError('Target reaches/exceeds fixed residual bounds; audit coverage, never silently clip labels')
        distribution, state = self.distribution(tokens, state, reset=reset)
        return self._decoded_log_prob(distribution, torch.atanh(scaled.flatten(1))), state

    def log_prob_sequence(self, tokens, residual, *, reset, valid):
        if residual.shape != (*tokens.shape[:2], *self.config.layout.command_shape):
            raise ValueError('Require whole decoded residual sequence targets [T,B,H,A]')
        residual = torch.where(valid[..., None, None], residual, 0.)
        scaled = residual/self.residual_radius
        if not torch.isfinite(scaled).all() or not (scaled.abs() < 1).all():
            raise ValueError('Target reaches/exceeds fixed residual bounds; audit coverage, never silently clip labels')
        features = self.encode_sequence(tokens, reset=reset, valid=valid)
        distribution = Normal(self.actor(features), self.log_std.clamp(-5, 2).exp())
        return self._decoded_log_prob(distribution, torch.atanh(scaled.flatten(2)))


class ChunkHistoryCritic(ChunkHistoryBackbone):
    """Q receives the entire current decoded residual; V receives only history."""
    def __init__(self, config, *, action_value):
        super().__init__(config)
        if type(action_value) is not bool:
            raise ValueError('Declare Q or V explicitly')
        self.action_value = action_value
        count = config.layout.chunk_length*config.layout.action_dim if action_value else 0
        self.head = nn.Sequential(nn.Linear(config.width+count, config.width), nn.GELU(), nn.Linear(config.width, 1))

    def step(self, tokens, state=None, *, action=None, reset=None):
        features, state = self.encode(tokens, state, reset=reset)
        if self.action_value:
            if (action is None or action.shape != (len(tokens), *self.config.layout.command_shape)
                    or not torch.isfinite(action).all()):
                raise ValueError('Q requires the full finite current residual [B,H,A]')
            features = torch.cat((features, action.flatten(1)), -1)
        elif action is not None:
            raise ValueError('V must not consume a current action or target')
        return self.head(features).squeeze(-1), state

    def sequence(self, tokens, *, action=None, reset, valid):
        features = self.encode_sequence(tokens, reset=reset, valid=valid)
        if self.action_value:
            if action is None or action.shape != (*tokens.shape[:2], *self.config.layout.command_shape):
                raise ValueError('Q requires whole decoded residual sequence actions')
            action = torch.where(valid[..., None, None], action, 0.)
            if not torch.isfinite(action).all():
                raise ValueError('Require finite valid Q actions')
            features = torch.cat((features, action.flatten(2)), -1)
        elif action is not None:
            raise ValueError('V must not consume a current action or target')
        return self.head(features).squeeze(-1)
