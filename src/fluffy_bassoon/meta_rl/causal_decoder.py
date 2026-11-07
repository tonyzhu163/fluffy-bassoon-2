"""Plain causal decoder for in-context history controllers.

A GPT-style stack as used by Algorithm Distillation, DPT and Vintix:
pre-normalized attention and GELU feed-forward blocks with ordinary residual
additions, a final layer norm, and ALiBi distance biases
(Press et al., arxiv.org/abs/2108.12409) so a sliding-window cache has no
positional drift. There is no gating and no segment recurrence.

The core consumes one token per call and caches each layer's inputs for the
previous ``memory_length`` tokens. Within that window, stepping through a
sequence equals full causal attention. State is functional and shaped like
``GatedTransformerXLCore`` state, so callers swap cores without other changes.
The caller owns trial resets.
"""

import torch
from torch import nn


def alibi_slopes(heads):
    """Geometric per-head slopes 2^(-8i/heads), i = 1..heads."""
    return torch.tensor([2 ** (-8 * (index + 1) / heads) for index in range(heads)])


class _DecoderBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, width)
        self.norm_attention, self.norm_ffn = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, current, memory, bias, valid):
        batch, width = current.shape
        joined = torch.cat((memory, current[:, None]), dim=1)
        # Sanitize padding before projection so masked NaN/Inf cannot leak via 0 * NaN.
        joined = torch.where(valid[:, :, None], joined, 0.)
        normalized = self.norm_attention(joined)
        query = self.query(normalized[:, -1]).reshape(batch, self.heads, self.head_dim)
        key, value = self.key_value(normalized).chunk(2, dim=-1)
        key, value = (entry.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
                      for entry in (key, value))
        scores = torch.einsum("bhd,bhsd->bhs", query, key) * self.head_dim ** -.5 + bias[None]
        scores = scores.masked_fill(~valid[:, None], -torch.inf)
        attended = torch.einsum("bhs,bhsd->bhd", scores.softmax(-1), value).reshape(batch, width)
        current = current + self.output(attended)
        return current + self.ffn(self.norm_ffn(current))

    def sequence(self, current, bias, allowed):
        """Train all tokens together without materializing a cache per step."""
        batch, length, width = current.shape
        normalized = self.norm_attention(current)
        query = self.query(normalized).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key, value = self.key_value(normalized).chunk(2, dim=-1)
        key, value = (entry.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
                      for entry in (key, value))
        scores = torch.einsum('bhtd,bhsd->bhts', query, key)*self.head_dim**-.5 + bias[None]
        scores = scores.masked_fill(~allowed[:, None], -torch.inf)
        attended = torch.einsum('bhts,bhsd->bhtd', scores.softmax(-1), value)
        current = current + self.output(attended.transpose(1, 2).reshape(batch, length, width))
        return current + self.ffn(self.norm_ffn(current))


class CausalDecoderCore(nn.Module):
    """Pre-norm causal decoder with ALiBi and a functional sliding-window cache.

    ``memory`` has shape [batch, layers, memory_length, width] and holds each
    layer's inputs for prior tokens; ``length`` is a per-lane long tensor.
    """

    def __init__(self, input_dim, width=128, memory_length=300, layers=2, heads=4):
        super().__init__()
        if (any(type(value) is not int or value < 1
                for value in (input_dim, width, memory_length, layers, heads)) or width % heads):
            raise ValueError("Require positive counts and width divisible by heads")
        self.input_dim, self.output_dim, self.memory_length = input_dim, width, memory_length
        self.input_projection = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList(_DecoderBlock(width, heads) for _ in range(layers))
        self.final_norm = nn.LayerNorm(width)
        distances = torch.arange(memory_length, -1, -1, dtype=torch.float32)
        self.register_buffer("alibi_bias", -alibi_slopes(heads)[:, None] * distances[None])

    def initial_state(self, batch):
        if type(batch) is not int or batch < 1:
            raise ValueError("Require a positive integer batch")
        weight = self.input_projection.weight
        return dict(memory=weight.new_zeros(batch, len(self.layers), self.memory_length, self.output_dim),
                    length=torch.zeros(batch, device=weight.device, dtype=torch.long))

    def sequence(self, observation, *, reset, valid):
        """Full causal training from empty history, with lane-specific resets.

        Inputs/outputs are time-major. This is the same layer/cache computation
        as repeated forward calls, including the sliding attention window and
        stored ALiBi biases, but padding scales with the selected sequence
        length rather than allocating memory_length slots at every time step.
        Incremental deployment continues to use forward and its owned cache.
        """
        weight = self.input_projection.weight
        if (not torch.is_tensor(observation) or observation.ndim != 3
                or min(observation.shape[:2]) < 1 or observation.shape[-1] != self.input_dim
                or observation.dtype != weight.dtype or observation.device != weight.device):
            raise ValueError('Require sequence [time,batch,input_dim] on the core dtype/device')
        for flag in (reset, valid):
            if (not torch.is_tensor(flag) or flag.shape != observation.shape[:2]
                    or flag.dtype != torch.bool or flag.device != observation.device):
                raise ValueError('Require explicit boolean sequence reset/valid masks')
        if not valid[0].all() or (valid[1:] & ~valid[:-1]).any():
            raise ValueError('Require a nonempty contiguous sequence prefix in each lane')
        clean = torch.where(valid[..., None], observation, 0.)
        if not torch.isfinite(clean).all():
            raise ValueError('Require finite valid sequence observations')
        length = len(observation)
        positions = torch.arange(length, device=observation.device)
        distances = positions[:, None]-positions[None]
        causal = (distances >= 0) & (distances <= self.memory_length)
        segments = reset.long().cumsum(0).transpose(0, 1)
        allowed = causal[None] & (segments[:, :, None] == segments[:, None, :])
        active = valid.transpose(0, 1)
        allowed = allowed & active[:, :, None] & active[:, None, :]
        # Trailing padded queries use a harmless self key; all-masked softmax
        # would create NaNs even though their losses are excluded downstream.
        allowed = allowed | (~active[:, :, None] & torch.eye(length, device=weight.device, dtype=torch.bool)[None])
        # Index the saved bias, preserving its original floating-point rounding.
        bias = self.alibi_bias[:, self.memory_length-distances.clamp(0, self.memory_length)].to(weight.dtype)
        current = self.input_projection(clean.transpose(0, 1))
        for layer in self.layers:
            current = layer.sequence(current, bias, allowed)
        return self.final_norm(current).transpose(0, 1)

    def forward(self, observation, state=None):
        weight = self.input_projection.weight
        if (not isinstance(observation, torch.Tensor) or observation.ndim != 2
                or observation.shape[0] < 1 or observation.shape[1] != self.input_dim
                or observation.dtype != weight.dtype or observation.device != weight.device):
            raise ValueError("Require observation [batch, input_dim] on the core dtype/device")
        batch = observation.shape[0]
        if state is None:
            state = self.initial_state(batch)
        if (not isinstance(state, dict) or state.keys() != {"memory", "length"}
                or not isinstance(state["memory"], torch.Tensor)
                or state["memory"].shape != (batch, len(self.layers), self.memory_length, self.output_dim)
                or state["memory"].dtype != weight.dtype or state["memory"].device != weight.device
                or not isinstance(state["length"], torch.Tensor) or state["length"].shape != (batch,)
                or state["length"].dtype != torch.long or state["length"].device != weight.device
                or ((state["length"] < 0) | (state["length"] > self.memory_length)).any()):
            raise ValueError("Require complete valid decoder layer caches and lengths")
        valid = torch.arange(self.memory_length + 1, device=weight.device)[None] >= (
            self.memory_length - state["length"][:, None])
        bias = self.alibi_bias.to(weight.dtype)
        current, memories = self.input_projection(observation), []
        for index, layer in enumerate(self.layers):
            memory = state["memory"][:, index]
            memories.append(torch.cat((memory[:, 1:], current[:, None]), dim=1))
            current = layer(current, memory, bias, valid)
        return self.final_norm(current), dict(memory=torch.stack(memories, dim=1),
                                              length=(state["length"] + 1).clamp(max=self.memory_length))

