"""Single-token Transformer-XL recurrence for the compact full-trial learner.

Relative attention follows arxiv.org/abs/1901.02860. Unlike detached language-
model segments, PPO replay retains the complete trial's gradient graph.
"""

import torch
from torch import nn


class _RelativeBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key_value = nn.Linear(width, 2 * width, bias=False)
        self.relative = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.norm_attention, self.norm_ffn = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.ReLU(), nn.Linear(4 * width, width))

    def forward(self, current, memory, positions, valid, content_bias, position_bias):
        batch, width = current.shape
        joined = torch.cat((memory, current[:, None]), dim=1)
        query = self.query(current).reshape(batch, self.heads, self.head_dim)
        key, value = self.key_value(joined).chunk(2, dim=-1)
        key, value = (x.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
                      for x in (key, value))
        relative = self.relative(positions).reshape(-1, self.heads, self.head_dim)
        scores = (torch.einsum('bhd,bhsd->bhs', query + content_bias, key)
                  + torch.einsum('bhd,shd->bhs', query + position_bias, relative))
        scores = (scores * self.head_dim ** -.5).masked_fill(~valid[:, None], -torch.inf)
        attended = torch.einsum('bhs,bhsd->bhd', scores.softmax(-1), value).reshape(batch, width)
        current = self.norm_attention(current + self.output(attended))
        return self.norm_ffn(current + self.ffn(current))


class TransformerXLCore(nn.Module):
    """Independent layers with relative positions and cached layer-input history."""

    def __init__(self, input_dim, width=64, ticks=1, memory_length=64, layers=2, heads=4):
        super().__init__()
        if (any(type(x) is not int or x < 1 for x in (input_dim, width, memory_length, layers, heads))
                or width % heads or width % 2 or type(ticks) is not int or ticks != 1):
            raise ValueError('Require positive counts, even width divisible by heads and exactly one tick')
        self.input_dim, self.output_dim, self.ticks = input_dim, width, ticks
        self.memory_length = memory_length
        self.input_projection = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList(_RelativeBlock(width, heads) for _ in range(layers))
        self.content_bias = nn.Parameter(torch.zeros(heads, width // heads))
        self.position_bias = nn.Parameter(torch.zeros(heads, width // heads))
        distances = torch.arange(memory_length, -1, -1, dtype=torch.float32)
        frequencies = 10000 ** (-torch.arange(0, width, 2, dtype=torch.float32) / width)
        phase = distances[:, None] * frequencies[None]
        self.register_buffer('relative_positions', torch.cat((phase.sin(), phase.cos()), dim=-1))

    def initial_state(self, batch):
        if type(batch) is not int or batch < 1:
            raise ValueError('Require a positive integer batch')
        weight = self.input_projection.weight
        return dict(memory=weight.new_zeros(batch, len(self.layers), self.memory_length, self.output_dim),
                    length=torch.zeros(batch, device=weight.device, dtype=torch.long))

    def forward(self, observation, state=None):
        weight = self.input_projection.weight
        if (not isinstance(observation, torch.Tensor) or observation.ndim != 2
                or observation.shape[0] < 1 or observation.shape[1] != self.input_dim
                or observation.dtype != weight.dtype or observation.device != weight.device):
            raise ValueError('Require observation [batch, input_dim] on the core dtype/device')
        batch = observation.shape[0]
        if state is None:
            state = self.initial_state(batch)
        if (not isinstance(state, dict) or state.keys() != {'memory', 'length'}
                or not isinstance(state['memory'], torch.Tensor)
                or state['memory'].shape != (batch, len(self.layers), self.memory_length, self.output_dim)
                or state['memory'].dtype != weight.dtype or state['memory'].device != weight.device
                or not isinstance(state['length'], torch.Tensor) or state['length'].shape != (batch,)
                or state['length'].dtype != torch.long or state['length'].device != weight.device
                or ((state['length'] < 0) | (state['length'] > self.memory_length)).any()):
            raise ValueError('Require complete valid Transformer-XL layer memories and lengths')
        valid = torch.arange(self.memory_length + 1, device=weight.device)[None] >= (
            self.memory_length - state['length'][:, None])
        current, memories = self.input_projection(observation), []
        for index, layer in enumerate(self.layers):
            memory = state['memory'][:, index]
            # Cache this layer's input, not its output: next-step keys must use the same layer.
            memories.append(torch.cat((memory[:, 1:], current[:, None]), dim=1))
            current = layer(current, memory, self.relative_positions, valid,
                            self.content_bias, self.position_bias)
        return current, dict(memory=torch.stack(memories, dim=1),
                             length=(state['length'] + 1).clamp(max=self.memory_length))
