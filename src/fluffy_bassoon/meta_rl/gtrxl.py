"""Single-observation Gated Transformer-XL for full-trial recurrent PPO.

The block follows Parisotto et al., arxiv.org/abs/1910.06764, equations 5--8
and the GRU gate in section 3.2: pre-normalized relative attention, ReLU
updates and two independent identity-favoring residual gates. Cached entries
are each layer's unnormalized inputs, as required by Transformer-XL.

Experiment-specific choices are a vector input projection, a compact default
128-wide/two-layer model and 300 prior observations of memory. There is no
dropout, final layer normalization or in-trial stop-gradient. The last choice
differs from detached segments in the paper: PPO replays a complete trial and
must backpropagate through its full history. The caller owns trial resets.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GRUResidualGate(nn.Module):
    """The paper's depth-wise GRU gate, including ``z = sigmoid(... - b)``.

    A positive learned update bias favors the residual input at initialization;
    it does not make the gate an exact identity. Affine transforms have no
    additional biases, matching the displayed gate equations.
    """

    def __init__(self, width, update_bias=2.):
        super().__init__()
        if (type(width) is not int or width < 1 or isinstance(update_bias, bool)
                or not isinstance(update_bias, (int, float)) or not math.isfinite(update_bias)
                or update_bias < 0):
            raise ValueError("Require a positive gate width and finite nonnegative update bias")
        self.reset_input = nn.Linear(width, width, bias=False)
        self.reset_update = nn.Linear(width, width, bias=False)
        self.update_input = nn.Linear(width, width, bias=False)
        self.update_update = nn.Linear(width, width, bias=False)
        self.candidate_input = nn.Linear(width, width, bias=False)
        self.candidate_update = nn.Linear(width, width, bias=False)
        self.update_bias = nn.Parameter(torch.full((width,), float(update_bias)))

    def forward(self, residual, update):
        reset = torch.sigmoid(self.reset_update(update) + self.reset_input(residual))
        mix = torch.sigmoid(self.update_update(update) + self.update_input(residual) - self.update_bias)
        candidate = torch.tanh(self.candidate_update(update) + self.candidate_input(reset * residual))
        return (1 - mix) * residual + mix * candidate


class _GatedRelativeBlock(nn.Module):
    def __init__(self, width, heads, gate_bias):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key_value = nn.Linear(width, 2 * width, bias=False)
        self.relative = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.content_bias = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.position_bias = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.norm_attention, self.norm_ffn = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.ReLU(), nn.Linear(4 * width, width))
        self.attention_gate = GRUResidualGate(width, gate_bias)
        self.ffn_gate = GRUResidualGate(width, gate_bias)

    def forward(self, current, memory, positions, valid):
        batch, width = current.shape
        joined = torch.cat((memory, current[:, None]), dim=1)
        # Invalid padding must not affect either scores or values. Sanitizing
        # before projection also prevents masked NaN/Inf padding from leaking
        # through the otherwise undefined product 0 * NaN in the value sum.
        joined = torch.where(valid[:, :, None], joined, 0.)
        normalized = self.norm_attention(joined)
        query = self.query(normalized[:, -1]).reshape(batch, self.heads, self.head_dim)
        key, value = self.key_value(normalized).chunk(2, dim=-1)
        key, value = (entry.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
                      for entry in (key, value))
        relative = self.relative(positions).reshape(-1, self.heads, self.head_dim)
        scores = (torch.einsum("bhd,bhsd->bhs", query + self.content_bias, key)
                  + torch.einsum("bhd,shd->bhs", query + self.position_bias, relative))
        scores = (scores * self.head_dim ** -.5).masked_fill(~valid[:, None], -torch.inf)
        attended = torch.einsum("bhs,bhsd->bhd", scores.softmax(-1), value).reshape(batch, width)
        current = self.attention_gate(current, F.relu(self.output(attended)))
        return self.ffn_gate(current, F.relu(self.ffn(self.norm_ffn(current))))


class GatedTransformerXLCore(nn.Module):
    """Relative-attention recurrence with independent GRU-gated layers.

    ``memory_length`` counts prior observations, in addition to the current
    query. The default covers all 300 noise decisions in a four-attempt Can
    trial. ``memory`` has shape [batch, layers, memory_length, width]; ``length``
    is a per-lane long tensor. State is functional, with no module-held cache.
    Existing ``RecurrentActorCritic`` reset masks reset both entries per lane.
    """

    def __init__(self, input_dim, width=128, ticks=1, memory_length=300, layers=2, heads=4,
                 gate_bias=2.):
        super().__init__()
        if (any(type(value) is not int or value < 1
                for value in (input_dim, width, memory_length, layers, heads))
                or width % heads or width % 2 or type(ticks) is not int or ticks != 1):
            raise ValueError("Require positive counts, even width divisible by heads and exactly one tick")
        if (isinstance(gate_bias, bool) or not isinstance(gate_bias, (int, float))
                or not math.isfinite(gate_bias) or gate_bias < 0):
            raise ValueError("Require a finite nonnegative gate bias")
        self.input_dim, self.output_dim, self.ticks = input_dim, width, ticks
        self.memory_length, self.gate_bias = memory_length, float(gate_bias)
        self.input_projection = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList(_GatedRelativeBlock(width, heads, gate_bias) for _ in range(layers))
        distances = torch.arange(memory_length, -1, -1, dtype=torch.float32)
        frequencies = 10000 ** (-torch.arange(0, width, 2, dtype=torch.float32) / width)
        phase = distances[:, None] * frequencies[None]
        self.register_buffer("relative_positions", torch.cat((phase.sin(), phase.cos()), dim=-1))

    def initial_state(self, batch):
        if type(batch) is not int or batch < 1:
            raise ValueError("Require a positive integer batch")
        weight = self.input_projection.weight
        return dict(memory=weight.new_zeros(batch, len(self.layers), self.memory_length, self.output_dim),
                    length=torch.zeros(batch, device=weight.device, dtype=torch.long))

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
            raise ValueError("Require complete valid GTrXL layer memories and lengths")
        valid = torch.arange(self.memory_length + 1, device=weight.device)[None] >= (
            self.memory_length - state["length"][:, None])
        current, memories = self.input_projection(observation), []
        for index, layer in enumerate(self.layers):
            memory = state["memory"][:, index]
            # Keys in this layer must consume previous inputs to this same
            # layer. Caching its outputs would silently change the recurrence.
            memories.append(torch.cat((memory[:, 1:], current[:, None]), dim=1))
            current = layer(current, memory, self.relative_positions, valid)
        return current, dict(memory=torch.stack(memories, dim=1),
                             length=(state["length"] + 1).clamp(max=self.memory_length))
