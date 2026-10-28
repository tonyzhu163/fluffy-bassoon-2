"""Continuous Memory Machines, arXiv:2610.07907v1, §3.2 and Appendix A.

Independent equation-based implementation and the repository's only CMM. The
linked authors' code was not accessible during implementation; this is not a
claim of upstream code parity. `fluffy_bassoon.meta_rl.cmm.CMMCore` is a thin
per-observation wrapper around this module.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from lerobot_policy_ctm.shared.sakana_reference import SynapseUNET


STATE_KEYS = ("activated", "trace", "long_term", "alpha", "beta")


@dataclass(frozen=True)
class CMMConfig:
    width: int = 128
    stm_length: int = 10
    ltm_slots: int = 32
    memory_layers: int = 1
    synapse_depth: int = 8
    heads: int = 4
    ticks: int = 1
    dropout: float = 0.
    learned_residual_gate: bool = False
    pair_seed: int = 1000
    norm_eps: float = 1e-5
    rms_eps: float = 1e-8

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in
               (self.width, self.stm_length, self.heads, self.synapse_depth, self.ticks)):
            raise ValueError("CMM dimensions must be positive integers")
        if (type(self.ltm_slots) is not int or self.ltm_slots < 0
                or type(self.memory_layers) is not int or self.memory_layers < 0):
            raise ValueError("Require nonnegative LTM slots and memory layers")
        if self.memory_layers == 0 and self.ltm_slots:
            raise ValueError("A CMM without memory layers has no LTM to update")
        if ((self.memory_layers and self.width % self.heads) or self.synapse_depth < 2
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1
                or min(self.norm_eps, self.rms_eps) <= 0
                or not all(math.isfinite(x) for x in (self.norm_eps, self.rms_eps))
                or type(self.pair_seed) is not int or not 0 <= self.pair_seed < 2**63
                or type(self.learned_residual_gate) is not bool):
            raise ValueError("Invalid CMM memory, attention, normalization or pairing settings")


class JointMemoryBlock(nn.Module):
    """Equations 7–12: Q/K-only positions, zero sink value, shared scalar gate."""

    def __init__(self, width, heads, dropout=0., gated=False, norm_eps=1e-5):
        super().__init__()
        if width < 1 or heads < 1 or width % heads or not 0 <= dropout < 1:
            raise ValueError("Require positive divisible width/heads and dropout in [0, 1)")
        self.heads, self.dropout = heads, dropout
        self.attention_norm = nn.LayerNorm(width, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(width, eps=norm_eps)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.ffn = nn.Sequential(nn.Linear(width, width, bias=False), nn.SiLU(),
                                 nn.Linear(width, width, bias=False))
        if gated:
            self.gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("gate", torch.ones(()))

    def forward(self, tokens, positions):
        batch, count, width = tokens.shape
        normalized = self.attention_norm(tokens)
        q = self.query(normalized + positions)
        k = self.key(normalized + positions)
        v = self.value(normalized)
        # Never overwrite the projected tensor in place: backward needs it.
        v = torch.cat((torch.zeros_like(v[:, :1]), v[:, 1:]), dim=1)

        def split(x):
            return x.reshape(batch, count, self.heads, width // self.heads).transpose(1, 2)

        attended = F.scaled_dot_product_attention(split(q), split(k), split(v))
        attended = F.dropout(attended.transpose(1, 2).reshape(batch, count, width), self.dropout, self.training)
        updated = tokens + self.gate * F.dropout(self.output(attended), self.dropout, self.training)
        return updated + self.gate * F.dropout(self.ffn(self.ffn_norm(updated)), self.dropout, self.training)


class NeuronLevelModels(nn.Module):
    """Equation 14: one private bias-free history-to-two-output map per neuron, then GLU/LN."""

    def __init__(self, width, stm_length, norm_eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, stm_length, 2))
        nn.init.uniform_(self.weight, -(stm_length+2)**-.5, (stm_length+2)**-.5)
        self.normalization = nn.LayerNorm(width, eps=norm_eps)

    def forward(self, trace):
        paired = torch.einsum("bdm,dmo->bdo", trace, self.weight)
        return self.normalization(F.glu(paired, dim=-1).squeeze(-1))


class ContinuousMemoryMachine(nn.Module):
    """Vector-observation recurrent core. A VLA encoder adapter is a separate method.

    State is functional: activated [B,d], raw STM trace [B,d,M_S], LTM long_term
    [B,M_L,d], and synchronization numerator alpha / denominator beta [B,d]. No
    state is detached implicitly, including the learned LTM initialization. With
    output_dim=None the step returns the normalized synchronization features.
    memory_layers=0 (no LTM) is the CTM-style control: neurons read the raw trace.
    """

    def __init__(self, input_dim, output_dim=None, config=CMMConfig(), *, synapse=None):
        super().__init__()
        if type(input_dim) is not int or input_dim < 1 or (
                output_dim is not None and (type(output_dim) is not int or output_dim < 1)):
            raise ValueError("CMM input/output widths must be positive integers")
        self.config, self.input_dim = config, input_dim
        width = config.width
        self.output_dim = width if output_dim is None else output_dim
        # Reuse the retained CTM synapse, with table 7's configurable depth.
        # Materialize the input layer without a dummy model forward.
        if synapse is None:
            self.synapses = SynapseUNET(width, config.synapse_depth, min(16, width), config.dropout)
            self.synapses.first_projection[0] = nn.Linear(width+input_dim, width)
        else:
            self.synapses = synapse
        self.register_buffer("initial_activated", torch.zeros(width))
        self.register_buffer("initial_trace", torch.zeros(width, config.stm_length))
        self.blocks = nn.ModuleList([
            JointMemoryBlock(width, config.heads, config.dropout, config.learned_residual_gate, config.norm_eps)
            for _ in range(config.memory_layers)])
        if config.memory_layers:
            self.initial_ltm = nn.Parameter(torch.empty(config.ltm_slots, width))
            nn.init.normal_(self.initial_ltm, std=.01)
            self.sink = nn.Parameter(torch.zeros(1, width))
            self.positions = nn.Parameter(torch.empty(1+config.ltm_slots+config.stm_length, width))
            nn.init.normal_(self.positions, std=.02)
            self.ltm_norm = nn.LayerNorm(width, eps=config.norm_eps)
        else:
            self.register_buffer("initial_ltm", torch.empty(0, width))
            self.register_buffer("sink", torch.empty(0, width))
            self.register_buffer("positions", torch.empty(0, width))
            self.ltm_norm = nn.Identity()
        self.neurons = NeuronLevelModels(width, config.stm_length, config.norm_eps)
        generator = torch.Generator().manual_seed(config.pair_seed)
        pairs = torch.randperm(width*width, generator=generator)[:width]
        self.register_buffer("pair_left", pairs // width)
        self.register_buffer("pair_right", pairs % width)
        # Per-pair decay rate r in [0,15], zero-initialized as in the CTM (r=0 averages
        # uniformly); the CMM paper does not state an initialization.
        self.decay_params = nn.Parameter(torch.zeros(width))
        self.synchronization_norm = nn.RMSNorm(width, eps=config.rms_eps)
        self.readout = None if output_dim is None else nn.Linear(width, output_dim)

    @property
    def width(self):
        return self.config.width

    @property
    def stm_length(self):
        return self.config.stm_length

    @property
    def ltm_tokens(self):
        return self.config.ltm_slots

    @property
    def ticks(self):
        return self.config.ticks

    def initial_state(self, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("A CMM state needs a positive batch size")
        return {"activated": self.initial_activated.expand(batch_size, -1),
                "trace": self.initial_trace.expand(batch_size, -1, -1),
                "long_term": self.initial_ltm.expand(batch_size, -1, -1),
                "alpha": self.initial_activated.new_zeros(batch_size, self.width),
                "beta": self.initial_activated.new_zeros(batch_size, self.width)}

    def validate_state(self, state, batch_size):
        shapes = {"activated": (batch_size, self.width), "trace": (batch_size, self.width, self.stm_length),
                  "long_term": (batch_size, self.ltm_tokens, self.width),
                  "alpha": (batch_size, self.width), "beta": (batch_size, self.width)}
        if not isinstance(state, dict) or state.keys() != shapes.keys():
            raise ValueError("Require the complete CMM state, or None at a trial start")
        for name, shape in shapes.items():
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.device != self.initial_ltm.device or value.dtype != self.initial_ltm.dtype):
                raise ValueError(f"Invalid CMM state tensor {name}")

    @torch.no_grad()
    def constrain_parameters(self):
        self.decay_params.clamp_(0, 15)

    def synchronization(self, state):
        # Appendix A.3: denominator is linear, never its square root.
        normalized = state["alpha"] / state["beta"].clamp_min(torch.finfo(state["beta"].dtype).tiny)
        return self.synchronization_norm(normalized + 1e-8)

    def step(self, observation, state=None):
        if (not isinstance(observation, torch.Tensor) or observation.ndim != 2
                or observation.shape[1] != self.input_dim or not observation.shape[0]
                or observation.dtype != self.initial_ltm.dtype or observation.device != self.initial_ltm.device):
            raise ValueError("Require observation [B,input_dim] on the CMM dtype/device")
        if state is None:
            state = self.initial_state(observation.shape[0])
        self.validate_state(state, observation.shape[0])
        activated, trace, long_term, alpha, beta = (state[k] for k in STATE_KEYS)
        decay = self.decay_params.clamp(0, 15).neg().exp()
        for _ in range(self.ticks):
            preactivation = self.synapses(torch.cat((activated, observation), dim=-1))
            if preactivation.shape != activated.shape:
                raise ValueError("Synapse must return [B,width]")
            trace = torch.cat((trace[:, :, 1:], preactivation.unsqueeze(-1)), dim=-1)
            if self.blocks:
                tokens = torch.cat((self.sink.expand(observation.shape[0], -1, -1),
                                    long_term, trace.transpose(1, 2)), dim=1)
                for block in self.blocks:
                    tokens = block(tokens, self.positions)
                ltm_end = 1 + self.ltm_tokens
                long_term = self.ltm_norm(tokens[:, 1:ltm_end])
                transformed = tokens[:, ltm_end:].transpose(1, 2)
            else:
                transformed = trace
            activated = self.neurons(transformed)
            product = activated[:, self.pair_left] * activated[:, self.pair_right]
            alpha, beta = decay * alpha + product, decay * beta + 1
        updated = dict(zip(STATE_KEYS, (activated, trace, long_term, alpha, beta)))
        features = self.synchronization(updated)
        return (features if self.readout is None else self.readout(features)), updated

    def forward(self, observations, state=None, *, reset_mask=None, valid_mask=None):
        """Scan chronological [B,T,I]; padding produces zero output and no transition."""
        if (observations.ndim != 3 or not observations.shape[0] or not observations.shape[1]
                or observations.shape[2] != self.input_dim
                or observations.dtype != self.initial_ltm.dtype or observations.device != self.initial_ltm.device):
            raise ValueError("Require a nonempty chronological [B,T,input_dim] sequence")
        batch, length, _ = observations.shape
        if state is None:
            state = self.initial_state(batch)
        self.validate_state(state, batch)
        shape = (batch, length)
        valid = torch.ones(shape, dtype=torch.bool, device=observations.device) if valid_mask is None else valid_mask
        resets = torch.zeros(shape, dtype=torch.bool, device=observations.device) if reset_mask is None else reset_mask
        for mask in (valid, resets):
            if mask.shape != shape or mask.dtype != torch.bool or mask.device != observations.device:
                raise ValueError("Masks must be boolean [B,T] on the observation device")
        outputs = []
        for t in range(length):
            indices = valid[:, t].nonzero().flatten()
            output = observations.new_zeros(batch, self.output_dim)
            if indices.numel():
                selected = {name: value.index_select(0, indices) for name, value in state.items()}
                reset = resets[:, t].index_select(0, indices)
                if reset.any():
                    initial = self.initial_state(len(indices))
                    selected = {name: torch.where(reset.reshape(-1, *([1]*(value.ndim-1))), initial[name], value)
                                for name, value in selected.items()}
                y, updated = self.step(observations[:, t].index_select(0, indices), selected)
                state = {name: value.index_copy(0, indices, updated[name]) for name, value in state.items()}
                output = output.index_copy(0, indices, y)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state
