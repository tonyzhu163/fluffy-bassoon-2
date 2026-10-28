"""Token-wise CTM modules shared by deterministic and stochastic direct experts."""
import numpy as np
import torch
from torch import nn
from .sakana_reference import SakanaCoreOperations
from .attention import ProjectedKVAttention


class TokenCTM(SakanaCoreOperations, nn.Module):
    def __init__(self, context_dim, *, population_count, readout_dim, action_dim=7, horizon=20, width=512, ticks=8,
                 heads=4, history_length=32, neuron_hidden=16, synapse_depth=2,
                 attention_pairs=1024, output_pairs=1024):
        nn.Module.__init__(self)
        if (min(context_dim, action_dim, horizon, population_count, readout_dim, width, ticks, heads, history_length, neuron_hidden) < 1
                or width % heads or width % 2 or synapse_depth < 2
                or attention_pairs != 2*width or output_pairs != 2*width):
            raise ValueError("Invalid direct CTM dimensions")
        self.context_dim, self.action_dim, self.chunk_size = context_dim, action_dim, horizon
        self.population_count, self.readout_dim = population_count, readout_dim
        self.d_model = self.hidden_dim = width
        self.history_length, self.num_ticks = history_length, ticks
        self.neuron_select_type = "random-pairing"
        self.n_synch_action = self.n_synch_out = 2*width
        self.synch_representation_size_action = self.synch_representation_size_out = 2*width
        self.context_proj = nn.Sequential(nn.Linear(context_dim, width), nn.LayerNorm(width))
        self.positions = nn.Parameter(torch.randn(population_count, width) * .02)
        self._build_recurrent_modules(width, heads, history_length, neuron_hidden, synapse_depth)

    def _build_recurrent_modules(self, width, heads, history_length, neuron_hidden, synapse_depth):
        self.conditioning_norm = nn.LayerNorm(width)
        self.slot_norm = nn.LayerNorm(width)
        self.q_proj = nn.Linear(2 * width, width)
        self.attention = ProjectedKVAttention(width, heads)
        self.synapses = self.get_synapses(synapse_depth, width, 0)
        self.synapses(torch.zeros(1, 2 * width))
        self.trace_processor = self.get_neuron_level_models(True, False, history_length, neuron_hidden, width, 0)
        self.start_activated_state = nn.Parameter(torch.empty(width).uniform_(-width**-.5, width**-.5))
        bound = (width + history_length)**-.5
        self.start_trace = nn.Parameter(torch.empty(width, history_length).uniform_(-bound, bound))
        numpy_state = np.random.get_state()
        try:
            np.random.seed(torch.initial_seed() % 2**32)
            self.set_synchronisation_parameters("action", self.n_synch_action, 0)
            self.set_synchronisation_parameters("out", self.n_synch_out, 0)
        finally:
            np.random.set_state(numpy_state)
        self.velocity_proj = nn.Linear(2 * width, self.readout_dim)

    def project_context(self, context, context_mask):
        """Return K/V for this graph, retaining state/context projection gradients."""
        if (context.ndim != 3 or not context.shape[0] or context.shape[-1] != self.context_dim
                or context_mask.shape != context.shape[:2] or context_mask.dtype != torch.bool
                or not context.is_floating_point()
                or any(x.device != self.start_trace.device for x in (context, context_mask))):
            raise ValueError("Expected context [B,P,C] and boolean mask [B,P] on the expert device")
        tokens = self.context_proj(context.masked_fill(~context_mask[..., None], 0).to(self.start_trace.dtype))
        return self.attention.project_kv(tokens)
