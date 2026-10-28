"""Shared token operations; architecture folders select their own block structure."""
import torch
from torch import nn


class TransformerExpert(nn.Module):
    population_count = 1  # One memory dictionary row per replay lane.

    def __init__(self, config):
        super().__init__()
        self.width = config.expert_width
        self.depth = config.expert_depth
        self.memory_count = config.memory_tokens
        self.horizon = config.chunk_size
        self.context_proj = nn.Linear(config.context_dim, self.width)
        self.initial_memory = nn.Parameter(torch.randn(1, self.memory_count, self.width) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(1, self.horizon, self.width) * 0.02)
        self.output_norm = nn.LayerNorm(self.width)
        self.readout = nn.Linear(self.width, config.action_feature.shape[0])
        blocked = torch.zeros(self.memory_count + self.horizon, self.memory_count + self.horizon,
                              dtype=torch.bool)
        blocked[:self.memory_count, self.memory_count:] = True
        self.register_buffer("blocked_attention", blocked, persistent=False)

    @staticmethod
    def make_block(config):
        return nn.TransformerDecoderLayer(config.expert_width, config.expert_heads,
            dim_feedforward=4 * config.expert_width, dropout=0, activation="gelu",
            batch_first=True, norm_first=True)

    def project_context(self, context, valid):
        # Padding is zeroed before projection so even nonfinite padded features cannot leak.
        context = self.context_proj(context.masked_fill(~valid[..., None], 0))
        return context, context

    def validate_memory(self, memory, batch):
        if memory is None:
            return
        if (not isinstance(memory, dict) or set(memory) != {"tokens"}
                or not isinstance(memory["tokens"], torch.Tensor)
                or memory["tokens"].shape != (batch, self.memory_count, self.width)
                or memory["tokens"].device != self.initial_memory.device
                or memory["tokens"].dtype != self.initial_memory.dtype):
            raise ValueError("Memory must contain tokens[B,M,D] on the expert device/dtype")

    def memory_input(self, memory, batch):
        return self.initial_memory.expand(batch, -1, -1)

    def forward(self, projected, context_mask, memory=None):
        context, _ = projected
        batch = context.shape[0]
        self.validate_memory(memory, batch)
        tokens = torch.cat((self.memory_input(memory, batch),
                            self.action_queries.expand(batch, -1, -1)), dim=1)
        for block in self.blocks():
            tokens = block(tokens, context, tgt_mask=self.blocked_attention,
                           memory_key_padding_mask=~context_mask)
        tokens = self.output_norm(tokens)
        return self.readout(tokens[:, self.memory_count:]), {"tokens": tokens[:, :self.memory_count]}
