"""Lyco attention with reusable observation projections and native MHA weights."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class ProjectedKVAttention(nn.MultiheadAttention):
    """Keep MHA initialization/state dict; expose its packed K/V projection.

    Caches belong to the caller's current forward graph, never to this module.
    The sequence-first projections and output layout follow PyTorch 2.7 MHA.
    """

    def __init__(self, width, heads):
        super().__init__(width, heads, dropout=0, batch_first=True)

    def project_kv(self, tokens):
        """Project [B,N,D] tokens without detaching their autograd history."""
        projected = F.linear(tokens.transpose(0, 1), self.in_proj_weight[self.embed_dim:],
                             self.in_proj_bias[self.embed_dim:])
        # Match native encoder-decoder attention's packed 2D projection.
        projected = (projected.unflatten(-1, (2, self.embed_dim)).unsqueeze(0)
                     .transpose(0, -2).squeeze(-2).contiguous())
        return projected[0].transpose(0, 1), projected[1].transpose(0, 1)

    def forward(self, query, keys, values, valid):
        """Attend to projected [B,N,D] K/V; True validity means an allowed key."""
        batch, length, width = query.shape
        query = F.linear(query.transpose(0, 1), self.in_proj_weight[:width], self.in_proj_bias[:width])
        query = query.reshape(length, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        keys = keys.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        mask = torch.zeros_like(valid, dtype=query.dtype).masked_fill(~valid, float('-inf'))
        with sdpa_kernel(SDPBackend.MATH):
            attended = F.scaled_dot_product_attention(query, keys, values, mask[:, None, None, :])
        attended = attended.permute(2, 0, 1, 3).contiguous().view(length * batch, width)
        return self.out_proj(attended).view(length, batch, width).transpose(0, 1)
