"""Paper-based METTA network (arXiv:2610.31656v1), not the authors' code.

The fast states below are tensors, not mutable Parameters. Analytical inner
gradients work under no_grad/inference_mode and remain differentiable for the
outer learner. Appendix B's chunk rule evaluates every write at the chunk's
initial memory, reads after the causal prefix update, and carries the final
memory into the next chunk. See docs/metaworld-metta.md for unspecified choices.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TTTMLP(nn.Module):
    """One residual GELU-MLP fast memory; Eqs. 3 and Appendix B.2."""

    def __init__(self, width=128, *, expansion=4, chunk_length=128, base_lr=1., eps=1e-6):
        super().__init__()
        if (any(type(x) is not int or x < 1 for x in (width, expansion, chunk_length))
                or not math.isfinite(base_lr) or base_lr <= 0
                or not math.isfinite(eps) or eps <= 0):
            raise ValueError("Require positive TTT dimensions, learning rate and epsilon")
        self.width, self.chunk_length, self.base_lr, self.eps = width, chunk_length, base_lr, eps
        self.initial_w1 = nn.Parameter(torch.randn(width, expansion * width) * .02)
        self.initial_b1 = nn.Parameter(torch.zeros(expansion * width))
        self.initial_w2 = nn.Parameter(torch.randn(expansion * width, width) * .02)
        self.initial_b2 = nn.Parameter(torch.zeros(width))
        self.norm = nn.LayerNorm(width, eps=eps)
        self.key, self.value, self.query = (nn.Linear(width, width, bias=False) for _ in range(3))
        self.learning_rate = nn.Linear(width, 1)
        nn.init.zeros_(self.learning_rate.weight)
        nn.init.zeros_(self.learning_rate.bias)

    def initial_state(self, batch):
        if type(batch) is not int or batch < 1:
            raise ValueError("Require a positive batch size")
        result = {}
        for name in ("w1", "b1", "w2", "b2"):
            value = getattr(self, "initial_" + name).unsqueeze(0).expand(batch, *getattr(self, "initial_" + name).shape)
            result[name] = value
            result["base_" + name] = value
        result["position"] = torch.zeros(batch, dtype=torch.long, device=self.initial_w1.device)
        return result

    def memory(self, u, weights):
        z = torch.bmm(u[:, None], weights["w1"]).squeeze(1) + weights["b1"]
        y = torch.bmm(F.gelu(z)[:, None], weights["w2"]).squeeze(1) + weights["b2"]
        return u + self.norm(y)

    def inner_gradients(self, key, target, weights):
        """d[.5 * ||key + LN(MLP(key)) - target||²]/d fast weights.

        No stop-gradient: derivatives through this expression include the
        Hessian terms needed to meta-learn the inner update.
        """
        z = torch.bmm(key[:, None], weights["w1"]).squeeze(1) + weights["b1"]
        activation = F.gelu(z)
        y = torch.bmm(activation[:, None], weights["w2"]).squeeze(1) + weights["b2"]
        centered = y - y.mean(-1, keepdim=True)
        inv_std = torch.rsqrt(centered.square().mean(-1, keepdim=True) + self.eps)
        unit = centered * inv_std
        error = key + unit * self.norm.weight + self.norm.bias - target
        g = error * self.norm.weight
        dy = (g - g.mean(-1, keepdim=True) - unit * (g * unit).mean(-1, keepdim=True)) * inv_std
        da = torch.bmm(dy[:, None], weights["w2"].transpose(1, 2)).squeeze(1)
        gelu_derivative = .5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-.5 * z.square()) / math.sqrt(2 * math.pi)
        dz = da * gelu_derivative
        return dict(w1=key[:, :, None] * dz[:, None, :], b1=dz,
                    w2=activation[:, :, None] * dy[:, None, :], b2=dy)

    def forward(self, x, state=None, *, online_updates=True):
        if x.ndim != 2 or x.shape[1] != self.width:
            raise ValueError("Require [batch, width] TTT inputs")
        if state is None:
            state = self.initial_state(x.shape[0])
        key, target, query = self.key(x), self.value(x), self.query(x)
        weights = {name: state[name] for name in ("w1", "b1", "w2", "b2")}
        if online_updates:
            base = {name: state["base_" + name] for name in weights}
            gradients = self.inner_gradients(key, target, base)
            eta = self.base_lr / self.width * self.learning_rate(x).sigmoid()
            weights = {name: value - eta.reshape(-1, *([1] * (value.ndim - 1))) * gradients[name]
                       for name, value in weights.items()}
        next_position = (state["position"] + 1) % self.chunk_length
        end = next_position == 0
        result = dict(weights, position=next_position)
        for name, value in weights.items():
            result["base_" + name] = torch.where(end.reshape(-1, *([1] * (value.ndim - 1))), value, state["base_" + name])
        return self.memory(query, weights), result


class MettaMemory(nn.Module):
    """Within-episode TTT stack plus cross-episode attention/EMA summaries.

    `episode_start` commits the *previous* episode's last representation before
    resetting fast weights. Full trial resets are expressed by initial_state.
    History capacity is explicit: overflow raises, rather than silently dropping
    old episodes. Slots are unordered; no positional features enter retrieval.
    """

    def __init__(self, input_dim, *, width=128, layers=2, chunk_length=128,
                 expansion=4, max_episodes=128, aggregation="attention", ema_alpha=.5,
                 base_lr=1., online_updates=True):
        super().__init__()
        if (any(type(x) is not int or x < 1 for x in (input_dim, width, layers, max_episodes))
                or aggregation not in ("attention", "ema", "none")
                or not 0 <= ema_alpha <= 1 or type(online_updates) is not bool):
            raise ValueError("Invalid METTA memory configuration")
        self.input_dim, self.output_dim, self.ticks = input_dim, width, 1
        self.max_episodes, self.aggregation = max_episodes, aggregation
        self.ema_alpha, self.online_updates = ema_alpha, online_updates
        self.embed = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList([TTTMLP(width, expansion=expansion, chunk_length=chunk_length,
                                           base_lr=base_lr) for _ in range(layers)])
        if aggregation == "attention":
            self.summary_query, self.summary_key, self.summary_value = (nn.Linear(width, width, bias=False) for _ in range(3))
            self.fusion = nn.LayerNorm(width)

    def initial_state(self, batch):
        reference = self.embed.weight
        state = dict(history=reference.new_zeros(batch, self.max_episodes, self.output_dim),
                     summaries=torch.zeros(batch, dtype=torch.long, device=reference.device),
                     last=reference.new_zeros(batch, self.output_dim), ema=reference.new_zeros(batch, self.output_dim),
                     active=torch.zeros(batch, dtype=torch.bool, device=reference.device))
        for i, layer in enumerate(self.layers):
            state.update({f"layer{i}_{key}": value for key, value in layer.initial_state(batch).items()})
        return state

    def retrieve(self, h, history, counts, ema):
        present = counts > 0
        if self.aggregation == "none":
            return h
        if self.aggregation == "ema":
            fused = self.ema_alpha * ema + (1 - self.ema_alpha) * h
        else:
            scores = torch.bmm(self.summary_key(history), self.summary_query(h)[:, :, None]).squeeze(-1) / math.sqrt(self.output_dim)
            valid = torch.arange(self.max_episodes, device=h.device)[None] < counts[:, None]
            # Empty histories must be finite in both forward and backward paths.
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            weights = scores.softmax(-1) * valid.to(scores.dtype)
            read = torch.bmm(weights[:, None], self.summary_value(history)).squeeze(1)
            fused = self.fusion(h + read)
        return torch.where(present[:, None], fused, h)

    def forward(self, interaction, state=None, *, episode_start=None):
        if interaction.ndim != 2 or interaction.shape[1] != self.input_dim:
            raise ValueError("Require [batch, input_dim] METTA inputs")
        batch = interaction.shape[0]
        if state is None:
            state = self.initial_state(batch)
        if episode_start is None:
            episode_start = torch.zeros(batch, dtype=torch.bool, device=interaction.device)
        if episode_start.shape != (batch,) or episode_start.dtype != torch.bool or episode_start.device != interaction.device:
            raise ValueError("Require boolean episode_start [batch] on the input device")
        commit = episode_start & state["active"]
        if bool((commit & (state["summaries"] >= self.max_episodes)).any()):
            raise ValueError("METTA episode-summary capacity exceeded")
        slot = F.one_hot(state["summaries"].clamp(max=self.max_episodes - 1), self.max_episodes).to(interaction.dtype)
        slot = slot * commit[:, None]
        history = state["history"] * (1 - slot[:, :, None]) + slot[:, :, None] * state["last"][:, None]
        counts = state["summaries"] + commit.long()
        updated_ema = torch.where((state["summaries"] == 0)[:, None], state["last"],
                                 self.ema_alpha * state["ema"] + (1 - self.ema_alpha) * state["last"])
        ema = torch.where(commit[:, None], updated_ema, state["ema"])
        h = self.embed(interaction)
        result = dict(history=history, summaries=counts, ema=ema,
                      active=torch.ones_like(state["active"]))
        for i, layer in enumerate(self.layers):
            prefix = f"layer{i}_"
            carried = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            fresh = layer.initial_state(batch)
            carried = {key: torch.where(episode_start.reshape(-1, *([1] * (value.ndim - 1))), fresh[key], value)
                       for key, value in carried.items()}
            h, carried = layer(h, carried, online_updates=self.online_updates)
            result.update({prefix + key: value for key, value in carried.items()})
        result["last"] = h
        return self.retrieve(h, history, counts, ema), result
