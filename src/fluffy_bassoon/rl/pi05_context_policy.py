"""Public history inside native pi0.5 attention; no external recurrent actor.

Prepared prefix *input embeddings* may be cached because image/text embedding
weights stay frozen. Native attention and suffix outputs must be recomputed.
The stochastic policy's actions are all sampled flow velocities, not decoded
robot commands. Its density includes every pass, horizon slot and model channel.
"""

import hashlib
import math

import torch
from torch import nn


PROTOCOL = "native_pi05_context_gaussian_velocity_v1"
FEEDBACK_DIM = 11


def attention_mask(valid, blocks):
    if valid.ndim != 2 or valid.dtype != torch.bool or blocks.shape != (valid.shape[1],):
        raise ValueError("Require boolean [batch,tokens] validity and token block boundaries")
    groups = blocks.to(torch.int64).cumsum(0)
    return ((groups[None, :] <= groups[:, None])[None]
            & valid[:, :, None] & valid[:, None, :])


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        if not isinstance(base, nn.Linear) or type(rank) is not int or rank < 1:
            raise ValueError("Require a native Linear and positive LoRA rank")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.down = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        delta = nn.functional.linear(nn.functional.linear(x.float(), self.down), self.up)
        return self.base(x) + delta.to(x.dtype)


class PublicHistory:
    """Bounded per-lane public buffer. Reset clears both data and validity."""
    def __init__(self, batch, capacity, *, device="cpu"):
        if any(type(v) is not int or v < 1 for v in (batch, capacity)):
            raise ValueError("Require positive batch/capacity")
        self.values = torch.zeros(batch, capacity, FEEDBACK_DIM, device=device)
        self.lengths = torch.zeros(batch, dtype=torch.long, device=device)

    def reset(self, lanes=None):
        if lanes is None:
            self.values.zero_()
            self.lengths.zero_()
        else:
            if lanes.shape != self.lengths.shape or lanes.dtype != torch.bool:
                raise ValueError("Reset lanes must be boolean [batch]")
            self.values[lanes] = 0
            self.lengths[lanes] = 0

    def append(self, feedback):
        if (feedback.shape != (len(self.lengths), FEEDBACK_DIM)
                or feedback.device != self.values.device or not feedback.isfinite().all()
                or ((feedback[:, 7:9] != 0) & (feedback[:, 7:9] != 1)).any()
                or ((feedback[:, 9:] < 0) | (feedback[:, 9:] >= 1)).any()):
            raise ValueError("Require finite public feedback, binary reward/boundary and normalized clocks")
        if (self.lengths >= self.values.shape[1]).any():
            raise ValueError("History capacity exceeded; never silently evict a trial")
        self.values[torch.arange(len(self.lengths), device=self.values.device), self.lengths] = feedback.detach()
        self.lengths += 1

    def snapshot(self):
        length = int(self.lengths.max())
        valid = torch.arange(length, device=self.values.device)[None] < self.lengths[:, None]
        return self.values[:, :length].clone(), valid


class NativeContextPi05(nn.Module):
    def __init__(self, motor, *, rank=4, noise_scale=.05):
        super().__init__()
        if not motor.pi05 or not math.isfinite(noise_scale) or noise_scale <= 0:
            raise ValueError("Require pi0.5 and positive velocity noise")
        if type(rank) is not int or rank < 1:
            raise ValueError("Require positive LoRA rank")
        self.motor = motor.eval().requires_grad_(False)
        # Keep original names/references for base-only fingerprints after wrapping.
        self._base_parameters = tuple(motor.named_parameters())
        self.rank, self.noise_scale = rank, float(noise_scale)
        width = motor.llm.configs[0].width
        reference = next(motor.parameters())
        self.feedback = nn.Linear(FEEDBACK_DIM, width, device=reference.device)
        nn.init.normal_(self.feedback.weight, std=.02)
        nn.init.zeros_(self.feedback.bias)
        # The initial public record is all zeros. A zero embedding stays zero
        # through causal self-attention/FFNs and repeated RMSNorm Jacobians can
        # overflow on backward. Give every history token a nonzero type vector
        # at roughly the RMS scale of native language input embeddings.
        self.history_type = nn.Parameter(torch.randn(width, device=reference.device))
        self.critic = nn.Linear(width, 1, device=reference.device)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)
        self.adapter_paths = []
        # Only expert 0; the action expert's projections remain untouched.
        for index, layer in enumerate(motor.llm.layers):
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                modules = getattr(layer.attn, projection)
                modules[0] = LoRALinear(modules[0], rank)
                self.adapter_paths.append(f"llm.layers.{index}.attn.{projection}.0")
        self.eval()

    def config(self):
        return dict(protocol=PROTOCOL, rank=self.rank, noise_scale=self.noise_scale,
                    adapter_paths=self.adapter_paths, history_fields=FEEDBACK_DIM,
                    likelihood="joint Gaussian velocities over every pass, model slot and channel",
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        parameters = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if parameters.keys() != state.keys() or any(parameters[n].shape != state[n].shape for n in state):
            raise ValueError("Context adapter checkpoint differs")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(state[name])

    def base_fingerprint(self):
        digest = hashlib.sha256()
        for name, value in self._base_parameters:
            if value.requires_grad or value.grad is not None:
                raise RuntimeError("Frozen native parameter has a gradient or is trainable")
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def embed_observation(self, prepared):
        return tuple(t.detach() for t in self.motor.embed_prefix(prepared))

    def build_prefix(self, inputs, history, history_mask):
        tokens, valid, blocks = inputs
        if (tokens.ndim != 3 or tokens.shape[:2] != valid.shape or valid.dtype != torch.bool
                or not valid.any(1).all() or blocks.shape != (tokens.shape[1],)
                or history.ndim != 3 or history.shape[0] != tokens.shape[0]
                or history.shape[-1] != FEEDBACK_DIM or history_mask.shape != history.shape[:2]
                or history_mask.dtype != torch.bool or history.device != tokens.device
                or history_mask.device != tokens.device):
            raise ValueError("Native prefix and public history shapes/devices differ")
        if history.shape[1]:
            public = history.masked_fill(~history_mask[..., None], 0)
            embedded = (self.feedback(public.float()) + self.history_type).to(tokens.dtype)
            tokens = torch.cat((embedded, tokens), 1)
            valid = torch.cat((history_mask, valid), 1)
            # Each older record is causal. All current image/language tokens
            # share a later block and can read all valid preceding history.
            native_blocks = blocks.clone()
            native_blocks[0] = True
            blocks = torch.cat((torch.ones(history.shape[1], device=tokens.device, dtype=torch.bool),
                                native_blocks))
        positions = valid.long().cumsum(1) - 1
        outputs, cache = self.motor.llm([tokens.detach() if not history.shape[1] else tokens, None],
            positions=positions, mask=attention_mask(valid, blocks), prefix_requires_grad=True)
        pooled = (outputs[0].float() * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        value = self.critic(pooled.detach()).squeeze(-1)
        return valid, cache, value

    def velocity(self, prepared, x, time, valid, cache):
        return self.motor.velocity_from_suffix(self.motor.run_suffix(prepared, x, time, cache, valid))

    def flow_loss(self, prepared, inputs, history, mask, target, noise, time, *, steps=5, active_dims=7):
        if (target.shape != noise.shape or target.ndim != 3 or time.shape != (target.shape[0],)
                or not 0 < steps <= target.shape[1] or not 0 < active_dims <= target.shape[2]):
            raise ValueError("Invalid flow targets or executed action prefix")
        valid, cache, _ = self.build_prefix(inputs, history, mask)
        x = (1-time[:, None, None]) * target.detach() + time[:, None, None] * noise
        prediction = self.velocity(prepared, x, time, valid, cache)
        return (prediction[:, :steps, :active_dims]
                - (noise-target.detach())[:, :steps, :active_dims]).square().mean()

    @torch.no_grad()
    def sample_chunk(self, prepared, inputs, history, mask, *, generator, num_steps=4,
                     deterministic=False, initial_noise=None):
        if type(num_steps) is not int or num_steps < 1:
            raise ValueError("Require positive flow steps")
        valid, cache, value = self.build_prefix(inputs, history, mask)
        shape = (history.shape[0], self.motor.action_horizon, self.motor.action_dim)
        x = (torch.randn(shape, device=history.device, generator=generator) if initial_noise is None
             else initial_noise.clone())
        initial = x.detach().cpu()
        states, samples, times, log_probs = [], [], [], []
        for i in range(num_steps):
            time = torch.full((shape[0],), 1-i/num_steps, device=x.device)
            mean = self.velocity(prepared, x, time, valid, cache)
            sampled = mean if deterministic else mean + self.noise_scale * torch.randn(
                shape, device=x.device, generator=generator)
            states.append(x.detach().cpu())
            samples.append(sampled.detach().cpu())
            times.append(time.detach().cpu())
            log_probs.append(torch.distributions.Normal(mean, self.noise_scale).log_prob(sampled).sum((-1,-2)))
            x = x-sampled/num_steps
        record = dict(protocol=PROTOCOL, deterministic=deterministic, initial_noise=initial,
                      states=torch.stack(states), velocities=torch.stack(samples), times=torch.stack(times),
                      old_log_prob=torch.stack(log_probs).sum(0).cpu(), old_value=value.cpu())
        return x, record

    def replay_chunk(self, prepared, inputs, history, mask, record):
        if record.get("protocol") != PROTOCOL or record.get("deterministic") is not False:
            raise ValueError("Require a stochastic native context record")
        valid, cache, value = self.build_prefix(inputs, history, mask)
        log_probs = []
        for x, sampled, time in zip(record["states"], record["velocities"], record["times"], strict=True):
            mean = self.velocity(prepared, x.to(valid.device), time.to(valid.device), valid, cache)
            log_probs.append(torch.distributions.Normal(mean, self.noise_scale).log_prob(
                sampled.to(mean.device)).sum((-1,-2)))
        return torch.stack(log_probs).sum(0), value
