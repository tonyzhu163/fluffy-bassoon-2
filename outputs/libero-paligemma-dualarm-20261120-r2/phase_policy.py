"""Public-observation PaliGemma motor for a single demonstrated Paddle route family."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "libero-paligemma-crossarm-20261120-r1"))
import motor as base

ARCH = "phase_paddle_v1"
SLOTS = base.HORIZON
PHASES = 3


def rotation_error(target, current):
    """Shortest world rotation vector log(target * inverse(current)), both xyzw."""
    v, w = target[..., :3], target[..., 3:]
    u, s = current[..., :3], current[..., 3:]
    vec = s * v - w * u - torch.linalg.cross(v.expand_as(u), u)
    scalar = w * s + (v * u).sum(-1, keepdim=True)
    vec = torch.where(scalar < 0, -vec, vec)
    length = vec.norm(dim=-1, keepdim=True)
    return vec * (2 * torch.atan2(length, scalar.abs().clamp_min(1e-8))
                  / length.clamp_min(1e-8))


class SkillHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.patch = nn.Linear(width, 24)
        self.spatial = nn.Conv2d(24, 16, kernel_size=3, stride=2, padding=1)
        self.text = nn.Linear(width, 32)
        self.visual = nn.Sequential(nn.Linear(2 * 8 * 8 * 16 + 32, 192),
                                    nn.LayerNorm(192), nn.SiLU())
        self.plate = nn.Linear(192, 2)
        self.plate_origin = nn.Parameter(torch.zeros(2))
        self.fusion = nn.Sequential(nn.Linear(192 + 8 + 2 + 2, 160),
                                    nn.LayerNorm(160), nn.SiLU())
        self.slots = nn.Parameter(torch.randn(SLOTS, 16) * 0.02)
        self.plan = nn.Sequential(nn.Linear(176, 96), nn.SiLU())
        self.progress = nn.Linear(96, PHASES)
        self.local = nn.Linear(96, PHASES * 11)
        self.motif = nn.Parameter(torch.zeros(PHASES, 3))
        self.hand = nn.Parameter(torch.tensor([0., 0., 0., 1.]))
        self.log_linear_gain = nn.Parameter(torch.zeros(PHASES, 3))
        self.log_angular_gain = nn.Parameter(torch.zeros(PHASES, 1))
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)

    def forward(self, context, mask, state, raw, ticks):
        b = len(raw)
        context = self.norm(context.float())
        patch = F.silu(self.patch(context[:, :512])).reshape(-1, 16, 16, 24)
        grid = F.silu(self.spatial(patch.permute(0, 3, 1, 2))).reshape(b, -1)
        last = context[torch.arange(b, device=raw.device), mask.sum(1) - 1]
        vision = self.visual(torch.cat((grid, F.silu(self.text(last))), -1))
        plate = self.plate_origin + 0.10 * self.plate(vision)
        # The second clock feature marks whether elapsed controls were supplied.
        known = ticks >= 0
        clock = torch.stack((ticks.clamp(0, 300) / 100, known.float()), -1)
        fused = self.fusion(torch.cat((vision, state, (plate - raw[:, :2]) / .10, clock), -1))
        slot = self.plan(torch.cat((fused[:, None].expand(-1, SLOTS, -1),
                                    self.slots.expand(b, -1, -1)), -1))
        logits = self.progress(slot)
        local = self.local(slot).reshape(b, SLOTS, PHASES, 11)
        # Motif stores plate-relative XY and absolute world Z in metres.
        anchor = F.pad(plate, (0, 1))[:, None, None, :]
        target = anchor + self.motif + .05 * local[..., :3]
        error = target - raw[:, None, None, :3]
        # Learn the residual force needed even when wrist motion is stalled.
        linear = error / .035 * self.log_linear_gain.clamp(-2, 2).exp() + local[..., 3:6]
        hand = F.normalize(self.hand + .25 * local[..., 6:10], dim=-1, eps=1e-8)
        current = F.normalize(raw[:, 3:7], dim=-1)[:, None, None, :].expand_as(hand)
        angular = rotation_error(hand, current) * self.log_angular_gain.clamp(-2, 2).exp()
        actions = torch.cat((linear.clamp(-1, 1), angular.clamp(-1, 1),
                             local[..., 10:].clamp(-1, 1)), -1)
        return dict(plate=plate, logits=logits, error=error, hand=hand, actions=actions)


def select(values, phase):
    indices = phase[..., None, None].expand(-1, -1, 1, values.shape[-1])
    return values.gather(2, indices).squeeze(2)


@dataclass
class Memory:
    tick: int = 0
    phase: int = 0

    def advance(self, proposed_phases, executed):
        if not 1 <= executed <= len(proposed_phases):
            raise ValueError("Advance by the number of controls actually executed")
        self.phase = int(proposed_phases[executed - 1])
        self.tick += executed


class PhaseMotor(base.Motor):
    def __init__(self, weights):
        super().__init__(weights)
        if self.identity["repo"] != "google/paligemma-3b-mix-224":
            raise ValueError("Expected the pinned PaliGemma MIX 224 snapshot")
        self.head = SkillHead(self.vlm.config.text_config.hidden_size).to(self.state_mean.device)

    def predict(self, observations, ticks):
        for obs in observations:
            if set(obs) != {"rgb", "wrist", "state", "prompt"}:
                raise ValueError("Expected exactly rgb, wrist, state and prompt")
        raw = np.stack([o["state"] for o in observations]).astype(np.float32, copy=False)
        if raw.shape != (len(observations), 8) or not np.isfinite(raw).all():
            raise ValueError("Expected finite public XYZ/XYZW/jaw")
        if not np.allclose(np.linalg.norm(raw[:, 3:7], axis=-1), 1., atol=1e-4):
            raise ValueError("Expected unit public XYZW quaternion")
        raw = torch.as_tensor(raw, device=self.state_mean.device)
        # Quaternion signs are arbitrary. Canonicalize the MLP input only.
        q = raw[:, 3:7]
        pivot = q.gather(1, q.abs().argmax(1, keepdim=True))
        canonical = torch.cat((raw[:, :3], torch.where(pivot < 0, -q, q), raw[:, 7:]), -1)
        state = ((canonical - self.state_mean) / self.state_scale).clamp(-10, 10)
        context, mask = self.prefix(observations)
        clock = torch.as_tensor(ticks, device=raw.device, dtype=raw.dtype).reshape(len(raw))
        return self.head(context, mask, state, raw, clock)

    @staticmethod
    def decode(pred, floors, ticks=None):
        learned = pred["logits"].argmax(-1)
        # Adjacent decoded slots must agree that the old stage has ended.
        # Each issued control can advance by at most one stage.
        following = torch.cat((learned[:, 1:], learned[:, -2:-1]), 1)
        supported = torch.minimum(learned, following)
        current = torch.as_tensor(floors, device=learned.device, dtype=torch.long)
        stages = []
        opening = (torch.as_tensor(ticks, device=learned.device) == 0
                   if ticks is not None else torch.zeros_like(current, dtype=torch.bool))
        for slot, wanted in enumerate(supported.unbind(1)):
            advance = wanted > current
            if slot == 0:
                advance = advance & ~opening
            current = current + advance.long()
            stages.append(current)
        phases = torch.stack(stages, dim=1)
        actions = select(pred["actions"], phases)
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Nonfinite phase policy command")
        return actions, phases

    def forward(self, observations):
        """Clock-free public interface, also used as an offline ablation."""
        return self.decode(self.predict(observations, [-1] * len(observations)),
                           [0] * len(observations))[0]

    @torch.inference_mode()
    def plan(self, observations, memories):
        if len(observations) != len(memories):
            raise ValueError("One memory per active simulator")
        return self.decode(self.predict(observations, [m.tick for m in memories]),
                           [m.phase for m in memories], [m.tick for m in memories])

    @classmethod
    def load(cls, checkpoint, weights):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["contract"]["architecture"] != ARCH:
            raise ValueError("Expected a phase skill checkpoint")
        model = cls(weights)
        if saved["contract"]["weights"] != model.identity:
            raise ValueError("Checkpoint base weights differ")
        model.restore_delta(saved["model"])
        return model.requires_grad_(False).eval()
