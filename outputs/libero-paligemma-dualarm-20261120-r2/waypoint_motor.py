"""Continuous visual waypoint policy for the public five-control source motor interface."""

import json
import math
from pathlib import Path

import motor as base
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCH = "plate_waypoint_v1"
MOTION_SCALE_M = 0.035
PLATE_SCALE_M = 0.10
LOOKAHEAD = 6
SLOTS = base.HORIZON


class WaypointHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.key = nn.Linear(width, 32)
        self.value = nn.Linear(width, 48)
        self.text = nn.Linear(width, 64)
        self.query = nn.Linear(64, 2 * 4 * 32)
        axis = torch.linspace(-1.0, 1.0, 16)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "xy", torch.stack((x.flatten(), y.flatten()), -1), persistent=False
        )
        self.visual = nn.Sequential(
            nn.Linear(2 * 4 * 50 + 64, 160), nn.LayerNorm(160), nn.SiLU()
        )
        self.plate = nn.Linear(160, 3)
        self.motor = nn.Sequential(
            nn.Linear(160 + 8 + 3, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, SLOTS * 7),
        )
        # Shared across bodies, poses and control offsets. Optimize under native
        # control imitation while physical motion labels constrain its input.
        self.log_gain = nn.Parameter(torch.zeros(3))

    def forward(self, context, mask, state, raw_xyz, center):
        context = self.norm(context.float())
        b = len(state)
        grids = context[:, :512].reshape(b, 2, 256, -1)
        prompt = F.silu(
            self.text(context[torch.arange(b, device=state.device), mask.sum(1) - 1])
        )
        queries = self.query(prompt).reshape(b, 2, 4, 32)
        attention = (
            (queries @ self.key(grids).transpose(-1, -2)) / math.sqrt(32)
        ).softmax(-1)
        landmarks = torch.cat(
            (attention @ self.xy, attention @ self.value(grids)), dim=-1
        )
        features = self.visual(torch.cat((landmarks.flatten(1), prompt), dim=-1))
        plate = center + PLATE_SCALE_M * self.plate(features)
        relative = (plate - raw_xyz) / PLATE_SCALE_M
        plan = self.motor(torch.cat((features, state, relative), -1)).reshape(
            b, SLOTS, 7
        )
        motion = plan[..., :3] * MOTION_SCALE_M
        gain = self.log_gain.clamp(-2, 2).exp()
        command = torch.cat(
            ((plan[..., :3] * gain).clamp(-1.0, 1.0), plan[..., 3:].tanh()), -1
        )
        return plate, motion, command


class WaypointMotor(base.Motor):
    def __init__(self, weights):
        super().__init__(weights)
        self.head = WaypointHead(self.vlm.config.text_config.hidden_size).to(
            self.state_mean.device
        )

    def predict(self, observations):
        # The allowlist is also the interface used by the native rollout.
        if any(set(obs) != {"rgb", "wrist", "state", "prompt"} for obs in observations):
            raise ValueError(
                "Expected exactly both public cameras, wrist/jaw and prompt"
            )
        raw = np.stack([o["state"] for o in observations]).astype(
            np.float32, copy=False
        )
        if raw.shape != (len(observations), 8) or not np.isfinite(raw).all():
            raise ValueError("Invalid public wrist/jaw state")
        raw = torch.from_numpy(raw).to(self.state_mean.device)
        state = ((raw - self.state_mean) / self.state_scale).clamp(-10, 10)
        context, mask = self.prefix(observations)
        return self.head(context, mask, state, raw[:, :3], self.state_mean[:3])

    def forward(self, observations):
        plate, motion, actions = self.predict(observations)
        if not (
            torch.isfinite(plate).all()
            & torch.isfinite(motion).all()
            & torch.isfinite(actions).all()
        ):
            raise FloatingPointError(
                "Nonfinite visual localization, waypoint or native action"
            )
        return actions

    @classmethod
    def load(cls, directory, weights):
        from motor_weights import sha256

        checkpoint = Path(directory) / "checkpoint.pt"
        report = json.loads((checkpoint.parents[1] / "dev/report.json").read_text())
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if (
            saved["schema"] != base.SCHEMA
            or saved["contract"]["architecture"] != ARCH
            or report["status"] != "complete"
            or not report["source_gate"]
            or report["motor_checkpoint_sha256"] != sha256(checkpoint)
            or report["fit_id"] != saved["fit_id"]
            or report["selected_step"] != saved["step"]
            or saved["best"]["step"] != saved["step"]
        ):
            raise ValueError(
                "Expected the exact promoted source-development checkpoint"
            )
        model = cls(weights)
        if model.identity != saved["contract"]["weights"]:
            raise ValueError("Pretrained weights differ from the selected fit")
        model.restore_delta(saved["best"]["model"])
        return model.requires_grad_(False).eval()
