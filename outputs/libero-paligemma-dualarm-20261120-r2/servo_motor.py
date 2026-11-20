"""Optional R2 supervised action head. The R2 entry points put the R1 motor on sys.path."""

import math

import torch
from torch import nn
from torch.nn import functional as F

import motor as base


BASELINE = "chunk_huber_v1"
SERVO = "spatial_bins_v1"
ARCHITECTURES = (BASELINE, SERVO)
AXES = 7
MODES = 3
INTERVALS = 9
BINS = INTERVALS + 2  # Two point masses at the controller limits.
RESIDUAL_WEIGHT = 0.5
VIEWS = 2
SIDE = 16
LANDMARKS = 4
KEY_WIDTH = 32
VALUE_WIDTH = 24


def architecture_name(contract):
    """Historical checkpoints lack this field and use the original R1 head."""
    name = contract.get("architecture", BASELINE)
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown motor architecture {name!r}")
    return name


def training_recipe(architecture):
    recipe = base.recipe()
    if architecture == BASELINE:
        return recipe
    if architecture != SERVO:
        raise ValueError(f"Unknown motor architecture {architecture!r}")
    del recipe["huber_beta"]
    recipe.update(
        action_objective="masked chunk mixture with categorical bins and within-bin Huber",
        chunk_modes=MODES,
        interior_intervals=INTERVALS,
        residual_huber_beta=1.0,
        residual_weight=RESIDUAL_WEIGHT,
        landmarks_per_view=LANDMARKS,
        decode="highest-prior chunk mode, highest-probability bin, bounded residual",
    )
    return recipe


class SpatialBinHead(nn.Module):
    """Public state and instruction query each camera's contextualized patch grid."""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.text = nn.Linear(width, 64)
        self.proprio = nn.Linear(8, 32)
        self.query = nn.Linear(96, VIEWS * LANDMARKS * KEY_WIDTH)
        self.key = nn.Linear(width, KEY_WIDTH)
        self.value = nn.Linear(width, VALUE_WIDTH)
        axis = torch.linspace(-1.0, 1.0, SIDE)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("xy", torch.stack((x.flatten(), y.flatten()), dim=-1), persistent=False)
        inputs = 96 + VIEWS * LANDMARKS * (VALUE_WIDTH + 2)
        self.fusion = nn.Sequential(nn.Linear(inputs, 192), nn.LayerNorm(192), nn.SiLU())
        self.prior = nn.Linear(192, MODES)
        # Reuse the same action decoder at each future control. Only the learned
        # offset embedding changes with the control's position in this proposal.
        self.time = nn.Parameter(torch.empty(base.HORIZON, 32))
        nn.init.normal_(self.time, std=0.1)
        self.decoder = nn.Sequential(
            nn.Linear(192 + 32, 128), nn.SiLU(), nn.Linear(128, MODES * AXES * (BINS + INTERVALS))
        )

    def forward(self, context, mask, state):
        batch = state.shape[0]
        context = self.norm(context.float())
        patches = context[:, :VIEWS * SIDE**2].reshape(batch, VIEWS, SIDE**2, -1)
        last = mask.sum(1) - 1
        text = context[torch.arange(batch, device=state.device), last]
        public = F.silu(torch.cat((self.text(text), self.proprio(state)), dim=-1))
        queries = self.query(public).reshape(batch, VIEWS, LANDMARKS, KEY_WIDTH)
        scores = queries @ self.key(patches).transpose(-1, -2)
        attention = (scores / math.sqrt(KEY_WIDTH)).softmax(-1)
        # Each view normalizes over its own 256 raster locations. Each landmark
        # supplies both its soft pixel location and the features found there.
        landmarks = torch.cat((attention @ self.xy, attention @ self.value(patches)), dim=-1)
        latent = self.fusion(torch.cat((public, landmarks.flatten(1)), dim=-1))
        slots = torch.cat((latent[:, None].expand(-1, base.HORIZON, -1),
                           self.time[None].expand(batch, -1, -1)), dim=-1)
        params = self.decoder(slots).reshape(batch, base.HORIZON, MODES, AXES, BINS + INTERVALS)
        logits, residuals = params.transpose(1, 2).split((BINS, INTERVALS), dim=-1)
        return self.prior(latent), logits, residuals


def _interval_center(labels):
    # Labels 1..INTERVALS denote equal-width intervals covering (-1, 1).
    return (2 * labels - 1).float() / INTERVALS - 1


def _chosen(values, indices):
    return values.gather(-1, indices.unsqueeze(-1)).squeeze(-1)


def decode(distribution):
    """Select one observation-conditioned mode for the entire [5, 7] chunk."""
    prior, logits, residuals = distribution
    # argmax and endpoint clipping can otherwise turn invalid logits into a
    # plausible finite command before the inherited act() checks its output.
    if not (torch.isfinite(prior).all() & torch.isfinite(logits).all() & torch.isfinite(residuals).all()):
        raise FloatingPointError("Nonfinite servo action distribution")
    batch = torch.arange(len(prior), device=prior.device)
    modes = prior.argmax(-1)
    labels = logits[batch, modes].argmax(-1)
    offsets = _chosen(residuals[batch, modes], (labels - 1).clamp(0, INTERVALS - 1))
    interior = _interval_center(labels) + offsets.clamp(-1, 1) / INTERVALS
    actions = torch.where(labels == 0, -1.0, torch.where(labels == BINS - 1, 1.0, interior))
    return actions.clamp(-1, 1)


def mixture_loss(distribution, target, mask):
    """One loss per anchor. The mode is shared over all valid times and axes."""
    prior, logits, residuals = distribution  # [B,M], [B,M,5,7,K], [B,M,5,7,I]
    valid = mask[:, None, :, None]
    interior = (target > -1) & (target < 1)
    interval = ((target + 1) * (INTERVALS / 2)).floor().long().clamp(0, INTERVALS - 1)
    labels = torch.where(target <= -1, 0, torch.where(target >= 1, BINS - 1, interval + 1))
    labels = labels[:, None].expand_as(logits[..., 0])
    predicted = _chosen(residuals, interval[:, None].expand_as(labels))
    offset = (target - _interval_center(interval + 1)) * INTERVALS
    regression = F.smooth_l1_loss(predicted, offset[:, None].expand_as(predicted), reduction="none")
    log_bin = _chosen(logits.log_softmax(-1), labels)
    score = log_bin - RESIDUAL_WEIGHT * regression * interior[:, None]
    log_joint = prior.log_softmax(-1) + (score * valid).sum(dim=(-2, -1))
    return -torch.logsumexp(log_joint, dim=1) / (mask.sum(1) * AXES)


class ServoMotor(base.Motor):
    def __init__(self, weights):
        super().__init__(weights)
        width = self.vlm.config.text_config.hidden_size
        self.head = SpatialBinHead(width).to(self.state_mean.device).train(self.training)

    def forward(self, observations):
        # The inherited forward does both real-camera encoding and normalized
        # state validation, then calls our distribution-valued head.
        return decode(super().forward(observations))

    def loss_per_anchor(self, observations, target, mask):
        return mixture_loss(super().forward(observations), target, mask)


def build_motor(weights, architecture):
    if architecture == BASELINE:
        return base.Motor(weights)
    if architecture == SERVO:
        return ServoMotor(weights)
    raise ValueError(f"Unknown motor architecture {architecture!r}")


def loss_per_anchor(motor, observations, target, mask):
    """One VLM pass, one equally weighted loss for each collated source anchor."""
    if isinstance(motor, ServoMotor):
        return motor.loss_per_anchor(observations, target, mask)
    errors = F.smooth_l1_loss(motor(observations), target, beta=0.1, reduction="none").mean(-1)
    return (errors * mask).sum(1) / mask.sum(1)


@torch.no_grad()
def load_selected(saved, weights):
    """Restore the checkpoint envelope already opened under the transfer gate's lock."""
    if saved["schema"] != base.SCHEMA or saved["best"] is None:
        raise ValueError("Expected a selected motor checkpoint")
    contract = saved["contract"]
    motor = build_motor(weights, architecture_name(contract))
    if motor.identity != contract["weights"]:
        raise ValueError("Base PaliGemma snapshot differs from the selected motor")
    motor.restore_delta(saved["best"]["model"])
    return motor.requires_grad_(False).eval()
