"""Full-action LIBERO reader. Import load_scorer in the collector's CPU worker."""

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# The pinned checkout also supplies the causal core when used from a reader venv.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fluffy_bassoon.meta_rl.causal_decoder import CausalDecoderCore

PREFIX = 400
GROUP = 20
CHANNELS = 13  # issued [7], post-control displacement [3], jaw [2], native reward
STATE_DIM = 8
PROPOSAL_SHAPE = (2, 10, 7)
WIDTH = 128
RECIPES = ("meta", "imitation")
SEEDS = (41, 42, 43)
FORMAT = "libero90-full-action-v1"


class Practice(NamedTuple):
    history: Tensor
    length: Tensor
    terminal: Tensor


def numeric(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iufb":
        raise ValueError(f"{name}: expected numeric {shape}, got {array.dtype} {array.shape}")
    array = np.asarray(array, dtype=np.float32, order="C")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: nonfinite input")
    return array


def pack_practice(tape: Mapping) -> Practice:
    """One completed practice, padded to the 400-control horizon. Keep its actual terminal outcome."""
    controls = np.asarray(tape["controls"])
    terminal = np.asarray(tape["terminal"])
    if controls.shape != () or controls.dtype.kind not in "iu" or int(controls) < 1:
        raise ValueError("Expected a positive integer support control count")
    if terminal.shape != () or terminal.item() not in (False, True):
        raise ValueError("Expected binary native support terminal success")
    total = int(controls)
    length = min(PREFIX, total)
    initial = numeric(tape["initial_position"], (3,), "initial_position")
    fields = []
    for key, tail in (("issued", (7,)), ("positions", (3,)), ("gripper", (2,)), ("rewards", ())):
        values = np.asarray(tape[key])
        if values.shape != (total, *tail):
            raise ValueError(f"{key}: support length or shape disagrees with controls")
        fields.append(numeric(values[:length], (length, *tail), key))
    issued, positions, jaw, rewards = fields
    motion = np.diff(positions, axis=0, prepend=initial[None])
    history = np.zeros((PREFIX, CHANNELS), dtype=np.float32)
    history[:length] = np.column_stack((issued, motion, jaw, rewards))
    return Practice(torch.from_numpy(history), torch.tensor(length), torch.tensor(float(terminal.item())))


class ActionReader(nn.Module):
    """Encode practice and current observation once, then score each full action independently."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("Expected a nonempty pooled embedding")
        self.embedding_dim = embedding_dim
        self.support_projection = nn.Linear(GROUP * (CHANNELS + 1), WIDTH)
        self.boundary = nn.Linear(1, WIDTH)
        self.query_projection = nn.Linear(STATE_DIM + embedding_dim, WIDTH)
        self.core = CausalDecoderCore(WIDTH, width=WIDTH, layers=2, heads=4,
                                     memory_length=PREFIX // GROUP + 1)
        self.action_projection = nn.Linear(10 * 7, WIDTH)
        self.score = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1))
        for key, size in (("history", CHANNELS), ("state", STATE_DIM)):
            self.register_buffer(key + "_mean", torch.zeros(size))
            self.register_buffer(key + "_scale", torch.ones(size))

    def forward(self, practice: Practice | None, query: Tensor, proposals: Tensor) -> Tensor:
        state = (query[:, :STATE_DIM] - self.state_mean) / self.state_scale
        pooled = F.layer_norm(query[:, STATE_DIM:], (self.embedding_dim,))
        current = self.query_projection(torch.cat((state, pooled), dim=1))
        rows = torch.arange(len(query), device=query.device)
        if practice is None:
            tokens = current[:, None]
            last = torch.zeros_like(rows)
        else:
            history, lengths, terminal = practice
            valid_control = torch.arange(PREFIX, device=query.device)[None] < lengths[:, None]
            history = (history - self.history_mean) / self.history_scale
            history = torch.where(valid_control[:, :, None], history, 0.0)
            marked = torch.cat((history, valid_control[:, :, None]), dim=-1)
            blocks = self.support_projection(marked.reshape(len(query), PREFIX // GROUP, -1))
            # Close the practice *after* its last valid block, including when success
            # occurs later than the prefix. The query can attend across this boundary.
            end = (lengths + GROUP - 1) // GROUP
            last = end + 1
            tokens = torch.cat((blocks, current.new_zeros((len(query), 2, WIDTH))), dim=1)
            tokens[rows, end] = self.boundary(terminal[:, None])
            tokens[rows, last] = current
        valid = torch.arange(tokens.shape[1], device=query.device)[:, None] <= last[None]
        encoded = self.core.sequence(tokens.transpose(0, 1), valid=valid, reset=torch.zeros_like(valid))
        context = encoded[last, rows]
        # Use the same matrix shape and row for both alternatives. Some CPU
        # kernels round duplicate rows of a flattened proposal batch differently.
        logits = []
        for proposal in proposals.unbind(dim=1):
            action = self.action_projection(proposal.flatten(1).contiguous())
            logits.append(self.score(torch.cat((context, action), dim=1)))
        return torch.cat(logits, dim=1)


def load_scorer(path: Path, recipe: str):
    """Load a self-contained ensemble on CPU. Returns stateless, ordered mean logits."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if recipe not in RECIPES or bundle["format"] != FORMAT or bundle["recipe"] != recipe:
        raise ValueError(f"Expected {FORMAT} frozen weights for {recipe!r}")
    if tuple(bundle["seeds"]) != SEEDS or len(bundle["models"]) != len(SEEDS):
        raise ValueError(f"Expected one frozen reader for each of {SEEDS}")
    dim = bundle["embedding_dim"]
    members = []
    for weights in bundle["models"]:
        reader = ActionReader(dim)
        reader.load_state_dict(weights, strict=True)
        members.append(reader.eval().requires_grad_(False))

    @torch.inference_mode()
    def score(support, state, embedding, proposals):
        practice = None
        if support is not None:
            practice = Practice(*(field.unsqueeze(0) for field in pack_practice(support)))
        query = np.concatenate((numeric(state, (STATE_DIM,), "state"),
                                numeric(embedding, (dim,), "embedding")))
        query = torch.from_numpy(query).unsqueeze(0)
        actions = torch.from_numpy(numeric(proposals, PROPOSAL_SHAPE, "proposals")).unsqueeze(0)
        duplicate = torch.equal(actions[:, 0], actions[:, 1])
        if duplicate:
            actions = actions[:, :1]
        logits = torch.stack([member(practice, query, actions)[0] for member in members]).mean(0)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite ensemble logits")
        return (logits.repeat(2) if duplicate else logits).numpy()

    return score
