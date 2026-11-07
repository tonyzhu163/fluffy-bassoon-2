"""Public, causal tokens for frozen-motor decoded-action residual policies.

The current frozen proposal is public. Feedback belongs to the previous
physical decision, including across episode resets in a full-history trial.
This module does not accept teacher versions, current labels or hidden state.
"""
from dataclasses import dataclass

import numpy as np


MODES = ("full", "episode", "static")


def _float32(value, shape, name):
    value = np.asarray(value)
    if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Require finite float32 {name} with shape {shape}")
    return value


@dataclass(frozen=True)
class ChunkHistoryLayout:
    feature_dim: int
    chunk_length: int
    action_dim: int

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in
               (self.feature_dim, self.chunk_length, self.action_dim)):
            raise ValueError("Require positive feature/chunk/action dimensions")

    @property
    def command_shape(self):
        return self.chunk_length, self.action_dim

    @property
    def current_dim(self):
        return self.feature_dim + self.chunk_length * self.action_dim

    @property
    def feedback_dim(self):
        # Full issued command, delivered mask, reward, terminated, truncated, seconds.
        return self.chunk_length * self.action_dim + self.chunk_length + 4

    @property
    def token_dim(self):
        return self.current_dim + self.feedback_dim + 1

    def residual(self, issued, base):
        """Keep the entire command tail: it can affect native TOPP execution."""
        issued = _float32(issued, self.command_shape, "issued commands")
        base = _float32(base, self.command_shape, "frozen proposal")
        result = issued - base
        if not np.isfinite(result).all():
            raise ValueError("Decoded residual overflow")
        return result


class ChunkHistoryEncoder:
    """The same token state machine for offline fitting and online evaluation.

    query returns (token, reset_attention). Call observe after the real physical
    decision, before the next query. A new trial clears both the encoder and,
    via reset_attention, the model memory. Physical episode boundaries must be
    supplied explicitly and agree with the preceding terminal feedback.
    """

    def __init__(self, layout, *, mode="full"):
        if not isinstance(layout, ChunkHistoryLayout) or mode not in MODES:
            raise ValueError("Require a chunk layout and declared history mode")
        self.layout, self.mode = layout, mode
        self.pending = False
        self.reset_trial()

    def reset_trial(self):
        if self.pending:
            raise ValueError("Observe the issued physical decision before resetting a trial")
        self.previous = np.zeros(self.layout.feedback_dim, dtype=np.float32)
        self.first = True
        self.previous_terminal = False

    def _check_query(self, episode_start):
        if self.pending:
            raise ValueError("A query already awaits physical feedback")
        if type(episode_start) not in (bool, np.bool_):
            raise ValueError("Require an explicit boolean physical episode start")
        if bool(episode_start) != (self.first or self.previous_terminal):
            raise ValueError("Physical episode start disagrees with terminal feedback")

    def query_parts(self, *, episode_start):
        """Return owned shifted feedback/reset for a device-resident prefix.

        Tensor callers compose their current public features/base on device;
        the physical feedback and reset state machine stays shared with query.
        """
        self._check_query(episode_start)
        clear = self.first or self.mode == "static" or self.mode == "episode" and episode_start
        feedback = np.zeros_like(self.previous) if clear else self.previous.copy()
        self.pending = True
        return feedback, bool(clear)

    def query(self, features, base, *, episode_start):
        self._check_query(episode_start)
        features = _float32(features, (self.layout.feature_dim,), "public features")
        base = _float32(base, self.layout.command_shape, "frozen proposal")
        feedback, clear = self.query_parts(episode_start=episode_start)
        token = np.concatenate((features, base.reshape(-1), feedback,
                                np.array([episode_start], dtype=np.float32)))
        return token, bool(clear)

    def observe(self, issued, delivered_mask, reward, *, terminated, truncated, physics_seconds):
        if not self.pending:
            raise ValueError("Physical feedback requires a preceding query")
        issued = _float32(issued, self.layout.command_shape, "issued commands")
        mask = np.asarray(delivered_mask)
        if (mask.dtype != np.bool_ or mask.shape != (self.layout.chunk_length,)
                or not mask.any() or np.any(mask[1:] & ~mask[:-1])):
            raise ValueError("Require the measured nonempty delivered target prefix")
        if any(type(v) not in (bool, np.bool_) for v in (terminated, truncated)):
            raise ValueError("Keep separate boolean termination and truncation")
        scalars = np.array([reward, terminated, truncated, physics_seconds], dtype=np.float32)
        if not np.isfinite(scalars).all() or physics_seconds <= 0:
            raise ValueError("Require finite real feedback and positive physical duration; audit no-ops separately")
        # Concatenation owns the snapshot, including the undelivered command tail.
        self.previous = np.concatenate((issued.reshape(-1), mask.astype(np.float32), scalars))
        self.previous_terminal = bool(terminated or truncated)
        self.pending, self.first = False, False


def trial_tokens(layout, *, features, bases, issued, delivered_masks, rewards,
                 terminated, truncated, physics_seconds, episode_starts, mode="full"):
    """Encode complete positive-duration physical episodes without current labels.

    For a sampled window, call this on that window: its first physical episode
    starts a new trial. Never slice precomputed full-trial tokens and retain
    feedback from outside the admitted context.
    """
    features = np.asarray(features)
    if features.ndim != 2 or len(features) < 1:
        raise ValueError("Require a nonempty sequence of public pre-action features")
    length = len(features)
    _float32(features, (length, layout.feature_dim), "public feature sequence")
    for value, name in ((bases, "frozen proposals"), (issued, "issued commands")):
        _float32(value, (length, *layout.command_shape), name)
    arrays = {name: np.asarray(value) for name, value in (
        ("delivered_masks", delivered_masks), ("rewards", rewards),
        ("terminated", terminated), ("truncated", truncated),
        ("physics_seconds", physics_seconds), ("episode_starts", episode_starts))}
    for name, value in arrays.items():
        shape = (length, layout.chunk_length) if name == "delivered_masks" else (length,)
        expected = np.float32 if name in ("rewards", "physics_seconds") else np.bool_
        if value.shape != shape or value.dtype != expected:
            raise ValueError(f"Invalid {name} sequence shape or dtype")
    if not bool(arrays["terminated"][-1] or arrays["truncated"][-1]):
        raise ValueError("Require complete physical episodes")
    encoder, tokens, resets = ChunkHistoryEncoder(layout, mode=mode), [], []
    for t in range(length):
        token, reset = encoder.query(features[t], bases[t], episode_start=arrays["episode_starts"][t])
        encoder.observe(issued[t], arrays["delivered_masks"][t], arrays["rewards"][t],
                        terminated=arrays["terminated"][t], truncated=arrays["truncated"][t],
                        physics_seconds=arrays["physics_seconds"][t])
        tokens.append(token)
        resets.append(reset)
    return np.stack(tokens), np.asarray(resets, dtype=np.bool_)
