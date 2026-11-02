"""Paired AD/DPT views of recorded residual-PPO learning streams.

Each stream follows one collection slot across successive policy versions.
Its episodes may have different scenes; this grouping does not claim repeated
attempts at one scene. Split slots before constructing causal windows.
"""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .vla_history import FIELDS, FrozenHistoryController, validate_trial


def split_learning_stream(trial, *, validation_slots):
    """Partition whole source episodes, retaining every policy version per stream."""
    validate_trial(trial, objective="ad")
    starts = np.flatnonzero(trial["episode_starts"])
    ends = np.r_[starts[1:], len(trial["observations"])]
    groups = {}
    for start, end in zip(starts, ends, strict=True):
        versions = trial["teacher_steps"][start:end]
        if np.any(versions != versions[0]):
            raise ValueError("PPO weights must remain fixed within each complete episode")
        groups.setdefault(int(versions[0]), []).append((int(start), int(end)))
    counts = {len(episodes) for episodes in groups.values()}
    if len(groups) < 2 or len(counts) != 1:
        raise ValueError("Require multiple chronological versions with equal collection slots")
    count = counts.pop()
    heldout = set(validation_slots)
    if (not heldout or len(heldout) >= count
            or any(type(slot) is not int or not 0 <= slot < count for slot in heldout)):
        raise ValueError("Require distinct train and validation collection slots")
    result = {"train": [], "validation": []}
    for slot in range(count):
        ranges = [episodes[slot] for episodes in groups.values()]
        stream = {key: np.concatenate([trial[key][start:end] for start, end in ranges]).copy()
                  for key in FIELDS}
        validate_trial(stream, objective="ad")
        partition = "validation" if slot in heldout else "train"
        result[partition].append((slot, stream))
    return result


@torch.no_grad()
def relabel_teacher(trials, teacher, *, deterministic, generator=None, batch_size=512):
    """Change only query targets using fixed teacher weights and normalization.

Mean labels match deterministic PPO evaluation. Sampled labels require a
private generator; neither convention changes actual historical feedback.
"""
    if batch_size < 1 or not deterministic and generator is None:
        raise ValueError("Use a positive label batch and a private sampling generator")
    reference = next(teacher.parameters())
    teacher.eval()
    result = []
    for trial in trials:
        labels = []
        for start in range(0, len(trial["observations"]), batch_size):
            features = torch.as_tensor(trial["observations"][start:start + batch_size],
                                       device=reference.device, dtype=reference.dtype)
            control, _, _ = teacher.act(features, deterministic=deterministic, generator=generator)
            labels.append(control.cpu().numpy().astype(np.float32))
        labeled = {key: value.copy() for key, value in trial.items()}
        labeled["target_noise"] = np.concatenate(labels)
        validate_trial(labeled, objective="dpt")
        result.append(labeled)
    return result


class HistoryResidualActor(nn.Module):
    """Use the common native residual rollout loop with a frozen history policy.

This evaluation adapter emits pre-tanh u, exactly as the PPO actor does.
Rewards passed to observe are actual installed feedback; evaluation-only
zero-shot episodes have zero nonterminal rewards and reset before each case.
"""

    def __init__(self, policy, *, mode="full", reset_each_episode=True, generator=None):
        super().__init__()
        self.policy = policy.eval().requires_grad_(False)
        self.config = SimpleNamespace(residual_dim=policy.config.noise_dim)
        self.controller = FrozenHistoryController(policy, mode=mode, generator=generator)
        self.reset_each_episode = reset_each_episode

    def begin_episode(self):
        if self.reset_each_episode:
            self.controller.reset_trial()

    @torch.no_grad()
    def act(self, features, *, deterministic=False, generator=None, action_noise=None):
        if len(features) != 1 or action_noise is not None:
            raise ValueError("History evaluation requires one causal lane without PPO noise overrides")
        control = self.controller.act(features[0].detach().cpu().numpy(), deterministic=deterministic)
        value = torch.as_tensor(control[None], device=features.device, dtype=features.dtype)
        unused = torch.zeros(1, device=features.device, dtype=features.dtype)
        return value, unused, unused

    def observe(self, control, reward, done):
        self.controller.observe(control, reward, done)

    def install_terminal_reward(self, reward):
        """Install the actual delayed episode label before a subsequent query."""
        if (self.controller.pending or not self.controller.previous_done
                or not np.isfinite(reward)):
            raise ValueError("Install a finite terminal label only after a completed episode")
        self.controller.previous_reward = float(reward)
