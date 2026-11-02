"""Core causal projection and teacher-label invariants for residual histories."""
import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.residual_history import (
    HistoryResidualActor, relabel_teacher, split_learning_stream,
)
from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy, trial_tokens


def trial():
    n = 24  # Three policy versions, four episodes of two decisions each.
    return dict(observations=np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        executed_noise=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        target_noise=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        rewards=np.tile(np.array([0., 1.], np.float32), n // 2),
        dones=np.tile(np.array([False, True]), n // 2),
        episode_starts=np.tile(np.array([True, False]), n // 2),
        teacher_steps=np.repeat(np.arange(3, dtype=np.int64), 8))


def test_whole_episode_split_keeps_versions_and_covers_each_source_row_once():
    original = trial()
    split = split_learning_stream(original, validation_slots=[1])
    assert [slot for slot, _ in split["train"]] == [0, 2, 3]
    assert [slot for slot, _ in split["validation"]] == [1]
    seen = []
    for streams in split.values():
        for _, stream in streams:
            assert stream["teacher_steps"].tolist() == [0, 0, 1, 1, 2, 2]
            assert stream["episode_starts"].tolist() == [True, False] * 3
            seen.extend(stream["observations"][:, 0].tolist())
    assert sorted(seen) == original["observations"][:, 0].tolist()
    broken = trial()
    broken["teacher_steps"][1] = 1
    with pytest.raises(ValueError):
        split_learning_stream(broken, validation_slots=[1])


def test_teacher_targets_cannot_rewrite_context_or_refit_teacher_buffers():
    original = trial()
    teacher = ResidualActorCritic(ResidualPPOConfig(3, 2, width=8, layers=1))
    with torch.no_grad():
        teacher.actor[-1].bias.fill_(.7)
        teacher.feature_mean.fill_(5.)
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    labeled = relabel_teacher([original], teacher, deterministic=True, batch_size=3)[0]
    assert np.allclose(labeled["target_noise"], .7)
    for field in original:
        if field != "target_noise":
            np.testing.assert_array_equal(original[field], labeled[field])
    np.testing.assert_array_equal(trial_tokens(original), trial_tokens(labeled))
    for key, value in before.items():
        torch.testing.assert_close(teacher.state_dict()[key], value)


def test_online_adapter_resets_cases_and_consumes_only_observed_feedback():
    torch.manual_seed(8)
    policy = HistoryNoisePolicy(HistoryConfig(3, 2, width=8, layers=1, heads=2, context_length=4))
    actor = HistoryResidualActor(policy)
    features = torch.ones(1, 3)
    actor.begin_episode()
    first = actor.act(features, deterministic=True)[0]
    with pytest.raises(RuntimeError):
        actor.act(features, deterministic=True)
    actor.observe(first[0].numpy(), 0., True)
    actor.begin_episode()
    repeated = actor.act(features, deterministic=True)[0]
    torch.testing.assert_close(first, repeated)
