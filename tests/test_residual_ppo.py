"""Bounded action residual and memoryless residual PPO; run on allocated Linux resources."""

import json

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.residual_ppo import (HistoryResidualController, ResidualActorCritic,
    ResidualPPOConfig, collect_episodes, history_trial, stack_records, update)
from fluffy_bassoon.meta_rl.vla_history import (FrozenHistoryController, HistoryConfig,
    HistoryNoisePolicy, validate_trial)
from fluffy_bassoon.rl.action_residual import ResidualBound, action_std_from_norm_stats


ARM = (0, 1, 2, 3, 4, 5)


def bound(fraction=.25):
    return ResidualBound.from_std(np.arange(1, 13, dtype=float), indices=ARM, fraction=fraction, action_dim=12)


def test_zero_control_is_identity_and_only_arm_dims_move():
    rng = np.random.default_rng(0)
    chunk = rng.normal(size=(50, 12))
    b = bound()
    assert np.array_equal(b.apply(chunk, np.zeros(6)), chunk)
    moved = b.apply(chunk, np.full(6, 100.))
    np.testing.assert_allclose(moved[:, :6] - chunk[:, :6], np.broadcast_to(b.scale, (50, 6)), rtol=1e-12)
    assert np.array_equal(moved[:, 6:], chunk[:, 6:])


def test_residual_never_exceeds_bound_and_matches_torch():
    b = bound(.5)
    control = np.random.default_rng(1).normal(scale=50, size=(7, 6))
    delta = b.delta(control)
    assert (np.abs(delta) <= np.asarray(b.scale) + 1e-12).all()
    torch.testing.assert_close(b.delta(torch.as_tensor(control)), torch.as_tensor(delta))
    batched = b.apply(torch.zeros(7, 50, 12, dtype=torch.float64), torch.as_tensor(control))
    torch.testing.assert_close(batched[:, 0, :6], torch.as_tensor(delta))


def test_bound_validation_and_norm_stats(tmp_path):
    with pytest.raises(ValueError):
        ResidualBound(12, (0, 0), (1., 1.))
    with pytest.raises(ValueError):
        ResidualBound.from_std(np.zeros(12), indices=ARM, fraction=.25, action_dim=12)
    with pytest.raises(ValueError):
        bound().apply(np.zeros((50, 12)), np.zeros(5))
    path = tmp_path / "norm_stats.json"
    std = list(np.linspace(.1, 3.2, 32))
    path.write_text(json.dumps({"norm_stats": {"actions": {"mean": [0.] * 32, "std": std}}}))
    b = ResidualBound.from_std(action_std_from_norm_stats(path), indices=ARM, fraction=.25, action_dim=12)
    np.testing.assert_allclose(b.scale, .25 * np.asarray(std[:6]))


class LineMotor:
    """Frozen 'VLA' that always proposes zero motion; features are the position."""

    def __init__(self, environment):
        self.environment = environment

    def infer(self, observation):
        return np.zeros((5, 12)), np.asarray([observation, 1.], np.float32), 0


class LineEnvironment:
    """Reach a target at +1 within four chunks; terminal reward is success."""

    def reset(self):
        self.position, self.steps = 0., 0
        return self.position

    def step_chunk(self, chunk):
        self.position += float(chunk[:, 0].sum())
        self.steps += 1
        done = self.steps == 4
        return self.position, done, dict(success=abs(self.position - 1.) < .25)


def make_policy():
    torch.manual_seed(0)
    return ResidualActorCritic(ResidualPPOConfig(feature_dim=2, residual_dim=6, width=32, minibatch_size=16))


def test_initial_deterministic_policy_reproduces_frozen_motor():
    policy, environment = make_policy(), LineEnvironment()
    records = collect_episodes(policy, bound(), LineMotor(environment), environment,
                               lambda info: float(info["success"]), episodes=2, deterministic=True)
    assert all(np.array_equal(record["control"], np.zeros(6)) for record in records)
    assert [record["episode_start"] for record in records[:4]] == [True, False, False, False]
    assert [record["reward"] for record in records] == [0.] * 8


def test_ppo_learns_to_reach_target_and_exports_history():
    policy, environment = make_policy(), LineEnvironment()
    # Five steps per chunk at scale 0.25 * std(=1) reach about +1 in four chunks with tanh(u) near 0.2.
    b = bound()
    motor, reward = LineMotor(environment), lambda info: float(info["success"])
    policy.fit_feature_normalization(np.asarray([[0., 1.], [1., 1.]], np.float32))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    history, before = [], None
    for iteration in range(60):
        records = collect_episodes(policy, b, motor, environment, reward, episodes=16, generator=generator)
        success = np.mean([record["reward"] for record in records if record["done"]])
        before = success if before is None else before
        history.append((iteration, records))
        update(policy, optimizer, stack_records(records), generator=generator)
    final = collect_episodes(policy, b, motor, environment, reward, episodes=16, deterministic=True)
    assert np.mean([record["reward"] for record in final if record["done"]]) > max(before, .5)
    trial = history_trial(history)
    validate_trial(trial, objective="ad")


def test_history_controller_adapter_applies_bound():
    b = bound()
    policy = HistoryNoisePolicy(HistoryConfig(2, 6, width=8, heads=2, layers=1, context_length=8))
    adapter = HistoryResidualController(FrozenHistoryController(policy), b)
    chunk = np.zeros((5, 12))
    executed, control = adapter.act(chunk, np.asarray([0., 1.], np.float32), deterministic=True)
    np.testing.assert_allclose(executed[0, :6], b.delta(control))
    adapter.observe(control, 1., True)
    with pytest.raises(ValueError):
        HistoryResidualController(FrozenHistoryController(
            HistoryNoisePolicy(HistoryConfig(2, 5, width=8, heads=2, layers=1, context_length=8))), b)
