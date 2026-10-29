"""Accounting/reset tests; run on allocated Linux, not the editing host."""
import numpy as np
import pytest

from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv


class FakeBackend:
    observation_dim = 3
    noise_shape = (4, 2)

    def __init__(self):
        self.calls = []
        self.early_success = False

    def reset(self, task_name, seed):
        self.calls.append((task_name, seed))
        return np.zeros(3), dict(requested_seed=seed)

    def step(self, noise, action_count):
        self.calls.append((noise.copy(), action_count))
        count = min(2, action_count) if self.early_success else action_count
        return np.ones(3), float(self.early_success), self.early_success, False, dict(
            primitive_steps=count, issued_primitive_steps=action_count, success=self.early_success)

    def close(self):
        self.calls.append("closed")


def test_native_horizon_final_partial_chunk_and_reset():
    backend = FakeBackend()
    env = RobotwinNoiseEnv(episode_limit=10, backend=backend)
    np.testing.assert_array_equal(env.reset("handover_mic", 5), np.zeros(3))
    for count in (4, 4, 2):
        _, _, _, truncated, info = env.step(np.zeros(8))
        assert info["primitive_steps"] == count
    assert truncated and env.elapsed_steps == 10
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(8))
    env.reset("scan_object", 6)
    assert env.elapsed_steps == 0 and env.last_reset_info["requested_seed"] == 6


def test_early_success_counts_executed_not_issued():
    backend = FakeBackend()
    backend.early_success = True
    env = RobotwinNoiseEnv(backend=backend)
    env.reset("handover_mic", 1)
    _, reward, terminated, truncated, info = env.step(np.zeros(8))
    assert reward == 1 and terminated and not truncated
    assert info["primitive_steps"] == 2 and info["issued_primitive_steps"] == 4
    assert env.elapsed_steps == 2


def test_exact_external_budget_cap_and_unbounded_native_reference():
    backend = FakeBackend()
    env = RobotwinNoiseEnv(backend=backend)
    env.reset("handover_mic", 1)
    _, _, terminated, truncated, info = env.step(np.full(8, 2.5), max_primitive_steps=3)
    assert not terminated and truncated and info["budget_truncated"]
    assert info["primitive_steps"] == 3
    np.testing.assert_array_equal(backend.calls[-1][0], np.full((4, 2), 2.5))


def test_no_fallback_to_issued_count():
    backend = FakeBackend()
    backend.step = lambda noise, count: (np.zeros(3), 0., False, False, {})
    env = RobotwinNoiseEnv(backend=backend)
    env.reset("handover_mic", 1)
    with pytest.raises(RuntimeError, match="exact valid"):
        env.step(np.zeros(8))


def test_natural_episode_end_takes_precedence_over_equal_budget_cap():
    env = RobotwinNoiseEnv(episode_limit=4, backend=FakeBackend())
    env.reset("handover_mic", 1)
    _, _, terminated, truncated, info = env.step(np.zeros(8), max_primitive_steps=4)
    assert not terminated and truncated
    assert info["primitive_steps"] == 4 and not info["budget_truncated"]


@pytest.mark.parametrize("noise", [np.zeros(7), np.full(8, np.nan)])
def test_invalid_noise(noise):
    env = RobotwinNoiseEnv(backend=FakeBackend())
    env.reset("handover_mic", 1)
    with pytest.raises(ValueError, match="finite noise"):
        env.step(noise)


def test_explicit_inventory_and_seed_validation():
    env = RobotwinNoiseEnv(backend=FakeBackend())
    with pytest.raises(ValueError, match="inventory"):
        env.reset("invented_task", 1)
    with pytest.raises(ValueError, match="nonnegative"):
        env.reset("handover_mic", -1)
