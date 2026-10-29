"""Accounting/reset tests; run on allocated Linux, not the editing host."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.rl.robotwin_noise import (
    RobotwinNoiseEnv, _reset_with_retry_limit, _setup_task_with_retry_limit,
)


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


class UnstableScene(Exception):
    pass


def retry_fixture(failures):
    attempts = []
    closed = []

    class Task:
        def setup_demo(self, *, seed, now_ep_num, **kwargs):
            assert seed == now_ep_num
            attempts.append((seed, kwargs))
            if len(attempts) <= len(failures):
                raise failures[len(attempts) - 1]

        def get_info(self):
            return {"actual_seed": attempts[-1][0]}

        def close_env(self, **kwargs):
            closed.append(kwargs)

    subenv = SimpleNamespace(
        task=Task(), task_name="handover_mic", env_seed=7, args={"step_lim": 400},
        global_lock=nullcontext(), lock=nullcontext(), reset_count=0,
        clear_cache_freq=8, create_instruction=lambda: "pass the microphone",
        close=lambda: None,
    )
    return subenv, Task, attempts, closed


def test_setup_retry_guard_preserves_seed_order_and_success():
    subenv, task, attempts, _ = retry_fixture([ValueError("bad scene")])
    _setup_task_with_retry_limit(subenv, lambda name: task(), max_attempts=3)
    assert [seed for seed, _ in attempts] == [7, 8]
    assert all(kwargs["is_test"] for _, kwargs in attempts)
    assert subenv.episode_info_list == [{"actual_seed": 8}]
    assert subenv.env_seed == 7


def test_setup_retry_guard_exposes_last_error_and_stops():
    cause = FileNotFoundError("assets/objects/018_microphone/base0.glb")
    subenv, task, attempts, _ = retry_fixture([cause] * 4)
    with pytest.raises(RuntimeError, match="3 seeds.*018_microphone") as error:
        _setup_task_with_retry_limit(subenv, lambda name: task(), max_attempts=3)
    assert len(attempts) == 3 and error.value.__cause__ is cause
    assert not hasattr(subenv, "episode_info_list")


def test_reset_retry_guard_preserves_unstable_seed_selection_and_counters():
    subenv, _, attempts, closed = retry_fixture([UnstableScene("unstable")])
    _reset_with_retry_limit(subenv, UnstableScene, env_seed=11, max_attempts=3)
    assert [seed for seed, _ in attempts] == [11, 12]
    assert subenv.env_seed == 11 and subenv.reset_count == 1
    assert subenv.task.step_lim == 400
    assert subenv.task.run_steps == subenv.task.reward_step == 0
    assert subenv.args["instruction"] == "pass the microphone"
    assert closed == [{"clear_cache": False}, {"clear_cache": True}]


def test_reset_retry_guard_bounds_instability_and_keeps_other_errors_immediate():
    subenv, _, attempts, _ = retry_fixture([UnstableScene("unstable")] * 4)
    with pytest.raises(RuntimeError, match="unstable for 3 seeds"):
        _reset_with_retry_limit(subenv, UnstableScene, max_attempts=3)
    assert len(attempts) == 3
    subenv, _, attempts, _ = retry_fixture([FileNotFoundError("missing object")])
    with pytest.raises(FileNotFoundError, match="missing object"):
        _reset_with_retry_limit(subenv, UnstableScene, max_attempts=3)
    assert len(attempts) == 1
