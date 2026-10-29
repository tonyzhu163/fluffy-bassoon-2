"""Capture-side-effect tests; execute on allocated Linux, never the editor host.

These fixtures supply native action-count outcomes independently of the capture
adapter. They test whether the adapter forwards those outcomes and preserves
scene/light updates while avoiding intermediate camera reads. They do not
establish real ray-tracer image equivalence.
"""
from contextlib import nullcontext
import random
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.rl.robotwin_noise import (
    RobotwinNoiseEnv,
    _SynchronousExecutor,
    _install_final_observation_capture,
    _step_with_final_observation,
)


class FakeNativeTask:
    """A physical trace with externally specified native completion metadata."""

    def __init__(self, captures=3, count=3, *, reuse_last=True,
                 failed_control=False, failed_camera=False, missing_count=False,
                 success=True, advance_after_capture=False):
        self.captures = captures
        self.native_count = count
        self.reuse_last = reuse_last
        self.failed_control = failed_control
        self.failed_camera = failed_camera
        self.missing_count = missing_count
        self.success = success
        self.advance_after_capture = advance_after_capture
        self.rng = random.Random(319)
        self.lighting = None
        self.physics_steps = self.render_updates = self.camera_reads = 0
        self.collect_modes = []

    def _update_render(self):
        self.render_updates += 1
        self.lighting = self.rng.random()

    def get_obs(self):
        self._update_render()
        if self.failed_camera:
            raise RuntimeError("camera read failed")
        self.camera_reads += 1
        self.now_obs = dict(position=self.physics_steps, lighting=self.lighting)
        return dict(self.now_obs)

    def get_instruction(self):
        return "put the object on the stand"

    def gen_sparse_reward_data(self, actions, *, collect_obs=False):
        self.collect_modes.append(collect_obs)
        assert collect_obs  # The sparse native mode has no exact count.
        rows = []
        for _ in range(self.captures):
            self.physics_steps += 2
            self._update_render()  # The native control loop syncs every step.
            rows.append(self.get_obs())
        if self.failed_control:
            raise RuntimeError("control loop failed")
        if ((not self.reuse_last or not rows) and self.native_count != 0
                or self.advance_after_capture):
            self.physics_steps += 1  # Success can occur inside the next action.
            self._update_render()
        if len(rows) < len(actions):
            final = rows[-1] if self.reuse_last and rows else self.get_obs()
            rows.extend([final] * (len(actions) - len(rows)))
        if not self.success:
            # Native normal completion syncs rendering once after the final
            # collected image. A renderer's temporal state need not be pixels.
            self._update_render()
        info = {"success": self.success, "native_marker": object()}
        if not self.missing_count:
            info["executed_action_count"] = self.native_count
        self.last_info = info
        return np.array([float(self.success)]), np.array([self.success]), np.array([False]), info, rows


def fake_subenv(task):
    return SimpleNamespace(task=task, lock=nullcontext(),
                           get_instruction=lambda: "already reset",
                           reset=lambda **kwargs: pytest.fail("Unexpected reset"))


@pytest.mark.parametrize("captures,count,reuse_last", [
    (3, 3, True),  # Success exactly at a completed action boundary.
    (2, 3, False),  # Native count includes the partly executed next action.
    (0, 0, False),  # Already terminal: no action is charged.
    (8, 8, True),  # Every issued action completed.
])
def test_deferred_capture_preserves_native_counts_and_render_side_effects(
        captures, count, reuse_last):
    actions = np.zeros((8, 14), dtype=np.float32)
    reference = FakeNativeTask(captures, count, reuse_last=reuse_last)
    expected_reward, expected_end, expected_trunc, expected_info, full_rows = (
        reference.gen_sparse_reward_data(actions, collect_obs=True)
    )
    task = FakeNativeTask(captures, count, reuse_last=reuse_last)
    result = _step_with_final_observation(fake_subenv(task), actions, dict)
    assert result["info"] is task.last_info
    assert result["info"]["executed_action_count"] == count
    assert result["info"]["success"] == expected_info["success"]
    np.testing.assert_array_equal(result["reward"], expected_reward)
    np.testing.assert_array_equal(result["terminated"], expected_end)
    np.testing.assert_array_equal(result["truncated"], expected_trunc)
    assert task.physics_steps == reference.physics_steps
    assert task.render_updates == reference.render_updates
    assert task.rng.getstate() == reference.rng.getstate()
    assert result["obs"]["position"] == full_rows[-1]["position"]
    assert result["obs"]["lighting"] == full_rows[-1]["lighting"]
    assert result["obs"]["instruction"] == task.get_instruction()
    assert task.camera_reads == 1
    assert reference.camera_reads == max(1, captures + int(not reuse_last))
    assert task.collect_modes == [True]
    assert "get_obs" not in vars(task) and "_update_render" not in vars(task)


@pytest.mark.parametrize("failure", ["control", "camera"])
def test_native_exceptions_restore_both_methods(failure):
    task = FakeNativeTask(failed_control=failure == "control",
                          failed_camera=failure == "camera")
    with pytest.raises(RuntimeError, match=("control loop" if failure == "control" else "camera read")):
        _step_with_final_observation(fake_subenv(task), np.zeros((8, 14)), dict)
    assert "get_obs" not in vars(task) and "_update_render" not in vars(task)
    assert task.get_obs.__func__ is FakeNativeTask.get_obs
    assert task._update_render.__func__ is FakeNativeTask._update_render


def test_normal_completion_preserves_temporal_side_effects_without_claiming_pixels():
    actions = np.zeros((8, 14))
    reference = FakeNativeTask(captures=8, count=8, success=False)
    _, _, _, _, reference_rows = reference.gen_sparse_reward_data(actions, collect_obs=True)
    task = FakeNativeTask(captures=8, count=8, success=False)
    result = _step_with_final_observation(fake_subenv(task), actions, dict)
    assert result["info"]["executed_action_count"] == 8
    assert not bool(result["terminated"][0])
    assert task.physics_steps == reference.physics_steps
    assert task.render_updates == reference.render_updates
    assert task.rng.getstate() == reference.rng.getstate()
    assert task.camera_reads == 1 and reference.camera_reads == 8
    assert result["obs"]["position"] == reference_rows[-1]["position"]
    # This deliberate renderer-state fixture demonstrates why the real
    # allocated pixel/feature comparison remains mandatory.
    assert result["obs"]["lighting"] != reference_rows[-1]["lighting"]


def test_hypothetical_stale_native_frame_is_not_claimed_equivalent():
    # The pinned loop cannot take this branch: reuse_last is true only when
    # a capture occurred after the CURRENT physics step. Explicitly model a
    # future incompatible native change so qualification must detect it.
    actions = np.zeros((8, 14))
    reference = FakeNativeTask(captures=2, count=2, reuse_last=True,
                               advance_after_capture=True)
    _, _, _, _, stale_rows = reference.gen_sparse_reward_data(actions, collect_obs=True)
    task = FakeNativeTask(captures=2, count=2, reuse_last=True,
                          advance_after_capture=True)
    result = _step_with_final_observation(fake_subenv(task), actions, dict)
    assert result["info"]["executed_action_count"] == 2
    assert task.physics_steps == reference.physics_steps
    assert result["obs"]["position"] == task.physics_steps
    assert result["obs"]["position"] != stale_rows[-1]["position"]


def test_already_terminal_zero_count_does_not_execute_physics():
    task = FakeNativeTask(captures=0, count=0, reuse_last=False)
    result = _step_with_final_observation(fake_subenv(task), np.zeros((8, 14)), dict)
    assert result["info"]["executed_action_count"] == 0
    assert bool(result["terminated"][0])
    assert task.physics_steps == 0 and task.camera_reads == 1


def test_existing_instance_overrides_are_restored_by_identity():
    task = FakeNativeTask()
    get_obs = task.get_obs
    update_render = task._update_render
    task.get_obs = get_obs
    task._update_render = update_render
    _step_with_final_observation(fake_subenv(task), np.zeros((8, 14)), dict)
    assert task.get_obs is get_obs and task._update_render is update_render


@pytest.mark.parametrize("count", [None, -1, 9, True, 2.5])
def test_no_fallback_to_issued_length_or_placeholder_length(count):
    task = FakeNativeTask(count=count, missing_count=count is None)
    with pytest.raises(RuntimeError, match="exact executed_action_count"):
        _step_with_final_observation(fake_subenv(task), np.zeros((8, 14)), dict)
    assert task.camera_reads == 0
    assert "get_obs" not in vars(task) and "_update_render" not in vars(task)


def test_capture_installation_is_per_instance_and_requires_synchronous_lane():
    first = fake_subenv(FakeNativeTask())
    second = fake_subenv(FakeNativeTask())
    native_step = lambda actions: "native"
    first.step = native_step
    second.step = native_step
    venv = SimpleNamespace(n_envs=1, env_thread_pool=_SynchronousExecutor(), envs=[first])
    _install_final_observation_capture(venv, update_obs=dict)
    wrapped = first.step
    _install_final_observation_capture(venv, update_obs=dict)
    assert first.step is wrapped and second.step is native_step
    assert first.step(np.zeros((8, 14)))["info"]["executed_action_count"] == 3
    with pytest.raises(RuntimeError, match="one synchronous"):
        _install_final_observation_capture(SimpleNamespace(n_envs=2), update_obs=dict)
    with pytest.raises(RuntimeError, match="one synchronous"):
        _install_final_observation_capture(
            SimpleNamespace(n_envs=1, env_thread_pool=object()), update_obs=dict)


def test_fast_capture_rejects_randomized_configuration_before_backend_loading():
    with pytest.raises(ValueError, match="demo_clean only"):
        RobotwinNoiseEnv(task_config_name="demo_randomized", final_observation_only=True)
    with pytest.raises(ValueError, match="must be a bool"):
        RobotwinNoiseEnv(final_observation_only=1)
    task = FakeNativeTask()
    task.crazy_random_light = True
    with pytest.raises(RuntimeError, match="randomized lights"):
        _step_with_final_observation(fake_subenv(task), np.zeros((8, 14)), dict)
    assert task.camera_reads == task.physics_steps == 0
