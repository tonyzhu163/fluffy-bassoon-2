"""CPU checks of the released-motor Meta-World noise screen (fake env; Meta-World where installed)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import noise_screen as ns
from fluffy_bassoon.rl import metaworld_pi05 as mw


def test_goals_are_separable_and_scenes_keep_distance():
    for name in mw.TASKS:
        _, goals = mw.task_spec(name)
        positions = mw.scenes(name, 40, 3)
        assert positions.shape == (40, 3)
        np.testing.assert_array_equal(positions, mw.scenes(name, 40, 3))
        assert np.linalg.norm(goals[None, :, :2] - positions[:, None, :2], axis=-1).min() >= mw.MIN_OBJECT_GOAL_XY
    with pytest.raises(ValueError):
        mw.task_spec("door-open-v3")


def test_arm_noise_is_seeded_distinct_and_shaped():
    noise = mw.arm_noise([5, 6, 7])
    assert noise.shape == (3, *mw.NOISE_SHAPE) and noise.dtype == np.float32
    np.testing.assert_array_equal(noise[1], mw.arm_noise([6])[0])
    assert not np.array_equal(noise[0], noise[1])
    with pytest.raises(ValueError):
        mw.arm_noise([1.5])


class FakeEnv:
    """Succeeds once the summed x action passes the goal's x; the goal never enters obs."""

    def __init__(self):
        self.steps = 0

    def reset(self):
        self._target_pos = self._last_rand_vec[3:].copy()
        self.position, self.steps = 0.0, 0
        return np.zeros(39), {}

    def step(self, action):
        self.position += float(action[0])
        self.steps += 1
        success = self.position >= self._target_pos[0] + 1
        return np.zeros(39), 0.0, False, self.steps >= 500, dict(success=float(success))

    def render(self):
        return np.full((4, 4, 3), int(self.position) % 255, dtype=np.uint8)


def test_rollout_settles_executes_chunks_and_hashes_actions():
    env, calls = FakeEnv(), []

    def act(batch):
        calls.append(batch)
        assert batch["prompt"] == mw.TASKS["reach-v3"]["prompt"] and batch["observation/state"].shape == (4,)
        return np.full((5, 32), .1, dtype=np.float32)

    goal = np.array([.1, .8, .15])
    success, digest, issued = mw.rollout(env, act, "reach-v3", goal, np.array([0., .6, .02]))
    assert success and issued.shape[1:] == (mw.CHUNK, mw.ACTION_DIM)
    assert env.steps == mw.SETTLE + 11  # 15 zero settle steps, then 11 steps of 0.1 reach 1.1
    assert len(calls) == 3
    again = mw.rollout(FakeEnv(), act, "reach-v3", goal, np.array([0., .6, .02]))
    assert again[1] == digest
    failed, _, issued = mw.rollout(FakeEnv(), lambda b: np.zeros((5, 32)), "reach-v3", goal, np.zeros(3), steps=20)
    assert not failed and len(issued) == 4
    with pytest.raises(ValueError):
        mw.rollout(FakeEnv(), lambda b: np.zeros((3, 32)), "reach-v3", goal, np.zeros(3))


def test_reset_rejects_a_misplaced_goal():
    env = FakeEnv()
    env.reset = lambda: (setattr(env, "_target_pos", np.zeros(3)), (np.zeros(39), {}))[1]
    with pytest.raises(RuntimeError):
        mw.reset(env, "reach-v3", np.array([.1, .8, .15]), np.zeros(3))


def rows_for(success):
    """success[goal][arm] -> rows over 4 scenes and 2 repeats, arm index -1 for native."""
    rows = []
    for g, per_arm in enumerate(success):
        for arm, value in per_arm.items():
            for scene in range(4):
                for repeat in range(2):
                    rows.append(dict(row=len(rows), variant=g, scene_index=scene, arm=arm, repeat=repeat,
                                     success_once=bool(value), issued_actions_sha256=f"{g}{arm}{scene}",
                                     slot=0))
    return rows


def test_goal_coverage_needs_every_goal_and_more_than_one_preferred():
    multimodal = [{-1: 0, 0: 1, 1: 0, 2: 0}, {-1: 1, 0: 0, 1: 1, 2: 0}, {-1: 0, 0: 0, 1: 0, 2: 1}]
    result = mw.goal_coverage(rows_for(multimodal))
    assert result["passed"] and result["goals_preferred_by_some_arm"] == 3
    collapsed = [{-1: 1, 0: 1, 1: 1, 2: 1}, {-1: 0, 0: 0, 1: 0, 2: 0}, {-1: 0, 0: 0, 1: 0, 2: 0}]
    assert not mw.goal_coverage(rows_for(collapsed))["passed"]


def test_cache_analysis_runs_the_noise_gates_on_goal_conditions(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "mw_cache", Path(__file__).resolve().parents[1]/"scripts/diagnostics/metaworld_pi05_noise_cache.py")
    cache = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache)
    rows = rows_for([{-1: 0, 0: 1, 1: 0, 2: 0}, {-1: 0, 0: 0, 1: 1, 2: 0},
                     {-1: 0, 0: 0, 1: 0, 2: 1}, {-1: 0, 0: 1, 1: 0, 2: 0}])
    (tmp_path/"trajectories").mkdir()
    (tmp_path/"outcomes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    issued = np.stack([np.full((1, 5, 4), r["arm"], dtype=np.float32) for r in rows])
    np.savez_compressed(tmp_path/"trajectories/first-chunks.npz", row=np.arange(len(rows)), issued=issued)
    result = cache.analyse(tmp_path, [0, 1, 2, 3])
    assert result["summary"]["g1_headroom"]["mean"] == pytest.approx(1.0)
    assert result["goal_coverage"]["passed"] and result["determinism"]["deterministic"]
    assert json.loads((tmp_path/"gates.json").read_text())["all_passed"] == result["all_passed"]
    assert cache.episode_seed(7, 0, True, 1, 0) != cache.episode_seed(7, 0, False, 1, 0)


@pytest.mark.skipif(importlib.util.find_spec("metaworld") is None, reason="Meta-World not installed")
def test_real_env_places_goals_and_hides_the_marker():
    position = mw.scenes("reach-v3", 1, 0)[0]
    hidden = mw.make_env("reach-v3", goal_visible=False)
    _, goals = mw.task_spec("reach-v3")
    obs = mw.reset(hidden, "reach-v3", goals[2], position)
    assert np.allclose(hidden._target_pos, goals[2]) and not np.any(obs[-3:])
    assert mw.goal_visibility(hidden, "reach-v3", position) == 0
    visible = mw.make_env("reach-v3", goal_visible=True)
    assert mw.goal_visibility(visible, "reach-v3", position) > 0
    batch = mw.policy_input(hidden, obs, "Reach a goal position")
    assert batch["observation/image"].shape == (mw.RESOLUTION, mw.RESOLUTION, 3)
