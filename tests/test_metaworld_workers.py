"""Serial/subprocess trajectory parity; run only in an allocated Linux step."""

from collections import namedtuple
import os
import pickle
import sys
import time

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.metaworld_env import benchmark_pools, make_lanes, task_digest
from fluffy_bassoon.meta_rl.metaworld_workers import (
    MetaWorldWorkerError, batch_reset, batch_step, close_lanes, make_worker_lanes,
)


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Simulator qualification requires allocated Linux")
Task = namedtuple("Task", "env_name data")


class FakeSawyer:
    def set_task(self, task):
        self.task = pickle.loads(task.data)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def observation(self):
        state = np.zeros(39, dtype=np.float32)
        state[0] = self.rng.uniform()
        state[1:5] = self.last_action
        state[-3:] = self.task["goal"]
        return state

    def reset(self):
        self.steps, self.last_action = 0, np.zeros(4, dtype=np.float32)
        return self.observation(), {}

    def step(self, action):
        self.steps += 1
        self.last_action = action.copy()
        return self.observation(), float(self.steps) + float(action[0]), False, False, {"success": self.steps == 2}

    def close(self):
        pass


class BrokenSawyer(FakeSawyer):
    def step(self, action):
        raise ValueError("intentional worker simulator failure")


class SlowSawyer(FakeSawyer):
    def step(self, action):
        time.sleep(.25)
        return super().step(action)


def fake_pool(factory=FakeSawyer):
    classes = {f"family-{i}": factory for i in range(10)}
    tasks = [Task(name, pickle.dumps(dict(partially_observable=False, goal=[i, x, 3])))
             for i, name in enumerate(classes) for x in (1, 2, 3)]
    return classes, tasks


def assert_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def compare_lanes(serial, parallel):
    for left, right in zip(serial, parallel, strict=True):
        assert left.observation_space == right.observation_space
        assert left.action_space == right.action_space
        assert left.family == right.family
        assert (left.attempts, left.horizon) == (right.attempts, right.horizon)
        assert task_digest(left.tasks) == task_digest(right.tasks)
        assert task_digest([left.task]) == task_digest([right.task])
        assert_equal(left.outcomes, right.outcomes)
        assert_equal(left.native_returns, right.native_returns)
        assert left.finished == right.finished


@pytest.mark.parametrize("workers", [1, 2])
def test_persistent_worker_groups_match_serial_boundaries_and_next_reset(workers):
    pool = fake_pool()
    serial = make_lanes(pool, 10, benchmark="ml10", attempts=2, horizon=3)
    parallel = make_worker_lanes(pool, 10, benchmark="ml10", attempts=2, horizon=3,
                                workers=workers, allocated_cpus=2)
    owner = parallel[0]._worker_batch
    pids = owner.manifest["worker_pids"]
    rng = np.random.default_rng(3)
    actions = rng.uniform(-1, 1, (6, 10, 4)).astype(np.float32)
    try:
        for seeds in (np.arange(10) + 11, np.arange(10) + 500):
            assert_equal(batch_reset(serial, seeds), batch_reset(parallel, seeds))
            compare_lanes(serial, parallel)
            for step_actions in actions:
                left, right = batch_step(serial, step_actions), batch_step(parallel, step_actions)
                assert_equal(left, right)
                for index, result in enumerate(right):
                    np.testing.assert_array_equal(result[0]["previous_action"], step_actions[index])
                compare_lanes(serial, parallel)
            assert owner.manifest["worker_pids"] == pids
            assert all(process.is_alive() for process in owner._processes)
        assert [env.family for env in parallel] == list(pool[0])
        assert owner.manifest["start_method"] == "spawn"
        assert owner.manifest["workers"] <= owner.manifest["allocated_cpus"]
    finally:
        close_lanes(serial)
        # Preserve compatibility with the runner's existing cleanup loop.
        for env in parallel:
            env.close()
    assert owner.closed and not any(process.is_alive() for process in owner._processes)


def test_subsets_and_reordered_lanes_preserve_per_lane_state():
    lanes = make_worker_lanes(fake_pool(), 10, benchmark="ml10", attempts=2, horizon=3,
                              workers=2, allocated_cpus=2)
    try:
        order = [7, 2, 9, 0]
        chosen = [lanes[index] for index in order]
        first = batch_reset(chosen, [12, 13, 14, 15])
        repeated = batch_reset(chosen[::-1], [15, 14, 13, 12])
        assert_equal(first, repeated[::-1])
        action = np.zeros((4, 4), dtype=np.float32)
        results = batch_step(chosen, action)
        assert all(row[0]["clock"][1] == 1 for row in results)
        # Single-lane proxy operations stay available, but callers must use
        # batch_step for actual group parallelism.
        lane_result = chosen[0].step(action[0])
        assert lane_result[0]["clock"][1] == 2
    finally:
        close_lanes(lanes)


def test_worker_error_closes_partial_batch_and_surfaces_original_cause():
    lanes = make_worker_lanes(fake_pool(BrokenSawyer), 10, benchmark="ml10", attempts=2, horizon=3,
                              workers=2, allocated_cpus=2)
    owner = lanes[0]._worker_batch
    batch_reset(lanes, np.arange(10))
    with pytest.raises(MetaWorldWorkerError, match="intentional worker simulator failure"):
        batch_step(lanes, np.zeros((10, 4), dtype=np.float32))
    assert owner.closed and not any(process.is_alive() for process in owner._processes)
    with pytest.raises(MetaWorldWorkerError, match="closed"):
        batch_reset(lanes, np.arange(10))
    close_lanes(lanes)


def test_worker_bound_uses_scheduler_cpu_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    with pytest.raises(ValueError, match="worker count"):
        make_worker_lanes(fake_pool(), 10, benchmark="ml10", workers=3, allocated_cpus=8)


def test_request_timeout_closes_owned_children():
    lanes = make_worker_lanes(fake_pool(SlowSawyer), 10, benchmark="ml10", attempts=2, horizon=3,
                              workers=2, allocated_cpus=2)
    owner = lanes[0]._worker_batch
    try:
        batch_reset(lanes, np.arange(10))
        owner.timeout_seconds = .02
        with pytest.raises(MetaWorldWorkerError, match="timed out"):
            batch_step(lanes, np.zeros((10, 4), dtype=np.float32))
        assert owner.closed and not any(process.is_alive() for process in owner._processes)
    finally:
        close_lanes(lanes)


def test_batch_helpers_reject_mixed_owners_and_duplicate_lanes():
    lanes = make_worker_lanes(fake_pool(), 10, benchmark="ml10", attempts=2, horizon=3,
                              workers=1, allocated_cpus=2)
    serial = make_lanes(fake_pool(), 10, benchmark="ml10", attempts=2, horizon=3)
    try:
        with pytest.raises(ValueError, match="mix worker"):
            batch_reset([lanes[0], serial[0]], [1, 2])
        with pytest.raises(ValueError, match="distinct lanes"):
            batch_reset([lanes[0], lanes[0]], [1, 2])
        with pytest.raises(ValueError, match="one executed action"):
            batch_step(lanes, np.zeros((1, 4), dtype=np.float32))
    finally:
        close_lanes(lanes)
        close_lanes(serial)


@pytest.mark.skipif(os.environ.get("FLUFFY_METAWORLD_REAL_QUALIFICATION") != "1",
                    reason="Real K2 H500 ML10 parity is an opt-in allocated Linux qualification")
def test_real_ml10_k2_h500_full_trajectory_and_next_reset_parity():
    pool = benchmark_pools("ml10", 0)["train"]
    serial = make_lanes(pool, 10, benchmark="ml10", attempts=2, horizon=500)
    parallel = make_worker_lanes(pool, 10, benchmark="ml10", attempts=2, horizon=500,
                                workers=2, allocated_cpus=2)
    rng = np.random.default_rng(61)
    actions = rng.uniform(-1, 1, (1000, 10, 4)).astype(np.float32)
    seeds = np.arange(10) + 1000
    try:
        assert_equal(batch_reset(serial, seeds), batch_reset(parallel, seeds))
        compare_lanes(serial, parallel)
        for step_actions in actions:
            left, right = batch_step(serial, step_actions), batch_step(parallel, step_actions)
            assert_equal(left, right)
            compare_lanes(serial, parallel)
            for index, result in enumerate(right):
                np.testing.assert_array_equal(result[0]["previous_action"], step_actions[index])
        assert all(len(env.outcomes) == 2 for env in parallel)
        assert_equal(batch_reset(serial, seeds + 1_000_000), batch_reset(parallel, seeds + 1_000_000))
        compare_lanes(serial, parallel)
        assert_equal(batch_step(serial, actions[0]), batch_step(parallel, actions[0]))
        compare_lanes(serial, parallel)
    finally:
        close_lanes(serial)
        close_lanes(parallel)
