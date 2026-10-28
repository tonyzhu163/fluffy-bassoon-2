"""Boundary, goal-visibility, likelihood and complete-recovery qualification."""

from collections import namedtuple
import os
import pickle

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.checkpoint import restore_checkpoint, save_checkpoint
from fluffy_bassoon.meta_rl.metaworld_env import MetaWorldTrial, benchmark_pools, make_lanes
from fluffy_bassoon.meta_rl.metaworld_policy import make_policy
from fluffy_bassoon.meta_rl.metaworld_train import core_settings, replay_error
from fluffy_bassoon.meta_rl.ppo import collect_trials, update_ppo


Task = namedtuple("Task", "env_name data")


class FakeSawyer:
    def set_task(self, task):
        self.task = pickle.loads(task.data)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def observation(self):
        state = np.zeros(39)
        state[0] = self.rng.uniform()
        if not self.task["partially_observable"]:
            state[-3:] = self.task["goal"]
        return state

    def reset(self):
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        self.steps += 1
        return self.observation(), float(self.steps), False, False, {"success": self.steps == 1}

    def close(self):
        pass


def fake_env(**kwargs):
    tasks = [Task("push-v3", pickle.dumps(dict(partially_observable=True, goal=[x, 2, 3])))
             for x in (1, 2, 3)]
    return MetaWorldTrial(FakeSawyer, tasks, attempts=2, horizon=3, **kwargs)


def test_success_is_latched_and_terminal_feedback_survives_physical_reset():
    env = fake_env()
    obs, _ = env.reset(seed=4)
    selected = env.task
    for step in range(6):
        obs, reward, done, truncated, info = env.step(np.zeros(4, np.float32))
        assert done == (step == 5)
        assert not truncated
        assert info["attempt_end"] == (step % 3 == 2)
        assert reward == pytest.approx((step % 3 + 1) * .1)
        assert obs["previous_reward"].item() == pytest.approx(reward)
        assert env.observation_space.contains(obs)
        assert env.task == selected
        assert np.all(obs["state"][-3:] == 0)
    assert env.outcomes == [True, True]
    assert env.native_returns == [6., 6.]


@pytest.mark.parametrize("visible", [False, True])
def test_goal_visibility_and_seeded_lanes(visible):
    left, right = fake_env(goal_visible=visible), fake_env(goal_visible=visible)
    for env in (left, right):
        env.reset(seed=91)
    for _ in range(6):
        a, *_ = left.step(np.zeros(4, np.float32))
        b, *_ = right.step(np.zeros(4, np.float32))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        assert bool(np.any(a["state"][-3:])) == visible


@pytest.mark.parametrize("core", ["gru", "persistent_ctm", "transformer_xl"])
def test_replay_and_exact_next_update_after_checkpoint(tmp_path, core):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    envs = [fake_env(), fake_env()]
    config = dict(core_settings(core), width=16)
    policy = make_policy(core, config, envs[0])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    sampler = np.random.default_rng(4)
    rollout = collect_trials(policy, envs, [3, 4])
    assert max(replay_error(policy, rollout).values()) < 2e-5
    update_ppo(policy, optimizer, rollout, epochs=1, minibatch_trials=2)
    policy.core.observe_after_update(rollout["observations"])
    save_checkpoint(tmp_path / "latest.pt", policy=policy, optimizer=optimizer,
                    environment_rng=sampler, iteration=1, counters={"updates": 1}, contract={"test": core})
    next_seeds = sampler.integers(0, 1000, 2)
    next_rollout = collect_trials(policy, envs, next_seeds)
    update_ppo(policy, optimizer, next_rollout, epochs=1, minibatch_trials=1)
    expected = {key: value.clone() for key, value in policy.state_dict().items()}
    restore_checkpoint(tmp_path / "latest.pt", policy=policy, optimizer=optimizer,
                       environment_rng=sampler, contract={"test": core})
    np.testing.assert_array_equal(sampler.integers(0, 1000, 2), next_seeds)
    restored = collect_trials(policy, envs, next_seeds)
    for key in restored:
        torch.testing.assert_close(restored[key], next_rollout[key], rtol=0, atol=0)
    update_ppo(policy, optimizer, restored, epochs=1, minibatch_trials=1)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FLUFFY_METAWORLD_REAL_QUALIFICATION") != "1",
                    reason="Real simulator qualification belongs in an allocated Linux job")
@pytest.mark.parametrize("benchmark", ["ml1-push", "ml10"])
def test_real_simulator_trial_and_disjoint_pools(benchmark):
    pools = benchmark_pools(benchmark)
    envs = make_lanes(pools["train"], 10, benchmark=benchmark, attempts=2, horizon=500)
    try:
        env = envs[0]
        first, _ = env.reset(seed=321)
        for _ in range(1000):
            obs, reward, done, truncated, _ = env.step(np.zeros(4, np.float32))
            assert not truncated and np.isfinite(reward)
        assert done and len(env.outcomes) == 2
        replay, _ = env.reset(seed=321)
        for key in first:
            np.testing.assert_array_equal(first[key], replay[key])
    finally:
        for env in envs:
            env.close()
