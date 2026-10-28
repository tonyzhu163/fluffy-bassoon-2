"""Hidden-task, early termination, parallel replay and simulator qualification."""

import copy
import json
import os

import gymnasium as gym
import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.metta_env import (MemoryTrial, distant_pairs, memory_pools, task)
from fluffy_bassoon.meta_rl.metaworld_env import make_lanes
from fluffy_bassoon.meta_rl.metaworld_policy import make_policy
from fluffy_bassoon.meta_rl.metaworld_train import core_settings, replay_error
from fluffy_bassoon.meta_rl.ppo import collect_trials, update_ppo
from fluffy_bassoon.meta_rl.checkpoint import restore_checkpoint, save_checkpoint


class FakePoint:
    action_space = gym.spaces.Box(-1., 1., (2,), np.float32)
    observation_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
        "desired_goal": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)})

    def reset(self, *, seed, options):
        self.steps = 0
        self.goal = np.random.default_rng(seed).uniform(size=2)
        return self.observation(), {}

    def observation(self):
        return dict(observation=np.array([0., 0., self.steps, 0.], np.float32),
                    desired_goal=self.goal.copy())

    def step(self, action):
        self.steps += 1
        return self.observation(), 1., True, False, {"success": True}

    def close(self):
        pass


class FakeCheetah:
    action_space = gym.spaces.Box(-1., 1., (6,), np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (17,), np.float32)

    def reset(self, *, seed):
        self.steps = 0
        return np.zeros(17, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(17, np.float32), 999., False, False, {"x_velocity": .75}

    def close(self):
        pass


def fake_point():
    return MemoryTrial(FakePoint, [task("pointmaze", start=[1, 1], goal=[5, 5], reset_seed=42)],
                       benchmark="pointmaze", attempts=2, horizon=3)


def test_task_splits_are_disjoint_deterministic_and_path_constrained():
    assert distant_pairs() and all(distance >= 10 for _, _, distance in distant_pairs())
    for name in ("pointmaze", "halfcheetah-vel"):
        pools = memory_pools(name, 123)
        assert pools == memory_pools(name, 123)
        assert len(pools["train"][1]) == 100
        assert len(pools["test"][1]) == 24
        values = [t.data for _, tasks in pools.values() for t in tasks]
        assert len(values) == len(set(values))
        if name == "pointmaze":
            def pairs(split):
                return {(tuple(json.loads(t.data)["start"]), tuple(json.loads(t.data)["goal"]))
                        for t in pools[split][1]}
            assert not pairs("train").intersection(pairs("test"))
            assert pairs("train_eval") <= pairs("train")


def test_goal_is_hidden_and_early_success_resets_immediately_then_only_pads_suffix():
    env = fake_point()
    obs, _ = env.reset(seed=1)
    assert set(obs) == {"state", "previous_action", "previous_reward", "attempt_end", "clock"}
    assert obs["state"].shape == (4,)
    goal = env.env.goal.copy()
    obs, reward, done, _, info = env.step(np.zeros(2, np.float32))
    assert reward == 1. and not done and info["attempt_end"]
    assert obs["previous_reward"] == 1.
    np.testing.assert_array_equal(env.env.goal, goal)
    _, _, done, _, info = env.step(np.zeros(2, np.float32))
    assert done and info["simulator_step"] and env.outcomes == [True, True]
    for _ in range(4):
        _, reward, done, _, info = env.step(np.zeros(2, np.float32))
        assert done and reward == 0 and not info["simulator_step"] and not info["attempt_end"]
    assert [x["steps"] for x in env.episode_metrics] == [1, 1]


def test_hidden_velocity_reward_and_fixed_target_across_episodes():
    env = MemoryTrial(FakeCheetah, [task("halfcheetah-vel", velocity=1.25)],
                      benchmark="halfcheetah-vel", attempts=2, horizon=3)
    obs, _ = env.reset(seed=3)
    assert obs["state"].shape == (17,)
    for i in range(6):
        obs, reward, done, _, info = env.step(np.ones(6, np.float32))
        assert reward == pytest.approx(-.5 - .05 * 6)
        assert done == (i == 5) and info["attempt_end"] == (i % 3 == 2)
        assert env.setting["velocity"] == 1.25
    assert env.native_returns == pytest.approx([-2.4, -2.4])
    assert env.episode_metrics[-1]["absolute_velocity_error"] == .5


@pytest.mark.parametrize("core", ["gru", "persistent_ctm", "cmm", "transformer_xl"])
@pytest.mark.parametrize("chunk", [0, 3])
def test_variable_trial_replay_and_padding_count(core, chunk):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    envs = [fake_point(), fake_point()]
    policy = make_policy(core, dict(core_settings(core), width=16), envs[0])
    rollout = collect_trials(policy, envs, [1, 2], replay_chunk_length=chunk)
    assert rollout["action_mask"].sum() == 4
    assert rollout["trial_ends"][1:].all()
    assert max(replay_error(policy, rollout, chunk).values()) < 2e-5
    update_ppo(policy, torch.optim.Adam(policy.parameters()), rollout, epochs=1,
               minibatch_trials=2, replay_chunk_length=chunk)
    policy.core.observe_after_update(rollout["observations"], mask=rollout["action_mask"])
    assert policy.core.state_count.item() == pytest.approx(4.0001)


def test_padding_has_no_effect_on_ppo_parameter_update():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    envs = [fake_point(), fake_point()]
    policy = make_policy("gru", dict(width=16, ticks=1), envs[0])
    other = copy.deepcopy(policy)
    rollout = collect_trials(policy, envs, [1, 2])
    altered = {key: value.clone() for key, value in rollout.items()}
    padding = ~rollout["action_mask"]
    for key in ("observations", "raw_actions", "log_probs", "values"):
        altered[key][padding] = 42.
    for model, data in ((policy, rollout), (other, altered)):
        torch.manual_seed(77)
        update_ppo(model, torch.optim.Adam(model.parameters()), data, epochs=1, minibatch_trials=2)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("core", ["gru", "cmm", "persistent_ctm"])
def test_complete_variable_trial_checkpoint_recovery(tmp_path, core):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    envs = [fake_point(), fake_point()]
    policy = make_policy(core, dict(core_settings(core), width=16), envs[0])
    optimizer = torch.optim.Adam(policy.parameters())
    sampler = np.random.default_rng(88)
    contract = {"variable_trial_recovery": core}
    save_checkpoint(tmp_path / "latest.pt", policy=policy, optimizer=optimizer,
                    environment_rng=sampler, iteration=0, counters={}, contract=contract)
    expected = None
    for repeat in range(2):
        if repeat:
            restore_checkpoint(tmp_path / "latest.pt", policy=policy, optimizer=optimizer,
                               environment_rng=sampler, contract=contract)
        rollout = collect_trials(policy, envs, sampler.integers(0, 1000, 2), replay_chunk_length=3)
        update_ppo(policy, optimizer, rollout, epochs=1, minibatch_trials=1, replay_chunk_length=3)
        policy.core.observe_after_update(rollout["observations"], mask=rollout["action_mask"])
        actual = {name: value.clone() for name, value in policy.state_dict().items()}
        if repeat:
            for name, value in actual.items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        else:
            expected = actual


@pytest.mark.skipif(os.environ.get("FLUFFY_METTA_REAL_QUALIFICATION") != "1",
                    reason="Real simulator episodes require allocated Linux resources")
@pytest.mark.parametrize("benchmark", ["pointmaze", "halfcheetah-vel"])
def test_real_simulator_and_serial_parallel_trajectory_parity(benchmark):
    from fluffy_bassoon.meta_rl.metaworld_workers import make_worker_lanes, batch_reset, batch_step
    pools = memory_pools(benchmark, 0)
    serial = make_lanes(pools["train"], 2, benchmark=benchmark, attempts=2, horizon=20)
    parallel = make_worker_lanes(pools["train"], 2, benchmark=benchmark, attempts=2, horizon=20,
                                 workers=2, allocated_cpus=4)
    try:
        left, right = batch_reset(serial, [9, 10]), batch_reset(parallel, [9, 10])
        for _ in range(40):
            for a, b in zip(left, right, strict=True):
                for key in a[0]:
                    np.testing.assert_array_equal(a[0][key], b[0][key])
                assert a[1:] == b[1:]
            commands = [np.zeros(env.action_space.shape, np.float32) for env in serial]
            left, right = batch_step(serial, commands), batch_step(parallel, commands)
        assert all(row[2] for row in left)
        assert [e.outcomes for e in serial] == [e.outcomes for e in parallel]
        assert [e.native_returns for e in serial] == [e.native_returns for e in parallel]
        assert [e.episode_metrics for e in serial] == [e.episode_metrics for e in parallel]
    finally:
        for env in serial + parallel:
            env.close()
