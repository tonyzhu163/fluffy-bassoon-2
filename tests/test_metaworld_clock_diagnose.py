"""Clock intervention changes only the later episode index, with frozen weights."""

from collections import namedtuple
import pickle

import gymnasium as gym
from gymnasium.spaces.utils import flatten
import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.metaworld_clock_diagnose import CappedEpisodeClock, evaluate_condition, fingerprint
from fluffy_bassoon.meta_rl.metaworld_env import MetaWorldTrial
from fluffy_bassoon.meta_rl.metaworld_policy import make_policy
from fluffy_bassoon.meta_rl.metaworld_train import core_settings


class ToySawyer:
    def set_task(self, task):
        pass

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = np.zeros(39, np.float32)
        self.state[:4] = self.rng.normal(size=4)
        return self.state.copy(), {}

    def step(self, action):
        self.state[:4] += action * .1
        return self.state.copy(), float(self.state[0]), False, False, {"success": self.state[0] > 0}

    def close(self):
        pass


def lanes(attempts):
    task_type = namedtuple("Task", "env_name data")
    task = task_type("push-v3", pickle.dumps(dict(partially_observable=True)))
    envs = [MetaWorldTrial(ToySawyer, [task], attempts=attempts, horizon=3) for _ in range(2)]
    for env in envs:
        env.family = "push-v3"
    return envs


@pytest.mark.parametrize("core", ["gru", "persistent_ctm", "cmm"])
@pytest.mark.parametrize("reset", [False, True])
def test_frozen_prefix_parity_and_noop_within_training_range(core, reset):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    envs = lanes(4)
    # The original CTM runner predates the CMM CLI preset, while its shared
    # constructor already supports CMM. Specify that same preset explicitly.
    settings = (dict(ticks=1, stm_length=32, ltm_tokens=32, synapse_depth=8)
                if core == "cmm" else core_settings(core))
    policy = make_policy(core, dict(settings, width=16), envs[0]).eval()
    original = fingerprint(policy)
    raw = evaluate_condition(policy, envs, capped=False, reset=reset, training_attempts=2, seed_start=11)
    capped = evaluate_condition(policy, envs, capped=True, reset=reset, training_attempts=2, seed_start=11)
    assert raw["trained_prefix_sha256"] == capped["trained_prefix_sha256"]
    assert raw["complete_trajectory_sha256"] != capped["complete_trajectory_sha256"]
    assert fingerprint(policy) == original
    for env in envs:
        env.close()
    envs = lanes(2)
    raw = evaluate_condition(policy, envs, capped=False, reset=reset, training_attempts=2, seed_start=11)
    capped = evaluate_condition(policy, envs, capped=True, reset=reset, training_attempts=2, seed_start=11)
    assert raw["complete_trajectory_sha256"] == capped["complete_trajectory_sha256"]
    for env in envs:
        env.close()


def test_cap_preserves_step_clock_and_other_inputs_without_mutation():
    env = lanes(4)[0]
    policy = make_policy("gru", dict(width=16, ticks=1), env)
    wrapper = CappedEpisodeClock(policy, env.observation_space)
    obs, _ = env.reset(seed=1)
    obs["clock"] = np.array([3., .7], np.float32)
    source = torch.tensor(flatten(env.observation_space, obs))[None]
    before = source.clone()
    actual = wrapper.effective_observations(source)
    expected = source.clone()
    expected[:, wrapper.clock_index] = 1.
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(source, before, atol=0, rtol=0)
    with pytest.raises(ValueError, match="frozen evaluation"):
        wrapper.step(source, deterministic=False)
    with pytest.raises(ValueError, match="Missing public clock"):
        CappedEpisodeClock(policy, gym.spaces.Dict({"state": env.observation_space["state"]}))
    env.close()
