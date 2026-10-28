"""Trial boundaries must preserve feedback without exposing the hidden setting."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from fluffy_bassoon.meta_rl.environments import (BernoulliBandit, DarkRoom, HiddenGainReach,
                                                 HiddenRewardBandit, KeyToDoor,
                                                 NonstationaryBernoulliBandit)


@pytest.mark.parametrize("factory", [HiddenRewardBandit, BernoulliBandit, HiddenGainReach,
                                     NonstationaryBernoulliBandit, DarkRoom, KeyToDoor])
def test_gymnasium_contract(factory):
    check_env(factory(), skip_render_check=True)


def test_hidden_reward_has_fixed_boundaries_and_preserves_target(monkeypatch):
    env = HiddenRewardBandit()
    samples = []
    sample_hidden = env._sample_hidden
    def record_sample():
        sample_hidden()
        samples.append(env._target)
    monkeypatch.setattr(env, "_sample_hidden", record_sample)
    observation, info = env.reset(seed=12)
    target = env._target  # Oracle-only access; never part of policy inputs.
    assert info == {}
    assert observation["previous_reward"].item() == 0
    assert observation["attempt_end"].item() == 0
    for attempt, action in enumerate([1 - target, target, target, target]):
        observation, reward, terminated, truncated, info = env.step(action)
        assert env._target == target
        assert reward == float(action == target)
        assert terminated == (attempt == 3)
        assert not truncated
        assert info["attempt_end"] and info["attempt_index"] == attempt
        assert observation["attempt_end"].item() == 1
        assert observation["previous_action"][action] == 1
        assert observation["previous_reward"].item() == reward
        assert env.observation_space.contains(observation)
    assert samples == [target]
    with pytest.raises(gym.error.ResetNeeded):
        env.step(target)
    env.reset()
    assert len(samples) == 2


def test_attempt_reset_exposes_terminal_state_and_fresh_physical_state():
    env = HiddenGainReach(attempts=2, horizon=1)
    observation, _ = env.reset(seed=42)
    start = observation["state"].copy()
    gain = env._gain
    action = np.array([.25, -.5], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    expected = start.copy()
    expected[:2] += env.step_size * gain * action
    np.testing.assert_allclose(observation["transition_state"], expected)
    assert not np.array_equal(observation["state"], expected)
    assert env._gain == gain
    assert not terminated and not truncated and info["attempt_end"]
    assert observation["clock"].tolist() == [1., 0.]
    assert observation["previous_reward"].item() == reward
    np.testing.assert_array_equal(observation["previous_action"], action)
    observation["state"][:] = 1000
    observation["transition_state"][:] = 1000
    observation, _, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert terminated and env.observation_space.contains(observation)
    np.testing.assert_array_equal(observation["state"], observation["transition_state"])


def test_fixed_attempt_horizon_does_not_terminate_on_early_success():
    env = HiddenGainReach(attempts=1, horizon=8, gains=(1.,))
    observation, _ = env.reset(seed=5)
    for step in range(8):
        state = observation["state"]
        action = np.clip((state[2:] - state[:2]) / env.step_size, -1, 1).astype(np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        assert terminated == (step == 7)
        assert info["attempt_end"] == (step == 7)
        assert not truncated
        assert reward == float(step == 7)


@pytest.mark.parametrize("factory,action", [
    (HiddenRewardBandit, 0),
    (HiddenGainReach, np.array([.3, -.2], dtype=np.float32)),
])
def test_seeded_trials_are_reproducible_and_lanes_are_independent(factory, action):
    left, right, reference = factory(), factory(), factory()
    for env in (left, right, reference):
        env.reset(seed=27)
    right.reset(seed=999)
    for _ in range(left.attempts * left.horizon):
        a, ar, ad, at, ai = left.step(action)
        b, br, bd, bt, bi = reference.step(action)
        assert (ar, ad, at, ai) == (br, bd, bt, bi)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    left.reset(seed=27)
    reference.reset(seed=27)
    a, *_ = left.step(action)
    b, *_ = reference.step(action)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("gain", [-1., 1.])
def test_oracle_can_complete_every_attempt(gain):
    env = HiddenGainReach(gains=(gain,))
    observation, _ = env.reset(seed=71)
    total = 0
    for _ in range(env.attempts * env.horizon):
        state = observation["state"]
        action = np.clip((state[2:] - state[:2]) / (env.step_size * gain), -1, 1).astype(np.float32)
        observation, reward, *_ = env.step(action)
        total += reward
    assert total == env.attempts


@pytest.mark.parametrize("action", [np.array([np.nan, 0], dtype=np.float32),
                                  np.array([2., 0], dtype=np.float32),
                                  np.zeros(3, dtype=np.float32)])
def test_invalid_action_does_not_advance_the_trial(action):
    env = HiddenGainReach()
    observation, _ = env.reset(seed=3)
    with pytest.raises(ValueError, match="Action"):
        env.step(action)
    assert env._elapsed == 0 and env._attempt == 0
    np.testing.assert_array_equal(env._state, observation["state"])


def test_continuous_action_sequence_produces_float32_feedback():
    env = HiddenGainReach()
    env.reset(seed=3)
    observation, *_ = env.step([.25, -.5])
    assert observation["previous_action"].dtype == np.float32
    assert env.observation_space.contains(observation)


@pytest.mark.parametrize("kwargs", [{"attempts": 0}, {"attempts": True}, {"horizon": 1.5},
                                    {"gains": []}, {"gains": [0]}, {"gains": [np.inf]},
                                    {"step_size": 0}, {"tolerance": np.nan}])
def test_invalid_reaching_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        HiddenGainReach(**kwargs)


def test_reset_required_and_no_hidden_setting_override():
    env = HiddenRewardBandit()
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    with pytest.raises(ValueError, match="overrides"):
        env.reset(options={"target": 0})


def test_bernoulli_probabilities_persist_and_are_not_observable():
    env = BernoulliBandit(attempts=6, arms=2)
    observation, _ = env.reset(seed=1)
    env._means[:] = [0., 1.]
    for step in range(6):
        action = step % 2
        observation, reward, terminated, truncated, info = env.step(action)
        assert reward == action
        assert terminated == (step == 5) and not truncated
        np.testing.assert_array_equal(env._means, [0., 1.])
        assert set(observation) == set(env.observation_space.spaces)
        assert set(info) == {"attempt_end", "attempt_index", "is_success"}
    env.reset()
    assert ((env._means > 0) & (env._means < 1)).all()


def test_nonstationary_bandit_records_the_probabilities_behind_each_pull():
    env = NonstationaryBernoulliBandit(attempts=200, arms=3, hazard=.5)
    env.reset(seed=3)
    pulled = []
    for _ in range(200):
        pulled.append(env._means.copy())
        env.step(0)
    np.testing.assert_array_equal(env._mean_history, pulled)
    changes = (np.diff(env._mean_history, axis=0) != 0).any(1).mean()
    assert .35 < changes < .65
    assert not ({"_means", "_mean_history"} & set(env.observation_space.spaces))
    with pytest.raises(ValueError, match="hazard"):
        NonstationaryBernoulliBandit(hazard=0)


def walk(env, target):
    """Oracle moves from the current cell to target; returns the per-step rewards."""
    rewards = []
    while not (env._position == target).all():
        dx, dy = np.sign(target - env._position)
        action = 4 if dx > 0 else 3 if dx < 0 else 1 if dy > 0 else 2
        rewards.append(env.step(action)[1])
    return rewards


def test_dark_room_pays_on_goal_steps_and_keeps_the_goal_across_attempts():
    env = DarkRoom(attempts=2, horizon=20)
    observation, _ = env.reset(seed=5)
    goal = env._goal.copy()
    while (goal == 4).all():
        observation, _ = env.reset()
        goal = env._goal.copy()
    np.testing.assert_array_equal(observation["state"], [0, 0])
    rewards = walk(env, goal)
    assert rewards[:-1] == [0.] * (len(rewards) - 1) and rewards[-1] == 1.
    stay = [env.step(0) for _ in range(20 - len(rewards))]
    assert all(reward == 1. for _, reward, *_ in stay)
    observation, _, terminated, _, info = stay[-1]
    assert info["attempt_end"] and info["is_success"] and not terminated
    np.testing.assert_array_equal(observation["state"], [0, 0])
    np.testing.assert_array_equal(env._goal, goal)
    for step in range(20):
        _, reward, terminated, _, info = env.step(0)
        assert reward == 0.
    assert terminated and not info["is_success"]


def test_dark_room_clips_at_walls_and_never_exposes_the_goal():
    env = DarkRoom(attempts=1, horizon=20, size=5)
    env.reset(seed=0)
    env._goal = np.array([4, 4])
    for _ in range(4):
        observation, *_ = env.step(3)
    np.testing.assert_array_equal(observation["state"], [-1, 0])
    np.testing.assert_array_equal(observation["previous_action"], [0, 0, 0, 1, 0])
    assert set(observation) == set(env.observation_space.spaces)


def test_key_to_door_pays_key_then_door_once_per_attempt():
    env = KeyToDoor(attempts=2, horizon=50)
    env.reset(seed=11)
    key, door = env._key.copy(), env._door.copy()
    assert not (key == door).all()
    env._key, env._door = np.array([0, 0]), np.array([8, 8])
    assert sum(walk(env, np.array([8, 8]))) == 0  # The door pays nothing before the key.
    assert walk(env, np.array([0, 0]))[-1] == 1.
    assert sum(walk(env, np.array([8, 0]))) == 0
    rewards = walk(env, np.array([8, 8]))
    assert rewards[-1] == 1. and sum(rewards) == 1.
    used = 8 + 16 + 8 + 8
    outcomes = [env.step(0) for _ in range(50 - used)]
    assert all(reward == 0. for _, reward, *_ in outcomes)
    assert outcomes[-1][-1]["is_success"]
    env.step(0)
    assert not env._has_key and not env._opened
