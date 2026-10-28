"""Trial boundaries must preserve feedback without exposing the hidden setting."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from fluffy_bassoon.meta_rl.environments import BernoulliBandit, HiddenGainReach, HiddenRewardBandit


@pytest.mark.parametrize("factory", [HiddenRewardBandit, BernoulliBandit, HiddenGainReach])
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
