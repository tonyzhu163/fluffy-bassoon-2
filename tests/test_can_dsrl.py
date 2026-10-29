"""Meaningful contract checks, run on allocated Linux with the Can runtime."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.can_dsrl import (
    ACTION_DIM, CHUNK, HORIZON, NOISE_DIM, OBS_DIM, CanChunkEnv,
    collect_trial, fingerprint, rng_state, restore_rng, trial_inputs,
)
from fluffy_bassoon.meta_rl.can_dsrl_run import make_policy, save_ppo
from fluffy_bassoon.meta_rl.ppo import generalized_advantages, update_ppo


class FakeLow:
    def __init__(self):
        self.env = self
        self.count, self.scene = 0, None

    def reset(self, options):
        self.scene, self.count = options["seed"], 0
        return {"state": np.full(OBS_DIM, self.scene % 11, dtype=np.float32)}

    def step(self, action):
        self.count += 1
        # Success exists for a single primitive step within a chunk, testing
        # that the adapter doesn't infer success from shifted chunk reward.
        return {"state": np.full(OBS_DIM, self.count, np.float32)}, float(self.count == 2), False, {}

    def get_state(self):
        return dict(count=self.count, scene=self.scene)

    def reset_to(self, state):
        self.count, self.scene = state["count"], state["scene"]


def motor(observation, noise):
    assert noise.shape == (len(observation), CHUNK, ACTION_DIM)
    return noise.cpu().numpy()


def test_unchanged_chunk_reward_and_true_success():
    env = CanChunkEnv(low_env=FakeLow())
    obs, info = env.reset(seed=194)
    assert info == {"scene_seed": 194}
    assert np.all(obs == 194 % 11)
    _, reward, terminated, truncated, info = env.step(np.zeros(NOISE_DIM))
    assert reward == -3 and info["raw_reward"] == 1 and info["success_once"]
    assert info["primitive_steps"] == 4 and not terminated and not truncated
    for _ in range(HORIZON // CHUNK - 1):
        _, _, terminated, _, info = env.step(np.zeros(NOISE_DIM))
    assert terminated and info["success_once"]
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(np.zeros(NOISE_DIM))


def test_snapshot_restores_simulator_and_sampler():
    env = CanChunkEnv(low_env=FakeLow(), seed=37)
    env.reset()
    checkpoint = deepcopy(env.snapshot())
    expected = env.step(np.zeros(NOISE_DIM))
    next_reset = env.reset()[1]
    env.restore(checkpoint)
    actual = env.step(np.zeros(NOISE_DIM))
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    assert env.reset()[1] == next_reset
    env.step(np.zeros(NOISE_DIM))
    with pytest.raises(RuntimeError, match="boundary"):
        env.snapshot()


def test_trial_matched_dsrl_reset_sampling():
    env = CanChunkEnv(low_env=FakeLow(), seed=37, retries=4)
    seeds = [env.reset()[1]["scene_seed"] for _ in range(8)]
    assert len(set(seeds[:4])) == len(set(seeds[4:])) == 1
    assert seeds[0] != seeds[4]


def test_no_seed_or_privileged_state_in_policy_inputs():
    x = trial_inputs(np.zeros((2, OBS_DIM)), np.zeros((2, NOISE_DIM)), np.array([-4., 0.]), 1., 2, 0, 4)
    assert x.shape == (2, OBS_DIM + NOISE_DIM + 4)
    np.testing.assert_array_equal(x[:, OBS_DIM + NOISE_DIM], [-4, 0])
    np.testing.assert_allclose(x[:, -2], 2 / 3)


def test_full_trial_returns_cross_episode_boundary():
    rewards = torch.tensor([[0.], [0.], [0.], [1.]])
    ends = torch.tensor([[False], [False], [False], [True]])
    advantages, returns = generalized_advantages(rewards, torch.zeros_like(rewards), ends)
    torch.testing.assert_close(advantages, torch.ones_like(rewards))
    torch.testing.assert_close(returns, torch.ones_like(rewards))


@pytest.mark.parametrize("reset_memory", [False, True])
def test_trial_boundaries_replay_and_real_update(tmp_path, reset_memory):
    torch.manual_seed(7)
    envs = [CanChunkEnv(low_env=FakeLow()) for _ in range(2)]
    policy = make_policy(width=16, device="cpu")
    rollout, row = collect_trial(policy, motor, envs, [17, 18], attempts=2, reset_memory=reset_memory)
    assert row["primitive_steps"] == 1200
    assert row["initial_observation_sha256"][0] == row["initial_observation_sha256"][1]
    assert int(rollout["trial_ends"].sum()) == 2
    assert int(rollout["attempt_ends"].sum()) == 4
    logprob, _, _ = policy.evaluate_sequence(rollout["observations"], rollout["raw_actions"], rollout["resets"])
    torch.testing.assert_close(logprob, rollout["log_probs"], rtol=1e-6, atol=1e-6)
    boundary = HORIZON // CHUNK
    if reset_memory:
        assert rollout["resets"][boundary].all()
        assert not rollout["observations"][boundary, :, OBS_DIM:OBS_DIM + NOISE_DIM + 1].any()
    else:
        assert not rollout["resets"].any()
        assert rollout["observations"][boundary, :, OBS_DIM:OBS_DIM + NOISE_DIM].any()
    before = fingerprint(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    metrics = update_ppo(policy, optimizer, rollout, epochs=1, minibatch_trials=2)
    assert metrics["gradient_norm"] > 0 and fingerprint(policy) != before
    path = tmp_path / "recovery.pt"
    sampler = np.random.default_rng(77)
    save_ppo(path, policy, optimizer, sampler, 1, 1200, {"test": True})
    saved = torch.load(path, weights_only=False)
    restored = make_policy(width=16, device="cpu")
    restored.load_state_dict(saved["policy"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(saved["optimizer"])
    assert fingerprint(restored) == fingerprint(policy)
    assert len(restored_optimizer.state) == len(optimizer.state) > 0
    sampler2 = np.random.default_rng()
    sampler2.bit_generator.state = saved["sampler"]
    np.testing.assert_array_equal(sampler.integers(1000, size=8), sampler2.integers(1000, size=8))


def test_rng_recovery_and_reward_blind():
    torch.manual_seed(13)
    policy = make_policy(width=16, device="cpu")
    state = rng_state()
    roll, _ = collect_trial(policy, motor, [CanChunkEnv(low_env=FakeLow())], [9], attempts=2, reward_blind=True)
    restore_rng(state)
    repeated, _ = collect_trial(policy, motor, [CanChunkEnv(low_env=FakeLow())], [9], attempts=2, reward_blind=True)
    torch.testing.assert_close(roll["raw_actions"], repeated["raw_actions"], rtol=0, atol=0)
    assert not roll["observations"][..., OBS_DIM + NOISE_DIM].any()
