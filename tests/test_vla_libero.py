"""Collector semantics; execute on allocated Linux alongside native qualification."""
import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_libero import (expand_noise, public_observation,
                                               run_episode, sac_observation)


def observation():
    return dict(image=np.zeros((256, 256, 3), np.uint8),
                wrist=np.zeros((256, 256, 3), np.uint8), state=np.arange(8, dtype=np.float32))


class Motor:
    horizon = 10
    def __init__(self):
        self.noises = []
    def infer(self, obs, instruction, noise):
        self.noises.append(noise.copy())
        return np.zeros((self.horizon, 7), np.float32)


class Sim:
    metadata = dict(instruction="put the object in the bowl")
    def reset(self, initial_state, scene_seed):
        self.count = 0
        return dict(observation=observation(), initial_observation_sha256="fixed")
    def step(self, actions):
        self.count += 1
        # Success in the middle of the second proposed chunk.
        n, done = (2, True) if self.count == 2 else (len(actions), False)
        return dict(observation=observation(), rewards=[0.] * (n - 1) + [float(done)],
                    executed_actions=actions[:n], primitive_steps=n, done=done, success=done)


def test_noise_tying_preserves_coordinate_and_rejects_invalid():
    z = np.arange(32, dtype=np.float32)
    result = expand_noise(z, 10)
    assert result.shape == (10, 32)
    assert np.array_equal(result, np.repeat(z[None], 10, axis=0))
    result[0, 0] = -100
    assert z[0] == 0
    with pytest.raises(ValueError):
        expand_noise(np.zeros(31), 10)


def test_public_observation_excludes_object_state_and_does_not_mutate_quaternion():
    quat = np.array([0., 0., 0., 1.01])
    raw = dict(agentview_image=np.zeros((256, 256, 3), np.uint8),
               robot0_eye_in_hand_image=np.zeros((256, 256, 3), np.uint8),
               robot0_eef_pos=np.arange(3), robot0_eef_quat=quat,
               robot0_gripper_qpos=np.zeros(2), object=np.ones(100))
    public = public_observation(raw)
    assert set(public) == {"image", "wrist", "state"}
    assert public["state"].shape == (8,)
    assert quat[3] == 1.01


def test_sac_observation_matches_official_camera_state_layout():
    obs = sac_observation(observation(), batched=True)
    assert obs["pixels"].shape == (1, 64, 64, 3, 1)
    assert obs["pixels"].dtype == np.uint8
    assert obs["state"].shape == (1, 8, 1)


def test_collector_logs_only_executed_partial_chunk_and_actual_feedback():
    motor, feedback = Motor(), []
    rows, summary = run_episode(Sim(), motor, lambda _: np.full(32, 2., np.float32),
        initial_state=0, scene_seed=4, chunk=5,
        feedback=lambda *args: feedback.append(args))
    assert len(rows) == 2 and summary["primitive_steps"] == 7 and summary["success"]
    assert rows[-1]["executed_actions"].shape == (2, 7)
    assert rows[-1]["raw_rewards"] == [0., 1.]
    assert [r["reward"] for r in rows] == [-1., 0.]
    assert [f[2] for f in feedback] == [False, True]
    assert np.array_equal(feedback[0][0], np.ones(32))
    assert np.array_equal(motor.noises[0], np.ones((10, 32)))


def test_native_and_tied_gaussian_keep_their_distinct_noise_distributions():
    native, tied = Motor(), Motor()
    run_episode(Sim(), native, lambda _: np.full(32, 3., np.float32),
        initial_state=0, scene_seed=4, native=True, rng=np.random.default_rng(0), bounded=False)
    run_episode(Sim(), tied, lambda _: np.full(32, 3., np.float32),
        initial_state=0, scene_seed=4, bounded=False)
    assert not np.array_equal(native.noises[0][0], native.noises[0][1])
    assert np.array_equal(tied.noises[0], np.full((10, 32), 3.))
