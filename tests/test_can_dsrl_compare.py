"""Replay and scene-isolation contracts; execute on allocated Linux only."""
from copy import deepcopy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import can_dsrl, can_dsrl_compare as compare


class FakeReplay:
    def __init__(self, buffer_size=5, observation_space=None, action_space=None, *, n_envs=4,
                 device="cpu", optimize_memory_usage=False, handle_timeout_termination=True,
                 offline_mix_ratio=-1):
        self.buffer_size, self.n_envs = buffer_size, n_envs
        self.observation_space, self.action_space, self.device = observation_space, action_space, device
        self.optimize_memory_usage = optimize_memory_usage
        self.handle_timeout_termination, self.offline_mix_ratio = handle_timeout_termination, offline_mix_ratio
        self.pos, self.full, self.offline_steps, self.use_noise_actions = 0, False, -1, False
        for name in ("observations", "next_observations", "actions", "noise_actions"):
            setattr(self, name, np.zeros((buffer_size, n_envs, 2), np.float32))
        for name in ("rewards", "dones", "timeouts"):
            setattr(self, name, np.zeros((buffer_size, n_envs), np.float32))

    def add(self, *args):
        self.pos += 1


@pytest.mark.parametrize("full,pos", [(False, 3), (True, 2)])
def test_flatten_replay_preserves_each_transition_once(full, pos):
    source = FakeReplay()
    source.full, source.pos, source.offline_steps, source.use_noise_actions = full, pos, 1, True
    identifiers = np.arange(source.buffer_size * source.n_envs).reshape(source.buffer_size, source.n_envs)
    fields = ("observations", "next_observations", "actions", "noise_actions", "rewards", "dones", "timeouts")
    for offset, name in enumerate(fields):
        array = getattr(source, name)
        values = identifiers + 100 * offset
        array[:] = values[..., None] if array.ndim == 3 else values
    original = deepcopy(source)
    converted = compare.flatten_replay_buffer(source, extra_transitions=12, buffer_class=FakeReplay)
    indices = (np.arange(source.buffer_size) + pos) % source.buffer_size if full else np.arange(pos)
    count = len(indices) * source.n_envs
    assert converted.n_envs == 1 and converted.pos == count and not converted.full
    assert converted.buffer_size > count + 12
    assert converted.offline_steps == 4 and converted.use_noise_actions
    for name in fields:
        array = getattr(source, name)
        expected = array[indices].reshape(count, 1, *array.shape[2:])
        np.testing.assert_array_equal(getattr(converted, name)[:count], expected)
        np.testing.assert_array_equal(array, getattr(original, name))
    assert len(np.unique(converted.rewards[:count])) == count
    converted.observations[0] = -999
    np.testing.assert_array_equal(source.observations, original.observations)


@pytest.mark.parametrize("attribute,value", [("optimize_memory_usage", True), ("offline_mix_ratio", .5)])
def test_flatten_rejects_unsupported_replay(attribute, value):
    source = FakeReplay()
    source.pos = 1
    setattr(source, attribute, value)
    with pytest.raises(ValueError, match="ordinary DSRL replay"):
        compare.flatten_replay_buffer(source, extra_transitions=3, buffer_class=FakeReplay)


class FakeOptimizer:
    def __init__(self):
        self.state = {"parameter": {"step": 0}}

    def state_dict(self):
        return {"state": deepcopy(self.state), "param_groups": [{"lr": .0003}]}


class FakeNetwork:
    def __init__(self):
        self.value = 0.
        self.optimizer = FakeOptimizer()

    def state_dict(self):
        return {"weight": torch.tensor([self.value])}


class FakeAgent:
    def __init__(self):
        for name in ("actor", "critic", "critic_target", "critic_noise"):
            setattr(self, name, FakeNetwork())
        self.ent_coef_optimizer = FakeOptimizer()
        self.log_ent_coef = torch.tensor(0.)
        self._n_updates, self.num_timesteps, self.batch_size = 100, 1000, 8
        self.gradient_steps, self.noise_critic_grad_steps, self.actor_gradient_steps = 20, 10, -1
        self.replay_buffer = FakeReplay(buffer_size=100, n_envs=1)
        self.replay_buffer.pos = 3
        self.actions = []

    def predict_diffused(self, obs, deterministic):
        assert not deterministic
        self.actions.append(self.actor.value)
        return np.full((1, 28), self.actor.value, np.float32), None

    def train(self, gradient_steps, batch_size):
        assert gradient_steps == 20 and batch_size == self.batch_size
        for name, optimizer in compare._optimizers(self).items():
            optimizer.state["parameter"]["step"] += 10 if name == "noise_critic" else gradient_steps
        self._n_updates += gradient_steps
        self.actor.value += 1
        self.critic.value += 1
        self.critic_noise.value += 1


class FakeEnv:
    closed = []

    def __init__(self, *args):
        self.count, self.scene = 0, None

    def reset(self, *, seed):
        self.count, self.scene = 0, seed
        return np.full(23, seed % 11, np.float32), {"scene_seed": seed}

    def step(self, action):
        self.count += can_dsrl.CHUNK
        return (np.full(23, self.count, np.float32), -4., self.count == can_dsrl.HORIZON, False,
                dict(primitive_steps=can_dsrl.CHUNK, success_once=bool(action[0] > 0),
                     release_logged_success=False))

    def close(self):
        self.closed.append(self.scene)


def test_online_trial_counts_real_steps_and_actual_optimizer_updates(monkeypatch):
    monkeypatch.setattr(can_dsrl, "HORIZON", 8)
    agent = FakeAgent()
    row = compare.run_online_trial(agent, FakeEnv(), scene_seed=17, attempts=2,
                                   deadline=time.monotonic() + 60, seed=3)
    assert row["completed"] and row["primitive_steps"] == 16 and row["chunks"] == 4
    assert row["adaptation_train_calls"] == 4 and row["adaptation_learner_updates"] == 80
    assert row["adaptation_optimizer_updates"] == dict(actor=80, critic=80, noise_critic=40, entropy=80)
    assert row["adaptation_optimizer_steps_total"] == 280
    assert row["initial_observation_sha256"][0] == row["initial_observation_sha256"][1]
    assert agent.actions == [0, 1, 2, 3]
    assert row["rng_seed"] == 20


def test_deadline_finishes_started_episode_and_records_next_attempt(monkeypatch):
    monkeypatch.setattr(can_dsrl, "HORIZON", 8)
    agent = FakeAgent()
    # Calls: start, first attempt, next attempt, finish. A deadline that expires
    # during an episode must retain that episode's full two chunks and updates.
    readings = iter([0., 0., 2., 2.])
    row = compare.run_online_trial(agent, FakeEnv(), scene_seed=17, attempts=4,
                                   deadline=1., clock=lambda: next(readings))
    assert not row["completed"] and row["stop_reason"] == "deadline_before_attempt"
    assert row["primitive_steps"] == 8 and row["chunks"] == 2
    assert row["next_attempt_index"] == 1 and row["attempt_records"][0]["adaptation_complete"]
    assert row["adaptation_optimizer_steps_total"] == 140
    assert row["deadline_overrun_seconds"] == 1.


def test_evaluation_reloads_learner_and_replay_for_every_scene(tmp_path, monkeypatch):
    monkeypatch.setattr(can_dsrl, "HORIZON", 8)
    monkeypatch.setattr(can_dsrl, "CanChunkEnv", FakeEnv)
    reference = tmp_path / "reference"
    checkpoint = reference / "checkpoints" / "generation"
    checkpoint.mkdir(parents=True)
    motor = SimpleNamespace(base_policy=FakeNetwork())
    (checkpoint / "recovery.zip").write_bytes(b"immutable fake agent")
    (checkpoint / "replay.pkl").write_bytes(b"immutable fake replay")
    torch.save(dict(contract={"frozen_motor_sha256": can_dsrl.fingerprint(motor.base_policy)},
                    num_timesteps=1000, updates=100), checkpoint / "runtime.pt")
    hashes = {name: can_dsrl.sha256(checkpoint / name) for name in ("recovery.zip", "replay.pkl", "runtime.pt")}
    (reference / "recovery-manifest.json").write_text(json.dumps(dict(directory="checkpoints/generation", files=hashes)))
    loaded = []

    def load_fresh(files, env, shared_motor, extra_transitions):
        assert files["recovery.zip"] == checkpoint / "recovery.zip"
        assert shared_motor is motor and extra_transitions == 4
        agent = FakeAgent()
        loaded.append(agent)
        return agent

    monkeypatch.setattr(compare, "_load_fresh_learner", load_fresh)
    args = SimpleNamespace(dsrl_reference=reference, dsrl_root=tmp_path, normalization=tmp_path,
                           output=tmp_path, attempts=2, eval_trials=2, seed=7)
    before = can_dsrl.rng_state()
    row = compare.evaluate_online(args, motor, time.monotonic() + 60)
    after = can_dsrl.rng_state()
    assert row["completed"] and row["reference_unchanged"] and row["motor_unchanged"]
    assert len(loaded) == 2 and loaded[0] is not loaded[1]
    assert loaded[0].actions == loaded[1].actions == [0, 1, 2, 3]
    scenes = row["scenes"]
    assert [scene["scene_seed"] for scene in scenes] == [2_000_000, 2_000_001]
    assert [scene["starting_replay_transitions"] for scene in scenes] == [3, 3]
    assert scenes[0]["starting_learner_sha256"] == scenes[1]["starting_learner_sha256"]
    assert row["primitive_steps"] == 32 and row["adaptation_optimizer_steps_total"] == 560
    assert compare._hash_structure(before) == compare._hash_structure(after)
