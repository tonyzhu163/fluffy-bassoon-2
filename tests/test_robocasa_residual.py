"""Allocated-Linux numerical/core tests. Native physics is qualified separately."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.robocasa_residual import (
    FrozenRoboCasaMotor, RoboCasaResidualEnvironment, collect_pending,
    install_terminal_labels, restore_checkpoint, save_checkpoint,
)
from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig, stack_records, update
from fluffy_bassoon.rl.action_residual import ResidualBound


def bound():
    return ResidualBound(12, tuple(range(6)), (.25,) * 6)


class MotorModel(torch.nn.Module):
    action_horizon, num_steps, action_env_dim = 50, 10, 12
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = 0
        self.prefix_calls = 0

    def build_prefix_cache(self, observation):
        self.prefix_calls += 1
        # A NaN in padding must not contaminate the pooled valid features.
        return torch.tensor([[[1., 3.], [3., 1.], [float("nan"), float("nan")]]]), torch.tensor([[True, True, False]]), ()

    def predict_action_batch(self, observation, mode, rng):
        self.calls += 1
        assert mode == "eval" and not self.training and not torch.is_grad_enabled()
        assert "success" not in observation and "task_metadata" not in observation
        self.build_prefix_cache(observation)
        return torch.randn((1, 50, 12), generator=rng), {}


def motor():
    stats = {key: dict(mean=list(np.arange(32) / 10), std=[2.] * 32) for key in ("state", "actions")}
    return FrozenRoboCasaMotor(MotorModel(), stats, seed=7)


def test_native_draw_replay_features_and_zero_residual_identity():
    m = motor()
    observation = dict(states=np.arange(25, dtype=np.float32)[None], success=True, task_metadata="private")
    chunk, features, seed = m.infer(observation, noise_seed=123)
    reference, _, _ = m.infer(observation, noise_seed=seed)
    assert np.array_equal(bound().apply(chunk, np.zeros(6, np.float32)), reference)
    assert m.model.calls == m.model.prefix_calls == 2
    assert all(not parameter.requires_grad for parameter in m.model.parameters())
    state = np.r_[np.arange(11, 25), [7, 8]]
    np.testing.assert_allclose(features[:16], (state - np.arange(16) / 10) / (2 + 1e-6), atol=1e-6)
    np.testing.assert_allclose(features[16:28], (chunk[0] - np.arange(12) / 10) / (2 + 1e-6), atol=1e-6)
    np.testing.assert_allclose(features[28:40], (chunk.mean(0) - np.arange(12) / 10) / (2 + 1e-6), atol=1e-6)
    np.testing.assert_allclose(features[40:], [1., 1.], atol=1e-6)


def test_motor_restores_hook_on_native_prediction_failure():
    m = motor()
    original = m.model.build_prefix_cache
    m.model.predict_action_batch = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("native failure"))
    with pytest.raises(RuntimeError, match="native failure"):
        m.infer(dict(states=np.zeros((1, 25), np.float32)))
    assert m.model.build_prefix_cache == original


def test_zero_residual_native_qualification_executes_one_episode_not_two(tmp_path):
    m = motor()
    policy = ResidualActorCritic(ResidualPPOConfig(42, 6, width=8))
    env = environment(tmp_path)
    rows, episodes = collect_pending(policy, bound(), m, env, episodes=1,
        generator=torch.Generator().manual_seed(1), deterministic=True, verify_zero=True)
    assert len(episodes) == 1 and len(rows) == 15
    assert all(row["zero_residual_verified"] for row in rows)
    assert env.native.steps == 750 and m.model.calls == 30


class Journal:
    def begin(self, maximum, phase):
        return maximum

    def finish(self, intent, actual):
        assert actual <= intent


class Writer:
    def __init__(self, path, **kwargs):
        self.path, self.count = path, 0

    def append_data(self, frame):
        self.count += 1

    def close(self):
        from pathlib import Path
        Path(self.path).write_bytes(b"fixture-video")


class Native:
    auto_reset, ignore_terminations = False, True
    task_descriptions = ["open the drawer"]
    current_info_list = [dict(ep_meta={})]

    def __init__(self, early=False):
        self.early = early
        self.count = dict(controls=3, reset_controls=3, physics_ticks=30, control_time=0., clock=dict(control_freq=20.))
        self.env = SimpleNamespace(get_env_attr=lambda key: [deepcopy(self.count)])

    def observation(self):
        return dict(main_images=np.zeros((1, 4, 5, 3), np.uint8), states=np.zeros((1, 25), np.float32),
                    task_descriptions=self.task_descriptions, success=True, task_metadata="private")

    def reset(self):
        self.steps = 0
        return self.observation(), {}

    def step(self, actions, auto_reset):
        assert actions.shape == (1, 12) and auto_reset is False
        self.steps += 1
        self.count["controls"] += 1
        self.count["physics_ticks"] += 10
        self.count["control_time"] += .05
        return self.observation(), np.ones(1), np.array([self.early]), np.array([self.steps == 750]), dict(episode=dict(success_at_end=np.array([True])))

    def close(self):
        pass


def environment(tmp_path, early=False):
    return RoboCasaResidualEnvironment(lambda case: Native(early), [dict(native_seed=1, task="OpenDrawer")],
        horizon=750, output=tmp_path / "videos", monitoring_root=tmp_path / "monitor", journal=Journal(), writer_factory=Writer)


def test_full_native_horizon_and_video_are_truth_isolated(tmp_path):
    env = environment(tmp_path)
    observation = env.reset()
    assert "success" not in observation and "task_metadata" not in observation
    for index in range(15):
        observation, done, info = env.step_chunk(np.zeros((50, 12), np.float32))
        assert done == (index == 14)
        if not done:
            assert info == {}
    assert "simulator_final_success" not in info and "success" not in info
    assert info["frame_clock"]["frame_count"] == 751 and env.writer.count == 751
    import json
    assert json.loads((tmp_path / "monitor/episode-000000.json").read_text())["simulator_final_success"] is True
    with pytest.raises(ValueError):
        env.step_chunk(np.zeros((50, 12)))
    assert env.state_dict()["cursor"] == 1


def test_success_mask_or_short_episode_cannot_be_scored(tmp_path):
    env = environment(tmp_path, early=True)
    env.reset()
    with pytest.raises(ValueError, match="masks"):
        env.step_chunk(np.zeros((50, 12)))
    with pytest.raises(ValueError, match="boundaries"):
        env.state_dict()
    assert not list((tmp_path / "monitor").glob("*.json"))
    env.close()


class TinyEnvironment:
    def __init__(self):
        self.cursor = 0

    def reset(self):
        self.steps = 0
        self.cursor += 1
        return 0.

    def step_chunk(self, chunk):
        self.steps += 1
        return 0., self.steps == 2, dict(id=str(self.cursor)) if self.steps == 2 else {}

    def state_dict(self):
        return dict(cursor=self.cursor)

    def load_state_dict(self, state):
        self.cursor = state["cursor"]


class TinyMotor:
    def __init__(self):
        self.rng = np.random.default_rng(3)

    def infer(self, observation):
        seed = int(self.rng.integers(2**31))
        return np.full((50, 12), .1, np.float32), np.array([seed / 2**31, 1.], np.float32), seed

    def state_dict(self):
        return deepcopy(self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.rng.bit_generator.state = deepcopy(state)


def learner():
    torch.manual_seed(8)
    p = ResidualActorCritic(ResidualPPOConfig(2, 6, width=8, epochs=1, minibatch_size=4))
    return p, torch.optim.Adam(p.parameters(), lr=p.config.learning_rate), TinyMotor(), TinyEnvironment(), {
        "action": torch.Generator().manual_seed(4), "update": torch.Generator().manual_seed(5)}


def test_pending_batch_has_no_labels_and_install_is_atomic():
    p, _, m, e, g = learner()
    records, episodes = collect_pending(p, bound(), m, e, episodes=2, generator=g["action"])
    assert all(row["reward"] == 0 for row in records)
    assert all("base_chunk" in row and "executed_chunk" in row for row in records)
    with pytest.raises(ValueError):
        install_terminal_labels(records, episodes, {"1": .25}, arm="F")
    assert all(row["reward"] == 0 for row in records)
    labeled = install_terminal_labels(records, episodes, {"1": .25, "2": .75}, arm="F")
    assert [row["reward"] for row in labeled] == [0., .25, 0., .75]
    with pytest.raises(ValueError):
        install_terminal_labels(records, episodes, {"1": .25, "2": .75}, arm="S")


def test_pending_recovery_reproduces_next_optimizer_update_and_final_checkpoint(tmp_path):
    p, opt, m, e, g = learner()
    records, episodes = collect_pending(p, bound(), m, e, episodes=2, generator=g["action"])
    contract = dict(motor="immutable-motor-reference", norm="immutable-norm-reference", arm="F")
    path = tmp_path / "pending.pt"
    save_checkpoint(path, policy=p, optimizer=opt, bound=bound(), motor=m, environment=e, generators=g,
        contract=contract, iteration=0, stage="pending_labels", records=records, episodes=episodes)
    labels = {"1": .25, "2": .75}
    update(p, opt, stack_records(install_terminal_labels(records, episodes, labels, arm="F")), generator=g["update"])
    expected = deepcopy(p.state_dict())
    next_motor, next_action = m.infer(0.)[2], torch.randn(3, generator=g["action"])
    p2, opt2, m2, e2, g2 = learner()
    saved = restore_checkpoint(path, policy=p2, optimizer=opt2, bound=bound(), motor=m2, environment=e2, generators=g2, contract=contract)
    update(p2, opt2, stack_records(install_terminal_labels(saved["pending_records"], saved["pending_episodes"], labels, arm="F")), generator=g2["update"])
    for key, value in expected.items():
        torch.testing.assert_close(p2.state_dict()[key], value, rtol=0, atol=0)
    assert m2.infer(0.)[2] == next_motor
    torch.testing.assert_close(torch.randn(3, generator=g2["action"]), next_action, rtol=0, atol=0)
    final = tmp_path / "final.pt"
    save_checkpoint(final, policy=p2, optimizer=opt2, bound=bound(), motor=m2, environment=e2, generators=g2,
        contract=contract, iteration=1, stage="boundary", final=True)
    actual = torch.load(final, map_location="cpu", weights_only=False)
    assert actual["final"] and actual["optimizer"]["state"] and actual["iteration"] == 1
    assert actual["contract"]["motor"] == contract["motor"]
