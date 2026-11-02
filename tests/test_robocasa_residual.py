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
        self.path, self.count, self.values = path, 0, []

    def append_data(self, frame):
        self.count += 1
        self.values.append(int(frame[0, 0, 0]))

    def close(self):
        from pathlib import Path
        Path(self.path).write_bytes(b"fixture-video")


class Native:
    auto_reset, ignore_terminations = False, True
    task_descriptions = ["open the drawer"]
    current_info_list = [dict(ep_meta={})]

    def __init__(self, early=False, horizon=750, success=None):
        self.early = early
        self.horizon = horizon
        self.success = success or (lambda step: True)
        self.count = dict(controls=3, reset_controls=3, physics_ticks=30, control_time=0., clock=dict(control_freq=20.))
        self.env = SimpleNamespace(get_env_attr=lambda key: [deepcopy(self.count)])

    def observation(self):
        return dict(main_images=np.full((1, 4, 5, 3), 11, np.uint8),
                    wrist_images=np.full((1, 4, 5, 3), 22, np.uint8),
                    extra_view_images=np.full((1, 1, 4, 5, 3), 33, np.uint8),
                    states=np.zeros((1, 25), np.float32),
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
        return self.observation(), np.ones(1), np.array([self.early]), np.array([self.steps == self.horizon]), dict(episode=dict(success_at_end=np.array([self.success(self.steps)])))

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
    assert [writer.count for writer in env.writer.writers] == [751, 751, 751]
    assert [set(writer.values) for writer in env.writer.writers] == [{11}, {22}, {33}]
    assert [ref['camera'] for ref in info['videos']] == info['camera_order']
    from fluffy_bassoon.meta_rl.rlinf_flora_ppo import flora_jobs
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import protocol
    assert flora_jobs([info], protocol('OpenDrawer'))[0]['videos'] == [ref['path'] for ref in info['videos']]
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


@pytest.mark.parametrize('steps', [(), (137,), (137, 138, 750), (750,)])
def test_first_success_counts_primitive_controls_and_preserves_transient_success(tmp_path, steps):
    cases = [dict(native_seed=1, task='OpenDrawer')] * 2
    env = RoboCasaResidualEnvironment(lambda case: Native(success=lambda step: step in steps), cases,
        horizon=750, output=tmp_path/'videos', monitoring_root=tmp_path/'monitor',
        journal=Journal(), writer_factory=Writer)
    import json
    for episode_index in range(2):
        env.reset()
        for _ in range(15):
            observation, _, episode = env.step_chunk(np.zeros((50, 12), np.float32))
            assert 'first_success_step' not in observation and 'first_success_step' not in episode
        row = json.loads((tmp_path/f'monitor/episode-{episode_index:06d}.json').read_text())
        assert row['first_success_step'] == (steps[0] if steps else None)
        assert row['success_steps'] == list(steps)
        assert row['simulator_success_once'] is bool(steps)
        assert row['simulator_final_success'] is (750 in steps)
    env.close()


@pytest.mark.parametrize('task,horizon', [('TurnOnMicrowave',450),('CoffeeSetupMug',600),('OpenCabinet',1050)])
def test_declared_native_horizons_keep_full_masks_and_all_three_frame_clocks(tmp_path,task,horizon):
    cases=[dict(native_seed=1,task=task)]
    env=RoboCasaResidualEnvironment(lambda case:Native(horizon=horizon),cases,horizon=horizon,
        output=tmp_path/'videos',monitoring_root=tmp_path/'monitor',journal=Journal(),writer_factory=Writer)
    env.reset()
    for index in range(horizon//50):
        _,done,episode=env.step_chunk(np.zeros((50,12),np.float32))
        assert done == (index==horizon//50-1)
    assert episode['horizon']==horizon and episode['frame_clock']['frame_count']==horizon+1
    assert [writer.count for writer in env.writer.writers]==[horizon+1]*3
    assert env.state_dict()['cursor']==1
    env.close()
    with pytest.raises(ValueError,match='native full episode horizon'):
        RoboCasaResidualEnvironment(lambda case:Native(),cases,horizon=750,
            output=tmp_path/'wrong',monitoring_root=tmp_path/'bad-monitor',journal=Journal(),writer_factory=Writer)


def test_mixed_task_native_horizons_recover_only_at_full_episode_boundary(tmp_path):
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS
    cases=[dict(task='CoffeeSetupMug',native_seed=1),dict(task='OpenDrawer',native_seed=2)]
    factory=lambda case:Native(horizon=TASKS[case['task']][0])
    env=RoboCasaResidualEnvironment(factory,cases,horizon=None,output=tmp_path/'first',
        monitoring_root=tmp_path/'monitor',journal=Journal(),writer_factory=Writer)
    env.reset()
    for _ in range(12):_,done,episode=env.step_chunk(np.zeros((50,12),np.float32))
    assert done and episode['horizon']==600 and episode['frame_clock']['frame_count']==601
    state=env.state_dict();assert state['horizon'] is None and state['cursor']==1
    env.close()
    recovered=RoboCasaResidualEnvironment(factory,cases,horizon=None,output=tmp_path/'recovered',
        monitoring_root=tmp_path/'monitor',journal=Journal(),writer_factory=Writer)
    recovered.load_state_dict(state);recovered.reset()
    with pytest.raises(ValueError,match='boundaries'):recovered.state_dict()
    for _ in range(15):_,done,episode=recovered.step_chunk(np.zeros((50,12),np.float32))
    assert done and episode['horizon']==750 and episode['frame_clock']['frame_count']==751
    assert recovered.state_dict()['cursor']==2
    recovered.close()


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


class LaneModel(MotorModel):
    action_dim = 32

    def build_prefix_cache(self, observation):
        self.prefix_calls += 1
        batch = len(observation["states"])
        return (torch.tensor([[[1., 3.], [3., 1.], [float("nan"), float("nan")]]]).expand(batch, -1, -1),
                torch.tensor([[True, True, False]]).expand(batch, -1), ())

    def predict_action_batch(self, observation, mode, rng=None, noise=None):
        self.calls += 1
        assert mode == "eval" and "success" not in observation
        self.build_prefix_cache(observation)
        states = torch.as_tensor(np.asarray(observation["states"]))
        if noise is None:
            noise = torch.randn((len(states), 50, 32), generator=rng)
        return noise[..., :12] + states[:, None, :12], {}


class StepNative(Native):
    def observation(self):
        value = super().observation()
        value["states"] = np.full((1, 25), getattr(self, "steps", 0) / 1000, np.float32)
        return value


def test_lane_collection_matches_serial_draws_ids_and_records(tmp_path):
    from fluffy_bassoon.meta_rl.robocasa_residual import collect_pending_lanes
    from fluffy_bassoon.meta_rl.robocasa_residual_parallel import draw_schedule, native_chunk_offsets
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS
    stats = {key: dict(mean=[0.] * 32, std=[2.] * 32) for key in ("state", "actions")}
    cases = [dict(task=task, native_seed=index) for index, task in
             enumerate(["CoffeeSetupMug", "OpenDrawer", "OpenDrawer", "CoffeeSetupMug", "OpenDrawer"])]
    torch.manual_seed(0)
    policy = ResidualActorCritic(ResidualPPOConfig(42, 6, width=8))
    plan_motor = FrozenRoboCasaMotor(LaneModel(), stats, seed=3)
    seeds, noise = draw_schedule(plan_motor, policy, torch.Generator().manual_seed(4),
                                 native_chunk_offsets(cases)[-1], deterministic=False)

    def env(root):
        return RoboCasaResidualEnvironment(lambda case: StepNative(horizon=TASKS[case["task"]][0]), cases,
            horizon=None, output=root / "videos", monitoring_root=root / "monitor", journal=Journal(),
            writer_factory=Writer, episode_id_prefix="case-000000-rank-00-")

    serial_records, serial_episodes = collect_pending(policy, bound(), FrozenRoboCasaMotor(LaneModel(), stats),
        env(tmp_path / "serial"), episodes=len(cases), generator=None,
        planned_motor_seeds=seeds, planned_action_noise=noise)
    lane_motor = FrozenRoboCasaMotor(LaneModel(), stats)
    lane_records, lane_episodes = collect_pending_lanes(policy, bound(), lane_motor,
        [env(tmp_path / "lanes") for _ in range(2)], planned_motor_seeds=seeds, planned_action_noise=noise)
    # Background resets make the forward count timing dependent; draws are keyed by case and chunk.
    assert lane_motor.model.calls >= 42
    assert [e["id"] for e in lane_episodes] == [e["id"] for e in serial_episodes]
    assert [e["round"] for e in lane_episodes] == list(range(5))
    assert len(lane_records) == len(serial_records) == sum(TASKS[c["task"]][0] // 50 for c in cases)
    for lane, serial in zip(lane_records, serial_records):
        for key in ("noise_seed", "reward", "done", "episode_start"):
            assert lane[key] == serial[key]
        for key in ("features", "control", "base_chunk", "executed_chunk"):
            np.testing.assert_array_equal(lane[key], serial[key])
        assert lane["log_prob"] == serial["log_prob"]


def test_native_reset_child_cannot_keep_another_lanes_video_pipe_open():
    """Refilling a lane cannot prevent a finished encoder from receiving EOF."""
    import subprocess
    import sys
    code = r'''
import multiprocessing as mp
import os
import select
import time
from fluffy_bassoon.meta_rl.robocasa_residual import native_environment_factory

mp.set_start_method("fork", force=True)
reader, writer = os.pipe()
native_environment_factory({})  # Production factory selects the child context.
child = mp.Process(target=time.sleep, args=(30,))
child.start()
try:
    os.close(writer)
    assert select.select([reader], [], [], 2)[0], "reset child retained the encoder write end"
    assert os.read(reader, 1) == b"" and child.is_alive()
finally:
    os.close(reader)
    child.terminate()
    child.join(timeout=5)
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
