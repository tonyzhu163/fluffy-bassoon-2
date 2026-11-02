"""Continuous native lanes preserve complete records, GAE and the next Adam update."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig, stack_records, update
from fluffy_bassoon.meta_rl.robocasa_residual import collect_pending, collect_pending_lanes, install_terminal_labels
from fluffy_bassoon.meta_rl.robocasa_residual_parallel import ParallelResidualEnvironment, draw_schedule, native_chunk_offsets
from fluffy_bassoon.rl.action_residual import ResidualBound


class Motor:
    def __init__(self):
        self.seed_rng = np.random.default_rng(42)
        self.batch_sizes = []

    def infer(self, observation, *, noise_seed):
        state = observation['states'][0]
        chunk = np.full((50, 12), (noise_seed % 101)*.001+state[1]*.0001, np.float32)
        feature = np.array([state[0]*.1, state[1]*.01, (noise_seed % 101)*.01], np.float32)
        return chunk, feature, noise_seed

    def infer_batch(self, observation, *, noise_seeds):
        self.batch_sizes.append(len(noise_seeds))
        rows = [self.infer(dict(states=observation['states'][i:i+1]), noise_seed=seed)
                for i, seed in enumerate(noise_seeds)]
        return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]), noise_seeds


class Lane:
    def __init__(self, index, case, *, cases=None, fail=False):
        self.cases = deepcopy([case] if cases is None else cases)
        self.index, self.fail = index, fail
        self.steps, self.cursor, self.resets = 0, 0, 0
        self.closed, self.active = False, False
        self.completed = []

    def seek(self, index):
        assert not self.active and self.cursor <= index < len(self.cases)
        self.index, self.cursor = index, index

    def observation(self):
        return dict(states=np.array([[self.index, self.steps//50]], np.float32))

    def reset(self):
        from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS
        self.horizon = TASKS[self.cases[self.cursor]['task']][0]
        self.steps, self.active = 0, True
        self.cursor += 1; self.resets += 1
        return self.observation()

    def step_chunk(self, chunk):
        if self.fail: raise RuntimeError('native failure')
        assert chunk.shape == (50, 12)
        self.steps += 50
        done = self.steps == self.horizon
        if done:
            self.active = False; self.completed.append(self.index)
        return self.observation(), done, dict(id=str(self.index), round=self.index) if done else {}

    def state_dict(self):
        if self.active: raise ValueError('Incomplete native episode')
        return dict(index=self.index, cursor=self.cursor, steps=self.steps)

    def close(self):
        self.closed = True


def learner():
    torch.manual_seed(42)
    policy = ResidualActorCritic(ResidualPPOConfig(3, 6, width=8, epochs=2, minibatch_size=8))
    return policy, torch.optim.Adam(policy.parameters(), lr=policy.config.learning_rate)


def test_mixed_horizon_batch_preserves_serial_records_and_next_adam_update():
    cases = [dict(task=task) for task in ('OpenDrawer', 'CoffeeSetupMug', 'OpenDrawer')]
    offsets = native_chunk_offsets(cases)
    policy, optimizer = learner(); bound = ResidualBound(12, tuple(range(6)), (.25,)*6)
    seeds, noise = draw_schedule(Motor(), policy, torch.Generator().manual_seed(8), offsets[-1], deterministic=False)
    serial, serial_episodes = [], []
    for i, case in enumerate(cases):
        rows, episodes = collect_pending(policy, bound, Motor(), Lane(i, case), episodes=1, generator=None,
            planned_motor_seeds=seeds[offsets[i]:offsets[i+1]], planned_action_noise=noise[offsets[i]:offsets[i+1]])
        serial.extend(rows); serial_episodes.extend(episodes)
    lanes = [Lane(0, cases[0], cases=cases) for _ in range(2)]
    actual, episodes = collect_pending_lanes(policy, bound, Motor(), lanes,
        planned_motor_seeds=seeds, planned_action_noise=noise)
    assert episodes == serial_episodes and len(actual) == 42
    for left, right in zip(serial, actual):
        for key in left:
            if isinstance(left[key], np.ndarray): np.testing.assert_array_equal(left[key], right[key])
            else: assert left[key] == right[key]
    second, second_optimizer = learner(); labels = {row['id']: i%2 for i, row in enumerate(episodes)}
    for model, opt, records in ((policy, optimizer, serial), (second, second_optimizer, actual)):
        update(model, opt, stack_records(install_terminal_labels(records, episodes, labels, arm='S')),
               generator=torch.Generator().manual_seed(43))
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    for a, b in zip(optimizer.state_dict()['state'].values(), second_optimizer.state_dict()['state'].values()):
        for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_eight_lanes_refill_without_padding_and_pin_recovery(tmp_path):
    cases = [dict(task='CoffeeSetupMug' if i%5==0 else 'OpenDrawer') for i in range(10)]
    offsets = native_chunk_offsets(cases); policy, _ = learner(); motor = Motor()
    seeds, _ = draw_schedule(motor, policy, torch.Generator().manual_seed(8), offsets[-1], deterministic=True)
    lanes = [Lane(0, cases[0], cases=cases) for _ in range(8)]
    records, episodes = collect_pending_lanes(policy, ResidualBound(12, tuple(range(6)), (.25,)*6), motor, lanes,
        deterministic=True, verify_zero=True, planned_motor_seeds=seeds)
    assert len(records) == offsets[-1] and [e['id'] for e in episodes] == list(map(str, range(10)))
    assert sum(motor.batch_sizes) == offsets[-1] and 1 < max(motor.batch_sizes) <= 8
    assert sorted(slot for lane in lanes for slot in lane.completed) == list(range(10))
    assert sum(lane.resets for lane in lanes) == 10 and all(not lane.active for lane in lanes)
    assert all(r['zero_residual_verified'] for r in records[:12])
    p8 = ParallelResidualEnvironment(cases, horizon=None, output=tmp_path, monitoring_root=tmp_path,
        profile=dict(rollout_lanes_per_gpu=8), code=tmp_path, report=lambda *a, **k: None)
    p1 = ParallelResidualEnvironment(cases, horizon=None, output=tmp_path, monitoring_root=tmp_path,
        profile={}, code=tmp_path, report=lambda *a, **k: None)
    with pytest.raises(ValueError): p1.load_state_dict(p8.state_dict())


def test_failed_lane_cannot_produce_complete_records_or_recovery():
    cases = [dict(task='OpenDrawer')]*2; lanes = [Lane(0, cases[0], cases=cases, fail=i==1) for i in range(2)]
    policy, _ = learner()
    with pytest.raises(RuntimeError, match='native failure'):
        try:
            collect_pending_lanes(policy, ResidualBound(12, tuple(range(6)), (.25,)*6), Motor(), lanes,
                                  deterministic=True, planned_motor_seeds=list(range(30)))
        finally:
            for lane in lanes: lane.close()
    assert all(lane.closed for lane in lanes) and any(lane.active for lane in lanes)
    with pytest.raises(ValueError): next(lane for lane in lanes if lane.active).state_dict()


def test_incomplete_draw_plan_is_rejected_before_any_native_reset():
    cases = [dict(task='OpenDrawer')]; lanes = [Lane(0, cases[0], cases=cases)]
    policy, _ = learner()
    with pytest.raises(ValueError):
        collect_pending_lanes(policy, ResidualBound(12, tuple(range(6)), (.25,)*6), Motor(), lanes,
            deterministic=False, planned_motor_seeds=list(range(15)), planned_action_noise=torch.zeros(14, 6))
    assert all(lane.resets==0 for lane in lanes)


def test_native_lane_seek_retains_original_episode_id_and_round(tmp_path):
    from test_robocasa_residual import environment
    env = environment(tmp_path)
    env.cases.append(dict(native_seed=2, task='OpenDrawer', round=1))
    env.seek(1); env.reset()
    for _ in range(15):
        _, done, episode = env.step_chunk(np.zeros((50, 12), np.float32))
    assert done and episode['id'].endswith('episode-000001') and episode['round'] == 1
    assert env.state_dict()['cursor'] == 2
