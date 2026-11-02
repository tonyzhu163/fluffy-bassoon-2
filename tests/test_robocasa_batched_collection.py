"""Mixed-horizon records, next Adam update and complete worker recovery."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig, stack_records, update
from fluffy_bassoon.meta_rl.robocasa_residual import collect_pending, install_terminal_labels
from fluffy_bassoon.meta_rl.robocasa_residual_batch import BatchedRoboCasaResidualEnvironment
from fluffy_bassoon.meta_rl.robocasa_residual_parallel import (
    ParallelResidualEnvironment, draw_schedule, native_chunk_offsets,
)
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
    def __init__(self, index, case, *, fail=False):
        from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS
        self.index, self.horizon, self.fail = index, TASKS[case['task']][0], fail
        self.steps, self.closed = 0, False

    def observation(self):
        return dict(states=np.array([[self.index, self.steps//50]], np.float32))

    def reset(self):
        assert self.steps == 0
        return self.observation()

    def step_chunk(self, chunk):
        if self.fail: raise RuntimeError('native failure')
        assert chunk.shape == (50, 12)
        self.steps += 50
        done = self.steps == self.horizon
        return self.observation(), done, dict(id=str(self.index), round=self.index) if done else {}

    def state_dict(self):
        return dict(index=self.index, steps=self.steps)

    def close(self):
        self.closed = True


def learner():
    torch.manual_seed(42)
    policy = ResidualActorCritic(ResidualPPOConfig(3, 6, width=8, epochs=2, minibatch_size=8))
    return policy, torch.optim.Adam(policy.parameters(), lr=policy.config.learning_rate)


def test_mixed_horizon_batch_preserves_serial_records_and_next_adam_update():
    cases = [dict(task=task) for task in ('OpenDrawer', 'CoffeeSetupMug', 'OpenDrawer')]
    offsets = native_chunk_offsets(cases)
    policy, optimizer = learner()
    bound = ResidualBound(12, tuple(range(6)), (.25,)*6)
    seeds, noise = draw_schedule(Motor(), policy, torch.Generator().manual_seed(8), offsets[-1], deterministic=False)
    serial, serial_episodes = [], []
    for i, case in enumerate(cases):
        rows, episodes = collect_pending(policy, bound, Motor(), Lane(i, case), episodes=1, generator=None,
            planned_motor_seeds=seeds[offsets[i]:offsets[i+1]], planned_action_noise=noise[offsets[i]:offsets[i+1]])
        serial.extend(rows); serial_episodes.extend(episodes)
    env = BatchedRoboCasaResidualEnvironment(Lane, cases, horizon=None, environments_per_gpu=2)
    actual, episodes = collect_pending(policy, bound, Motor(), env, episodes=3, generator=None,
        planned_motor_seeds=seeds, planned_action_noise=noise)
    assert episodes == serial_episodes and len(actual) == 42
    for left, right in zip(serial, actual):
        for key in left:
            if isinstance(left[key], np.ndarray): np.testing.assert_array_equal(left[key], right[key])
            else: assert left[key] == right[key]
    second, second_optimizer = learner()
    labels = {row['id']: i%2 for i, row in enumerate(episodes)}
    for model, opt, records in ((policy, optimizer, serial), (second, second_optimizer, actual)):
        update(model, opt, stack_records(install_terminal_labels(records, episodes, labels, arm='S')),
               generator=torch.Generator().manual_seed(43))
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    for a, b in zip(optimizer.state_dict()['state'].values(), second_optimizer.state_dict()['state'].values()):
        for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_eight_lanes_shrink_at_native_ends_without_padding_and_pin_recovery(tmp_path):
    cases = [dict(task='CoffeeSetupMug' if i%5==0 else 'OpenDrawer') for i in range(10)]
    offsets = native_chunk_offsets(cases)
    policy, _ = learner(); motor = Motor()
    seeds, _ = draw_schedule(motor, policy, torch.Generator().manual_seed(8), offsets[-1], deterministic=True)
    env = BatchedRoboCasaResidualEnvironment(Lane, cases, horizon=None, environments_per_gpu=8)
    records, episodes = collect_pending(policy, ResidualBound(12, tuple(range(6)), (.25,)*6), motor, env,
        episodes=10, generator=None, deterministic=True, verify_zero=True, planned_motor_seeds=seeds)
    assert len(records) == offsets[-1] and [e['id'] for e in episodes] == list(map(str, range(10)))
    assert motor.batch_sizes[:12] == [8, 8]*6  # First12 chunks include same-batch zero-residual replay.
    assert motor.batch_sizes[24:27] == [6, 6, 6]  # Both coffee lanes ended at600.
    assert motor.batch_sizes[-15:] == [2]*15
    assert all(r['zero_residual_verified'] for r in records[:12])
    state = env.state_dict(); assert state['cursor'] == 10 and len(state['completed_states']) == 10
    restored = BatchedRoboCasaResidualEnvironment(Lane, cases, horizon=None, environments_per_gpu=8)
    restored.load_state_dict(deepcopy(state)); assert restored.state_dict() == state
    wrong = BatchedRoboCasaResidualEnvironment(Lane, cases, horizon=None, environments_per_gpu=4)
    with pytest.raises(ValueError): wrong.load_state_dict(state)
    p8 = ParallelResidualEnvironment(cases, horizon=None, output=tmp_path, monitoring_root=tmp_path,
        profile=dict(rollout_envs_per_gpu=8), code=tmp_path, report=lambda *a, **k: None)
    p1 = ParallelResidualEnvironment(cases, horizon=None, output=tmp_path, monitoring_root=tmp_path,
        profile={}, code=tmp_path, report=lambda *a, **k: None)
    with pytest.raises(ValueError): p1.load_state_dict(p8.state_dict())


def test_failed_lane_keeps_bank_incomplete_and_closes_native_children():
    cases = [dict(task='OpenDrawer')]*2; created = []
    def make(index, case):
        env = Lane(index, case, fail=index==1); created.append(env); return env
    env = BatchedRoboCasaResidualEnvironment(make, cases, horizon=None, environments_per_gpu=2)
    policy, _ = learner(); seeds = list(range(30))
    with pytest.raises(RuntimeError, match='native failure'):
        collect_pending(policy, ResidualBound(12, tuple(range(6)), (.25,)*6), Motor(), env,
            episodes=2, generator=None, deterministic=True, planned_motor_seeds=seeds)
    assert all(lane.closed for lane in created) and env.cursor == 0
    with pytest.raises(ValueError): env.state_dict()


def test_incomplete_draw_plan_is_rejected_before_any_native_reset():
    created = []
    env = BatchedRoboCasaResidualEnvironment(lambda *args: created.append(args), [dict(task='OpenDrawer')],
                                           horizon=None, environments_per_gpu=8)
    policy, _ = learner()
    with pytest.raises(ValueError):
        env.collect(policy, ResidualBound(12, tuple(range(6)), (.25,)*6), Motor(), episodes=1,
                    generator=None, deterministic=False, planned_motor_seeds=list(range(15)))
    assert not created


def test_native_lane_retains_original_episode_id_round_and_recovery_offset(tmp_path):
    from test_robocasa_residual import environment
    env = environment(tmp_path/'first')
    env.episode_index_offset = 5
    env.reset()
    for _ in range(15):
        _, done, episode = env.step_chunk(np.zeros((50, 12), np.float32))
    assert done and episode['id'].endswith('episode-000005') and episode['round'] == 5
    state = env.state_dict(); assert state['episode_index_offset'] == 5
    other = environment(tmp_path/'second')
    with pytest.raises(ValueError): other.load_state_dict(state)
