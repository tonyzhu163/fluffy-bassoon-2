"""PPO raw transition persistence and total physical action accounting."""
import gzip
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_rl2 import RL2Config, collect_rl2_trial, make_rl2_policy, train_rl2
from fluffy_bassoon.meta_rl.vla_source40_ppo import JournaledSimulator, episode_sink, physical_charge


class Simulator:
    metadata = dict(settle_steps=10)
    def __init__(self):
        self.resets, self.steps = 0, 0
    def reset(self, state, seed):
        self.resets += 1
        return dict(initial_observation_sha256='a' * 64)
    def step(self, actions):
        self.steps += 1
        return dict(primitive_steps=2, success=True, done=True)


def test_physical_journal_charges_unknown_steps_and_resets(tmp_path):
    (tmp_path / 'intent-a.json').write_text(json.dumps(dict(maximum_primitive_steps=10)))
    (tmp_path / 'step-a.json').write_text(json.dumps(dict(primitive_steps=2)))
    (tmp_path / 'intent-b.json').write_text(json.dumps(dict(maximum_primitive_steps=10)))
    (tmp_path / 'reset-a.json').write_text(json.dumps(dict(settling_steps=10, returned=True)))
    (tmp_path / 'reset-b.json').write_text(json.dumps(dict(settling_steps=10, returned=False)))
    assert physical_charge(tmp_path) == dict(known_primitive_steps=2, unresolved_primitive_ceiling=10,
        settling_step_charge=20, known_settling_steps=10, total_action_charge=32)


def test_action_ceiling_prevents_rpc_before_spending(tmp_path):
    sim = Simulator()
    wrapped = JournaledSimulator(sim, tmp_path, 15)
    wrapped.reset(0, 1)
    with pytest.raises(RuntimeError, match='ceiling'):
        wrapped.step(np.zeros((10, 7)))
    assert sim.resets == 1 and sim.steps == 0
    wrapped.step(np.zeros((3, 7)))
    assert physical_charge(tmp_path)['total_action_charge'] == 12
    with pytest.raises(RuntimeError, match='ceiling'):
        wrapped.reset(0, 1)
    assert sim.resets == 1


def episode(sim, motor, choose, *, initial_state, scene_seed, chunk, bounded, feedback):
    obs = dict(image=np.zeros((2, 2, 3), np.uint8), wrist=np.zeros((2, 2, 3), np.uint8),
               state=np.zeros(8, np.float32))
    rows = []
    for index in range(2):
        action = choose(obs)
        done = index == 1
        reward = 0. if done else -1.
        feedback(action, reward, done)
        rows.append(dict(motor_observation=obs, motor_next_observation=obs,
            public_features=np.zeros(4104, np.float32), executed_noise=action,
            native_initial_noise=np.repeat(action[None], 10, axis=0),
            executed_actions=np.zeros((1, 7), np.float32), reward=reward,
            raw_rewards=[float(done)], done=done, success=done, primitive_steps=1))
    return rows, dict(initial_state=initial_state, scene_seed=scene_seed, success=True,
        primitive_steps=2, initial_observation_sha256=f'{initial_state}:{scene_seed}')


def test_raw_episode_keeps_actual_transitions_and_behavior_likelihood(tmp_path):
    policy = make_rl2_policy(RL2Config(observation_dim=4104, width=8, layers=1, heads=2, context_length=8))
    sink = episode_sink(tmp_path, contract=dict(source_only=True))
    rollout, _ = collect_rl2_trial(policy, SimpleNamespace(metadata=dict(settle_steps=10)), None,
        lambda obs: np.zeros(4104, np.float32), initial_state=0, scene_seed=1, attempts=2, chunk=10,
        episode_runner=episode, episode_sink=lambda **values: sink(trial=1, behavior_update=4,
            behavior_policy_sha256='b' * 64, **values))
    files = sorted(tmp_path.glob('*.pkl.gz'))
    assert len(files) == 2
    with gzip.open(files[1], 'rb') as stream:
        saved = pickle.load(stream)
    assert saved['metadata']['behavior_update'] == 4
    assert saved['metadata']['behavior_policy_sha256'] == 'b' * 64
    assert isinstance(saved['ppo']['initial_recurrent_state'], dict)
    assert saved['ppo']['initial_recurrent_state']['length'].item() == 2
    assert saved['ppo']['terminated'].tolist() == [False, True]
    assert not saved['ppo']['truncated'].any()
    assert saved['ppo']['primitive_durations'].tolist() == [1, 1]
    assert saved['rows'][0]['motor_next_observation']['image'].shape == (2, 2, 3)
    torch.testing.assert_close(saved['ppo']['log_probs'], rollout['log_probs'][2:], rtol=0, atol=0)
    assert torch.isfinite(saved['ppo']['pre_squash_log_probs']).all()
    assert saved['ppo']['attempt_ends'][:, 0].tolist() == [False, True]
    assert saved['ppo']['trial_ends'][:, 0].tolist() == [False, True]
    with gzip.open(files[0], 'rb') as stream:
        first = pickle.load(stream)
    assert first['ppo']['initial_recurrent_state'] is None
    assert not first['ppo']['trial_ends'].any()


def test_raw_sink_rejects_missing_real_transition_fields(tmp_path):
    sink = episode_sink(tmp_path, contract={})
    with pytest.raises(ValueError, match='omitted public'):
        sink(trial=1, attempt=1, rows=[dict(done=True)], summary={}, behavior={},
            behavior_update=0, behavior_policy_sha256='b' * 64, initial_recurrent_state='zero')


def test_raw_sink_preserves_actual_timeout_instead_of_calling_it_success(tmp_path):
    sink = episode_sink(tmp_path, contract={})
    row = dict(motor_observation={}, motor_next_observation={}, public_features=np.zeros(4104),
        executed_noise=np.zeros(32), native_initial_noise=np.zeros((10, 32)),
        executed_actions=np.zeros((3, 7)), reward=-1., raw_rewards=[0., 0., 0.],
        done=True, success=False, primitive_steps=3)
    sink(trial=1, attempt=1, rows=[row], summary=dict(primitive_steps=3),
        behavior=dict(raw_actions=torch.zeros(1, 1, 32), log_probs=torch.zeros(1, 1)),
        behavior_update=0, behavior_policy_sha256='b' * 64, initial_recurrent_state=None)
    with gzip.open(next(tmp_path.glob('*.pkl.gz')), 'rb') as stream:
        saved = pickle.load(stream)
    assert saved['ppo']['terminated'].tolist() == [False]
    assert saved['ppo']['truncated'].tolist() == [True]
    assert saved['ppo']['primitive_durations'].tolist() == [3]


def test_total_trial_budget_and_motor_rng_share_one_recovery(tmp_path):
    spent = 0
    def runner(*args, **kwargs):
        nonlocal spent
        spent += 12
        return episode(*args, **kwargs)
    policy = make_rl2_policy(RL2Config(observation_dim=4104, width=8, layers=1, heads=2, context_length=8))
    _, summary = train_rl2(policy, SimpleNamespace(metadata=dict(settle_steps=10)), None,
        lambda obs: np.zeros(4104, np.float32), train_states=[0], heldout_states=[40], attempts=2,
        chunk=10, episode_horizon=2, max_primitive_steps=8, max_total_actions=24,
        physical_cost=lambda: spent, updates=2, output_dir=tmp_path / 'run',
        source_contract=dict(source_commit='a' * 40), epochs=1, episode_runner=runner,
        runtime_state=lambda: dict(motor_rng=dict(key_data=np.array([1, 2], np.uint32))),
        restore_runtime=lambda value: None)
    assert summary['updates'] == 1 and spent == 24
    saved = torch.load(tmp_path / 'run/latest.pt', map_location='cpu', weights_only=False)
    assert saved['iteration'] == 1
    assert set(saved) >= {'policy', 'optimizer', 'environment_rng', 'rng', 'runtime'}
    np.testing.assert_array_equal(saved['runtime']['motor_rng']['key_data'], [1, 2])
