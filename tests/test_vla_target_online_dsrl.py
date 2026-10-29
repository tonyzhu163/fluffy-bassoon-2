"""Target-online DSRL control flow; allocated Linux only, no backbone needed."""
import gzip
import json
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest

import fluffy_bassoon.meta_rl.vla_target_online_dsrl as module


class Simulator:
    metadata = dict(instruction='move the object', task_id=0, suite='libero_90', horizon=400,
                    settle_steps=10, bddl_file='task.bddl')
    def __init__(self, fail_on=None):
        self.calls, self.resets, self.fail_on = 0, 0, fail_on
    def observation(self):
        return dict(image=np.zeros((64, 64, 3), np.uint8), wrist=np.zeros((64, 64, 3), np.uint8),
                    state=np.zeros(8, np.float32))
    def reset(self, state, seed):
        self.resets += 1
        self.in_episode = 0
        return dict(observation=self.observation(), initial_state_vector_sha256='fixed-state',
                    initial_observation_sha256='fixed-rgb')
    def step(self, actions):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError('interrupted actual RPC')
        self.in_episode += len(actions)
        return dict(observation=self.observation(), primitive_steps=len(actions),
            executed_actions=np.asarray(actions), rewards=[0.] * len(actions),
            success=False, done=self.in_episode == 400)


class Motor:
    horizon, feature_encoder = 10, None
    def __init__(self):
        self.policy = SimpleNamespace(_rng=7)
        self.noises = []
    def infer(self, observation, instruction, noise):
        self.noises.append(noise.copy())
        self.policy._rng += 1
        return np.zeros((10, 7), np.float32)


class Learner:
    def __init__(self):
        self.agent = SimpleNamespace(_rng=11)
        self.replay, self.updates = [], 0
    def frozen_signature(self):
        return str(self.updates)
    def action(self, observation):
        self.agent._rng += 1
        return np.full(32, self.updates * 1e-5, np.float32)
    def update(self, batch_size, sampler):
        assert batch_size == 256 and len(self.replay) > 500
        sampler.integers(len(self.replay), size=batch_size)
        self.updates += 1
        return dict(critic_loss=1., actor_loss=2., temperature=.1)
    def save(self, path, sampler, contract, **progress):
        with path.open('wb') as stream:
            pickle.dump(dict(replay=self.replay, updates=self.updates, agent_rng=self.agent._rng,
                sampler=sampler.bit_generator.state, contract=contract, progress=progress,
                numpy=np.random.get_state(), python=random.getstate()), stream)
    def restore(self, path, sampler, contract):
        with path.open('rb') as stream:
            saved = pickle.load(stream)
        assert saved['contract'] == contract
        self.replay, self.updates, self.agent._rng = saved['replay'], saved['updates'], saved['agent_rng']
        sampler.bit_generator.state = saved['sampler']
        np.random.set_state(saved['numpy'])
        random.setstate(saved['python'])
        return saved['progress']


@pytest.fixture(autouse=True)
def fake_jax(monkeypatch):
    monkeypatch.setattr(module, '_key', int)
    monkeypatch.setattr(module, '_save_motor', lambda motor: motor.policy._rng)
    monkeypatch.setattr(module, '_restore_motor', lambda motor, value: setattr(motor.policy, '_rng', value))


def run(tmp_path, simulator=None, resume=False):
    learner, motor = Learner(), Motor()
    result = module.run_case(simulator or Simulator(), motor, learner, np.random.default_rng(0),
        target=module.case(0), output=tmp_path / 'case-00', root=tmp_path,
        contract=dict(target_only=True, fixed=1), resume=resume)
    return result, learner, motor


def test_exact_published_warmup_updates_and_independent_frozen_queries(tmp_path):
    result, learner, motor = run(tmp_path)
    assert result['completed'] and result['training_episodes'] == 15
    assert result['training_chunks'] == len(learner.replay) == 600
    assert learner.updates == 2400
    assert [row['target_optimizer_updates_before'] for row in result['training_receipts']] == [0] * 13 + [800, 1600]
    assert [row['target_optimizer_updates'] for row in result['query_receipts']] == [0, 0, 800, 2400]
    assert [row['training_action_ceiling'] for row in result['query_receipts']] == [0, 410, 5330, 6150]
    assert [row['actual_training_actions'] for row in result['query_receipts']] == [0, 410, 5330, 6150]
    assert [row['query_behavior'] for row in result['query_receipts']] == ['native', 'native', 'frozen_target_sac', 'frozen_target_sac']
    assert result['costs']['training']['known_actions'] == 6150
    assert result['costs']['query']['known_actions'] == 1640
    assert result['total_action_charge'] == 7790
    assert result['target_only'] and not result['source_assembly_admission'] and not result['source_checkpoint_reused']
    for receipt in result['query_receipts']:
        assert receipt['optimizer_updates_during_query'] == 0 and not receipt['query_data_added_to_replay']
        assert module.sha256(tmp_path / 'case-00' / receipt['full_recovery_path']) == receipt['full_recovery_sha256']
    assert not np.array_equal(motor.noises[0][0], motor.noises[0][1])  # Published initial evaluation is full-H.
    assert np.array_equal(motor.noises[40][0], motor.noises[40][1])  # Collection warmup is tied32D.


def test_interrupted_episode_is_charged_but_not_replayed_as_training_data(tmp_path):
    with pytest.raises(RuntimeError, match='interrupted actual RPC'):
        run(tmp_path, Simulator(fail_on=45))
    # Query0 completed; only four returned training chunks plus one uncertain RPC.
    first = json.loads((tmp_path / 'case-00/index.json').read_text())
    assert first['training_episodes'] == first['replay_rows'] == 0
    cost = module.costs(tmp_path / 'case-00')
    assert cost['training']['known_actions'] == 50  # Reset10 + returned40.
    assert cost['training']['uncertain_actions_ceiling'] == 10
    result, learner, _ = run(tmp_path, resume=True)
    assert result['completed'] and learner.updates == 2400 and len(learner.replay) == 600
    assert result['total_action_charge'] == 7850
    assert len(result['query_receipts']) == 4  # Complete query0 was not rerun.
    assert result['costs']['training']['uncertain_actions_ceiling'] == 10


def test_frozen_query_restores_agent_motor_and_global_rng_and_never_adds_replay(tmp_path):
    learner, motor, sim = Learner(), Motor(), Simulator()
    learner.updates = 1
    wrapped = module.JournaledSimulator(sim, tmp_path / 'case-00', tmp_path)
    wrapped.role, wrapped.episode_id = 'query', 'query-test'
    before_numpy, before_python = np.random.get_state(), random.getstate()
    rows, receipt = module.frozen_query(wrapped, motor, learner, module.case(0))
    assert learner.agent._rng == 11 and motor.policy._rng == 7
    assert len(learner.replay) == 0 and learner.updates == 1
    assert receipt['weights_unchanged'] and len(rows) == 40
    for a, b in zip(before_numpy, np.random.get_state(), strict=True):
        np.testing.assert_array_equal(a, b)
    assert random.getstate() == before_python


def test_joint_and_case_caps_block_rpc_and_keep_actual_payloads(tmp_path):
    sim = Simulator()
    wrapped = module.JournaledSimulator(sim, tmp_path / 'case-00', tmp_path, case_cap=20, total_cap=20)
    wrapped.reset(0, 123)
    wrapped.context = dict(executed_noise=np.zeros(32, np.float32))
    wrapped.step(np.zeros((10, 7), np.float32))
    assert module.charged(tmp_path / 'case-00') == module.joint_charge(tmp_path) == 20
    assert not wrapped.reserve_episode()
    with pytest.raises(RuntimeError, match='ceiling'):
        wrapped.step(np.zeros((10, 7), np.float32))
    assert sim.calls == 1
    other = module.JournaledSimulator(Simulator(), tmp_path / 'case-01', tmp_path, case_cap=20, total_cap=20)
    with pytest.raises(RuntimeError, match='ceiling'):
        other.reset(0, 123)
    rows = [json.loads(path.read_text()) for path in wrapped.journal.glob('*.json')]
    step = next(row for row in rows if row['kind'] == 'step')
    assert module.sha256(wrapped.journal / step['input_path']) == step['input_sha256']
    assert module.sha256(wrapped.journal / step['outcome_path']) == step['outcome_sha256']
    with gzip.open(wrapped.journal / step['input_path'], 'rb') as stream:
        payload = pickle.load(stream)
    np.testing.assert_array_equal(payload['context']['executed_noise'], np.zeros(32, np.float32))


def test_deadline_stops_at_complete_recovery_without_new_actions(tmp_path):
    learner, motor, sim = Learner(), Motor(), Simulator()
    result = module.run_case(sim, motor, learner, np.random.default_rng(0), target=module.case(0),
        output=tmp_path / 'case-00', root=tmp_path, contract=dict(fixed=1), deadline=0.)
    assert not result['completed'] and result['replay_rows'] == 0 and sim.calls == sim.resets == 0
    assert (tmp_path / 'case-00/target-recovery.pkl').exists()
    with pytest.raises(ValueError, match='contract changed'):
        module.run_case(sim, motor, learner, np.random.default_rng(0), target=module.case(0),
            output=tmp_path / 'case-00', root=tmp_path, contract=dict(fixed=2), resume=True)
