"""Allocated CPU recovery tests with no JAX/model/simulator construction.

Fakes exercise the real task57 driver, legacy sparse-transition collector and
physical RPC journal. Preseeded dummy replay is only a unit-test setup to reach
the post-warmup branch without hundreds of redundant physical-episode fixtures.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest


class FakeSAC:
    def __init__(self):
        self.replay = [{} for _ in range(500)]
        self.updates = 0
        self.agent = SimpleNamespace(_rng=3)
        self.update_calls = []

    def frozen_signature(self):
        return hashlib.sha256(str(self.updates).encode()).hexdigest()

    def action(self, observation, deterministic=False):
        del observation, deterministic
        self.agent._rng += 1
        return np.linspace(-2., 2., 32, dtype=np.float32) + np.float32(self.agent._rng % 7 / 100)

    def update(self, batch, sampler):
        assert batch == 256
        sampler.integers(0, len(self.replay), batch)  # Actual sampler consumption.
        self.updates += 1; self.agent._rng += 17
        self.update_calls.append(self.updates)
        return dict(actor_loss=-.25, critic_loss=.5, temperature=.2)

    def save(self, path, sampler, contract, **progress):
        payload = dict(contract=contract, updates=self.updates, replay=self.replay,
            agent_rng=self.agent._rng, sampler=sampler.bit_generator.state,
            numpy_rng=np.random.get_state(), python_rng=random.getstate(), progress=progress)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(path).with_suffix('.tmp')
        temporary.write_bytes(pickle.dumps(payload, protocol=5))
        temporary.replace(path)  # Match qualified OfficialSAC; preserve hardlinks.

    def restore(self, path, sampler, contract):
        saved = pickle.loads(Path(path).read_bytes())
        assert saved['contract'] == contract
        self.updates, self.replay, self.agent._rng = saved['updates'], saved['replay'], saved['agent_rng']
        sampler.bit_generator.state = saved['sampler']
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        return copy.deepcopy(saved['progress'])


class FakeMotor:
    horizon = 10
    feature_encoder = None

    def __init__(self):
        self.policy = SimpleNamespace(_rng=7)
        self.metadata = dict(weights_frozen=True, receipt_sha256='ec064-fake-unit-fixture')

    def infer(self, observation, instruction, noise):
        del observation, instruction
        self.policy._rng += 1
        return np.asarray(noise, np.float32)[:, :7].copy()


class InjectedQueryFailure(RuntimeError):
    pass


class FakeSimulator:
    def __init__(self, learner, *, interrupt_panel=False, chunks_per_episode=2):
        self.learner = learner
        self.interrupt_panel = interrupt_panel
        self.chunks_per_episode = chunks_per_episode
        self.resets = []
        self.metadata = dict(suite='libero_90', task_id=57, instruction='task57 fixture',
            horizon=400, settle_steps=10, initial_states=50, fresh_environment_on_reset=True,
            bddl_file='/fixture/task57.bddl', bddl_sha256='fixture-bddl')

    def observation(self):
        return dict(image=np.full((8, 8, 3), self.state, np.uint8),
            wrist=np.full((8, 8, 3), self.state+1, np.uint8),
            state=np.asarray([self.state, self.chunk, 0, 0, 0, 0, 0, 0], np.float32))

    def reset(self, state, seed):
        self.resets.append(dict(state=state, seed=seed, updates=self.learner.updates))
        # Fail after the first learned-panel query has already committed.
        if self.interrupt_panel and self.learner.updates >= 40 and state == 41:
            self.interrupt_panel = False
            raise InjectedQueryFailure('query41 reset interrupted after query40 committed')
        self.state, self.chunk = int(state), 0
        np.random.rand(); random.random()  # Query isolation must rewind these.
        nominal = hashlib.sha256(str(state).encode()).hexdigest()
        public = hashlib.sha256(f'{state}/{seed}'.encode()).hexdigest()
        return dict(observation=self.observation(), initial_observation_sha256=public,
            initial_state_vector_sha256=nominal, initial_simulator_state_sha256=nominal,
            initial_simulator_state=np.asarray([state], np.float64),
            initial_public_observation_hashes=dict(image=public, wrist=public, state=public))

    def step(self, actions):
        assert np.asarray(actions).shape == (10, 7)
        self.chunk += 1
        done = self.chunk == self.chunks_per_episode
        success = bool(done and self.state % 2 == 0)
        return dict(observation=self.observation(), primitive_steps=2,
            executed_actions=np.asarray(actions[:2], np.float32).copy(),
            rewards=[0., float(success)], done=done, success=success)


def patch_rng(monkeypatch, profile):
    # The real query function still snapshots/restores all training state; only
    # JAX key representation is replaced so this CPU fixture imports no JAX.
    monkeypatch.setattr(profile.onset, '_key', lambda seed: int(seed))
    monkeypatch.setattr(profile.onset, '_save_motor', lambda motor: int(motor.policy._rng))
    monkeypatch.setattr(profile.onset, '_restore_motor', lambda motor, key: setattr(motor.policy, '_rng', int(key)))


def load_raw(path):
    with gzip.open(path, 'rb') as handle:
        return pickle.load(handle)


def contract(profile):
    return profile.scientific_contract('a'*40, 'b'*64)


def paths(tmp_path, name):
    root = tmp_path/name/'experiment'
    return root, root/'case-57'


def training_rows(output, result):
    episodes = [load_raw(output/receipt['trace']) for receipt in result['training']['receipts']]
    return episodes, [row for episode in episodes for row in episode['rows']]


def test_native_view_and_learned_query_isolate_actual_learner_and_all_rng(monkeypatch):
    from fluffy_bassoon.meta_rl import vla_target_online_dsrl_task57 as profile
    patch_rng(monkeypatch, profile)
    learner, motor = FakeSAC(), FakeMotor()
    learner.updates = 40
    sim = FakeSimulator(learner)
    before = pickle.dumps(dict(updates=learner.updates, replay=learner.replay,
        agent_rng=learner.agent._rng, motor_rng=motor.policy._rng,
        numpy=np.random.get_state(), python=random.getstate()), protocol=5)
    rows, summary = profile.native_query(sim, motor, learner, profile.target(40))
    assert summary['native_sampling_view_only'] is True
    assert summary['target_optimizer_updates'] == summary['underlying_actor_optimizer_updates'] == 40
    assert summary['query_behavior'] == 'native'
    assert all(row['proposed_noise'] is None and row['executed_noise'] is None for row in rows)
    assert all(row['native_initial_noise'].shape == (10, 32) for row in rows)
    assert any(not np.array_equal(row['native_initial_noise'][0], row['native_initial_noise'][1]) for row in rows)
    after = pickle.dumps(dict(updates=learner.updates, replay=learner.replay,
        agent_rng=learner.agent._rng, motor_rng=motor.policy._rng,
        numpy=np.random.get_state(), python=random.getstate()), protocol=5)
    assert before == after
    learned_rows, learned = profile.onset.frozen_query(sim, motor, learner, profile.target(40))
    assert learned['query_behavior'] == 'frozen_target_sac' and learned['optimizer_updates_during_query'] == 0
    assert all(np.max(np.abs(row['executed_noise'])) <= 1 for row in learned_rows)
    assert all(np.array_equal(row['native_initial_noise'], np.repeat(row['executed_noise'][None], 10, axis=0))
        for row in learned_rows)
    assert before == pickle.dumps(dict(updates=learner.updates, replay=learner.replay,
        agent_rng=learner.agent._rng, motor_rng=motor.policy._rng,
        numpy=np.random.get_state(), python=random.getstate()), protocol=5)


def test_partial_panel_resume_keeps_query_cursor_training_rng_and_physical_intents(monkeypatch, tmp_path):
    from fluffy_bassoon.meta_rl import vla_target_online_dsrl_task57 as profile
    patch_rng(monkeypatch, profile)
    root, output = paths(tmp_path, 'interrupted')
    initial_numpy, initial_python = np.random.get_state(), random.getstate()
    learner, motor, sampler = FakeSAC(), FakeMotor(), np.random.default_rng(0)
    sim = FakeSimulator(learner, interrupt_panel=True)
    with pytest.raises(InjectedQueryFailure):
        profile.run_profile(sim, motor, learner, sampler, output=output, root=root,
            contract=contract(profile), stop_updates=80, panel_interval=40,
            query_states=(40, 41), max_actions=5000)
    cursor = json.loads((output/'state.json').read_text())
    assert cursor['phase'] == 'learned_panel' and cursor['active_panel']['query_cursor'] == 1
    assert cursor['active_panel']['actual_optimizer_updates'] == 40
    assert cursor['training']['episodes'] == 1 and cursor['model']['updates'] == 40
    first_query_path = output/cursor['active_panel']['receipts'][0]['trace']
    first_query_sha = profile.onset.sha256(first_query_path)
    unresolved = [json.loads(path.read_text()) for path in (output/'physical-journal').glob('*.json')
        if not json.loads(path.read_text())['returned']]
    assert len(unresolved) == 1 and unresolved[0]['role'] == 'query' and unresolved[0]['maximum_steps'] == 10
    old_charge = profile.onset.charged(output)
    restored, new_motor, new_sampler = FakeSAC(), FakeMotor(), np.random.default_rng(999)
    resumed_sim = FakeSimulator(restored)
    result = profile.run_profile(resumed_sim, new_motor, restored, new_sampler, output=output, root=root,
        contract=contract(profile), resume=True, stop_updates=80, panel_interval=40,
        query_states=(40, 41), max_actions=5000)
    assert result['completed'] and restored.updates == 80 and len(restored.replay) == 504
    assert [panel['name'] for panel in result['panels']] == [
        'initial_native-000000', 'learned-000040', 'learned-000080', 'final_native-000080']
    assert sum(len(panel['receipts']) for panel in result['panels']) == 8
    assert profile.onset.sha256(first_query_path) == first_query_sha
    assert result['panels'][1]['receipts'][0]['trace'] == cursor['active_panel']['receipts'][0]['trace']
    assert resumed_sim.resets[0]['state'] == 41  # Resume the second query, not query40.
    assert result['total_action_charge'] > old_charge
    assert result['costs']['query']['uncertain_actions_ceiling'] == 10

    # Compare training behavior to uninterrupted execution from exactly the
    # same initial global/sampler state; query/control flow must not perturb it.
    np.random.set_state(initial_numpy); random.setstate(initial_python)
    control_root, control_output = paths(tmp_path, 'control')
    control_sac, control_motor = FakeSAC(), FakeMotor()
    control = profile.run_profile(FakeSimulator(control_sac), control_motor, control_sac,
        np.random.default_rng(0), output=control_output, root=control_root, contract=contract(profile),
        stop_updates=80, panel_interval=40, query_states=(40, 41), max_actions=5000)
    episodes, rows = training_rows(output, result)
    control_episodes, control_rows = training_rows(control_output, control)
    assert len(rows) == len(control_rows) == 4
    assert [episode['summary']['initial_state'] for episode in episodes] == [0, 1]
    for row, expected in zip(rows, control_rows, strict=True):
        assert np.array_equal(row['executed_noise'], expected['executed_noise'])
        assert np.array_equal(row['native_initial_noise'], expected['native_initial_noise'])
        assert np.array_equal(row['motor_observation']['state'], expected['motor_observation']['state'])
        assert row['behavior_optimizer_updates'] == expected['behavior_optimizer_updates']
        assert row['native_initial_noise'].shape == (10, 32)
        assert row['executed_actions'].shape == (2, 7) and row['primitive_steps'] == 2
        assert row['valid_primitive_mask'] == [True, True]+[False]*8
        assert row['data_role'] == 'target-online-baseline' and not row['source_assembly_admission']
    assert [row['behavior_optimizer_updates'] for row in rows] == [0, 0, 40, 40]
    assert result['model']['signature'] == restored.frozen_signature()
    assert profile.onset.charged(output) == profile.onset.charged(control_output)+10


def test_checkpoint_replacement_before_cursor_keeps_old_atomic_boundary(monkeypatch, tmp_path):
    from fluffy_bassoon.meta_rl import vla_target_online_dsrl_task57 as profile
    patch_rng(monkeypatch, profile)
    root, output = paths(tmp_path, 'atomic')
    real_write = profile.onset.write_json
    crashed = False
    def crash_before_new_cursor(path, value):
        nonlocal crashed
        if (Path(path).name == 'state.json' and not crashed
                and value.get('training', {}).get('episodes') == 1):
            crashed = True
            raise RuntimeError('injected after full-slot replacement, before state cursor')
        return real_write(path, value)
    monkeypatch.setattr(profile.onset, 'write_json', crash_before_new_cursor)
    learner, motor = FakeSAC(), FakeMotor()
    with pytest.raises(RuntimeError, match='before state cursor'):
        profile.run_profile(FakeSimulator(learner), motor, learner, np.random.default_rng(0),
            output=output, root=root, contract=contract(profile), stop_updates=40,
            panel_interval=40, query_states=(40, 41), max_actions=5000)
    cursor = json.loads((output/'state.json').read_text())
    assert cursor['training']['episodes'] == cursor['model']['updates'] == 0
    assert profile.onset.sha256(output/cursor['model']['path']) == cursor['model']['sha256']
    assert (output/'recovery-1.pkl').is_file()  # Newer orphan is not authoritative.
    initial_archive = output/'milestones/initial_native-000000-recovery.pkl'
    original_archive_sha = profile.onset.sha256(initial_archive)
    monkeypatch.setattr(profile.onset, 'write_json', real_write)
    recovered = FakeSAC()
    result = profile.run_profile(FakeSimulator(recovered), FakeMotor(), recovered,
        np.random.default_rng(777), output=output, root=root, contract=contract(profile),
        resume=True, stop_updates=40, panel_interval=40, query_states=(40, 41), max_actions=5000)
    assert result['completed'] and result['training']['episodes'] == 1 and recovered.updates == 40
    assert profile.onset.sha256(initial_archive) == original_archive_sha
    assert result['costs']['training']['primitive_steps'] == 8  # Both actual executions stay charged.
    assert result['training']['primitive_steps'] == 4  # Only the committed episode enters learning history.


def test_cycle_covers_and_repeats_all40_states_and_keeps_queries_disjoint(monkeypatch, tmp_path):
    from fluffy_bassoon.meta_rl import vla_target_online_dsrl_task57 as profile
    patch_rng(monkeypatch, profile)
    root, output = paths(tmp_path, 'coverage')
    learner = FakeSAC()
    result = profile.run_profile(FakeSimulator(learner), FakeMotor(), learner, np.random.default_rng(0),
        output=output, root=root, contract=contract(profile), stop_updates=1640, panel_interval=1640,
        query_states=(40, 41), max_actions=2000)
    assert result['completed'] and result['training']['episodes'] == 41
    assert len(result['training']['state_hashes']) == 40
    assert [receipt['initial_state'] for receipt in result['training']['receipts']] == list(range(40))+[0]
    assert set(result['query_state_hashes']) == {'40', '41'}
    assert all(receipt['initial_state'] >= 40 for panel in result['panels'] for receipt in panel['receipts'])


def test_postburst_thresholds_and_stop_are_optimizer_steps_not_actions(monkeypatch, tmp_path):
    from fluffy_bassoon.meta_rl import vla_target_online_dsrl_task57 as profile
    patch_rng(monkeypatch, profile)
    root, output = paths(tmp_path, 'bursts')
    learner = FakeSAC()
    # One3-chunk episode must retain all60 updates, crossing BOTH20 and40.
    result = profile.run_profile(FakeSimulator(learner, chunks_per_episode=3), FakeMotor(), learner,
        np.random.default_rng(0), output=output, root=root, contract=contract(profile),
        stop_updates=40, panel_interval=20, query_states=(40, 41), max_actions=5000)
    assert result['completed'] and result['target_optimizer_updates'] == 60
    assert result['training']['episodes'] == 1 and result['training']['chunks'] == 3
    assert result['training']['primitive_steps'] == 6
    learned = [panel for panel in result['panels'] if panel['kind'] == 'learned']
    assert [panel['threshold_updates'] for panel in learned] == [20, 40]
    assert [panel['actual_optimizer_updates'] for panel in learned] == [60, 60]
    assert all(len(panel['receipts']) == 2 for panel in learned)
    assert learner.update_calls == list(range(1, 61))
    assert result['panels'][-1]['kind'] == 'final_native'
    assert result['panels'][-1]['actual_optimizer_updates'] == 60
