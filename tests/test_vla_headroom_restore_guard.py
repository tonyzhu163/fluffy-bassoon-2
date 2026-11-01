"""Allocated-Linux regressions using real Flax bytes, without any NN or SGD."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
from types import SimpleNamespace

import pytest
if sys.platform != 'linux':
    pytest.skip('Real serialization/RNG regressions require allocated Linux', allow_module_level=True)
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from fluffy_bassoon.meta_rl import vla_headroom_restore_guard as guard
from fluffy_bassoon.meta_rl import vla_headroom_execution_migration as migration
from fluffy_bassoon.meta_rl import vla_target_online_dsrl as onset
from fluffy_bassoon.meta_rl.vla_source40_corpus import save_motor_rng


def tree(reverse=False):
    result = dict(actor=dict(params=dict(z=np.array([1., 2.], np.float32), a=np.array([3.], np.float32)),
        opt_state=dict(mu=np.array([.2, .3], np.float32), count=np.array(17, np.int32))),
        critic=dict(params=dict(q=np.array([4., 5.], np.float32))),
        target_critic_params=dict(q=np.array([6., 7.], np.float32)),
        temp=dict(params=dict(log_temp=np.array(.1, np.float32))))
    def reorder(value):
        return {k: reorder(v) for k, v in reversed(list(value.items()))} if isinstance(value, dict) else value
    return reorder(result) if reverse else result


class SerializedLearner:
    """The same Flax template restoration convention as OfficialSAC.restore."""
    def __init__(self, fault=None):
        self.agent = SimpleNamespace(_save_dict=tree(True), _rng=jax.random.PRNGKey(0))
        self.replay, self.updates, self.fault = [], 0, fault

    def restore(self, path, sampler, contract):
        payload = pickle.loads(path.read_bytes())
        assert payload['contract'] == contract
        self.agent._save_dict = serialization.from_bytes(self.agent._save_dict, payload['agent'])
        self.agent._rng = jnp.asarray(payload['agent_rng'])
        self.replay, self.updates = copy.deepcopy(payload['replay']), payload['updates']
        sampler.bit_generator.state = copy.deepcopy(payload['sampler'])
        np.random.set_state(payload['numpy_rng']); random.setstate(payload['python_rng'])
        saved = copy.deepcopy(payload['progress'])
        state = self.agent._save_dict
        if self.fault == 'actor': state['actor']['params']['z'] = state['actor']['params']['z'] + 1
        elif self.fault == 'adam': state['actor']['opt_state']['mu'] = state['actor']['opt_state']['mu'] + 1
        elif self.fault == 'target': state['target_critic_params']['q'] = state['target_critic_params']['q'] + 1
        elif self.fault == 'dtype': state['actor']['params']['z'] = state['actor']['params']['z'].astype(np.float64)
        elif self.fault == 'shape': state['actor']['params']['z'] = state['actor']['params']['z'][:1]
        elif self.fault == 'updates': self.updates += 1
        elif self.fault == 'training': saved['training']['episodes'] += 1
        elif self.fault == 'replay_length': self.replay.append({})
        elif self.fault == 'replay_values': self.replay[0]['reward'][0] += 1
        elif self.fault == 'sampler': sampler.integers(0, 100)
        elif self.fault == 'agent_rng': self.agent._rng = jax.random.PRNGKey(12)
        return saved


def fixture(tmp_path):
    contract = dict(source_commit=migration.LEGACY_SOURCE, fixed='unchanged science')
    training = dict(episodes=1720, receipts=[], state_hashes={'0': 'fixed'})
    motor = SimpleNamespace(policy=SimpleNamespace(_rng=jax.random.key(3)))
    payload = dict(agent=serialization.to_bytes(tree()), agent_rng=np.array([2, 3], np.uint32),
        replay=[dict(reward=np.array([0.], np.float32))], updates=301700,
        sampler=np.random.default_rng(4).bit_generator.state, numpy_rng=np.random.get_state(),
        python_rng=random.getstate(), contract=contract,
        progress=dict(training=training, motor_rng=save_motor_rng(motor.policy._rng)))
    path = tmp_path/'recovery.pkl'; path.write_bytes(pickle.dumps(payload, protocol=5))
    progress = dict(training=copy.deepcopy(training), model=dict(path=path.name,
        sha256=onset.sha256(path), signature=hashlib.sha256(payload['agent']).hexdigest(),
        updates=301700, replay_rows=1))
    return path, progress, contract, motor


def test_real_flax_order_change_accepts_exact_leaves_without_rewriting_inputs(tmp_path):
    path, progress, contract, motor = fixture(tmp_path)
    before = path.read_bytes(); original = copy.deepcopy(progress)
    learner, sampler = SerializedLearner(), np.random.default_rng(999)
    saved, proof = guard.restore_checked(learner, sampler, motor, path, progress, contract)
    assert proof['restored_raw_signature'] != proof['legacy_signature']
    assert proof['exact_leaf_differences'] == 0 and all(proof['non_network_checks'].values())
    assert proof['optimizer_updates_during_restore'] == 0 and learner.updates == 301700
    assert path.read_bytes() == before and progress == original and saved['training'] == progress['training']
    payload = pickle.loads(before)
    expected = np.random.default_rng(); expected.bit_generator.state = payload['sampler']
    assert sampler.integers(0, 10000) == expected.integers(0, 10000)


@pytest.mark.parametrize('fault', ['actor', 'adam', 'target', 'dtype', 'shape', 'updates',
    'training', 'replay_length', 'replay_values', 'sampler', 'agent_rng'])
def test_changed_state_counters_replay_or_rng_is_rejected(tmp_path, fault):
    path, progress, contract, motor = fixture(tmp_path)
    with pytest.raises(ValueError, match='differ'):
        guard.restore_checked(SerializedLearner(fault), np.random.default_rng(0), motor, path, progress, contract)


@pytest.mark.parametrize('fault', ['checkpoint', 'signature'])
def test_legacy_byte_bindings_are_still_required(tmp_path, fault):
    path, progress, contract, motor = fixture(tmp_path)
    progress['model']['sha256' if fault == 'checkpoint' else 'signature'] = '0'*64
    with pytest.raises(ValueError, match='differs|differ'):
        guard.restore_checked(SerializedLearner(), np.random.default_rng(0), motor, path, progress, contract)


def test_only_exact_proven_panel_signature_pair_is_accepted():
    proof = dict(legacy_signature='stored', restored_raw_signature='restored')
    assert guard.panel_signature_matches('restored', 'stored', proof)
    assert guard.panel_signature_matches('restored', 'restored', proof)
    assert not guard.panel_signature_matches('restored', 'unrelated', proof)
    assert not guard.panel_signature_matches('changed', 'stored', proof)
    assert not guard.panel_signature_matches('restored', 'stored', None)


def migration_fixture(tmp_path, monkeypatch):
    from fluffy_bassoon.meta_rl import vla_target_online_dsrl_headroom as phase
    from fluffy_bassoon.meta_rl.vla_source40_schedule import content_hash
    root = tmp_path/'phase'; case = root/'experiment/case-learn-44'; case.mkdir(parents=True)
    screen, runtime = '7'*40, 'c'*40
    plan = phase.plan(screen)
    refs = {}
    def write(name, path, value):
        onset.write_json(path, value); refs[name] = dict(path=str(path), sha256=onset.sha256(path))
    write('screen_plan', root/'experiment/plan.json', plan)
    write('selection', root/'experiment/frozen-selection.json', dict(task_id=44, optimizer_updates=0,
        selected_before_any_sgd=True, plan_sha256=refs['screen_plan']['sha256']))
    write('screen_admission', root/'screen-to-learning-source.json',
        dict(screen_source=screen, learning_source=migration.LEGACY_SOURCE))
    write('learning_contract', case/'contract.json', dict(source_commit=migration.LEGACY_SOURCE,
        selected_task=44, frozen_selection_sha256=refs['selection']['sha256']))
    checkpoint = root/'original.pkl'; checkpoint.write_bytes(b'immutable closed full state')
    refs['checkpoint'] = dict(path=str(checkpoint), sha256=onset.sha256(checkpoint))
    write('cursor', case/'state.json', dict(model=dict(sha256=refs['checkpoint']['sha256'],
        signature='f'*64, updates=301700, replay_rows=15579), training=dict(episodes=1720)))
    prior = dict(worker_seconds=57600, contract=dict(plan, run_id='same-phase'), attempts=[
        dict(path=str(root/'attempts/old'), status='finished', charged_seconds=28217,
            reserved_seconds=28217, started_unix=1., ended_unix=28158.)])
    write('prior_worker_clock', root/'prior-clock.json', prior)
    onset.write_json(root/'worker-time.json', prior)
    write('diagnosis', root/'diagnosis.json', dict(
        classification='identical_saved_state_raw_representation_differs',
        qualified_learner_source=migration.LEGACY_SOURCE, legacy_saved_signature_matches=True,
        canonical_full_state_exact=True, exact_difference_count=0, all_inputs_unchanged=True,
        RNG={key: True for key in ('agent_exact', 'sampler_exact', 'numpy_global_exact',
            'python_global_exact', 'motor_key_roundtrip_exact')}, replay=dict(exact=True),
        legacy_cursor_signature='f'*64, saved_agent_bytes_SHA='f'*64,
        canonical_expected_SHA='e'*64, canonical_actual_SHA='e'*64, new_SGD=0, new_robot_actions=0,
        checkpoint=refs['checkpoint'], cursor=refs['cursor'],
        original_guard_predicates=dict(training_equal=True, updates_equal=True,
            replay_rows_equal=True, raw_restored_signature_equal=False)))
    for name in ('CHECKPOINT', 'CURSOR', 'CONTRACT'):
        monkeypatch.setattr(migration, name+'_SHA', refs[dict(CONTRACT='learning_contract').get(name, name.lower())]['sha256'])
    value = dict(format='pi05_headroom_exact_restore_execution_migration_v1', admitted=True,
        legacy_execution_source=migration.LEGACY_SOURCE, scientific_trainer_source=migration.LEGACY_SOURCE,
        execution_source=runtime, phase_root=str(root),
        allowed_change='Exact restore identity; MsgPack mapping order only', scientific_contract_unchanged=True,
        original_worker_contract_sha256=content_hash(prior['contract']), **refs)
    path = root/'migration.json'; onset.write_json(path, value)
    return root, screen, runtime, path, onset.sha256(path), value


def test_explicit_execution_migration_preserves_science_and_cumulative_clock(tmp_path, monkeypatch):
    from fluffy_bassoon.meta_rl import vla_target_online_dsrl_headroom as phase
    root, screen, runtime, path, digest, _ = migration_fixture(tmp_path, monkeypatch)
    original = json.loads((root/'worker-time.json').read_text())
    phase.prepare_attempt(root, runtime, 'same-phase', slice_seconds=600, started_unix=30000.,
        resume=True, stage='learn', screen_source=screen,
        execution_migration=path, execution_migration_sha256=digest)
    current = json.loads((root/'worker-time.json').read_text())
    assert current['contract'] == original['contract'] and current['attempts'][:-1] == original['attempts']
    assert current['worker_seconds'] == 57600 and current['attempts'][-1]['reserved_seconds'] == 660
    assert migration.validate_migration(root, runtime, path, digest, stage='run', screen_source=screen) == migration.LEGACY_SOURCE


@pytest.mark.parametrize('fault', ['runtime_source', 'science_source', 'clock_prefix', 'cursor'])
def test_arbitrary_source_alias_or_ancestry_drift_is_rejected(tmp_path, monkeypatch, fault):
    root, screen, runtime, path, digest, value = migration_fixture(tmp_path, monkeypatch)
    if fault == 'runtime_source': runtime = 'd'*40
    elif fault == 'science_source':
        value['scientific_trainer_source'] = 'd'*40
        onset.write_json(path, value); digest = onset.sha256(path)
    elif fault == 'clock_prefix':
        clock = json.loads((root/'worker-time.json').read_text()); clock['attempts'][0]['charged_seconds'] -= 1
        onset.write_json(root/'worker-time.json', clock)
    else:
        cursor = root/'experiment/case-learn-44/state.json'
        # Cursor reference is an immutable saved copy, as in actual production.
        original = root/'saved-cursor.json'; original.write_bytes(cursor.read_bytes())
        value['cursor']['path'] = str(original); onset.write_json(path, value); digest = onset.sha256(path)
        cursor.write_text('{}\n')
    with pytest.raises(ValueError):
        migration.validate_migration(root, runtime, path, digest, stage='begin', screen_source=screen)
