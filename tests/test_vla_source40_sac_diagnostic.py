"""Source-bank competence contracts; execute on allocated Linux only."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

import fluffy_bassoon.meta_rl.vla_source40_sac_diagnostic as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path)


def alias_fixture(tmp_path):
    parent = tmp_path / 'source-validation'
    source = 'b' * 40
    index = dict(format='source40_frozen_ppo_validation_v1', completed=True,
        weights_unchanged=True, optimizer_updates=0, total_action_charge=12000,
        contract=dict(source_commit=source))
    index_path = Path(write(parent / 'experiment/index.json', index))
    clock = dict(worker_seconds=14400, attempts=[dict(status='finished', exit_code=0,
        charged_seconds=1800, reserved_seconds=3660)], contract=dict(operation='frozen_source_ppo_validation',
        run_id=parent.name, source_commit=source, maximum_total_actions=56000))
    clock_path = Path(write(parent / 'worker-time.json', clock))
    (parent / 'exit-code.txt').write_text('0\n')
    ledger = tmp_path / 'canonical.json'
    write(ledger, dict(campaign_sha256='campaign', reservations={parent.name: dict(
        maximum_total_actions=56000, source_commit=source, campaign_sha256='campaign')}))
    value = dict(format=module.ALIAS_FORMAT, run_id='diagnostic', source_commit='a' * 40,
        maximum_total_actions=20000, maximum_worker_seconds=3600, parent_output_root=str(parent),
        canonical_ppo_ledger_path=str(ledger), parent_terminal_index_sha256=module.file_sha256(index_path),
        parent_worker_time_sha256=module.file_sha256(clock_path), parent_total_action_charge=12000,
        parent_worker_charge_seconds=1800, original_action_ceiling=56000, original_worker_seconds=14400)
    path = tmp_path / 'alias.json'
    write(path, value)
    return path, value


def test_exact48_queries_use_same_cases_and_no_target_states():
    rows = module.query_plan()
    assert len(rows) == 48
    assert 48 * 410 == 19680 < module.MAX_ACTIONS == 20000
    for start in range(0, 48, 3):
        group = rows[start:start + 3]
        assert [row['method'] for row in group] == ['native', 'mean', 'sample']
        assert group[0]['case'] == group[1]['case'] == group[2]['case']
        assert group[0]['case']['initial_state'] in (40, 41)
        assert group[0]['task_key'] in module.KEYS


def test_alias_cannot_spend_more_actions_or_time_or_duplicate_original_hold(tmp_path):
    path, value = alias_fixture(tmp_path)
    assert module.load_alias(path, module.file_sha256(path), run_id='diagnostic', source_commit='a' * 40) == value
    for field, changed in [('maximum_total_actions', 20001), ('maximum_worker_seconds', 14400),
                           ('parent_total_action_charge', 36001), ('source_commit', 'c' * 40)]:
        bad = dict(value, **{field: changed})
        write(path, bad)
        with pytest.raises(ValueError):
            module.load_alias(path, module.file_sha256(path), run_id='diagnostic', source_commit='a' * 40)
    write(path, value)
    ledger = Path(value['canonical_ppo_ledger_path'])
    v = module.read(ledger)
    v['reservations']['diagnostic'] = dict(maximum_total_actions=20000)
    write(ledger, v)
    with pytest.raises(ValueError, match='duplicate reservation'):
        module.load_alias(path, module.file_sha256(path), run_id='diagnostic', source_commit='a' * 40)


def test_alias_requires_terminal_completed_parent_not_a_live_or_partial_validation(tmp_path):
    path, value = alias_fixture(tmp_path)
    parent_index = Path(value['parent_output_root']) / 'experiment/index.json'
    changed = module.read(parent_index)
    changed['completed'] = False
    write(parent_index, changed)
    value['parent_terminal_index_sha256'] = module.file_sha256(parent_index)
    write(path, value)
    with pytest.raises(ValueError, match='terminal successful'):
        module.load_alias(path, module.file_sha256(path), run_id='diagnostic', source_commit='a' * 40)


def bank_fixture(tmp_path):
    tasks = [dict(task_key=f'{suite}:{number:02d}', suite=suite, task_id=number,
        bddl_sha256=f'bddl-{suite}-{number}') for suite in module.SOURCE_SUITES for number in range(10)]
    schedule = dict(tasks=tasks, schedule_sha256='schedule')
    records = {}
    for key in module.KEYS:
        run = key.replace(':', '-') + '-train'
        root = tmp_path / 'outputs' / run / 'experiment'
        root.mkdir(parents=True)
        task = next(row for row in tasks if row['task_key'] == key)
        contract = dict(source_commit='a' * 40, source_only=True, source_data_from_targets=False,
            partition='train', motor=dict(receipt_sha256=module.PI05_RECEIPT), horizon=400, chunk=10,
            task=task, feature_contract=dict(kind='qualified-prefix'))
        write(root / 'contract.json', contract)
        checkpoint = root / 'case-00000-source.pkl'
        checkpoint.write_bytes(b'full-source-recovery')
        index = dict(completed=True, collection_qualified=True, source_only=True, partition='train',
            task_key=key, source_contract_sha256=module.content_hash(contract), schedule_sha256='schedule',
            publications=[dict(status='paired_ad_dpt', through_case=0, source_checkpoint=checkpoint.name,
                source_checkpoint_sha256=module.file_sha256(checkpoint), episodes=4, transitions=600)],
            episode_receipts=[dict(initial_state=0) for _ in range(4)], chunks=600,
            source_updates=2000, total_action_budget_charge=6000)
        write(root / 'index.json', index)
        records[key + '/train'] = dict(run_id=run, source_commit='a' * 40)
        # A tempting validation entry must never be selected.
        records[key + '/validation'] = dict(run_id='wrong-validation-actor', source_commit='a' * 40)
    ledger = tmp_path / 'source-ledger.json'
    write(ledger, dict(reservations=records))
    return ledger, schedule


def test_source_bank_maps_each_goal_to_its_own_train_actor_and_binds_hashes(tmp_path):
    ledger, schedule = bank_fixture(tmp_path)
    bank = module.source_bank(ledger, tmp_path / 'outputs', schedule)
    assert list(bank) == module.KEYS
    assert len({spec['checkpoint_path'] for spec in bank.values()}) == 8
    assert all(spec['canonical_reservation']['run_id'].endswith('-train') for spec in bank.values())
    Path(bank[module.KEYS[0]]['checkpoint_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='matching canonical'):
        module.source_bank(ledger, tmp_path / 'outputs', schedule)


def test_mean_and_sample_call_official_choice_then_execute_bounded_tied_noise(tmp_path, monkeypatch):
    class Learner:
        updates = 2000
        def __init__(self):
            self.modes = []
        def action(self, observation, deterministic=False):
            self.modes.append(deterministic)
            return np.linspace(-2, 2, 32, dtype=np.float32)
    class Motor:
        horizon = 10
        def infer(self, observation, instruction, noise):
            self.last_noise = noise.copy()
            return np.zeros((10, 7), np.float32)
    class Simulator:
        metadata = dict(settle_steps=10, horizon=400, instruction='source task')
        def reset(self, state, seed):
            return dict(observation=dict(image=np.zeros((8, 8, 3), np.uint8), state=np.zeros(8, np.float32)),
                initial_observation_sha256='obs', initial_state_vector_sha256='state')
        def step(self, actions):
            return dict(observation=dict(image=np.zeros((8, 8, 3), np.uint8), state=np.zeros(8, np.float32)),
                primitive_steps=10, executed_actions=actions, rewards=[0.] * 10, done=True, success=False)
    monkeypatch.setattr(module.FrozenTeacher, 'set_episode_seed', lambda self, seed: None)
    learner = Learner()
    for deterministic in (True, False):
        controller = module.FrozenTeacher(learner, deterministic=deterministic)
        motor = Motor()
        ledger = module.CostLedger(tmp_path / str(deterministic), max_primitive_steps=20000,
            max_settling_steps=20000, max_total_actions=20000)
        rows, receipt, _ = module.rollout_episode(Simulator(), motor, controller,
            case=module.SourceCase(**module.query_plan()[0]['case']), method='frozen_source_sac', role='query',
            ordinal=0, episode_id='test', ledger=ledger, reset_motor_rng=lambda seed: None)
        assert learner.modes[-1] is deterministic
        assert np.max(np.abs(rows[0]['proposed_noise'])) == 2
        assert np.max(np.abs(rows[0]['executed_noise'])) == 1
        np.testing.assert_array_equal(motor.last_noise, np.repeat(rows[0]['executed_noise'][None], 10, axis=0))
        assert receipt['completed'] and not controller.pending
