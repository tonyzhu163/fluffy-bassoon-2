"""Six-case canonical baseline preserves onset/recovery and physical budgets."""
import gzip
import json
import pickle

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import vla_target_online_dsrl as onset
from fluffy_bassoon.meta_rl.vla_target_online_dsrl_six import (
    TASKS, MILESTONES, CASE_CAP, TOTAL_CAP, RECIPE, plan, validate_completed)
from test_vla_target_online_dsrl import Learner, Motor, Simulator


@pytest.fixture(autouse=True)
def fake_jax(monkeypatch):
    monkeypatch.setattr(onset, '_key', int)
    monkeypatch.setattr(onset, '_save_motor', lambda motor: motor.policy._rng)
    monkeypatch.setattr(onset, '_restore_motor', lambda motor, value: setattr(motor.policy, '_rng', value))


class PrefixMotor(Motor):
    def __init__(self):
        super().__init__()
        self.feature_encoder = lambda observation, instruction: np.full(4104, .125, np.float32)


def run(tmp_path, *, simulator=None, resume=False):
    learner, motor = Learner(), PrefixMotor()
    contract = dict(case=onset.case(0), fixed='six-case-test', target_only=True)
    result = onset.run_case(simulator or Simulator(), motor, learner, np.random.default_rng(0),
        target=onset.case(0), output=tmp_path / 'case-00', root=tmp_path, contract=contract,
        resume=resume, milestones=MILESTONES, case_cap=CASE_CAP, total_cap=TOTAL_CAP,
        record_features=True)
    return result, learner, contract


def test_predeclared_six_cases_fit_action_time_contract_without_implicit_source_admission():
    value = plan()
    assert TASKS == (0, 10, 20, 30, 40, 50)
    assert [row['task_id'] for row in value['cases']] == list(TASKS)
    assert all(row['initial_state'] == 0 and row['scene_seed'] == 123 for row in value['cases'])
    assert [row['training_episodes'] for row in value['milestones']] == [0, 1, 4, 8, 13, 16]
    assert value['planning']['nominal_complete_matrix_actions'] == 54180
    assert value['planning']['sum_individual_case_caps'] == 57000 <= TOTAL_CAP == 64000
    assert value['planning']['training_episodes'] == 96 and value['planning']['query_episodes'] == 36
    assert value['maximum_gpu_seconds'] == 7200
    assert not value['source_assembly_admission'] and value['target_only']
    assert RECIPE['batch_size'] == 256 and RECIPE['update_to_data'] == 20
    assert 'NOT actor input' in RECIPE['interface']


def test_six_milestones_clear_canonical_warmup_and_exclude_queries_from_replay(tmp_path):
    result, learner, contract = run(tmp_path)
    assert result['completed'] and result['training_episodes'] == 16
    assert learner.updates == 3200 and len(learner.replay) == result['training_chunks'] == 640
    assert [row['target_optimizer_updates'] for row in result['query_receipts']] == [0, 0, 0, 0, 800, 3200]
    assert [row['actual_training_episodes'] for row in result['query_receipts']] == [0, 1, 4, 8, 13, 16]
    assert result['costs']['training']['known_actions'] == 6560
    assert result['costs']['query']['known_actions'] == 2460
    assert result['total_action_charge'] == 9020 and result['maximum_total_actions'] == 9500
    assert all(row['optimizer_updates_during_query'] == 0 and not row['query_data_added_to_replay']
        for row in result['query_receipts'])
    validate_completed(result, contract, tmp_path / 'case-00')


def test_recorded_prefix_features_do_not_replace_authors_pixel_actor_observations(tmp_path):
    result, learner, _ = run(tmp_path)
    assert set(learner.replay[0]['observations']) == {'pixels', 'state'}
    assert learner.replay[0]['observations']['pixels'].shape == (64, 64, 3, 1)
    for receipt in (result['training_receipts'][0], result['query_receipts'][0]):
        with gzip.open(tmp_path / 'case-00' / receipt['trace'], 'rb') as stream:
            saved = pickle.load(stream)
        assert all(row['public_features'].shape == (4104,) for row in saved['rows'])
        assert all('motor_observation' in row and 'motor_next_observation' in row for row in saved['rows'])


def test_interrupted_trial_retains_old_physical_cost_and_exact_milestone_cursor(tmp_path):
    with pytest.raises(RuntimeError, match='interrupted actual RPC'):
        run(tmp_path, simulator=Simulator(fail_on=45))
    first = json.loads((tmp_path / 'case-00/index.json').read_text())
    assert first['next_milestone'] == 1 and first['training_episodes'] == 0
    result, learner, _ = run(tmp_path, resume=True)
    assert result['completed'] and learner.updates == 3200
    assert len(result['query_receipts']) == 6 and len(learner.replay) == 640
    assert result['total_action_charge'] == 9080
    assert result['costs']['training']['uncertain_actions_ceiling'] == 10


def test_completed_receipt_requires_pinned_full_recovery_hash(tmp_path):
    result, _, contract = run(tmp_path)
    changed = dict(result, latest_recovery_sha256='0' * 64)
    with pytest.raises(ValueError, match='immutable contract'):
        validate_completed(changed, contract, tmp_path / 'case-00')
