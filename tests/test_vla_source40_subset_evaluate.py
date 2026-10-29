"""Predeclared target grid, joint budget and source-SAC observation routing."""
import json

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_source40_evaluate import CostLedger, TargetCase, evaluate_case
from fluffy_bassoon.meta_rl.vla_source40_subset_evaluate import comparison_plan, charged_actions
from test_vla_source40_evaluate import Simulator, Motor, Policy


def test_primary_grid_compares_ad_dpt_and_memory_lesions_in_accounted_reserve():
    plan = comparison_plan()
    primary = [row for row in plan if row['primary']]
    assert len(primary) == 28
    assert sum(3 * 410 for row in primary) == 34440 < 35046
    assert {row['case']['task_id'] for row in primary} == {0, 1}
    assert {row['case']['initial_state'] for row in primary} == {0}
    assert {row['case']['scene_seed'] for row in primary} == {123}
    assert {row['updates'] for row in primary if row['method'] == 'ad_full'} == {25, 100, 500}
    assert {row['method'] for row in primary if row['method'].startswith('dpt')} == {'dpt_full','dpt_static','dpt_episode_reset'}
    assert [(row['case']['task_id'], row['updates']) for row in primary if row['method'] == 'ad_episode_reset'] == [(0, 500),(1,500)]
    assert {row['method'] for row in primary if row['updates'] is None} == {'native','tied_gaussian','zero','frozen_source_sac'}
    assert all(row['primary'] for row in plan[:28]) and all(not row['primary'] for row in plan[28:])


def test_zero_noise_control_uses_no_learned_policy(tmp_path):
    import gzip
    import pickle
    result = evaluate_case(Simulator(), Motor(), {},
        case=TargetCase('libero90', 'declared', 'libero_90', 57, 0, 123), output=tmp_path,
        provenance=dict(group_maximum_total_actions=100), max_primitive_steps=100,
        max_settling_steps=100, support_budgets=(0,1), methods=('zero',))
    assert len(result['episodes']) == 3 and result['weights_unchanged']
    for receipt in result['episodes']:
        with gzip.open(tmp_path / receipt['trace'], 'rb') as stream:
            rows = pickle.load(stream)
        assert all(np.array_equal(row['native_initial_noise'],np.zeros((10,32),np.float32)) for row in rows)


def test_joint_physical_cap_includes_settling_and_unresolved_execution(tmp_path):
    ledger = CostLedger(tmp_path / 'group-000/physical-journal', max_primitive_steps=100,
        max_settling_steps=100, max_total_actions=12)
    reset = ledger.intent(kind='reset', role='query', episode_id='q0', maximum_steps=3)
    ledger.returned(reset, 3)
    with pytest.raises(RuntimeError, match='joint'):
        ledger.intent(kind='step', role='query', episode_id='q0', maximum_steps=10)
    event = ledger.intent(kind='step', role='query', episode_id='q0', maximum_steps=9)
    assert charged_actions(tmp_path) == 12
    ledger.returned(event, 4)
    assert charged_actions(tmp_path) == 7
    other = CostLedger(tmp_path / 'group-001/physical-journal', max_primitive_steps=100, max_settling_steps=100)
    other.intent(kind='reset', role='support', episode_id='s0', maximum_steps=3)
    assert charged_actions(tmp_path) == 10


def test_frozen_source_sac_uses_public_images_and_state_without_history_update(tmp_path):
    sim, motor = Simulator(), Motor()
    class SourceSAC:
        public_observation_input = True
        policy = Policy()
        def __init__(self):
            self.actions, self.feedback, self.pending = [], [], False
        def set_episode_seed(self, seed):
            self.seed = seed
        def act(self, observation):
            assert isinstance(observation, dict) and 'step' in observation
            assert not self.pending
            self.actions.append(observation['step'])
            self.pending = True
            return np.zeros(32, np.float32)
        def observe(self, **outcome):
            assert self.pending
            self.feedback.append(outcome)
            self.pending = False
        def snapshot(self):
            return dict(pending=self.pending)
    controller = SourceSAC()
    result = evaluate_case(sim, motor, dict(frozen_source_sac=dict(policy=controller.policy)),
        case=TargetCase('libero90', 'declared', 'libero_90', 57, 0, 123), output=tmp_path,
        provenance=dict(group_maximum_total_actions=100), max_primitive_steps=100,
        max_settling_steps=100, support_budgets=(1,), methods=('frozen_source_sac',),
        controller_factory=lambda spec: controller)
    assert controller.actions == [0, 1, 0, 1] and len(controller.feedback) == 4
    assert result['weights_unchanged'] and len(result['episodes']) == 2
    assert result['trials'][0]['support_budget'] == 1
