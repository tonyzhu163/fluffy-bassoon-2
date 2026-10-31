"""Paired support sharing, reset interventions and complete-episode recovery."""
from copy import deepcopy
import json
import pickle

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_source40_evaluate import TargetCase
from fluffy_bassoon.meta_rl.vla_source40_memory_replication import (
    MAX_ACTIONS, PHASES, POLICY_SHA256, paired_replication, replication_plan, same_public)
from test_vla_source40_evaluate import Controller, Motor, Policy, Simulator


class ReplicationSimulator(Simulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata['task_id'] = 0

    def reset(self, state, seed):
        value = super().reset(state, seed)
        value['initial_state_vector_sha256'] = f'nominal-state-{state}'
        return value


def cases(count=1):
    return [TargetCase('libero90', f'libero90:00/state-{state}/seed-123', 'libero_90', 0, state, 123)
            for state in range(1, count + 1)]


def run(output, *, sim=None, motor=None, policy=None, resume=False, case_count=1, cap=1000):
    sim, motor, policy = sim or ReplicationSimulator(), motor or Motor(), policy or Policy()
    controllers = []
    def factory():
        value = Controller(dict(policy=policy, mode='full'), sim)
        controllers.append(value)
        return value
    result = paired_replication(sim, motor, policy, cases=cases(case_count), output=output,
        source_provenance=dict(qualification_subset=True), policy_sha256=POLICY_SHA256,
        max_total_actions=cap, resume=resume, controller_factory=factory,
        motor_rng_state=lambda: dict(key='saved motor RNG'))
    return result, sim, motor, controllers


def test_plan_predeclares_all_six_states_without_outcome_filtering():
    plan = replication_plan()
    assert [case.initial_state for case in plan] == [1, 2, 3, 4, 5, 6]
    assert {case.task_id for case in plan} == {0}
    assert {case.scene_seed for case in plan} == {123}
    assert len({case.case_id for case in plan}) == 6
    assert 6 * 3 * 410 == 7380 < MAX_ACTIONS == 8000


def test_shared_support_executes_once_then_full_and_reset_use_same_fixed_weights(tmp_path):
    result, sim, _, controllers = run(tmp_path)
    assert sim.episodes == 3 and [row['phase'] for row in result['episodes']] == list(PHASES)
    assert controllers[1].acts[0][1] == 2  # Actual terminal support feedback reaches full query.
    assert controllers[2].acts[0][1] == 0  # Same trained policy, erased support context.
    assert all(value.policy is controllers[0].policy for value in controllers)
    assert result['charged_total_actions'] == 3 * (20 + 3) == 69
    assert result['support_counted_once'] and result['counterfactual_paired_probe']
    assert result['deployment_optimizer_updates'] == 0 and result['weights_unchanged']
    full, reset = result['episodes'][1:]
    assert full['episode_random_seed'] == reset['episode_random_seed']
    assert full['initial_public_arrays_sha256'] == reset['initial_public_arrays_sha256']
    assert result['comparisons'][0]['initial_public_arrays_exactly_equal']


def test_support_snapshot_and_raw_rpc_data_survive_after_case_completion(tmp_path):
    result, _, _, _ = run(tmp_path)
    support = result['episodes'][0]
    with (tmp_path / support['context']).open('rb') as stream:
        saved = pickle.load(stream)
    assert len(saved['controller_after_episode']['feedback']) == 2
    assert saved['motor_rng'] == dict(key='saved motor RNG')
    assert saved['actual_episode_random_seed'] == support['episode_random_seed']
    assert saved['policy_sha256'] == POLICY_SHA256
    assert list((tmp_path / 'raw-rpc').glob('*-inference.pkl.gz'))
    assert list((tmp_path / 'raw-rpc').glob('*-outcome.pkl.gz'))
    with (tmp_path / 'paired-recovery.pkl').open('rb') as stream:
        recovery = pickle.load(stream)
    assert recovery['case_index'] == 1 and recovery['phase'] == 0
    assert recovery['support_context'] is None
    assert set(recovery['global_rng']) == {'python', 'numpy', 'torch', 'cuda'}


def test_interrupted_query_resumes_from_support_boundary_without_repeating_support(tmp_path):
    sim, motor, policy = ReplicationSimulator(fail_on_step=3), Motor(), Policy()
    with pytest.raises(RuntimeError, match='injected RPC failure'):
        run(tmp_path, sim=sim, motor=motor, policy=policy)
    with (tmp_path / 'paired-recovery.pkl').open('rb') as stream:
        saved = pickle.load(stream)
    assert saved['case_index'] == 0 and saved['phase'] == 1
    assert len(saved['records']) == 1 and saved['records'][0]['phase'] == 'shared_support'
    result, _, _, controllers = run(tmp_path, sim=sim, motor=motor, policy=policy, resume=True)
    assert sim.episodes == 4  # One support, failed query, two completed query branches.
    assert [row['phase'] for row in result['episodes']].count('shared_support') == 1
    assert controllers[0].acts[0][1] == 2
    assert result['charged_total_actions'] == 82  #69 committed +3 settle +10 uncertain.
    assert result['interrupted_or_uncertain_action_charge'] == 13
    assert list((tmp_path / 'raw-rpc').glob('*-inference.pkl.gz'))


def test_joint_budget_stops_before_an_unfunded_complete_case(tmp_path):
    result, sim, _, _ = run(tmp_path, case_count=2, cap=69)
    assert not result['completed'] and len(result['comparisons']) == 1
    assert result['stop_reason'] == 'remaining_complete_episode_reservation'
    assert sim.episodes == 3 and result['charged_total_actions'] == 69
    assert len(result['contract']['cases']) == 2  # Unvisited cases are never silently filtered.


def test_actual_public_array_comparison_checks_dtype_and_values():
    left = dict(image=np.zeros((2, 2, 3), np.uint8), wrist=np.ones((2, 2, 3), np.uint8), state=np.zeros(8, np.float32))
    assert same_public(left, deepcopy(left))
    changed = deepcopy(left)
    changed['wrist'][0, 0, 0] = 2
    assert not same_public(left, changed)
    changed = deepcopy(left)
    changed['state'] = changed['state'].astype(np.float64)
    assert not same_public(left, changed)


def test_changed_context_file_is_rejected_before_resume_interactions(tmp_path):
    result, sim, motor, _ = run(tmp_path)
    context = tmp_path / result['episodes'][0]['context']
    context.write_bytes(b'changed saved support context')
    before = sim.episodes
    with pytest.raises(ValueError, match='context checksum'):
        run(tmp_path, sim=sim, motor=motor, resume=True)
    assert sim.episodes == before
