"""Actual support/query semantics for the fixed two-source-goal comparison."""
from copy import deepcopy
import json
import pickle

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_history import file_sha256
from fluffy_bassoon.meta_rl.vla_source40_schedule import content_hash
from fluffy_bassoon.meta_rl.vla_source40_subset_evaluate import charged_actions, comparison_plan
import fluffy_bassoon.meta_rl.vla_source40_two_goal_evaluate as module
from test_vla_source40_evaluate import Controller, Motor, Policy, Simulator


class TargetSimulator(Simulator):
    def __init__(self, task, **kwargs):
        super().__init__(**kwargs)
        self.metadata['task_id'] = task
        self.closed = False

    def reset(self, state, seed):
        result = super().reset(state, seed)
        result['initial_state_vector_sha256'] = content_hash(dict(task=self.metadata['task_id'], state=state))
        result['initial_observation_sha256'] = f'initial-public-{self.metadata["task_id"]}'
        result['observation'].update(rgb=np.zeros((2, 2, 3), np.uint8), wrist=np.zeros((2, 2, 3), np.uint8))
        return result

    def step(self, actions):
        result = super().step(actions)
        result['observation'].update(rgb=np.zeros((2, 2, 3), np.uint8), wrist=np.zeros((2, 2, 3), np.uint8))
        return result

    def close(self):
        self.closed = True


def candidates():
    values = {}
    for objective in ('ad', 'dpt'):
        full, static = Policy(), Policy()
        values[objective + '_full'] = dict(policy=full, mode='full', sha256=objective + '-full')
        values[objective + '_static'] = dict(policy=static, mode='static', sha256=objective + '-static')
        values[objective + '_episode_reset'] = dict(policy=full, mode='episode', sha256=objective + '-full')
    return values


def contract(plan=None, cap=1000):
    return dict(format=module.FORMAT, groups=module.two_goal_plan() if plan is None else plan,
        maximum_total_actions=cap, maximum_seconds_per_invocation=60.)


def run(output, *, plan=None, cap=1000, sims=None, resume=False, sim_factory=None):
    controllers, servers = [], []
    def server_factory(case):
        sim = sims.pop(0) if sims else sim_factory(case) if sim_factory else TargetSimulator(case.task_id)
        servers.append(sim)
        return sim
    def factory(spec, sim):
        value = Controller(spec, sim)
        controllers.append(value)
        return value
    result = module.evaluate_groups(Motor(), candidates(), server_factory=server_factory,
        output=output, contract=contract(plan, cap), provenance=dict(assembly_sha256='two-goal-assembly'),
        controller_factory=factory, resume=resume, motor_rng_state=lambda: dict(key=[1, 2]))
    return result, controllers, servers


def test_fixed500_grid_has_original_cases_and_six_matched_methods():
    plan = module.two_goal_plan()
    old = {row['case']['task_id']: row['case'] for row in comparison_plan() if row['primary']}
    assert len(plan) == 12 and 12 * 3 == 36
    assert 36 * 410 == 14760 < module.MAX_ACTIONS == 16000
    assert {row['updates'] for row in plan} == {500}
    assert {row['method'] for row in plan} == set(module.METHODS)
    assert not any('sac' in row['method'] or row['method'] == 'native' for row in plan)
    assert all(row['case'] == old[row['case']['task_id']] for row in plan)


def test_full_grid_executes36_episodes_and_records_full_contexts_raw_arrays_and_rng(tmp_path):
    output = tmp_path / 'comparison'
    result, controllers, servers = run(output)
    assert result['completed'] and len(result['completed_groups']) == 12
    assert sum(sim.episodes for sim in servers) == 36 and all(sim.closed for sim in servers)
    assert result['charged_total_actions'] == 36 * 23 == 828
    assert result['deployment_optimizer_updates'] == 0 and result['support_query_separate']
    assert len(controllers) == 24
    # Each pair of controller wrappers runs independent K0 and K1 trials.
    for zero, one in zip(controllers[::2], controllers[1::2]):
        assert zero.acts[0][1] == one.acts[0][1] == 0
        assert one.acts[2][1] == (2 if one.mode == 'full' else 0)
    for group in result['completed_groups']:
        path = output / f'group-{result["completed_groups"].index(group):03d}'
        archive = json.loads((path / 'contexts/index.json').read_text())
        assert len(archive) == 3
        with (path / 'contexts' / archive[-1]['path']).open('rb') as stream:
            recovery = pickle.load(stream)
        assert recovery['global_rng'] and recovery['motor_rng'] == dict(key=[1, 2])
        assert recovery['controller_before_episode'] and recovery['controller_after_episode']
        assert list((path / 'raw-rpc').glob('*-inference.pkl.gz'))
        assert list((path / 'raw-rpc').glob('*-outcome.pkl.gz'))
    assert len(result['actual_initial_identities']) == 2
    with (output / 'comparison-recovery.pkl').open('rb') as stream:
        assert pickle.load(stream)['next_group'] == 12


def test_query_failure_resumes_actual_support_context_and_charges_unknown_rpc(tmp_path):
    output = tmp_path / 'comparison'
    plan = module.two_goal_plan()[:1]
    first = TargetSimulator(0, fail_on_step=5)  # K0 query and K1 support already committed.
    with pytest.raises(RuntimeError, match='injected RPC'):
        run(output, plan=plan, sims=[first])
    with (output / 'group-000/evaluation-recovery.pkl').open('rb') as stream:
        saved = pickle.load(stream)
    assert saved['trial_index'] == saved['next_episode'] == 1
    assert saved['controller']['controller']['feedback']
    result, controllers, servers = run(output, plan=plan, resume=True)
    assert result['completed'] and result['charged_total_actions'] == 82
    assert first.episodes == 3 and servers[0].episodes == 1
    assert controllers[0].acts[0][1] == 2  # Saved actual support retained.
    group = json.loads((output / 'group-000/index.json').read_text())
    assert not group['budget_matched_primary_eligible']


def test_stop_before_an_unfunded_complete_method_case_group(tmp_path):
    output = tmp_path / 'comparison'
    result, _, servers = run(output, plan=module.two_goal_plan()[:2], cap=69)
    assert not result['completed'] and len(result['completed_groups']) == 1
    assert result['stop_reason'] == 'complete_group_physical_reservation'
    assert sum(sim.episodes for sim in servers) == 3 and charged_actions(output) == 69


def test_same_case_methods_require_identical_actual_initial_wrist_arrays(tmp_path):
    class ChangedWrist(TargetSimulator):
        def reset(self, state, seed):
            value = super().reset(state, seed)
            if self.episodes > 3:
                value['observation']['wrist'][:] = 1
            return value
    output = tmp_path / 'comparison'
    with pytest.raises(ValueError, match='Methods changed the actual'):
        run(output, plan=module.two_goal_plan()[:2], sim_factory=lambda case: ChangedWrist(case.task_id))
    assert charged_actions(output) == 138


def test_completed_group_context_tampering_rejects_without_new_interactions(tmp_path):
    output = tmp_path / 'comparison'
    plan = module.two_goal_plan()[:1]
    run(output, plan=plan)
    index = json.loads((output / 'group-000/contexts/index.json').read_text())
    (output / 'group-000/contexts' / index[0]['path']).write_bytes(b'changed')
    def forbidden(_):
        raise AssertionError('No new simulator should start while verifying a completed group')
    with pytest.raises(ValueError, match='context checksum'):
        run(output, plan=plan, resume=True, sim_factory=forbidden)


def test_budget_reader_never_creates_or_expands_a_reservation(tmp_path):
    path = tmp_path / 'ppo-ledger.json'
    record = dict(run_id='declared-run', source_commit='a' * 40, maximum_total_actions=16000,
        purpose='target_evaluation', source_pretraining_data=False, fresh_ppo_training_data=False,
        campaign_sha256='campaign')
    path.write_text(json.dumps(dict(campaign_sha256='campaign', reservations={'declared-run': record})))
    before = path.read_bytes()
    assert module.budget_reservation(path, 'declared-run', 'a' * 40) == record
    with pytest.raises(ValueError, match='existing exact16k'):
        module.budget_reservation(path, 'different-run', 'a' * 40)
    with pytest.raises(ValueError, match='existing exact16k'):
        module.budget_reservation(path, 'declared-run', 'b' * 40)
    assert path.read_bytes() == before


def test_loader_requires_two_source_exposures_and_fixed500_weights(tmp_path, monkeypatch):
    matrix = tmp_path / 'matrix.json'
    matrix.write_text(json.dumps(dict(contract=dict(assembly_sha256='assembly'),
        controllers=[dict(updates=500)] * 4)))
    assembly = dict(small_pilot=True, source_task_keys=['libero_10:00', 'libero_spatial:00'],
        validation_task_keys=['libero_10:00', 'libero_spatial:00'], selected_sources=dict(train=[], validation=[]),
        assembly_sha256='assembly')
    value = dict(assembly_path='assembly.json', assembly_sha256='assembly', source_exposure=dict(training_task_keys=assembly['source_task_keys'],
        validation_task_keys=assembly['validation_task_keys'], selected_source_histories=assembly['selected_sources']),
        scales=[dict(updates=500, index_path=str(matrix), index_sha256=file_sha256(matrix))])
    monkeypatch.setattr(module, 'load_scales', lambda *args: value)
    monkeypatch.setattr(module, 'load_assembly', lambda *args, **kwargs: assembly)
    assert module.two_goal_scale('ignored', 'ignored')[1]['updates'] == 500
    invalid = deepcopy(assembly)
    invalid['source_task_keys'] = ['libero_spatial:00']
    monkeypatch.setattr(module, 'load_assembly', lambda *args, **kwargs: invalid)
    with pytest.raises(ValueError, match='exactly two source goals'):
        module.two_goal_scale('ignored', 'ignored')
    monkeypatch.setattr(module, 'load_assembly', lambda *args, **kwargs: assembly)
    matrix.write_text(json.dumps(dict(contract=dict(assembly_sha256='assembly'), controllers=[dict(updates=100)] * 4)))
    value['scales'][0]['index_sha256'] = file_sha256(matrix)
    with pytest.raises(ValueError, match='Every selected controller'):
        module.two_goal_scale('ignored', 'ignored')
