"""Shared-motor source-condition isolation, frozen-target routing and costs."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_history import file_sha256
from fluffy_bassoon.meta_rl.vla_source40_evaluate import TargetCase, _seed
from fluffy_bassoon.meta_rl.vla_source40_schedule import content_hash
from fluffy_bassoon.meta_rl.vla_source40_subset_evaluate import charged_actions
import fluffy_bassoon.meta_rl.vla_common_runtime_evaluate as module
from test_vla_source40_evaluate import Controller, Motor, Policy
from test_vla_source40_two_goal_evaluate import TargetSimulator, candidates


class TargetActor:
    public_observation_input = True
    def __init__(self):
        self.policy, self.pending, self.observations = Policy(), False, []
    def frozen_signature(self):
        return 'fixed-target-parameters-and-optimizer'
    def set_episode_seed(self, seed):
        self.seed = seed
    def act(self, observation):
        assert isinstance(observation, dict) and 'wrist' in observation and not self.pending
        self.observations.append(observation)
        self.pending = True
        return np.ones(32, np.float32)
    def observe(self, **outcome):
        assert self.pending
        self.pending = False
    def snapshot(self):
        return dict(pending=self.pending, seed=self.seed)
    def restore(self, value):
        self.pending, self.seed = value['pending'], value['seed']


def run(output, *, plan=None, cap=2000, runtime='one-process', resume=False, sims=None):
    plan = module.common_plan() if plan is None else plan
    banks = dict(source1=candidates(), source2=candidates())
    actors = {task: TargetActor() for task in (0, 1)}
    targets = {task: dict(policy=actor, kind='frozen_target_sac', mode='frozen', sha256=f'target-{task}') for task, actor in actors.items()}
    servers, controllers = [], []
    def factory(case):
        sim = sims.pop(0) if sims else TargetSimulator(case.task_id)
        servers.append(sim)
        return sim
    def controller(spec, sim):
        if spec.get('kind') == 'frozen_target_sac':
            return spec['policy']
        value = Controller(spec, sim)
        controllers.append(value)
        return value
    result = module.run_common(Motor(), banks, targets, output=output,
        contract=dict(format=module.FORMAT, groups=plan, maximum_total_actions=cap, maximum_seconds_per_invocation=60.),
        server_factory=factory, provenance={name: dict(bank=name) for name in ('source1', 'source2', 'native', 'target_online_final')},
        controller_factory=controller, runtime_id=runtime, resume=resume)
    return result, actors, controllers, servers


def test_plan_has76_actual_episodes_and_condition_never_changes_case_or_rng():
    plan = module.common_plan()
    assert len(plan) == 28
    assert sum(sum(budget + 1 for budget in row['support_budgets']) for row in plan) == 76
    assert 76 * 410 == 31160 < module.MAX_ACTIONS == 32000
    assert sum(row['method'] == 'native' for row in plan) == 2
    assert sum(row['method'] == 'frozen_target_sac' for row in plan) == 2
    assert sum(row['source_condition'] == 'source1' for row in plan) == 12
    assert sum(row['source_condition'] == 'source2' for row in plan) == 12
    for task in (0, 1):
        selected = [row for row in plan if row['case']['task_id'] == task]
        assert len({content_hash(row['case']) for row in selected}) == 1
        assert len({_seed(TargetCase(**row['case']), 'query', 0) for row in selected}) == 1
        assert all('source_condition' not in row['case'] for row in selected)
        students = [row for row in selected if row['updates'] == 500]
        assert [row['source_condition'] for row in students] == ['source1', 'source2'] * 6


def test_all_comparators_execute_in_one_motor_process_with_public_target_sac_input(tmp_path):
    result, actors, controllers, servers = run(tmp_path / 'common')
    assert result['completed'] and result['strict_common_runtime_primary_eligible']
    assert result['actual_runtime_ids'] == ['one-process']
    assert result['charged_total_actions'] == 76 * 23 == 1748
    assert sum(sim.episodes for sim in servers) == 76 and all(sim.closed for sim in servers)
    assert len(controllers) == 48  # Two independent0/1-support trials per student group.
    assert all(len(actor.observations) == 2 for actor in actors.values())
    assert result['deployment_optimizer_updates'] == 0
    assert len(result['actual_initial_identities']) == 2
    for row in result['completed_groups']:
        assert row['runtime_ids'] == ['one-process']
        assert row['audit']['initial_public_arrays_sha256']


def test_fresh_process_resume_keeps_support_and_cost_but_loses_strict_common_runtime_status(tmp_path):
    output = tmp_path / 'common'
    group = next(row for row in module.common_plan() if row['source_condition'] == 'source1')
    first = TargetSimulator(0, fail_on_step=5)
    with pytest.raises(RuntimeError, match='injected RPC'):
        run(output, plan=[group], runtime='process-a', sims=[first])
    result, _, controllers, servers = run(output, plan=[group], runtime='process-b', resume=True)
    assert result['completed'] and result['charged_total_actions'] == 82
    assert first.episodes == 3 and servers[0].episodes == 1
    assert controllers[0].acts[0][1] == 2
    assert result['actual_runtime_ids'] == ['process-a', 'process-b']
    assert not result['strict_common_runtime_primary_eligible']


def test_unfunded_group_stops_without_an_extra_physical_episode(tmp_path):
    output = tmp_path / 'common'
    result, _, _, servers = run(output, plan=module.common_plan()[:2], cap=23)
    assert not result['completed'] and result['stop_reason'] == 'complete_group_physical_reservation'
    assert charged_actions(output) == 23 and sum(sim.episodes for sim in servers) == 1


def test_32k_reservation_is_exact_and_read_only(tmp_path):
    path = tmp_path / 'ledger.json'
    row = dict(run_id='run', source_commit='a' * 40, maximum_total_actions=32000,
        purpose='target_evaluation', source_pretraining_data=False, fresh_ppo_training_data=False, campaign_sha256='campaign')
    path.write_text(json.dumps(dict(campaign_sha256='campaign', reservations={'run': row})))
    before = path.read_bytes()
    assert module.reservation(path, 'run', 'a' * 40) == row
    with pytest.raises(ValueError, match='exact32k'):
        module.reservation(path, 'other', 'a' * 40)
    assert path.read_bytes() == before


def test_target_manifest_binds_full_recovery_and_separates_original15episode_cost(tmp_path, monkeypatch):
    root = tmp_path / 'case-00'
    root.mkdir()
    case = next(row['case'] for row in module.common_plan() if row['case']['task_id'] == 0)
    contract = dict(case=case, recipe=module.RECIPE, motor_receipt_sha256=module.PI05_RECEIPT,
        target_only=True, source_assembly_admission=False, learner_seed=0)
    (root / 'contract.json').write_text(json.dumps(contract))
    (root / 'target-recovery.pkl').write_bytes(b'full target actor, critic, optimizer, replay and RNG fixture')
    recovery_sha = file_sha256(root / 'target-recovery.pkl')
    index = dict(completed=True, case=case, contract_sha256=content_hash(contract), source_checkpoint_reused=False,
        training_episodes=15, target_optimizer_updates=2400, training_chunks=583, replay_rows=583,
        latest_recovery_sha256=recovery_sha, training_primitive_steps=5826, training_settling_steps=150,
        costs=dict(training=dict(known_actions=5976), query=dict(known_actions=1640)), total_action_charge=7626)
    (root / 'index.json').write_text(json.dumps(index))
    pins = dict(index=file_sha256(root / 'index.json'), contract=file_sha256(root / 'contract.json'), recovery=recovery_sha)
    monkeypatch.setitem(module.TARGET_PINS, 0, pins)
    value = module.target_manifest(tmp_path, 0)
    assert value['original_training_episodes'] == 15 and value['original_training_actions'] == 5976
    assert value['target_optimizer_updates'] == 2400 and value['original_total_action_charge'] == 7626
    (root / 'target-recovery.pkl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='full recovery hash'):
        module.target_manifest(tmp_path, 0)
