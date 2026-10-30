"""Allocated-Linux tests of source quota/ancestry migration, not new model runs."""
import copy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import vla_source40_topup as topup
from fluffy_bassoon.meta_rl.vla_libero import structure_digest
from fluffy_bassoon.meta_rl.vla_source40_schedule import (
    LIBERO_COMMIT, SOURCE_SUITES, build_schedule, load_schedule)


def prepared(tmp_path, monkeypatch, mode='production'):
    rows = [dict(slot=key, run_id='old-' + str(index)) for index, key in enumerate(topup.expected_slots())]
    source = tmp_path / 'receipts.json'
    source.write_text(json.dumps(dict(rows=rows)))
    def parent(root, slot):
        val_index = topup.expected_slots().index(slot) - 40
        actions = 8000 if slot.endswith('/train') else (10660 if val_index == 7 else 10663)
        return dict(slot=slot, root=str(Path(root).resolve()), actions=actions,
            source_updates=5000, chunks=750, next_case=20, episodes=80)
    monkeypatch.setattr(topup, 'parent_record', parent)
    monkeypatch.setattr(topup, 'cost_evidence', lambda *args: dict(actions=topup.EXTRAS, total=sum(topup.EXTRAS.values())))
    return topup.make_plan(source, project_root=tmp_path, evidence_root=tmp_path, source_commit='a' * 40, mode=mode)


def rehash(plan):
    plan['plan_sha256'] = topup.content_hash({k: v for k, v in plan.items() if k != 'plan_sha256'})
    return plan


def test_canonical_full_caps_include_all_old_qualifications_and_controls(tmp_path, monkeypatch):
    value = prepared(tmp_path, monkeypatch)
    assert sum(topup.EXTRAS.values()) == 21253
    assert value['maximum_stream_action_charge'] == 957301
    assert value['maximum_charged_including_overhead'] == 998554 and value['unassigned_actions'] == 1446
    assert value['final_teacher_competence']['maximum_total_actions'] == 20000
    assert len(value['new_case_admitted_slots']) == 40 and len(value['retained_only_slots']) == 8
    path = tmp_path / 'plan.json'
    topup.write(path, value)
    assert topup.load_plan(path) == value
    value['historical_extra_costs']['actions'] = dict(topup.EXTRAS, collector_speed_qualification=0)
    topup.write(path, rehash(value))
    with pytest.raises(ValueError, match='cumulative charge'):
        topup.load_plan(path)


def test_qualification_admits_only_one_goal_and_five_thousand_new_actions(tmp_path, monkeypatch):
    plan = prepared(tmp_path, monkeypatch, 'qualification')
    assert plan['new_case_admitted_slots'] == ['libero_spatial:00/train']
    assert sum(row['maximum_new_actions'] for row in plan['slots']) == 5000
    assert plan['maximum_charged_including_overhead'] - plan['inherited_action_charge'] == 25000


def test_missing_or_target_streams_rejected_before_parent_reads(tmp_path, monkeypatch):
    path = tmp_path / 'bad.json'
    rows = [dict(slot=key, run_id='unused') for key in topup.expected_slots()]
    rows[0]['slot'] = 'libero_90:00/train'
    path.write_text(json.dumps(dict(rows=rows)))
    monkeypatch.setattr(topup, 'parent_record', lambda *args: pytest.fail('must not read model/source parents'))
    with pytest.raises(ValueError, match='all40'):
        topup.make_plan(path, project_root=tmp_path, evidence_root=tmp_path, source_commit='a' * 40)


def test_unique_leaf_replacement_counts_inherited_qualification_once(tmp_path, monkeypatch):
    plan = prepared(tmp_path, monkeypatch, 'qualification')
    ledger, first = tmp_path / 'ledger.json', tmp_path / 'qualification'
    slot = 'libero_spatial:00/train'
    topup.claim(plan, slot, output=first, ledger=ledger)
    topup.claim(plan, slot, output=first, ledger=ledger)  # Exact source resume.
    with pytest.raises(ValueError, match='Another leaf'):
        topup.claim(plan, slot, output=tmp_path / 'duplicate', ledger=ledger)
    production = prepared(tmp_path, monkeypatch)
    selected = production['slots'][0]
    selected['parent'].update(root=str(first.resolve()), actions=10000)
    selected['maximum_new_actions'] = 11800
    rehash(production)
    topup.claim(production, slot, output=tmp_path / 'production', ledger=ledger)
    status = topup.status(ledger)
    assert status['within_ceiling'] and status['historical_extra_actions'] == 21253
    assert status['rows'][0]['actions'] == 10000  # Not original8000 + current10000.


def test_shared_compute_holds_cannot_restart_qualification_budget(tmp_path, monkeypatch):
    plan = prepared(tmp_path, monkeypatch, 'qualification')
    ledger, root = tmp_path / 'ledger.json', tmp_path / 'q'
    topup.time_hold(plan, ledger, root, 1000)
    with pytest.raises(ValueError, match='compute hold'):
        topup.time_hold(plan, ledger, tmp_path / 'second', 1000)
    topup.time_hold(plan, ledger, root, 1000, actual=300)
    production = prepared(tmp_path, monkeypatch)
    for index in range(40):
        topup.time_hold(production, ledger, tmp_path / f'worker-{index}', 2100)
    assert sum(row['charged_or_reserved_seconds'] for row in topup.read(ledger)['time_holds'].values()) == 84300
    with pytest.raises(ValueError, match='compute hold'):
        topup.time_hold(production, ledger, tmp_path / 'too-many', 2101)


def test_repaired_qualification_shares_prior_failed_charge(tmp_path, monkeypatch):
    plan = prepared(tmp_path, monkeypatch, 'qualification')
    ledger = tmp_path / 'ledger.json'
    topup.time_hold(plan, ledger, tmp_path / 'failed', 1000)
    topup.time_hold(plan, ledger, tmp_path / 'failed', 1000, actual=146)
    topup.time_hold(plan, ledger, tmp_path / 'repaired', 854)
    with pytest.raises(ValueError, match='compute hold'):
        topup.time_hold(plan, ledger, tmp_path / 'extra', 1)
    assert sum(row['charged_or_reserved_seconds'] for row in
        topup.read(ledger)['time_holds'].values()) == 1000


def source_fixture(tmp_path):
    root = tmp_path / 'original'
    output = root / 'experiment'
    (output / 'raw').mkdir(parents=True)
    cases = [dict(case_id=f'case-{i}', initial_state=i % 40, scene_seed=i) for i in range(512)]
    contract = dict(task=dict(task_key='libero_spatial:00'), partition='train', cases=cases,
        history_id='unchanged-source-history', source_recipe=dict(warmup_chunks=500, batch_size=256, utd=20),
        learner_seed=19, horizon=400, chunk=10, feature_contract=dict(observation_dim=4104),
        noise_contract='same32D', source_commit='b' * 40)
    payload = dict(contract=contract, agent=b'all-optimizer-and-parameter-bytes', agent_rng=np.asarray([1, 2], np.uint32),
        replay=[dict(observations=np.zeros(2), actions=np.ones(32)) for _ in range(742)], updates=4920,
        sampler=np.random.default_rng(19).bit_generator.state, numpy_rng=np.random.get_state(),
        python_rng=('full-python-state-fixture',), boundary='complete physical episode',
        progress=dict(case_index=20, episode_in_case=0, motor_rng=dict(format='jax_motor_key_v1',
            implementation='threefry2x32', key_data=np.asarray([3, 4], np.uint32))))
    with (output / 'source-recovery.pkl').open('wb') as stream:
        pickle.dump(payload, stream)
    topup.write(output / 'contract.json', contract)
    topup.write(output / 'index.json', dict(completed=True))
    (output / 'raw/episode-000000.pkl.gz').write_bytes(b'actual-old-raw-bytes')
    parent = dict(root=str(root.resolve()), experiment=str(output.resolve()), slot='libero_spatial:00/train',
        actions=7892, source_updates=4920, chunks=742, episodes=80,
        recovery_sha256=topup.sha(output / 'source-recovery.pkl'))
    return parent, contract, payload


def test_migration_preserves_all_numeric_state_and_old_raw_then_changes_only_contract(tmp_path, monkeypatch):
    parent, old, payload = source_fixture(tmp_path)
    monkeypatch.setattr(topup, 'parent_record', lambda *args: parent)
    plan = dict(plan_sha256='f' * 64, mode='qualification')
    selected = dict(parent=parent, slot=parent['slot'], maximum_total_actions=12892, maximum_new_actions=5000)
    original = Path(parent['experiment'])
    before = {str(path.relative_to(original)): topup.sha(path) for path in original.rglob('*') if path.is_file()}
    child = topup.Continuation(plan, selected, tmp_path / 'new', stop_unix=1e12)
    child.prepare()
    updated = dict(old, source_commit='a' * 40, maximum_total_actions=12892)
    new_contract = child.bind_contract(updated)
    with (child.output / 'source-recovery.pkl').open('rb') as stream:
        migrated = pickle.load(stream)
    assert migrated['contract'] == new_contract
    assert structure_digest({k: v for k, v in migrated.items() if k != 'contract'}) == structure_digest(
        {k: v for k, v in payload.items() if k != 'contract'})
    assert child.bind_contract(dict(old, source_commit='a' * 40, maximum_total_actions=12892)) == new_contract
    for relative, digest in before.items():
        assert topup.sha(original / relative) == digest
    assert (child.output / 'raw/episode-000000.pkl.gz').read_bytes() == b'actual-old-raw-bytes'
    migration = topup.read(child.output / 'migration.json')
    assert topup.sha(child.output / migration['archive'] / 'source-recovery.pkl') == parent['recovery_sha256']


def test_migration_rejects_changed_case_bank_features_or_recipe(tmp_path, monkeypatch):
    parent, old, _ = source_fixture(tmp_path)
    monkeypatch.setattr(topup, 'parent_record', lambda *args: parent)
    child = topup.Continuation(dict(plan_sha256='f' * 64), dict(parent=parent, slot=parent['slot']), tmp_path / 'child', stop_unix=1e12)
    child.prepare()
    changed = copy.deepcopy(old)
    changed['cases'][20]['scene_seed'] += 1
    with pytest.raises(ValueError, match='cases'):
        child.bind_contract(changed)
    changed = copy.deepcopy(old)
    changed['source_recipe']['utd'] = 1
    with pytest.raises(ValueError, match='recipe'):
        child.bind_contract(changed)
    assert not (child.output / 'source-recovery.pkl').exists()


def test_new_episodes_require_actual_settled_hash_and_fixed_postwarmup_behavior(tmp_path):
    child = topup.Continuation({}, dict(parent={}), tmp_path / 'root', stop_unix=1e12)
    summary = dict(initial_state_vector_sha256='a' * 64, initial_simulator_state_sha256='b' * 64)
    valid = [dict(source_stage='sac_online', source_updates_before=5000) for _ in range(3)]
    child.validate_episode(summary, valid)
    with pytest.raises(RuntimeError, match='settled-state'):
        child.require_physical_state(dict(initial_state_vector_sha256='a' * 64))
    with pytest.raises(RuntimeError, match='warmup'):
        child.validate_episode(summary, [dict(source_stage='warmup_gaussian', source_updates_before=0)])
    valid[-1]['source_updates_before'] += 20
    with pytest.raises(RuntimeError, match='weights'):
        child.validate_episode(summary, valid)


def test_materialized_budget_keeps_original_case_ids_and_unused_bank(tmp_path, monkeypatch):
    source = tmp_path / 'libero'
    mapping = {suite: [f'task_{i}' for i in range(10)] for suite in SOURCE_SUITES}
    taskmap = source / 'libero/libero/benchmark/libero_suite_task_map.py'
    taskmap.parent.mkdir(parents=True)
    taskmap.write_text('libero_task_map = ' + repr(mapping))
    for kind, suffix in [('bddl_files', '.bddl'), ('init_files', '.pruned_init')]:
        path = source / f'libero/libero/{kind}/libero_spatial/task_0{suffix}'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'static original fixture')
    monkeypatch.setattr('fluffy_bassoon.meta_rl.vla_source40_schedule.subprocess.check_output',
        lambda *args, **kwargs: LIBERO_COMMIT + '\n')
    original = build_schedule(source, train_cases=512, validation_cases=512, episodes_per_case=4,
        collection_mode='physical_budget', train_action_budget=9500, validation_action_budget=12000,
        publish_every_cases=16, task_keys=['libero_spatial:00'])
    parent = tmp_path / 'parent'
    parent.mkdir()
    topup.write(parent / 'schedule.json', original)
    path = tmp_path / 'continued.json'
    value = topup.materialize_schedule(dict(root=str(parent), slot='libero_spatial:00/train'), path, 21800)
    assert value['tasks'] == original['tasks']
    assert len(value['tasks'][0]['partitions']['train']) == 512
    assert value['planned_requested_cases'] is None
    assert value['partition_budgets']['train']['maximum_total_actions'] == 21800
    assert load_schedule(path) == value


def test_new_final_competence_escrow_is_unique_and_does_not_reuse_old_ppo_remainder(tmp_path, monkeypatch):
    from fluffy_bassoon.meta_rl import vla_source40_topup_competence as competence
    plan = prepared(tmp_path, monkeypatch)
    plan_path = tmp_path / 'production.json'
    topup.write(plan_path, plan)
    ledger = tmp_path / 'canonical.json'
    topup.claim(plan, 'libero_spatial:00/train', output=tmp_path / 'train', ledger=ledger)
    fake_source, fake_schedule = tmp_path / 'source.json', tmp_path / 'schedule.json'
    topup.write(fake_source, dict(reservations={}))
    topup.write(fake_schedule, dict(static='fixture'))
    alias = dict(format=competence.FORMAT, run_id='new-final-control', source_commit=plan['source_commit'],
        source_plan_source_commit=plan['source_commit'],
        plan_path=str(plan_path), plan_sha256=plan['plan_sha256'], canonical_topup_ledger=str(ledger),
        source_leaves_sha256=topup.content_hash(topup.read(ledger)['leaves']),
        source_ledger=str(fake_source), source_ledger_sha256=topup.sha(fake_source),
        schedule=str(fake_schedule), schedule_sha256=topup.sha(fake_schedule), outputs_root=str(tmp_path),
        maximum_total_actions=20000, maximum_worker_seconds=1400)
    alias_path = tmp_path / 'alias.json'
    topup.write(alias_path, alias)
    actual = competence.load_alias(alias_path, topup.sha(alias_path), run_id=alias['run_id'], source_commit=plan['source_commit'])
    assert actual == alias
    value = topup.read(ledger)
    assert value['historical_extra_actions'] == 21253
    assert value['final_competence']['maximum_total_actions'] == 20000
    alias['run_id'] = 'duplicate-final-control'
    topup.write(alias_path, alias)
    with pytest.raises(ValueError, match='another immutable owner'):
        competence.load_alias(alias_path, topup.sha(alias_path), run_id=alias['run_id'], source_commit=plan['source_commit'])


def test_final_competence_requires_production_new_actors_not_the_old_controls(tmp_path, monkeypatch):
    from fluffy_bassoon.meta_rl import vla_source40_topup_competence as competence
    plan = prepared(tmp_path, monkeypatch, 'qualification')
    plan_path, ledger = tmp_path / 'qual.json', tmp_path / 'ledger.json'
    topup.write(plan_path, plan)
    topup.claim(plan, 'libero_spatial:00/train', output=tmp_path / 'qualified', ledger=ledger)
    with pytest.raises(ValueError, match='production source ledger'):
        competence.prepare(plan_path, ledger, tmp_path / 'final-alias', run_id='new-final', source_commit='a' * 40)
