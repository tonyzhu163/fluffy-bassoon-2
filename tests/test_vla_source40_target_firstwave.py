"""First-wave immutable inputs, disjoint budget/time and actual episode audit."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from fluffy_bassoon.meta_rl.vla_history import file_sha256
from fluffy_bassoon.meta_rl.vla_source40_schedule import content_hash
from fluffy_bassoon.meta_rl.vla_source40_target_firstwave_plan import build_plan
import fluffy_bassoon.meta_rl.vla_source40_target_firstwave as module
from fluffy_bassoon.meta_rl.vla_source40_subset_evaluate import charged_actions
from test_vla_common_runtime_evaluate import run


def plan():
    return build_plan(worktree='/tmp/planning-worktree', source_commit='a' * 40)


def ledger(path, value):
    rows = {f'worker-{w["worker"]}': dict(run_id=f'worker-{w["worker"]}', worker=w['worker'],
        source_commit='a' * 40, wave_sha256=value['firstwave_sha256'], maximum_total_actions=w['action_cap'],
        worker_seconds=3600, purpose='target_evaluation', source_pretraining_data=False, fresh_ppo_training_data=False)
        for w in value['workers']}
    data = dict(format=module.LEDGER_FORMAT, wave_sha256=value['firstwave_sha256'], maximum_total_actions=250000,
        maximum_gpu_seconds=21600, source_campaign_immutable=True, scope_amendment='Authorized target screen; old source cap unchanged',
        reservations=rows)
    path.write_text(json.dumps(data))
    return data


def test_wave_plan_is_exact_and_six_assignments_are_disjoint(tmp_path):
    value = plan()
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(value))
    assert module.load_plan(path, file_sha256(path), 'a' * 40) == value
    shards = [module.worker_groups(value, i) for i in range(6)]
    assert sum(sum(sum(b + 1 for b in g['support_budgets']) for g in shard) for shard in shards) == 626
    assert len({g['group_id'] for shard in shards for g in shard}) == 558
    assert sum(g['maximum_actions'] for shard in shards for g in shard) == 243920
    value['cases'][0]['simulator']['horizon'] = 280
    value['firstwave_sha256'] = content_hash({k: v for k, v in value.items() if k != 'firstwave_sha256'})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='predeclaration'):
        module.load_plan(path, file_sha256(path), 'a' * 40)


def test_only_separate_exact_target_scope_is_admitted_and_never_mutated(tmp_path):
    value, path = plan(), tmp_path / 'ledger.json'
    original = ledger(path, value)
    before = path.read_bytes()
    result = module.reservation(path, value, 2, 'worker-2', 'a' * 40)
    assert result['record']['worker'] == 2 and path.read_bytes() == before
    original['reservations']['worker-1']['worker'] = 2
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='scope amendment'):
        module.reservation(path, value, 2, 'worker-2', 'a' * 40)


def test_failed_attempts_use_only_original_remaining_hour(tmp_path, monkeypatch):
    data = dict(worker_seconds=3600, contract=dict(escrow_record_sha256='sha'), attempts=[
        dict(status='finished', charged_seconds=500, reserved_seconds=1060),
        dict(status='started', slice_seconds=3000, reserved_seconds=3060, started_unix=1000)])
    (tmp_path / 'worker-time.json').write_text(json.dumps(data))
    monkeypatch.setattr(module.time, 'time', lambda: 1100)
    assert module.remaining_seconds(tmp_path, 'sha') == 2840
    data['attempts'][0]['charged_seconds'] = 600
    (tmp_path / 'worker-time.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='cumulative'):
        module.remaining_seconds(tmp_path, 'sha')


def test_firstwave_one_budget_audit_checks_trace_and_context(tmp_path, monkeypatch):
    import fluffy_bassoon.meta_rl.vla_common_runtime_evaluate as common
    monkeypatch.setattr(common, 'audit_group', module.audit_firstwave)
    group = dict(module.worker_groups(plan(), 0)[1])
    # Fixture simulator is original task0/state0 with H20, while production
    # predeclaration reserves H400. Reuse its real episode journal machinery.
    group['source_condition'] = 'source1'
    result, _, _, _ = run(tmp_path / 'screen', plan=[group], cap=200)
    row = result['completed_groups'][0]
    path = Path(row['result_path']).parent
    receipt = json.loads(Path(row['result_path']).read_text())
    audited = module.audit_firstwave(path, receipt, group)
    assert audited['within_group_initial_public_arrays_equal']
    assert charged_actions(tmp_path / 'screen') == 23
    trace = path / receipt['episodes'][0]['trace']
    trace.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='trace checksum'):
        module.audit_firstwave(path, receipt, group)


def test_k1_audit_requires_one_actual_support_and_a_separate_query(tmp_path, monkeypatch):
    import fluffy_bassoon.meta_rl.vla_common_runtime_evaluate as common
    monkeypatch.setattr(common, 'audit_group', module.audit_firstwave)
    group = dict(module.worker_groups(plan(), 0)[5])
    group['source_condition'] = 'source1'
    result, _, _, servers = run(tmp_path / 'screen', plan=[group], cap=200)
    row = result['completed_groups'][0]
    receipt = json.loads(Path(row['result_path']).read_text())
    assert [e['role'] for e in receipt['episodes']] == ['support', 'query']
    assert sum(s.episodes for s in servers) == 2
    assert charged_actions(tmp_path / 'screen') == 46


def test_fixed500_manifest_binds_all_four_recoveries(tmp_path):
    rows = []
    for objective in ('ad', 'dpt'):
        for mode in ('full', 'static'):
            final, recovery = tmp_path / f'{objective}-{mode}.pt', tmp_path / f'{objective}-{mode}-recovery.pt'
            final.write_bytes(b'candidate')
            recovery.write_bytes(b'optimizer/sampler/RNG')
            rows.append(dict(objective=objective, mode=mode, updates=500, seed=0, final_path=str(final),
                final_sha256=file_sha256(final), full_recovery_path=str(recovery), full_recovery_sha256=file_sha256(recovery)))
    matrix, assembly = tmp_path / 'matrix.json', tmp_path / 'assembly.json'
    matrix.write_text(json.dumps(dict(completed=True, controllers=rows)))
    assembly.write_text('{}')
    assert len(module.fixed_matrix(matrix, file_sha256(matrix), assembly, file_sha256(assembly))[1]) == 4
    Path(rows[0]['full_recovery_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='full recovery checksum'):
        module.fixed_matrix(matrix, file_sha256(matrix), assembly, file_sha256(assembly))
