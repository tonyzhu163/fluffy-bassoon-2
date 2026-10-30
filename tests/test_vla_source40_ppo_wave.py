"""Frozen PPO wave planning, terminal admission and actual support/context controls."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_common_runtime_evaluate import run_common
from fluffy_bassoon.meta_rl.vla_history import file_sha256
from fluffy_bassoon.meta_rl.vla_source40_ppo_target import TargetPPOController
from fluffy_bassoon.meta_rl.vla_source40_ppo_wave_plan import PANEL, build_plan, content_hash, terminal_validation
from fluffy_bassoon.meta_rl.vla_source40_subset_evaluate import charged_actions
import fluffy_bassoon.meta_rl.vla_source40_ppo_wave as module
from test_vla_source40_evaluate import Controller, Motor
from test_vla_source40_two_goal_evaluate import TargetSimulator
from test_vla_source40_ppo_target import specs


def proof():
    return dict(format='source40_successful_terminal_validation_v1', root='/terminal/validation', index_sha256='index',
        exports=dict(full={}, episode={}))


def plan():
    return build_plan(worktree='/allocated/ppo-worker', source_commit='a' * 40, source_validation=proof())


def test_all90_zero_shot_and_prespecified9_one_shot_fit_two_balanced_original_hours():
    value = plan()
    assert value['counts'] == dict(cases=90, groups=297, physical_episodes=324,
        nominal_actions_including_settling=132840, action_cap=140000, gpu_seconds=7200)
    assert sorted(e['case']['task_id'] for e in value['cases']) == list(range(90))
    assert [e['case']['task_id'] for e in value['cases'][:9]] == list(PANEL)
    for worker, episodes, actions, panels in ((0, 165, 67650, 5), (1, 159, 65190, 4)):
        selected = value['cases'][worker::2]
        assert sum(e['nominal_episodes'] for e in selected) == episodes
        assert sum(e['nominal_actions'] for e in selected) == actions < 70000
        assert sum(e['case']['task_id'] in PANEL for e in selected) == panels
        assert value['workers'][worker]['worker_seconds'] == 3600
    for entry in value['cases']:
        zero = [g for g in entry['groups'] if g['phase'] == 'A']
        assert [(g['method'], g['support_budgets']) for g in zero] == [('native',[0]), ('ppo_full',[0]), ('ppo_trained_episode',[0])]
        one = [g for g in entry['groups'] if g['phase'] == 'B']
        assert len(one) == (3 if entry['case']['task_id'] in PANEL else 0)
        assert all(g['support_budgets'] == [1] and g['episodes'] == 2 for g in one)
        assert entry['zero_shot_reuse']['ppo_episode_reset'] == zero[1]['group_id']


def test_empty_history_reuse_has_identical_full_weights_and_global_query_rng():
    selected = specs()
    full, reset = (TargetPPOController(selected[name]['policy'], mode=selected[name]['mode'])
        for name in ('ppo_full', 'ppo_episode_reset'))
    full.set_episode_seed(123)
    a = full.act(np.zeros(4104, np.float32))
    full.observe(executed_noise=a, reward=-1., done=False)
    reset.set_episode_seed(123)
    b = reset.act(np.zeros(4104, np.float32))
    np.testing.assert_array_equal(a, b)
    reset.observe(executed_noise=b, reward=-1., done=False)
    assert full.generator is reset.generator is None
    assert selected['ppo_full']['policy'] is selected['ppo_episode_reset']['policy']
    assert selected['ppo_full']['policy'] is not selected['ppo_trained_episode']['policy']


def test_loader_copies_exact_validated_exports_and_never_reexports_into_source(tmp_path, monkeypatch):
    source, output = tmp_path / 'source', tmp_path / 'worker'
    source.mkdir()
    exports, provenance = {}, {}
    for mode in ('full', 'episode'):
        path = source / f'frozen-{mode}.pt'
        path.write_bytes(('immutable-' + mode).encode())
        exports[mode] = dict(path=str(path), sha256=file_sha256(path), tensor_sha256=mode,
            recovery_sha256=mode + '-recovery', terminal_lineage=dict(mode=mode))
        provenance[mode] = dict(policy_sha256=file_sha256(path), tensor_sha256=mode,
            recovery_sha256=mode + '-recovery', terminal_lineage=dict(mode=mode))
    validation = dict(exports=exports, schedule_sha256='schedule', models_manifest_path='manifest', models_manifest_sha256='manifest-sha')
    def loader(manifest, digest, **kwargs):
        assert kwargs['output'] == output
        assert kwargs['tasks'] == ['source-goal']
        assert all(file_sha256(output / f'frozen-{m}.pt') == exports[m]['sha256'] for m in exports)
        return {'full': object(), 'episode': object()}, provenance
    monkeypatch.setattr(module, 'source_tasks', lambda schedule: [dict(task_key='source-goal')])
    monkeypatch.setattr(module, 'load_models', loader)
    selected, result = module.load_terminal_models(dict(contract=dict(source_validation=validation)),
        SimpleNamespace(), dict(schedule_sha256='schedule'), output, 'cpu')
    assert selected['ppo_full']['sha256'] == exports['full']['sha256']
    assert selected['ppo_episode_reset']['policy'] is selected['ppo_full']['policy']
    assert all(file_sha256(Path(spec['path'])) == spec['sha256'] for spec in exports.values())
    (output / 'frozen-full.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed on resume'):
        module.load_terminal_models(dict(contract=dict(source_validation=validation)),
            SimpleNamespace(), dict(schedule_sha256='schedule'), output, 'cpu')


def test_selected_panel_runs_actual_support_then_query_with_recorded_complete_context(tmp_path):
    group = next(g for g in plan()['cases'][0]['groups'] if g['phase'] == 'B' and g['method'] == 'ppo_full')
    # Use the existing event-logged fixture policy. It has the same act/observe
    # and complete-context protocol; separate tests exercise the real PPO RNG.
    from test_vla_source40_two_goal_evaluate import candidates
    fake = candidates()
    selected = {'ppo_full': fake['ad_full']}
    server = TargetSimulator(0)
    result = run_common(Motor(), {'source_ppo': selected}, {}, output=tmp_path / 'target',
        contract=dict(groups=[group], maximum_total_actions=200, maximum_seconds_per_invocation=60.),
        provenance={'source_ppo': {'frozen': True}}, server_factory=lambda case: server,
        controller_factory=lambda spec, sim: Controller(spec, sim), runtime_id='one-runtime', audit_fn=module.audit_ppo)
    assert result['completed'] and server.episodes == 2 and charged_actions(tmp_path / 'target') == 46
    saved = json.loads(Path(result['completed_groups'][0]['result_path']).read_text())
    assert [e['role'] for e in saved['episodes']] == ['support', 'query']
    assert saved['weights_unchanged']


def test_terminal_admission_rejects_failed_validation_without_partial_source_fallback(tmp_path):
    (tmp_path / 'launcher.lock').write_text('')
    (tmp_path / 'exit-code.txt').write_text('1')
    with pytest.raises(ValueError, match='successful terminal'):
        terminal_validation(tmp_path, 'unused')


def test_existing_target_ledger_read_only_and_exact_cap(tmp_path):
    value, path = plan(), tmp_path / 'ledger.json'
    rows = {f'worker-{i}': dict(run_id=f'worker-{i}', worker=i, source_commit='a' * 40,
        ppo_wave_sha256=value['ppo_wave_sha256'], maximum_total_actions=70000, worker_seconds=3600,
        purpose='target_evaluation', source_pretraining_data=False, fresh_ppo_training_data=False) for i in (0, 1)}
    ledger = dict(format=module.LEDGER_FORMAT, ppo_wave_sha256=value['ppo_wave_sha256'], maximum_total_actions=140000,
        maximum_gpu_seconds=7200, source_campaign_immutable=True, scope_amendment='Future freed validation hold, no extra32h budget', reservations=rows)
    path.write_text(json.dumps(ledger))
    before = path.read_bytes()
    assert module.reservation(path, value, 0, 'worker-0', 'a' * 40)['record'] == rows['worker-0']
    assert path.read_bytes() == before
    ledger['reservations']['worker-1']['maximum_total_actions'] = 72000
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='original allocation/action hold'):
        module.reservation(path, value, 0, 'worker-0', 'a' * 40)
