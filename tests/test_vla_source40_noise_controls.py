"""Contemporaneous exact warmup/no-noise controls; allocated Linux tests only."""
import gzip
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_common_runtime_evaluate import run_common
from fluffy_bassoon.meta_rl.vla_history import file_sha256
from fluffy_bassoon.meta_rl.vla_source40_corpus import OnlineSourceSimulator
from fluffy_bassoon.meta_rl.vla_source40_evaluate import expand_noise
from fluffy_bassoon.meta_rl.vla_source40_noise_plan import build_plan, content_hash
from fluffy_bassoon.meta_rl.vla_source40_subset_evaluate import charged_actions
import fluffy_bassoon.meta_rl.vla_source40_noise_controls as module
from test_vla_source40_evaluate import Motor
from test_vla_source40_two_goal_evaluate import TargetSimulator


def plan():
    return build_plan(worktree='/allocated/worker-0', source_commit='a' * 40)


def test_exact_all90_four_controls_disjoint_cases_and_settling_costs(tmp_path):
    value = plan()
    assert value['counts'] == dict(cases=90, groups=360, physical_episodes=360,
        nominal_actions_including_settling=147600, action_cap=152000, gpu_seconds=7200)
    assert [e['case']['task_id'] for e in value['cases']] == list(range(90))
    for entry in value['cases']:
        assert [g['method'] for g in entry['groups']] == ['native', 'zero', 'tied_gaussian', 'tied_gaussian_clipped']
        assert all(g['support_budgets'] == [0] for g in entry['groups'])
        assert entry['case']['initial_state'] == 0 and entry['case']['scene_seed'] == 123
    assert set(value['workers'][0]['case_ids']).isdisjoint(value['workers'][1]['case_ids'])
    assert sum(w['maximum_total_actions'] for w in value['workers']) == 152000
    assert sum(w['worker_seconds'] for w in value['workers']) == 7200
    assert value['contract']['qualification_extra_resets'] == 0
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(value))
    assert module.load_plan(path, file_sha256(path), 'a' * 40) == value
    value['contract']['noise_distributions']['tied_gaussian'] = 'clipped'
    value['noise_plan_sha256'] = content_hash({k: v for k, v in value.items() if k != 'noise_plan_sha256'})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='predeclaration'):
        module.load_plan(path, file_sha256(path), 'a' * 40)


def test_tied_noise_exactly_matches_actual_source_warmup_unbounded_std_one():
    sim = SimpleNamespace(metadata={}, reset=lambda state, seed: {})
    learner = SimpleNamespace(updates=0, replay=[])
    sampler = np.random.default_rng(123)
    source = OnlineSourceSimulator(sim, learner, sampler, lambda observation: observation)
    source.reset(0, 123)
    expected_rng = np.random.default_rng(123)
    for _ in range(3):
        noise = source.choose({'actual_public': 'fixture'})
        expected = expected_rng.standard_normal(32).astype(np.float32)
        np.testing.assert_array_equal(noise, expected)
        assert np.abs(noise).max() > 1  # Clipping would change the collected warmup.
        np.testing.assert_array_equal(expand_noise(noise, 10), np.repeat(expected[None], 10, axis=0))
        assert source.pending['stage'] == 'warmup_gaussian'


def run(output, groups=None, *, resume=False, runtime='one-process', sims=None):
    value = plan()
    groups = value['cases'][0]['groups'] if groups is None else groups
    servers = []
    def factory(case):
        sim = sims.pop(0) if sims else TargetSimulator(case.task_id)
        servers.append(sim)
        return sim
    result = run_common(Motor(), {}, {}, output=output,
        contract=dict(groups=groups, maximum_total_actions=500, maximum_seconds_per_invocation=60.),
        provenance={'noise_control': {'no_learning': True}}, server_factory=factory,
        runtime_id=runtime, resume=resume, audit_fn=module.audit_noise, action_ceiling=152000)
    return result, servers


def test_all_four_have_real_traces_rng_contexts_exact_noise_and_one_common_process(tmp_path):
    result, servers = run(tmp_path / 'controls')
    assert result['completed'] and result['strict_common_runtime_primary_eligible']
    assert result['actual_runtime_ids'] == ['one-process']
    assert sum(s.episodes for s in servers) == 4 and all(s.closed for s in servers)
    assert charged_actions(tmp_path / 'controls') == 92
    tied = json.loads(Path(result['completed_groups'][2]['result_path']).read_text())
    clipped = json.loads(Path(result['completed_groups'][3]['result_path']).read_text())
    tied_rows = module.trace_rows(Path(result['completed_groups'][2]['result_path']).parent, tied['episodes'][0])
    clipped_rows = module.trace_rows(Path(result['completed_groups'][3]['result_path']).parent, clipped['episodes'][0])
    for unbounded, bounded in zip(tied_rows, clipped_rows):
        np.testing.assert_array_equal(unbounded['proposed_noise'], bounded['proposed_noise'])
        np.testing.assert_array_equal(np.clip(unbounded['executed_noise'], -1., 1.), bounded['executed_noise'])
    for group in result['completed_groups']:
        assert group['audit']['noise_recipe_exact']
        assert group['audit']['public_arrays_match_first_group']
        path = Path(group['result_path']).parent
        index = json.loads((path / 'contexts/index.json').read_text())
        assert len(index) == 1
        with (path / 'contexts' / index[0]['path']).open('rb') as stream:
            context = pickle.load(stream)
        assert context['evaluation_recovery']['sampler'] and context['global_rng']
        assert list((path / 'raw-rpc/one-process').glob('*-inference.pkl.gz'))


def test_actual_clipped_tied_noise_is_rejected_even_with_a_rehashed_trace(tmp_path):
    result, _ = run(tmp_path / 'controls')
    group = result['completed_groups'][2]
    path = Path(group['result_path']).parent
    receipt = json.loads(Path(group['result_path']).read_text())
    episode = receipt['episodes'][0]
    trace = path / episode['trace']
    with gzip.open(trace, 'rb') as stream:
        rows = pickle.load(stream)
    assert np.abs(rows[0]['executed_noise']).max() > 1
    rows[0]['executed_noise'] = np.clip(rows[0]['executed_noise'], -1, 1)
    rows[0]['native_initial_noise'] = np.repeat(rows[0]['executed_noise'][None], 10, axis=0)
    with gzip.open(trace, 'wb') as stream:
        pickle.dump(rows, stream)
    episode['trace_sha256'] = file_sha256(trace)
    with pytest.raises(ValueError, match='unbounded source warmup'):
        module.audit_noise(path, receipt, group['group'])


def test_rpc_failure_resume_preserves_unknown_charge_and_invocation_identity(tmp_path):
    groups = plan()['cases'][0]['groups'][2:3]
    first = TargetSimulator(0, fail_on_step=1)
    with pytest.raises(RuntimeError, match='injected RPC'):
        run(tmp_path / 'controls', groups, runtime='a', sims=[first])
    result, _ = run(tmp_path / 'controls', groups, resume=True, runtime='b')
    assert result['completed'] and result['actual_runtime_ids'] == ['a', 'b']
    assert not result['strict_common_runtime_primary_eligible']
    assert charged_actions(tmp_path / 'controls') > 23


def test_separate_two_worker_ledger_never_reserves_or_refunds(tmp_path):
    value, path = plan(), tmp_path / 'ledger.json'
    rows = {f'worker-{i}': dict(run_id=f'worker-{i}', source_commit='a' * 40, worker=i,
        noise_plan_sha256=value['noise_plan_sha256'], maximum_total_actions=76000, worker_seconds=3600,
        purpose='target_evaluation', source_pretraining_data=False, fresh_ppo_training_data=False) for i in (0, 1)}
    ledger = dict(format=module.LEDGER_FORMAT, noise_plan_sha256=value['noise_plan_sha256'], maximum_total_actions=152000,
        maximum_gpu_seconds=7200, source_campaign_immutable=True, scope_amendment='Released target time inside32h; original source pool unchanged', reservations=rows)
    path.write_text(json.dumps(ledger))
    before = path.read_bytes()
    assert module.reservation(path, value, 0, 'worker-0', 'a' * 40)['record'] == rows['worker-0']
    assert path.read_bytes() == before
    ledger['reservations']['worker-1']['maximum_total_actions'] += 1
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='immutable diagnostic'):
        module.reservation(path, value, 0, 'worker-0', 'a' * 40)
