"""Saved-only report checks; no model construction or simulator execution."""
import json

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_source40_noise_plan import METHODS, build_plan
import fluffy_bassoon.meta_rl.vla_source40_noise_report as module


def row(method, seed=42):
    rng = np.random.default_rng(seed)
    proposed = None if method == 'native' else np.zeros(32, np.float32) if method == 'zero' else rng.standard_normal(32).astype(np.float32)
    executed = None if proposed is None else np.clip(proposed, -1., 1.) if method == 'tied_gaussian_clipped' else proposed
    noise = rng.standard_normal((10, 32)).astype(np.float32) if method == 'native' else np.repeat(executed[None], 10, axis=0)
    return dict(native_initial_noise=noise, proposed_noise=proposed, executed_noise=executed,
        executed_actions=np.zeros((10, 7), np.float32), primitive_steps=10, raw_rewards=[0.] * 10, done=True)


def groups():
    return [dict(task_id=task, method=method, case_id=f'task-{task}', success=(task + i) % 3 == 0,
        charged_actions=410, query_seed=task + 1, initial_observation_sha256=f'frame-{task}',
        initial_public_arrays_sha256=f'public-{task}', nominal_state_sha256=f'nominal-{task}',
        actual_post_settle_state_sha256=f'physics-{task}', runtime_id=f'worker-{task % 2}')
        for task in range(90) for i, method in enumerate(METHODS)]


def test_saved_noise_reconstruction_factors_tying_and_clipping():
    for method in METHODS:
        module.verify_noise([row(method)], method, 42)
    changed = row('tied_gaussian')
    changed['executed_noise'] = np.clip(changed['executed_noise'], -1, 1)
    changed['native_initial_noise'] = np.repeat(changed['executed_noise'][None], 10, axis=0)
    with pytest.raises(ValueError, match='exact Gaussian/std/clipping'):
        module.verify_noise([changed], 'tied_gaussian', 42)
    with pytest.raises(ValueError, match='exact RNG'):
        module.verify_noise([row('native')], 'native', 43)


def test_scores_are_contemporaneous_and_all_six_pairs_keep_task_discordances():
    scores, pairs, per_task = module.summarize(groups())
    assert all(scores[m]['successes'] == 30 and scores[m]['queries'] == 90 for m in METHODS)
    assert len(pairs) == 6 and len(per_task) == 90
    first = pairs[0]
    assert first['counts'] == dict(both_success=0, both_failure=30, left_only=30, right_only=30)
    assert first['task_ids']['left_only'] == list(range(0, 90, 3))
    assert module.REFERENCE['native_successes'] == 31 and 'reference only' in module.REFERENCE['status']


@pytest.mark.parametrize('field', ['query_seed', 'initial_observation_sha256', 'initial_public_arrays_sha256',
    'nominal_state_sha256', 'actual_post_settle_state_sha256', 'runtime_id'])
def test_actual_initial_or_process_mismatch_blocks_paired_report(field):
    values = groups()
    values[1][field] = 'changed'
    with pytest.raises(ValueError, match='Actual paired initialization/process mismatch'):
        module.summarize(values)


def test_missing_or_duplicate_query_never_produces_full90_score():
    with pytest.raises(ValueError, match='all360'):
        module.summarize(groups()[:-1])
    with pytest.raises(ValueError, match='Duplicate'):
        module.summarize(groups() + [groups()[0]])


def test_failed_or_nonterminal_worker_is_rejected_before_saved_arrays(tmp_path):
    plan = build_plan(worktree='/old/immutable/worktree', source_commit='a' * 40)
    plan_path, ledger_path = tmp_path / 'plan.json', tmp_path / 'ledger.json'
    plan_path.write_text(json.dumps(plan))
    rows, roots = {}, []
    for i in (0, 1):
        root = tmp_path / f'worker-{i}'
        root.mkdir()
        roots.append(root)
        (root / 'launcher.lock').write_text('')
        (root / 'exit-code.txt').write_text('1')
        rows[root.name] = dict(worker=i)
    ledger_path.write_text(json.dumps(dict(format=module.LEDGER_FORMAT,
        noise_plan_sha256=plan['noise_plan_sha256'], maximum_total_actions=152000,
        maximum_gpu_seconds=7200, source_campaign_immutable=True, reservations=rows)))
    before = {path: path.read_bytes() for root in roots for path in root.iterdir()}
    with pytest.raises(ValueError, match='failed/nonterminal'):
        module.aggregate(plan_path, module.sha(plan_path), ledger_path, module.sha(ledger_path), roots)
    assert all(path.read_bytes() == value for path, value in before.items())
