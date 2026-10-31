"""New headroom plan/selection/cost glue; actual tests on allocated Linux only."""
import json
import pytest
from fluffy_bassoon.meta_rl import vla_target_online_dsrl_headroom as phase


def receipts(counts):
    return [dict(case=phase.target(task, state, bank='screen'),
        success=state < counts.get(task, 0), initial_state_vector_sha256=f'{task}/{state}')
        for task in phase.CANDIDATES for state in range(20)]


def test_closed_candidates_and_separate_physical_query_bank():
    p = phase.plan('a' * 40)
    assert phase.CANDIDATES == (0, 29, 33, 38, 44, 53, 59, 64, 70, 77, 79, 82)
    assert len(p['screening_cases']) == 240
    assert all(c['initial_state'] < 20 for c in p['screening_cases'])
    bank = phase.query_bank(29)
    assert len(bank) == 20 and len({c['initial_state'] for c in bank}) == 10
    assert len({phase.onset._seed(c, 'query', 0) for c in bank}) == 20
    assert p['source_assembly_admission'] is False and p['learner_seed'] == 0
    assert p['maximum_job_gpu_seconds'] < 8 * 3600
    assert p['all_in_planned_action_bound'] < p['maximum_total_actions'] == 1000000


def test_native_band_nearest_four_and_fixed_id_tie_break():
    order, table = phase.ranked_candidates(receipts({0: 1, 29: 8, 33: 4, 38: 4, 44: 3, 53: 5, 59: 9}))
    assert order == [33, 38, 44, 53, 29]
    assert len(table) == 12 and all(row['episodes'] == 20 for row in table)
    assert phase.ranked_candidates(receipts({}))[0] == []
    with pytest.raises(ValueError, match='all20'):
        phase.ranked_candidates(receipts({33: 4})[:-1])


def test_job_and_stage_caps_keep_exact_cumulative_cost_on_retry(tmp_path):
    first = phase.prepare_attempt(tmp_path, 'a' * 40, 'headroom-fixture',
        slice_seconds=7140, started_unix=1000., stage='screen')
    phase.finish(tmp_path, first['attempt'], 1, now=1010.)
    second = phase.prepare_attempt(tmp_path, 'a' * 40, 'headroom-fixture',
        slice_seconds=25140, started_unix=2000., stage='screen', resume=True)
    assert second['slice_seconds'] == 10670  # 3h less70 charged and60 allowance.
    phase.finish(tmp_path, second['attempt'], 0, now=2670.)
    third = phase.prepare_attempt(tmp_path, 'a' * 40, 'headroom-fixture',
        slice_seconds=25140, started_unix=4000., stage='learn', resume=True)
    assert third['slice_seconds'] == 25140
    ledger = json.loads((tmp_path / 'worker-time.json').read_text())
    assert ledger['worker_seconds'] == 57600 and len(ledger['attempts']) == 3
    with pytest.raises(ValueError, match='below8h'):
        phase.prepare_attempt(tmp_path, 'a' * 40, 'headroom-fixture', slice_seconds=28800,
            resume=True, stage='learn')
