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


def test_native_screen_resumes_exact_cursor_and_rejects_ceiling_confirmation(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from test_vla_target_online_dsrl_task57_recovery import FakeMotor, FakeSimulator, patch_rng
    patch_rng(monkeypatch, phase)
    class NativeSim(FakeSimulator):
        failed = False
        def __init__(self, task):
            super().__init__(SimpleNamespace(updates=0))
            self.task = task
            self.metadata['task_id'] = task
        def reset(self, state, seed):
            assert seed == 123
            if self.task == 0 and state == 7 and not NativeSim.failed:
                NativeSim.failed = True
                raise RuntimeError('one actual reset interrupted')
            return super().reset(state, seed)
        def step(self, actions):
            result = super().step(actions)
            # All tasks screen4/20. Task0 confirms20/20 and MUST be rejected;
            # next ranked29 confirms4/20, genuinely weak on separate states.
            result['success'] = bool(result['done'] and
                (self.state < 4 or (self.state >= 40 and (self.task == 0 or self.state < 42))))
            return result
        def close(self):
            pass
    root = tmp_path / 'experiment'
    contract = phase.plan('a' * 40)
    with pytest.raises(RuntimeError, match='actual reset interrupted'):
        phase.run_screen(FakeMotor(), NativeSim, root=root, contract=contract, deadline=float('inf'))
    cursor = json.loads((root / 'screen-state.json').read_text())
    assert cursor['screen_cursor'] == 7 and cursor['phase'] == 'screen'
    result = phase.run_screen(FakeMotor(), NativeSim, root=root, contract=contract,
        deadline=float('inf'), resume=True)
    assert result['phase'] == 'selected' and result['selected_task'] == 29
    assert result['screen_cursor'] == len(result['screen_receipts']) == 240
    assert [row['successes'] for row in result['confirmations']] == [20, 4]
    frozen = json.loads((root / 'frozen-selection.json').read_text())
    assert frozen['selected_before_any_sgd'] and frozen['optimizer_updates'] == 0
    assert len(frozen['query_bank']) == 20 and frozen['source_assembly_admission'] is False
    assert result['all_in_actions'] == 3930  #280×(4actual+10settle)+one unresolved10.
