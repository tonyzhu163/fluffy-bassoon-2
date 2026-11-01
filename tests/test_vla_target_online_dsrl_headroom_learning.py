"""Mocked boundaries of selected-task learning; run on allocated Linux only."""
import json
import numpy as np
from fluffy_bassoon.meta_rl import vla_target_online_dsrl_headroom as phase
from fluffy_bassoon.meta_rl import vla_target_online_dsrl_headroom_learning as learning
from test_vla_target_online_dsrl_task57_recovery import FakeSAC, FakeMotor, FakeSimulator, patch_rng


def test_fresh_native_readmission_aborts_before_any_sgd_and_cannot_resume_past_it(monkeypatch, tmp_path):
    patch_rng(monkeypatch, phase)
    # This existing episode-flow fixture has a deliberately non-Flax fake SAC.
    # The production exact-Flax restore is covered in test_vla_headroom_restore_guard.
    from fluffy_bassoon.meta_rl import vla_headroom_restore_guard
    def restore_fake(learner, sampler, motor, checkpoint, progress, contract):
        saved = learner.restore(checkpoint, sampler, contract)
        phase.onset._restore_motor(motor, saved['motor_rng'])
        assert saved['training'] == progress['training'] and learner.updates == progress['model']['updates']
        assert len(learner.replay) == progress['model']['replay_rows']
        assert phase.onset._signature(learner) == progress['model']['signature']
        return saved, None
    monkeypatch.setattr(vla_headroom_restore_guard, 'restore_checked', restore_fake)
    # Qualified fake succeeds on even states:10/20, outside weak2..8/20 band.
    learner, motor = FakeSAC(), FakeMotor()
    sim = FakeSimulator(learner)
    root, output = tmp_path / 'experiment', tmp_path / 'experiment/case-learn-29'
    inherited = root / 'case-screen-29/physical-journal'
    inherited.mkdir(parents=True)
    phase.onset.write_json(inherited / 'prior-reset.json', dict(role='query', kind='reset',
        actual_steps=10, maximum_steps=10, returned=True))
    contract = dict(fixed='readmission-fixture')
    result = learning.run_learning(sim, motor, learner, np.random.default_rng(0),
        output=output, root=root, contract=contract, task_id=29,
        stop_updates=40, panel_interval=40)
    assert result['phase'] == 'aborted_before_sgd' and learner.updates == 0
    assert result['training']['episodes'] == 0 and len(result['panels']) == 1
    assert result['panels'][0]['threshold_updates'] == 0
    assert result['cumulative_whole_phase_charge'] == result['total_action_charge']
    assert result['total_action_charge'] == result['learning_case_action_charge'] + 10
    assert result['stop_reason'] == 'initial_native_off_weak_band_no_task_substitution'
    restored = FakeSAC()
    again = learning.run_learning(FakeSimulator(restored), FakeMotor(), restored,
        np.random.default_rng(999), output=output, root=root, contract=contract,
        task_id=29, stop_updates=40, panel_interval=40, resume=True)
    assert again['phase'] == 'aborted_before_sgd' and restored.updates == 0
    assert again['total_action_charge'] == result['total_action_charge']


def test_zero_update_controls_and_fixed_endpoint_do_not_select_best_checkpoint(monkeypatch, tmp_path):
    patch_rng(monkeypatch, phase)
    learner, motor = FakeSAC(), FakeMotor()
    root, output = tmp_path / 'experiment', tmp_path / 'experiment/case-learn-29'
    # Small internal fixture bank bypasses20-case weak-band admission only in
    # this unit fixture; public CLI always uses all20 cases from frozen JSON.
    cases = [phase.target(29, 40), phase.target(29, 41)]
    result = learning.run_learning(FakeSimulator(learner), motor, learner, np.random.default_rng(0),
        output=output, root=root, contract=dict(fixed='controls-fixture'), task_id=29,
        query_cases=cases, stop_updates=120, panel_interval=40)
    assert result['completed'] and learner.updates == 120
    assert [p['kind'] for p in result['panels']] == [
        'initial_native', 'initial_tied', 'initial_sac', 'learned', 'learned', 'learned', 'final_native']
    assert all(p['threshold_updates'] == p['actual_optimizer_updates'] == 0 for p in result['panels'][:3])
    assert result['training']['episodes'] == 3 and len(learner.replay) == 506
    summary = learning.endpoint_summary(result)
    assert summary['fixed_endpoint_optimizer_updates'] == 120
    assert summary['final_learned_successes'] == summary['final_native_successes'] == 1
    assert summary['primary_paired_success_count_gain'] == 0
    assert not summary['predeclared_six_of_twenty_gain_and_last_three_persistence']
    # Every control/learned receipt uses identical physical reset seeds/cases.
    assert all(r['case']['scene_seed'] == 123 for p in result['panels'] for r in p['receipts'])
    assert all(r['query_data_added_to_replay'] is False for p in result['panels'] for r in p['receipts'])
