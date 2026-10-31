"""Setup/failures/retries share the original two-hour baseline time budget."""
import json

import pytest

from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import finish
from fluffy_bassoon.meta_rl.vla_target_online_dsrl_clock import prepare_attempt, worker_contract
from fluffy_bassoon.meta_rl.vla_target_online_dsrl_six import invocation_deadline, scientific_contract


SOURCE, RUN = 'a' * 40, 'target-online-dsrl-six-clock-test'


def test_initial_slice_plus_scheduler_allowance_reserves_exactly_two_hours(tmp_path):
    value = prepare_attempt(tmp_path, SOURCE, RUN, started_unix=1000.)
    assert value['slice_seconds'] == 7140 and value['stop_unix'] == 8140
    assert value['driver_stop_unix'] == 8020 < value['outer_timeout_unix'] == 8080 < value['stop_unix']
    ledger = json.loads((tmp_path / 'worker-time.json').read_text())
    assert ledger['worker_seconds'] == ledger['attempts'][0]['reserved_seconds'] == 7200
    assert ledger['attempts'][0]['started_unix'] == 1000.
    assert ledger['contract'] == worker_contract(SOURCE, RUN)


def test_shorter_resume_slice_preserves_science_and_charges_finished_time(tmp_path):
    first = prepare_attempt(tmp_path, SOURCE, RUN, started_unix=1000.)
    finish(tmp_path, first['attempt'], 124, now=1500.)
    before = scientific_contract(SOURCE, 0)
    second = prepare_attempt(tmp_path, SOURCE, RUN, resume=True, started_unix=3000.)
    assert second['prior_charged_seconds'] == 560
    assert second['slice_seconds'] == 6580 and second['stop_unix'] == 9580
    assert scientific_contract(SOURCE, 0) == before
    assert 'stop_unix' not in before and 'slice_seconds' not in before
    ledger = json.loads((tmp_path / 'worker-time.json').read_text())
    assert ledger['contract'] == worker_contract(SOURCE, RUN)
    assert sum(row.get('charged_seconds', row['reserved_seconds']) for row in ledger['attempts']) == 7200


def test_setup_failure_is_charged_before_any_driver_or_simulator(tmp_path):
    first = prepare_attempt(tmp_path, SOURCE, RUN, started_unix=1000.)
    row = finish(tmp_path, first['attempt'], 1, now=1012.)
    assert row['charged_seconds'] == 72 and row['exit_code'] == 1
    second = prepare_attempt(tmp_path, SOURCE, RUN, resume=True, started_unix=2000.)
    assert second['slice_seconds'] == 7068 and second['prior_charged_seconds'] == 72


def test_unclosed_attempt_consumes_its_reservation_and_cannot_reset_the_clock(tmp_path):
    prepare_attempt(tmp_path, SOURCE, RUN, started_unix=1000.)
    with pytest.raises(RuntimeError, match='original two-hour budget'):
        prepare_attempt(tmp_path, SOURCE, RUN, resume=True, started_unix=2000.)
    with pytest.raises(ValueError, match='original source/run'):
        prepare_attempt(tmp_path, SOURCE, RUN, started_unix=2000.)
    assert len(json.loads((tmp_path / 'worker-time.json').read_text())['attempts']) == 1


def test_actual_overrun_is_never_truncated_to_fit_budget(tmp_path):
    first = prepare_attempt(tmp_path, SOURCE, RUN, started_unix=1000.)
    row = finish(tmp_path, first['attempt'], 143, now=9000.)
    assert row['charged_seconds'] == 8060 and row['slice_overrun']
    with pytest.raises(RuntimeError, match='original two-hour budget'):
        prepare_attempt(tmp_path, SOURCE, RUN, resume=True, started_unix=10000.)


def test_explicit_invocation_stop_changes_deadline_without_changing_scientific_limits():
    assert invocation_deadline(8140., now_unix=1000., now_monotonic=10.) == 7150.
    assert invocation_deadline(9580., now_unix=3000., now_monotonic=20.) == 6600.
    contract = scientific_contract(SOURCE, 0)
    assert contract['maximum_driver_seconds'] == 7200
    assert contract['maximum_total_actions'] == 64000 and contract['maximum_actions_per_case'] == 9500
    for stop in (1000., 999., float('nan'), 9000.):
        with pytest.raises(ValueError, match='explicit live invocation stop'):
            invocation_deadline(stop, now_unix=1000., now_monotonic=10.)


def test_resume_rejects_a_different_source_pin_and_does_not_reset_time(tmp_path):
    first = prepare_attempt(tmp_path, SOURCE, RUN, started_unix=1000.)
    finish(tmp_path, first['attempt'], 0, now=1100.)
    with pytest.raises(ValueError, match='original source/run'):
        prepare_attempt(tmp_path, 'b' * 40, RUN, resume=True, started_unix=2000.)
    assert len(json.loads((tmp_path / 'worker-time.json').read_text())['attempts']) == 1
