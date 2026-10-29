"""Immutable PPO slices cannot spend an extra action/time budget on resume."""
import json

import pytest

from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin, finish


def test_closed_attempt_charges_actual_time_and_resume_retains_the_contract(tmp_path):
    contract = dict(source_commit='a' * 40, maximum_total_actions=200000, memory_mode='full')
    first = begin(tmp_path, contract, slice_seconds=600, worker_seconds=1800, now=100)
    receipt = finish(tmp_path, first, 0, now=220)
    assert receipt['charged_seconds'] == 180 and not receipt['slice_overrun']
    second = begin(tmp_path, contract, slice_seconds=600, worker_seconds=1800, resume=True, now=300)
    assert second != first and (first / 'attempt.json').exists()
    with pytest.raises(ValueError, match='exact pinned'):
        begin(tmp_path, dict(contract, maximum_total_actions=300000),
              slice_seconds=600, worker_seconds=1800, resume=True, now=400)


def test_unresolved_attempt_retains_its_full_reservation(tmp_path):
    contract = dict(source_commit='a' * 40)
    begin(tmp_path, contract, slice_seconds=600, worker_seconds=1200, now=100)
    with pytest.raises(RuntimeError, match='cumulative'):
        begin(tmp_path, contract, slice_seconds=600, worker_seconds=1200, resume=True, now=1000)
    saved = json.loads((tmp_path / 'worker-time.json').read_text())
    assert saved['attempts'][0]['reserved_seconds'] == 660


def test_actual_overrun_is_reported_without_hiding_time(tmp_path):
    first = begin(tmp_path, {}, slice_seconds=600, worker_seconds=1200, now=100)
    receipt = finish(tmp_path, first, 124, now=900)
    assert receipt['charged_seconds'] == 860 and receipt['slice_overrun']
    with pytest.raises(ValueError, match='already closed'):
        finish(tmp_path, first, 0, now=1000)


def test_resumption_requires_the_original_worker_time_ledger(tmp_path):
    with pytest.raises(ValueError, match='without its original'):
        begin(tmp_path, {}, slice_seconds=600, worker_seconds=1200, resume=True, now=100)
