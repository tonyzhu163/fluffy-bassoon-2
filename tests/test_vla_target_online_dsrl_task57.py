"""Task57 grant/clock/storage glue; allocated Linux CPU fixtures only."""
import json
import os

import pytest

from fluffy_bassoon.meta_rl import vla_target_online_dsrl_task57 as task


SOURCE, RUN = 'a' * 40, 'dsrl-task57-clock-fixture'


def grant(tmp_path, **changes):
    value = dict(format='pi05_task57_dsrl_grant_v1', approved=True, source_commit=SOURCE,
        run_id=RUN, maximum_gpu_seconds=36000, maximum_total_actions=1000000,
        maximum_output_bytes=64 * 1024 ** 3, task_id=57, learner_seed=0,
        motor_receipt_sha256=task.PI05_RECEIPT, reviewed_main_commit=task.REVIEWED_MAIN,
        data_role='target-online-baseline', motor_parameter_sha256=task.MOTOR_PARAMETER_SHA256)
    value.update(changes)
    path = tmp_path / 'grant.json'
    path.write_text(json.dumps(value))
    return path, task.onset.sha256(path)


def test_contract_binds_authors_task_frozen_pi05_cost_units_and_no_source_admission(tmp_path):
    path, digest = grant(tmp_path)
    task.load_grant(path, digest, SOURCE, RUN)
    value = task.scientific_contract(SOURCE, digest)
    assert value['task_id'] == 57 and value['learner_seed'] == 0
    assert value['train_states'] == list(range(40)) and value['query_states'] == list(range(40, 50))
    assert value['entire_motor_frozen'] and value['native_horizon'] == value['execute_steps'] == 10
    assert value['nominal_optimizer_stop'] == 500000 and value['maximum_optimizer_updates'] == 500780
    assert value['recipe']['warmup_condition'].startswith('replay >500')
    assert value['recipe']['batch_size'] == 256 and value['recipe']['update_to_data'] == 20
    assert value['data_role'] == 'target-online-baseline' and not value['source_assembly_admission']
    assert value['reviewed_main_commit'] == 'f7dde2c94baee3ee59c4450a0ed30543d1cba67f'
    assert 'slice_seconds' not in value and 'pause_after_updates' not in value


@pytest.mark.parametrize('changes', [dict(approved=False), dict(maximum_gpu_seconds=36001),
    dict(maximum_total_actions=1000001), dict(task_id=0), dict(learner_seed=1),
    dict(data_role='source-training'), dict(source_commit='b' * 40)])
def test_grant_cannot_silently_expand_or_switch_scope(tmp_path, changes):
    path, digest = grant(tmp_path, **changes)
    with pytest.raises(ValueError, match='funding record'):
        task.load_grant(path, digest, SOURCE, RUN)


def test_ten_hour_clock_charges_setup_retry_and_keeps_science_unchanged(tmp_path):
    path, digest = grant(tmp_path)
    root = tmp_path / 'worker'
    first = task.prepare_attempt(root, SOURCE, RUN, path, digest, slice_seconds=7140, started_unix=1000.)
    assert first['stop_unix'] == 8140 and first['driver_stop_unix'] == 7960
    assert first['driver_stop_unix'] < first['outer_timeout_unix'] < first['stop_unix']
    task.finish(root, first['attempt'], 1, now=1012.)
    second = task.prepare_attempt(root, SOURCE, RUN, path, digest, slice_seconds=35940,
        resume=True, started_unix=2000.)
    assert second['prior_charged_seconds'] == 72 and second['slice_seconds'] == 35868
    ledger = json.loads((root / 'worker-time.json').read_text())
    assert ledger['worker_seconds'] == 36000
    assert sum(row.get('charged_seconds', row['reserved_seconds']) for row in ledger['attempts']) == 36000
    assert ledger['contract']['nominal_optimizer_stop'] == 500000
    with pytest.raises(ValueError, match='Insufficient cumulative'):
        task.prepare_attempt(root, SOURCE, RUN, path, digest, resume=True, started_unix=3000.)


def test_deadline_changes_only_invocation_time():
    assert task.invocation_deadline(2000., now_unix=1000., now_monotonic=50.) == 1050.
    for stop in (1000., 999., float('nan'), 37001.):
        with pytest.raises(ValueError, match='live invocation deadline'):
            task.invocation_deadline(stop, now_unix=1000., now_monotonic=50.)


def test_storage_counts_hardlinks_once_and_atomic_replacement_preserves_milestone(tmp_path):
    slot = tmp_path / 'recovery-0.pkl'
    slot.write_bytes(b'full-state')
    archive = tmp_path / 'milestone.pkl'
    os.link(slot, archive)
    assert task.output_bytes(tmp_path) == len(b'full-state')
    temporary = tmp_path / 'replacement.tmp'
    temporary.write_bytes(b'new-state')
    temporary.replace(slot)
    assert archive.read_bytes() == b'full-state' and slot.read_bytes() == b'new-state'
    assert task.output_bytes(tmp_path) == len(b'full-state') + len(b'new-state')
    status = task.storage_status(tmp_path, tmp_path)
    assert status['maximum_full_checkpoint_bytes'] == len(b'new-state')
    assert status['next_episode_reservation_bytes'] >= 4 * 1024 ** 3
