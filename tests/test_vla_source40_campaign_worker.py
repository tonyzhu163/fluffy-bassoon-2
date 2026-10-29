"""Unique stream ownership, exact resume identities and action-cap binding."""
import json
from pathlib import Path

import pytest

from fluffy_bassoon.meta_rl.vla_source40_campaign import ALL_TASKS, VALIDATION_TASKS
from fluffy_bassoon.meta_rl.vla_source40_campaign_worker import (
    action_charge, assigned_slots, child_command, prepare_stream, stream_action, stream_record)


def campaign():
    return dict(campaign_sha256='c' * 64, source_schedule_sha256='s' * 64,
        sac_slots=[dict(task_key=task, partition=split, maximum_total_actions=cap)
            for split, tasks, cap in (('train', ALL_TASKS, 9500), ('validation', VALIDATION_TASKS, 12000))
            for task in tasks])


def record():
    value = campaign()
    return stream_record(value['sac_slots'][0], value, {}, 'a' * 40)[0]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def index(output, current, **changes):
    write(output / 'experiment/contract.json', dict(source_commit=current['source_commit'],
        schedule_sha256=current['source_schedule_sha256'], task=dict(task_key=current['task_key']),
        partition=current['partition'], maximum_total_actions=current['maximum_total_actions'], source_only=True))
    row = dict(task_key=current['task_key'], partition=current['partition'],
        maximum_total_actions=current['maximum_total_actions'], total_action_budget_charge=8000,
        completed=False, collection_qualified=False)
    write(output / 'experiment/index.json', dict(row, **changes))


def test_eight_workers_cover_each_of_the_48_reservations_once():
    value = campaign()
    groups = [assigned_slots(value, worker, 8) for worker in range(8)]
    assert [len(group) for group in groups] == [6] * 8
    keys = [(row['task_key'], row['partition']) for group in groups for row in group]
    assert len(keys) == len(set(keys)) == 48
    assert set(keys) == {(row['task_key'], row['partition']) for row in value['sac_slots']}
    with pytest.raises(ValueError, match='valid shard'):
        assigned_slots(value, 8, 8)


def test_reserved_original_run_and_commit_survive_new_worker_source():
    value, previous = campaign(), record()
    previous.update(source_commit='b' * 40, run_id='libero40-1m-sac-20261029-s00-train-r1')
    selected, reserved = stream_record(value['sac_slots'][0], value,
        {'libero_spatial:00/train': previous}, 'a' * 40)
    assert reserved and selected == previous
    command = child_command(Path('/project'), selected, 'resume')
    assert command[-1] == '--resume'
    assert command[2] == 'b' * 40
    assert command[3] == previous['run_id']
    assert '9500' not in command  # The pinned child binds its original campaign cap.


@pytest.mark.parametrize('change', [
    dict(maximum_total_actions=9501), dict(task_key='libero_object:00'),
    dict(source_commit='z' * 40), dict(run_id='../other-run'), dict(campaign_sha256='x' * 64)])
def test_changed_reservation_cannot_be_reused(change):
    value, previous = campaign(), record()
    previous.update(change)
    with pytest.raises(ValueError, match='reservation changed'):
        stream_record(value['sac_slots'][0], value, {'libero_spatial:00/train': previous}, 'a' * 40)


def test_terminal_raw_only_stream_skips_without_claiming_a_qualified_teacher(tmp_path):
    current = record()
    index(tmp_path, current, completed=True, collection_qualified=False)
    assert stream_action(tmp_path, current, True) == 'skip_raw_only'
    index(tmp_path, current, completed=True, collection_qualified=True)
    assert stream_action(tmp_path, current, True) == 'skip_completed'
    index(tmp_path, current, completed=True, total_action_budget_charge=9501)
    with pytest.raises(ValueError, match='exceeds'):
        stream_action(tmp_path, current, True)


def test_failed_partial_resumes_but_absent_recovery_or_exit_requires_review(tmp_path):
    current = record()
    index(tmp_path, current)
    assert stream_action(tmp_path, current, True) == 'skip_pending_or_unowned'
    (tmp_path / 'exit-code.txt').write_text('1\n')
    assert stream_action(tmp_path, current, True) == 'skip_needs_review'
    (tmp_path / 'experiment/source-recovery.pkl').write_bytes(b'full trusted recovery')
    assert stream_action(tmp_path, current, True) == 'resume'
    (tmp_path / 'exit-code.txt').unlink()
    previous = dict(current, status='timed_out')
    assert stream_action(tmp_path, current, True, previous) == 'resume'
    with pytest.raises(ValueError, match='another source stream'):
        stream_action(tmp_path, current, True, dict(previous, source_commit='b' * 40))


def test_resume_cap_misbind_is_rejected_before_any_child_can_execute(tmp_path):
    current = record()
    index(tmp_path, current)
    contract = tmp_path / 'experiment/contract.json'
    value = json.loads(contract.read_text())
    value['maximum_total_actions'] += 1
    write(contract, value)
    (tmp_path / 'exit-code.txt').write_text('1\n')
    (tmp_path / 'experiment/source-recovery.pkl').write_bytes(b'recovery')
    with pytest.raises(ValueError, match='recovery contract differs'):
        stream_action(tmp_path, current, True)


def test_actively_locked_collector_is_never_resumed(tmp_path, monkeypatch):
    current = record()
    index(tmp_path, current)
    (tmp_path / 'exit-code.txt').write_text('1\n')  # A stale earlier attempt receipt.
    (tmp_path / 'experiment/source-recovery.pkl').write_bytes(b'recovery')
    monkeypatch.setattr('fluffy_bassoon.meta_rl.vla_source40_campaign_worker.collector_locked', lambda _: True)
    assert stream_action(tmp_path, current, True) == 'skip_running'


def test_joint_action_charge_keeps_uncertain_execution_and_settling(tmp_path):
    journal = tmp_path / 'experiment/raw/decision-journal'
    write(journal / 'intent-1.json', dict(maximum_primitive_steps=10))
    write(journal / 'decision-1.json', dict(primitive_steps=6))
    write(journal / 'intent-2.json', dict(maximum_primitive_steps=10))
    write(journal / 'reset-qualification-1.json', dict(settling_steps=10, returned=True))
    write(journal / 'reset-episode-1.json', dict(settling_steps=10, returned=False))
    assert action_charge(tmp_path) == dict(known_primitive_steps=6,
        uncertain_primitive_steps_ceiling=10, settling_step_charge=20, total_action_charge=36)


def test_preparation_does_not_recreate_or_migrate_missing_reserved_worktree(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('fluffy_bassoon.meta_rl.vla_source40_campaign_worker.subprocess.run',
                        lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match='recover it explicitly'):
        prepare_stream(tmp_path, tmp_path / 'repo', record(), reserved=True)
    assert not calls


def test_fresh_stream_output_collision_cannot_silently_restart(tmp_path):
    current = record()
    with pytest.raises(ValueError, match='collides'):
        stream_action(tmp_path, current, False)
