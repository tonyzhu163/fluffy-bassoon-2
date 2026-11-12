"""Finish Cabinet iterations9–10 from the verified iteration8 recovery boundary."""
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time

from teacher_local import COLD, HERE, PROJECT, archive, sha, snapshot, write

CABINET_ARCHIVE_SHA = 'acbeeeff229d8aeaaecf147c07302cccf232298431454855b85b7b5c8a1f79a6'


def restore_members(receipt, destination, selected):
    """Stage back only the recovery inputs, checking every used byte."""
    assert receipt['verified'] and receipt['all_stageback_members_verified']
    cold = Path(receipt['cold'])
    assert cold.stat().st_size == receipt['archive_bytes']
    expected = {row['path']: row for row in receipt['files'] if row['path'] in selected}
    assert set(expected) == selected and selected and not destination.exists()
    destination.mkdir(parents=True)
    seen = set()
    with tarfile.open(cold, 'r:') as packed:
        for member in packed:
            if member.name not in selected:
                continue
            rel = PurePosixPath(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            assert member.name not in seen
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with packed.extractfile(member) as a, target.open('xb') as b:
                shutil.copyfileobj(a, b, 8 << 20)
            assert target.stat().st_size == expected[member.name]['bytes']
            assert sha(target) == expected[member.name]['sha256']
            seen.add(member.name)
    assert seen == selected
    return dict(verified=True, files=len(seen), bytes=sum(x['bytes'] for x in expected.values()),
                archive_sha256=receipt['archive_sha256'],
                policy='Every selected member rehashed against the fully verified original archive inventory')


def main():
    pin, attempt = sys.argv[1:]
    assert attempt == 'attempt04'
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    code = HERE.parent.parent
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == pin
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    started = time.time()
    prior = COLD / 'teachers/OpenCabinet/attempt03'
    receipt_path = prior / 'ARCHIVE.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['archive_sha256'] == CABINET_ARCHIVE_SHA
    restored = Path(receipt['source'])
    assert restored == Path('/tmp') / HERE.name / 'teachers/OpenCabinet/attempt03'
    checkpoint_prefix = 'OpenCabinet/checkpoints/global_step_8/'
    selected = {row['path'] for row in receipt['files']
                if row['path'].startswith((checkpoint_prefix, 'canonical-eval/'))
                or row['path'] == 'before-evaluation.json'
                or (row['path'].startswith('environment/') and row['path'].endswith('/before-i000-round00-episodes.json'))}
    required_bytes = sum(x['bytes'] for x in receipt['files'] if x['path'] in selected)
    source = restored.parent / attempt
    assert not source.exists()
    source.parent.mkdir(parents=True, exist_ok=True)
    assert source.parent.stat().st_dev != PROJECT.stat().st_dev
    assert shutil.disk_usage(source.parent).free >= 270 * 2**30 + required_bytes
    remote = COLD / 'teachers/OpenCabinet' / attempt
    remote.mkdir(parents=True, exist_ok=False)
    write(remote / 'LAUNCH.json', dict(source_commit=pin, hostname=socket.gethostname(),
        task='OpenCabinet', attempt=attempt, local_output=str(source), started_unix=started,
        scheduler_maximum_minutes=45, reserved_GPU_hours=6,
        checkpoint_request_seconds=1500, phase_deadline_seconds=2100,
        process_timeout_seconds=2200, learner_settings_unchanged=True,
        previous_archive=str(receipt_path), previous_archive_receipt_sha256=sha(receipt_path),
        resumed_global_step=8, target_global_step=10))
    restored_receipt = restore_members(receipt, restored, selected)
    checkpoint = restored / checkpoint_prefix
    boundary_path = checkpoint / 'pilot-boundary.json'
    boundary = json.loads(boundary_path.read_text())
    assert boundary['completed'] and boundary['global_step'] == 8
    for row in boundary['actor_checkpoint_files'] + boundary['workers']:
        path = Path(row['path'])
        assert path.is_relative_to(checkpoint) and sha(path) == row['sha256']
    before = json.loads((restored / 'before-evaluation.json').read_text())
    assert before['completed']
    refs, successes = [], 0
    for rank in range(8):
        path = restored / 'environment' / f'rank-{rank:02d}/before-i000-round00-episodes.json'
        data = json.loads(path.read_text())
        assert data['completed']
        assert [row['seed'] for row in data['rows']] == list(range(690100 + rank * 8, 690108 + rank * 8))
        assert all(row['native_counts']['controls'] - row['native_counts']['reset_controls'] == 600 for row in data['rows'])
        successes += sum(row['success_once'] for row in data['rows'])
        refs.append(dict(path=str(path), sha256=sha(path)))
        canonical = restored / 'canonical-eval' / f'rank-{rank:02d}'
        assert sha(canonical / 'initial.pt') == json.loads((canonical / 'complete.json').read_text())['snapshot_sha256']
    assert successes / 64 == before['metrics']['success_once']
    reference = source.parent / (attempt + '-before-reference.json')
    assert not reference.exists()
    write(reference, dict(completed=True, cases=64, checkpoint=str(checkpoint),
        checkpoint_marker_sha256=sha(boundary_path), metrics=before['metrics'],
        episode_refs=refs, canonical_directory=str(restored / 'canonical-eval'),
        original_before_sha256=sha(restored / 'before-evaluation.json'),
        policy='Original SFT before panel retained; resume all optimizer, sampler and RNG state at iteration8'))
    write(remote / 'RESTORED.json', dict(**restored_receipt, checkpoint=str(checkpoint),
        checkpoint_marker_sha256=sha(boundary_path), before_reference_sha256=sha(reference)))
    env = dict(os.environ, TASK_LOCAL_OUTPUT=str(source), TASK_PHASE_SECONDS='2100',
               TASK_CHECKPOINT_SECONDS='1500', TASK_PROCESS_SECONDS='2200', TASK_MAXIMUM_GPU_HOURS='6',
               ROBOCASA_SIM_PPO_RESUME=str(checkpoint), ROBOCASA_SIM_PPO_BEFORE_REFERENCE=str(reference),
               ROBOCASA_SIM_PPO_BEFORE_SHA256=sha(reference))
    done = threading.Event()
    monitor = threading.Thread(target=snapshot, args=(source, remote, done), daemon=False)
    monitor.start()
    try:
        result = subprocess.run(['bash', str(HERE / 'teacher.sh'), pin, 'OpenCabinet', attempt], env=env)
        assert (source / 'EXIT.json').exists()
    finally:
        done.set()
        monitor.join()
    shutil.copyfile(reference, source / 'resume-before-reference.json')
    progress_path = source / 'progress.json'
    completed_step = json.loads(progress_path.read_text())['completed_iterations'] if progress_path.exists() else 8
    write(source / 'CHRONOLOGY.json', dict(previous_archive=str(prior / 'collection.tar'),
        previous_archive_sha256=CABINET_ARCHIVE_SHA, previous_training_iterations=[1, 8],
        planned_resumed_training_iterations=[9, 10], completed_global_step=completed_step,
        target_fixed_before_training=True, resume_inputs_verified=True,
        runtime_restore_demonstrated_by_completed_iteration=completed_step > 8,
        original_before_reference=str(reference)))
    archived = archive(source, remote)
    for name in ('progress.json', 'MANIFEST.json', 'before-evaluation.json', 'after-evaluation.json',
                 'operator-checkpoint-exit.json', 'EXIT.json', 'CHRONOLOGY.json'):
        path = source / name
        if path.exists():
            write(remote / name, json.loads(path.read_text()))
    write(remote / 'COMPLETE.json', dict(archive_verified=archived['verified'],
        learner_exit_code=result.returncode, started_unix=started, ended_unix=time.time(),
        complete_checkpoint_boundaries=archived['complete_checkpoint_boundaries']))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
