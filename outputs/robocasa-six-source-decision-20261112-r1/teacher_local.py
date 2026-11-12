"""Bounded PPO pilot with local raw output and a fully verified private archive.

The native learner, source seeds, ten-update target and recovery format are
unchanged. Earlier operational checkpointing reserves time to preserve every
byte before releasing the allocated container. No shared payload is moved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time

HERE = Path(__file__).resolve().parent
PROJECT = Path('/mowenjie/fluffy-bassoon-2')
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives') / HERE.name / 'local01'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def snapshot(source, remote, done):
    """Small progress copies; these are explicitly provisional until archival."""
    while not done.is_set():
        state = dict(checked_unix=time.time(), final_archive_verified=False,
                     local_free_bytes=shutil.disk_usage(source.parent).free)
        for name in ('progress.json', 'gpu-probe.json', 'before-evaluation.json',
                     'after-evaluation.json', 'operator-checkpoint-exit.json', 'EXIT.json'):
            p = source / name
            if p.exists():
                try:
                    state[name] = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
        write(remote / 'PROGRESS.json', state)
        done.wait(20)


def archive(source, remote):
    files = sorted(p for p in source.rglob('*') if p.is_file())
    assert files and not any(p.is_symlink() for p in source.rglob('*'))
    inventory = [dict(path=str(p.relative_to(source)), bytes=p.stat().st_size,
                      sha256=sha(p)) for p in files]
    write(remote / 'ARCHIVE-INVENTORY.json', dict(source=str(source), files=inventory,
                                               verified=False))
    # Stream directly to BOS: a second full local copy is unnecessary.
    cold = remote / 'collection.tar'
    assert not cold.exists()
    with cold.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w|') as out:
            for p in files:
                out.add(p, arcname=str(p.relative_to(source)), recursive=False)
        stream.flush()
        os.fsync(stream.fileno())
    digest = sha(cold)
    restored = source.parent / (source.name + '-stageback')
    restored.mkdir(exist_ok=False)
    expected = {x['path']: x for x in inventory}
    seen = set()
    with tarfile.open(cold, 'r|') as packed:
        for member in packed:
            rel = PurePosixPath(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            assert member.name in expected and member.name not in seen
            p = restored / member.name
            p.parent.mkdir(parents=True, exist_ok=True)
            with packed.extractfile(member) as a, p.open('xb') as b:
                shutil.copyfileobj(a, b, 8 << 20)
            row = expected[member.name]
            assert p.stat().st_size == row['bytes']
            assert sha(p) == sha(source / member.name) == row['sha256']
            seen.add(member.name)
            # Each restored duplicate has been checked against the original;
            # the original and BOS member remain. Bound scratch use to one file.
            p.unlink()
    assert seen == set(expected)
    boundaries = []
    for p in source.rglob('pilot-boundary.json'):
        d = json.loads(p.read_text())
        assert d['completed']
        for f in d['actor_checkpoint_files'] + d['workers']:
            assert sha(Path(f['path'])) == f['sha256']
        boundaries.append(dict(step=d['global_step'], path=str(p.relative_to(source))))
    receipt = dict(verified=True, source=str(source), cold=str(cold),
        archive_sha256=digest, archive_bytes=cold.stat().st_size,
        payload_bytes=sum(x['bytes'] for x in inventory), files=inventory,
        full_archive_SHA_readback=True, all_stageback_members_verified=True,
        all_original_sources_rehashed=True, complete_checkpoint_boundaries=boundaries,
        checked_unix=time.time(),
        restore_instruction='Restore this archive to the recorded local source path in a fresh allocation and verify every inventory member before resuming or reading raw data.')
    write(remote / 'ARCHIVE.json', receipt)
    assert json.loads((remote / 'ARCHIVE.json').read_text()) == receipt
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pin')
    parser.add_argument('task')
    parser.add_argument('attempt')
    parser.add_argument('--checkpoint-seconds', type=int, default=2100,
                        choices=(2100, 3000))
    args = parser.parse_args()
    pin, task, attempt = args.pin, args.task, args.attempt
    checkpoint_seconds = args.checkpoint_seconds
    phase_seconds, process_seconds = checkpoint_seconds + 600, checkpoint_seconds + 700
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    assert task in ('OpenCabinet', 'SlideDishwasherRack', 'PickPlaceCounterToStove')
    assert re.fullmatch(r'attempt\d+', attempt)
    code = HERE.parent.parent
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == pin
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    source = Path('/tmp') / HERE.name / 'teachers' / task / attempt
    assert not source.exists()
    source.parent.mkdir(parents=True, exist_ok=True)
    assert source.parent.stat().st_dev != PROJECT.stat().st_dev
    assert shutil.disk_usage(source.parent).free >= 270 * 2**30
    remote = COLD / 'teachers' / task / attempt
    remote.mkdir(parents=True, exist_ok=False)
    probe = remote / '.write-probe'
    probe.write_text('private-teacher-archive')
    assert probe.read_text() == 'private-teacher-archive'
    probe.unlink()
    started = time.time()
    write(remote / 'LAUNCH.json', dict(source_commit=pin, hostname=socket.gethostname(),
        task=task, attempt=attempt, local_output=str(source), started_unix=started,
        scheduler_maximum_minutes=75, reserved_GPU_hours=10,
        checkpoint_request_seconds=checkpoint_seconds, phase_deadline_seconds=phase_seconds,
        process_timeout_seconds=process_seconds, learner_settings_unchanged=True,
        storage='Local scratch raw output, private BOS verified archive; PFS native temporary XML allowed'))
    done = threading.Event()
    monitor = threading.Thread(target=snapshot, args=(source, remote, done), daemon=False)
    monitor.start()
    env = dict(os.environ, TASK_LOCAL_OUTPUT=str(source), TASK_PHASE_SECONDS=str(phase_seconds),
               TASK_CHECKPOINT_SECONDS=str(checkpoint_seconds), TASK_PROCESS_SECONDS=str(process_seconds),
               TASK_MAXIMUM_GPU_HOURS='10')
    try:
        result = subprocess.run(['bash', str(HERE / 'teacher.sh'), pin, task, attempt], env=env)
        assert (source / 'EXIT.json').exists(), 'Native launcher must retain its terminal receipt'
    finally:
        done.set()
        monitor.join()
    receipt = archive(source, remote)
    for name in ('progress.json', 'MANIFEST.json', 'before-evaluation.json',
                 'after-evaluation.json', 'operator-checkpoint-exit.json', 'EXIT.json'):
        p = source / name
        if p.exists():
            write(remote / name, json.loads(p.read_text()))
    write(remote / 'COMPLETE.json', dict(archive_verified=receipt['verified'],
        learner_exit_code=result.returncode, started_unix=started, ended_unix=time.time(),
        complete_checkpoint_boundaries=receipt['complete_checkpoint_boundaries']))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
