"""Archive verified obsolete source checkpoints; retain the latest complete recovery hot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

RUN = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1')
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives/robocasa-eight-hour-babysit-20261104-r1/source5-early-checkpoints-01')
JOBS = {'PickPlaceCounterToCabinet': 'job-p4vgvfgs7s29', 'TurnOnSinkFaucet': 'job-jo6kx7qu29r4'}
COLD_READ_RETRIES = []


def digest(path, *, cold=False, expected=None):
    for attempt in range(6 if cold else 1):
        try:
            size = path.stat().st_size
            sha, read_bytes = hashlib.sha256(), 0
            with path.open('rb') as stream:
                while block := stream.read(8 << 20):
                    sha.update(block)
                    read_bytes += len(block)
            actual = sha.hexdigest()
            if read_bytes != size or (expected is not None and actual != expected):
                raise ValueError('Closed archive read failed byte/hash verification')
            return actual
        except (OSError, ValueError) as error:
            transient = isinstance(error, ValueError) or error.errno in (errno.EIO, errno.EBUSY, errno.ESTALE)
            if not cold or not transient or attempt == 5:
                raise
            COLD_READ_RETRIES.append(dict(path=str(path), attempt=attempt+1,
                                          error=type(error).__name__, errno=getattr(error, 'errno', None)))
            time.sleep((1, 2, 4, 8, 8)[attempt])


def durable_json(path, value):
    payload = json.dumps(value, indent=2) + '\n'
    with path.open('x') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload.encode()).hexdigest()


def main(completed_task=None, archive_attempt=1):
    assert sys.platform == 'linux'
    assert 1 <= archive_attempt <= 9 and (completed_task is not None or archive_attempt == 1)
    archive_id = ('source5-early-checkpoints-01' if completed_task is None else
                  f'source5-{completed_task}-intermediate-after40-{archive_attempt:02d}')
    cold = COLD.with_name(archive_id)
    receipts = RUN / 'archives'
    receipts.mkdir(exist_ok=True)
    lock = (receipts / (archive_id + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not cold.exists()
    ledger = json.loads((RUN / 'SOURCE5-ALLOCATION-LEDGER.json').read_text())
    plans, hot_recovery = [], []
    jobs = JOBS if completed_task is None else {completed_task: next(
        row['job_id'] for row in ledger['allocations']
        if row['task'] == completed_task and row['kind'] == 'source_resume')}
    for task, job in jobs.items():
        allocation = next(row for row in ledger['allocations'] if row['job_id'] == job)
        assert allocation['task'] == task and allocation['scheduler_release_verified'] and allocation['released_shanghai']
        parent = RUN / 'source5' / task
        if completed_task is None:
            recovery = parent / 'attempt-01' / task / 'checkpoints/global_step_10'
            prepared = json.loads((parent / 'RESUME-PREPARED-01.json').read_text())
            assert digest(recovery / 'pilot-boundary.json') == prepared['checkpoint_marker_sha256']
            candidates = [(recovery.with_name(f'global_step_{step}'), step, allocation) for step in (0, 1)]
        else:
            output = parent / 'attempt-02'
            assert json.loads((output / 'progress.json').read_text())['completed_iterations'] == 40
            assert json.loads((output / 'after-evaluation.json').read_text())['completed']
            assert json.loads((output / 'clock.json').read_text())['exit_code'] == 0
            recovery = output / task / 'checkpoints/global_step_40'
            latest = json.loads((recovery / 'pilot-boundary.json').read_text())
            assert latest['completed'] and latest['global_step'] == 40 and len(latest['workers']) == 18
            export = recovery / 'actor/model_state_dict/full_weights.pt'
            assert export.is_file() and any(item['path'] == str(export) for item in latest['actor_checkpoint_files'])
            for item in latest['workers'] + latest['actor_checkpoint_files']:
                assert digest(Path(item['path'])) == item['sha256']
            for rank in (0, 1):
                update = json.loads((output / 'updates' / f'iteration-040-rank-{rank:02d}.json').read_text())
                assert update['completed'] and update['optimizer_steps_after'] == 960 and update['frozen_vlm_unchanged']
            original = next(row for row in ledger['allocations'] if row['job_id'] == JOBS[task])
            assert original['scheduler_release_verified'] and original['released_shanghai']
            candidates = [(parent / 'attempt-01' / task / 'checkpoints/global_step_10', 10, original)]
            candidates += [(recovery.with_name(f'global_step_{step}'), step, allocation) for step in (11, 20, 30)]
        assert recovery.is_dir() and not recovery.is_symlink()
        hot_recovery.append(str(recovery))
        for source, step, owner in candidates:
            assert source.is_dir() and not source.is_symlink()
            marker = source / 'pilot-boundary.json'
            boundary = json.loads(marker.read_text())
            assert boundary['completed'] and boundary['global_step'] == step and len(boundary['workers']) == 18
            expected = {Path(row['path']): row['sha256'] for row in boundary['workers'] + boundary['actor_checkpoint_files']}
            expected[marker] = digest(marker)
            entries = list(source.rglob('*'))
            assert not any(path.is_symlink() for path in entries)
            assert {path for path in entries if path.is_file()} == set(expected)
            plans.append(dict(source=source, target=cold / task / source.name, expected=expected,
                              job_id=owner['job_id'], released_shanghai=owner['released_shanghai']))
    cold.parent.mkdir(parents=True, exist_ok=True)
    probe = cold.parent / (archive_id + '.write-probe')
    payload = b'verified private cold checkpoint storage\n' * 32768
    with probe.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    probe_sha256 = hashlib.sha256(payload).hexdigest()
    assert digest(probe, cold=True, expected=probe_sha256) == probe_sha256
    probe.unlink()
    cold.mkdir()
    began = time.time()
    manifest = []
    for plan in plans:
        source, target = plan['source'], plan['target']
        target.mkdir(parents=True)

        def copy(item):
            path, expected = item
            stat = path.stat()
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            sha = hashlib.sha256()
            with path.open('rb') as incoming, destination.open('xb') as outgoing:
                while block := incoming.read(8 << 20):
                    sha.update(block)
                    outgoing.write(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            assert sha.hexdigest() == expected == digest(destination, cold=True, expected=expected)
            after = path.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            return dict(relative=str(path.relative_to(source)), bytes=stat.st_size, sha256=expected)

        with ThreadPoolExecutor(max_workers=2) as pool:
            files = list(pool.map(copy, plan['expected'].items()))
        plan['files'] = files
        manifest.append(dict(source=str(source), target=str(target), job_id=plan['job_id'],
                             released_shanghai=plan['released_shanghai'], files=files))
    cold_manifest = cold / 'ARCHIVE-MANIFEST.json'
    manifest_sha256 = durable_json(cold_manifest, dict(checkpoints=manifest, retained_hot_recovery=hot_recovery))
    assert digest(cold_manifest, cold=True, expected=manifest_sha256) == manifest_sha256
    swapped = []
    try:
        for plan in plans:
            source = plan['source']
            quarantine = source.with_name(source.name + '.verified-cold-01')
            assert not quarantine.exists()
            source.rename(quarantine)
            swapped.append((source, quarantine))
            source.symlink_to(plan['target'], target_is_directory=True)
            assert source.resolve() == plan['target']
            for row in plan['files']:
                assert digest(source / row['relative'], cold=True, expected=row['sha256']) == row['sha256']
    except BaseException:
        for source, quarantine in reversed(swapped):
            if source.is_symlink():
                source.unlink()
            quarantine.rename(source)
        raise
    for _, quarantine in swapped:
        shutil.rmtree(quarantine)
    assert all(Path(path).is_dir() and not Path(path).is_symlink() for path in hot_recovery)
    receipt = dict(completed=True, GPU_hours=0, checkpoints=manifest, retained_hot_recovery=hot_recovery,
        archived_bytes=sum(row['bytes'] for plan in plans for row in plan['files']),
        manifest_sha256=manifest_sha256, elapsed_seconds=time.time()-began,
        cold_read_retries=COLD_READ_RETRIES, cold_read_attempt_limit=6,
        original_paths_verified=True, complete_optimizer_sampler_RNG_retained=True,
        all_TRAIN_histories_remain_hot=True, cold_storage_requires_staging_for_future_GPU_use=True)
    durable_json(receipts / (archive_id + '.json'), receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'checkpoints'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-task', choices=tuple(JOBS))
    parser.add_argument('--archive-attempt', type=int, default=1)
    args = parser.parse_args()
    main(args.completed_task, args.archive_attempt)
