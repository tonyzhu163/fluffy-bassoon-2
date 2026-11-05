"""Cold-archive only steps0/1 of the two released source attempts; retain hot10."""
from concurrent.futures import ThreadPoolExecutor
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


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def durable_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    assert sys.platform == 'linux'
    receipts = RUN / 'archives'
    receipts.mkdir(exist_ok=True)
    lock = (receipts / 'source5-early-checkpoints-01.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not COLD.exists()
    ledger = json.loads((RUN / 'SOURCE5-ALLOCATION-LEDGER.json').read_text())
    plans, hot_recovery = [], []
    for task, job in JOBS.items():
        allocation = next(row for row in ledger['allocations'] if row['job_id'] == job)
        assert allocation['task'] == task and allocation['scheduler_release_verified'] and allocation['released_shanghai']
        parent = RUN / 'source5' / task
        recovery = parent / 'attempt-01' / task / 'checkpoints/global_step_10'
        prepared = json.loads((parent / 'RESUME-PREPARED-01.json').read_text())
        assert recovery.is_dir() and not recovery.is_symlink()
        assert digest(recovery / 'pilot-boundary.json') == prepared['checkpoint_marker_sha256']
        hot_recovery.append(str(recovery))
        for step in (0, 1):
            source = recovery.with_name(f'global_step_{step}')
            assert source.is_dir() and not source.is_symlink()
            marker = source / 'pilot-boundary.json'
            boundary = json.loads(marker.read_text())
            assert boundary['completed'] and boundary['global_step'] == step and len(boundary['workers']) == 18
            expected = {Path(row['path']): row['sha256'] for row in boundary['workers'] + boundary['actor_checkpoint_files']}
            expected[marker] = digest(marker)
            entries = list(source.rglob('*'))
            assert not any(path.is_symlink() for path in entries)
            assert {path for path in entries if path.is_file()} == set(expected)
            plans.append(dict(source=source, target=COLD / task / source.name, expected=expected,
                              job_id=job, released_shanghai=allocation['released_shanghai']))
    COLD.parent.mkdir(parents=True, exist_ok=True)
    probe = COLD.parent / 'source5-early-checkpoints-01.write-probe'
    payload = b'verified private cold checkpoint storage\n' * 32768
    with probe.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    assert digest(probe) == hashlib.sha256(payload).hexdigest()
    probe.unlink()
    COLD.mkdir()
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
            assert sha.hexdigest() == expected == digest(destination)
            after = path.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            return dict(relative=str(path.relative_to(source)), bytes=stat.st_size, sha256=expected)

        with ThreadPoolExecutor(max_workers=2) as pool:
            files = list(pool.map(copy, plan['expected'].items()))
        plan['files'] = files
        manifest.append(dict(source=str(source), target=str(target), job_id=plan['job_id'],
                             released_shanghai=plan['released_shanghai'], files=files))
    durable_json(COLD / 'ARCHIVE-MANIFEST.json', dict(checkpoints=manifest, retained_hot_recovery=hot_recovery))
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
                assert digest(source / row['relative']) == row['sha256']
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
        manifest_sha256=digest(COLD / 'ARCHIVE-MANIFEST.json'), elapsed_seconds=time.time()-began,
        original_paths_verified=True, complete_optimizer_sampler_RNG_retained=True,
        all_TRAIN_histories_remain_hot=True, cold_storage_requires_staging_for_future_GPU_use=True)
    durable_json(receipts / 'source5-early-checkpoints-01.json', receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'checkpoints'}))


if __name__ == '__main__':
    main()
