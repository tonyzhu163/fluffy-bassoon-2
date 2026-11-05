"""Tier closed, independently audited artifacts while preserving their original paths."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

RUN = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1')
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives/robocasa-eight-hour-babysit-20261104-r1')


def read(path):
    return json.loads(path.read_text())


def main(task, mode):
    assert sys.platform == 'linux'
    spec = importlib.util.spec_from_file_location('verified_archive', Path(__file__).with_name('SOURCE5-ARCHIVE-EARLY.py'))
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    ledger = read(RUN / 'SOURCE5-ALLOCATION-LEDGER.json')
    rows = [row for row in ledger['allocations'] if row['task'] == task]
    assert rows and all(row.get('scheduler_release_verified') and row.get('released_shanghai') for row in rows)
    assert not any(row['task'] == task for row in ledger.get('pending_submissions', []))
    # A whole-family closeout is required before any raw bank or final recovery moves.
    readiness = RUN / 'SOURCE5-READINESS.json'
    assert read(readiness)['completed']
    parent = RUN / 'source5-tests' / task
    binding = parent / 'ENDPOINT-RESULT-SOURCE.json'
    result_dir = Path(read(binding)['path']) if binding.exists() else parent / 'attempt-01'
    audit = read(result_dir / 'INDEPENDENT-AUDIT.json')
    assert audit['verified'] and not audit['qualification_only'] and audit['cases_per_policy'] == 512
    assert audit['combined_sha256'] == lib.digest(result_dir / 'RESULT.json')
    dependencies = read(RUN / 'SOURCE5-POSTTEST-ARCHIVE-DEPENDENCIES-01.json')
    assert dependencies['no_admitted_readers_of_final_recovery_or_test_banks']
    assert dependencies['canonical_histories_cache_and_inference_stay_hot']
    archive_id = f'source5-{task}-closed-{mode}-01'
    target_root = COLD / archive_id
    receipt = RUN / 'archives' / (archive_id + '.json')
    lock = receipt.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not receipt.exists() and not target_root.exists()
    plans, protected = [], []
    if mode == 'recovery':
        assert task != 'CoffeeSetupMug'
        root = RUN / 'source5' / task / 'attempt-02' / task / 'checkpoints/global_step_40'
        marker = root / 'pilot-boundary.json'
        boundary = read(marker)
        assert boundary['completed'] and boundary['global_step'] == 40 and len(boundary['workers']) == 18
        model = root / 'actor/model_state_dict/full_weights.pt'
        expected_model = next(row['sha256'] for row in boundary['actor_checkpoint_files'] if row['path'] == str(model))
        assert not model.is_symlink() and lib.digest(model) == expected_model
        protected = [(model, expected_model), (marker, lib.digest(marker))]
        for source in (root / 'actor/dcp_checkpoint', root / 'pilot'):
            expected = {Path(row['path']): row['sha256'] for row in boundary['workers'] + boundary['actor_checkpoint_files']
                        if Path(row['path']).is_relative_to(source)}
            plans.append((source, expected))
    else:
        assert mode == 'raw'
        for row in rows:
            if row['kind'] != 'endpoint_test':
                continue
            source = RUN / row['output'] / row.get('worker_subdir', 'node-00') / 'banks'
            assert source.is_dir()
            # Byte hashes are captured during the copy; all source writers are released.
            plans.append((source, {p: None for p in source.rglob('*') if p.is_file()}))
    for source, expected in plans:
        assert source.is_dir() and not source.is_symlink() and expected
        entries = list(source.rglob('*'))
        assert not any(p.is_symlink() for p in entries)
        assert {p for p in entries if p.is_file()} == set(expected)
    before = os.statvfs(RUN)
    free_before = before.f_bavail * before.f_frsize
    probe = COLD / (archive_id + '.write-probe')
    payload = dict(kind='closed-source5-archive-probe', source=archive_id)
    probe_sha = lib.durable_json(probe, payload)
    assert lib.digest(probe, cold=True, expected=probe_sha) == probe_sha
    probe.unlink()
    began = time.time()
    manifests = []
    for source, expected in plans:
        target = target_root / source.relative_to(RUN)
        target.mkdir(parents=True)

        def copy(item):
            path, expected_sha = item
            initial = path.stat()
            assert initial.st_nlink == 1, 'Shared hardlinks need separate dependency accounting'
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            import hashlib
            digest = hashlib.sha256()
            with path.open('rb') as incoming, destination.open('xb') as outgoing:
                while block := incoming.read(8 << 20):
                    digest.update(block)
                    outgoing.write(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            actual = digest.hexdigest()
            assert expected_sha is None or actual == expected_sha
            assert lib.digest(destination, cold=True, expected=actual) == actual
            final = path.stat()
            assert (initial.st_size, initial.st_mtime_ns) == (final.st_size, final.st_mtime_ns)
            return dict(relative=str(path.relative_to(source)), bytes=initial.st_size,
                        source_blocks_bytes=initial.st_blocks * 512, sha256=actual)

        with ThreadPoolExecutor(max_workers=2) as pool:
            files = list(pool.map(copy, expected.items()))
        manifests.append(dict(source=str(source), target=str(target), files=files))
    manifest_path = target_root / 'ARCHIVE-MANIFEST.json'
    manifest_sha = lib.durable_json(manifest_path, dict(artifacts=manifests, task=task, mode=mode,
        independent_audit_sha256=lib.digest(result_dir / 'INDEPENDENT-AUDIT.json'),
        readiness_sha256=lib.digest(readiness), dependency_sha256=lib.digest(RUN / 'SOURCE5-POSTTEST-ARCHIVE-DEPENDENCIES-01.json')))
    assert lib.digest(manifest_path, cold=True, expected=manifest_sha) == manifest_sha
    swapped = []
    try:
        for entry in manifests:
            source, target = Path(entry['source']), Path(entry['target'])
            quarantine = source.with_name(source.name + '.verified-cold-closed-01')
            assert not quarantine.exists()
            source.rename(quarantine)
            swapped.append((source, quarantine))
            source.symlink_to(target, target_is_directory=True)
            assert source.resolve() == target
            for row in entry['files']:
                assert lib.digest(source / row['relative'], cold=True, expected=row['sha256']) == row['sha256']
        for path, expected in protected:
            assert not path.is_symlink() and lib.digest(path) == expected
    except BaseException:
        for source, quarantine in reversed(swapped):
            if source.is_symlink():
                source.unlink()
            quarantine.rename(source)
        raise
    for _, quarantine in swapped:
        shutil.rmtree(quarantine)
    after = os.statvfs(RUN)
    record = dict(completed=True, task=task, mode=mode, GPU_hours=0,
        cold_manifest=str(manifest_path), cold_manifest_sha256=manifest_sha,
        archived_bytes=sum(f['bytes'] for m in manifests for f in m['files']),
        removed_source_blocks_bytes=sum(f['source_blocks_bytes'] for m in manifests for f in m['files']),
        free_before_bytes=free_before, free_after_bytes=after.f_bavail * after.f_frsize,
        shared_free_delta_is_not_attributed_reclaim=True, source_paths_verified=True,
        complete_recovery_retained=True, future_GPU_readers_require_explicit_PFS_staging=True,
        standalone_inference_canonical_histories_and_shared_cache_hot=True,
        elapsed_seconds=time.time()-began, cold_read_retries=lib.COLD_READ_RETRIES,
        cold_read_attempt_limit=6, artifacts=manifests)
    lib.durable_json(receipt, record)
    print(json.dumps({k:v for k,v in record.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('PickPlaceCounterToCabinet', 'TurnOnSinkFaucet', 'CoffeeSetupMug'), required=True)
    parser.add_argument('--mode', choices=('recovery', 'raw'), required=True)
    args = parser.parse_args()
    main(args.task, args.mode)
