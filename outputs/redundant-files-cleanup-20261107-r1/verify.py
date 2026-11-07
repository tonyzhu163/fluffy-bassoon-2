"""Final metadata/recovery reconciliation after the cleanup worker exits."""
from pathlib import Path
import hashlib, json, os, shutil, time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(name): return json.loads((OUT / name).read_text())


plan, result, rows, released = (read(n) for n in ['PLAN.json', 'RESULT.json', 'SOURCE-INVENTORY.json', 'RELEASED.json'])
assert result['status'] == 'complete' and result['released_files'] == 323
assert result['protected_hashes_and_stamps_unchanged'] and result['existing_cold_duplicate_verified']
assert result['script_sha256'] == sha(OUT / 'cleanup.py') == read('QUALIFICATION.json')['script_sha256']
assert result['source_commit'] == read('DEPLOYMENT.json')['source_commit']
assert {r['path'] for r in rows} == {str(Path(r['path']).relative_to(ROOT)) for r in plan['files']}
assert {r['path'] for r in released} == {r['path'] for r in rows}
assert len(released) == len(rows) == 323
source_hashes = {r['path']: r['sha256'] for r in rows}
assert all(r['sha256'] == source_hashes[r['path']] for r in released)
assert sum(r['released_allocated_bytes'] for r in released) == result['released_allocated_bytes']
assert all(not os.path.lexists(ROOT / r['path']) and (ROOT / r['path']).parent.is_dir() for r in released)
cold = Path(plan['cold_root']); recovery = read('RECOVERY.json')
assert recovery == result['recovery']
assert recovery['source_compare'] and recovery['full_cold_readback'] and recovery['full_restore_verified']
assert recovery['restored_files'] == 322 and Path(recovery['archive']).stat().st_size == recovery['archive_bytes']
assert json.loads((cold / 'RECOVERY.json').read_text()) == recovery
duplicate = next(r for r in rows if r['kind'] == 'duplicate_archive')
assert duplicate['sha256'] == plan['existing_cold_sha256']
assert Path(plan['existing_cold_archive']).stat().st_size == duplicate['stamp'][1]
for name in ['PLAN.json', 'SOURCE-INVENTORY.json', 'PROTECTED-INVENTORY.json', 'GIT-CLOSURE.json', 'RELEASED.json', 'QUALIFICATION.json', 'cleanup.py', 'SCHEDULER.json']:
    assert sha(OUT / name) == sha(cold / name), name
protected = read('PROTECTED-INVENTORY.json')
for row in protected:
    path = ROOT / row['path']; s = path.lstat()
    assert [s.st_mode, s.st_size, s.st_mtime_ns, s.st_uid, s.st_gid, s.st_dev, s.st_ino, s.st_nlink, s.st_blocks * 512] == row['stamp'], row['path']
    if 'target' in row: assert os.readlink(path) == row['target']
assert len(protected) == result['protected_entries']
assert all(Path(p).is_file() for p in plan['excluded_bundle_paths'])
pointers = [ROOT / ('COLD-CLEANUP-'+OUT.name+'.json'),
            ROOT / 'cache/robotwin-ppo-pilot-20261105-r1-restore02' / ('COLD-CLEANUP-'+OUT.name+'.json'),
            ROOT / 'cache/robotwin-sidney-objects-20261105-r1' / ('COLD-CLEANUP-'+OUT.name+'.json')]
for path in pointers:
    pointer = json.loads(path.read_text()); assert pointer['archive'] == recovery
    assert pointer['source_commit'] == result['source_commit'] and pointer['duplicate_sha256'] == duplicate['sha256']
assert not (Path('/tmp') / OUT.name).exists() and result['staging_removed']
assert not (Path('/proc') / str(read('LAUNCH.json')['pid'])).exists()
report = dict(status='passed', checked_unix=time.time(), released_files=323,
              released_allocated_bytes=result['released_allocated_bytes'], protected_entries=len(protected),
              cold_metadata_matches=True, all_released_paths_absent=True, protected_stamps_unchanged=True,
              excluded_bundles_present=True, pointers_match=True, staging_removed=True, worker_exited=True,
              archive_full_readback_and_complete_restore_verified=True, filesystem_now=shutil.disk_usage(ROOT)._asdict(),
              GPU_allocations=0, note='Runtime rehashed all protected regular files after release. Final reconciliation checks those complete receipts and unchanged metadata without repeating bulk reads.')
for base in [OUT, cold]:
    (base / 'FINAL-VERIFICATION.json').write_text(json.dumps(report, indent=2)+'\n')
(cold / 'selected-files.tar.zst.sha256').write_text(recovery['archive_sha256']+'  selected-files.tar.zst\n')
for name in ['verify.py', 'RESTORE.md']:
    data = (OUT / name).read_bytes(); target = cold / name
    assert not target.exists() or target.read_bytes() == data
    target.write_bytes(data); assert target.read_bytes() == data
print(json.dumps(report))
