"""Share exact immutable dataset copies without changing any path or payload."""
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat


ROOT = Path('/mowenjie/fluffy-bassoon-2')
BASE = ROOT / 'outputs/robocasa-three-task-meta-baseline-20261104-r2'
OUT = ROOT / 'outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/migrations/pfs-members-01'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode), path
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def main():
    ledger = json.loads((ROOT / 'outputs/robocasa-eight-hour-babysit-20261104-r1/META-ALLOCATION-LEDGER.json').read_text())
    assert all(row.get('released_shanghai') for row in ledger['allocations']), 'Pause owned allocations first'
    assembly_path = BASE / 'joint/preparation/ASSEMBLY.json'
    assert digest(assembly_path) == 'fc3f388ef3feddde4c44ebd45ae9d7e73dab4a0fc58252c506315697be2a4758'
    assembly = json.loads(assembly_path.read_text()); allowed = {}
    manifests = [(assembly_path, digest(assembly_path))]
    manifests += [(Path(row['path']), row['sha256']) for row in assembly['source_assemblies'].values()]
    for path, expected in manifests:
        assert digest(path) == expected
        for row in json.loads(path.read_text())['datasets'].values():
            item = Path(row['path']); assert item.is_relative_to(BASE) and item.suffix == '.npz'
            allowed[item] = row['sha256']
    plan = json.loads((OUT / 'DEDUP-PLAN.json').read_text())
    items = {Path(p): group['sha256'] for group in plan['groups']
             for p in [group['primary'], *group['duplicates']]}
    assert all(allowed[p] == expected for p, expected in items.items())
    frozen = {p: identity(p) for p in items}
    before = os.statvfs(BASE); before_free = before.f_bavail * before.f_frsize
    def verify(pair):
        path, expected = pair
        assert digest(path) == expected, path
        assert identity(path) == frozen[path], path
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(verify, items.items()))
    changed, reclaimed = [], 0
    for group in plan['groups']:
        primary = Path(group['primary'])
        for name in group['duplicates']:
            duplicate = Path(name)
            assert identity(primary) == frozen[primary] and identity(duplicate) == frozen[duplicate]
            if os.path.samefile(primary, duplicate):
                continue
            original = duplicate.stat(); temp = duplicate.with_name(duplicate.name + f'.dedup-{os.getpid()}')
            assert not temp.exists()
            os.link(primary, temp)
            try:
                assert identity(duplicate) == frozen[duplicate] and identity(primary) == frozen[primary]
                os.replace(temp, duplicate)
            finally:
                temp.unlink(missing_ok=True)
            assert os.path.samefile(primary, duplicate)
            changed.append(dict(path=str(duplicate), canonical=str(primary), sha256=group['sha256'],
                previous_inode=original.st_ino, previous_links=original.st_nlink,
                reclaimed_allocated_bytes=original.st_blocks*512 if original.st_nlink == 1 else 0))
            reclaimed += changed[-1]['reclaimed_allocated_bytes']
    for group in plan['groups']:
        assert digest(Path(group['primary'])) == group['sha256']
        assert all(os.path.samefile(group['primary'], p) for p in group['duplicates'])
    after = os.statvfs(BASE)
    record = dict(completed=True, checked_shanghai=datetime.datetime.now().astimezone().isoformat(),
        every_input_SHA256_verified=True, all_paths_and_payloads_unchanged=True,
        files_changed=changed, owned_reclaimed_allocated_bytes=reclaimed,
        free_bytes_before=before_free, free_bytes_after=after.f_bavail*after.f_frsize,
        restore='To restore separate physical copies, copy the canonical file to a temporary file and atomically replace the alias; bytes remain identical',
        GPU_allocations_during_operation=0)
    (OUT / 'DEDUP-RESULT.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
