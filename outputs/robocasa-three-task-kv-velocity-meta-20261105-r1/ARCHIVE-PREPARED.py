"""Recoverably archive closed per-task preparation arrays, preserving path aliases."""
from concurrent.futures import ThreadPoolExecutor, wait
import datetime
import hashlib
import json
import os
from pathlib import Path
import threading
import time


ROOT = Path('/mowenjie/fluffy-bassoon-2')
BASE = ROOT / 'outputs/robocasa-three-task-meta-baseline-20261104-r2'
OUT = ROOT / 'outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/migrations/pfs-members-01'
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives/robocasa-three-task-kv-velocity-meta-20261105-r1/closed-preparation-pfs01')


def now():
    return datetime.datetime.now().astimezone().isoformat()


def digest(path):
    last = None
    for attempt in range(5):
        try:
            with path.open('rb') as f:
                return hashlib.file_digest(f, 'sha256').hexdigest()
        except OSError as error:
            last = error; time.sleep(min(.25*2**attempt, 2.))
    raise last


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n');temp.replace(path)


def identity(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def main():
    COLD.mkdir(parents=True, exist_ok=True)
    probe = COLD / '.write-probe'
    with probe.open('wb') as f:
        f.write(b'verified closed preparation archival\n'); f.flush(); os.fsync(f.fileno())
    assert probe.read_bytes() == b'verified closed preparation archival\n';probe.unlink()
    expected = {}; files = []
    for task in ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug'):
        folder = BASE / task / 'preparation'
        assembly = json.loads((folder / 'ASSEMBLY.json').read_text())
        expected.update({Path(row['path']): row['sha256'] for row in assembly['datasets'].values()})
        files.extend(sorted(folder.rglob('*.npz')))
    abandoned = BASE / 'joint/preparation-pool-cpu-01-abandoned/ad-train.npz'
    files.append(abandoned)
    assert all(p.is_relative_to(BASE) and p.suffix == '.npz' for p in files)
    assert not any('preparation-uncompressed-cpu02' in p.parts for p in files)
    # Existing stable cold aliases are complete; never rewrite an active payload.
    regular = [p for p in files if not p.is_symlink()]
    groups = {}
    for p in regular:
        assert p.is_file()
        key = p.stat().st_dev, p.stat().st_ino
        groups.setdefault(key, []).append(p)
    states = {}; mutex = threading.Lock()
    plan = dict(checked_shanghai=now(),scope='Closed per-task RAW/FEATURES parent/shard arrays, per-task corpus arrays and abandoned compressed joint array only',
        excluded='All current pooled arrays, PFS source members, original BOS source archives, models, environments and checkpoints',
        potential_allocated_bytes=sum(paths[0].stat().st_blocks*512 for paths in groups.values()),
        groups=[[str(p) for p in paths] for paths in groups.values()],
        dependency='Current TRAIN.py loads only joint/preparation datasets; per-task ASSEMBLY.json and TAR-INDEX.json stay hot and unchanged')
    save(OUT / 'COLD-ARCHIVE-PLAN.json', plan)

    def snapshot(**extra):
        with mutex:
            result = dict(checked_shanghai=now(),completed_groups=len(states),
                total_groups=len(groups),own_reclaimed_allocated_bytes=sum(row['reclaimed_allocated_bytes'] for row in states.values()),
                free_PFS_bytes=os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize,
                groups=list(states.values()),**extra)
        save(OUT / 'COLD-ARCHIVE-PROGRESS.json', result)
        return result

    def archive(paths):
        primary = paths[0]; before = {p: identity(p) for p in paths}; initial = primary.stat()
        assert initial.st_nlink == len(paths), 'Preserve any unknown external hardlink dependency'
        sha = digest(primary)
        for p in paths:
            assert p not in expected or expected[p] == sha
            assert identity(p) == before[p]
        target = COLD / (sha + '.npz')
        if not target.exists():
            with primary.open('rb') as source, target.open('xb') as dest:
                h = hashlib.sha256()
                while chunk := source.read(16*2**20):
                    dest.write(chunk); h.update(chunk)
                dest.flush(); os.fsync(dest.fileno())
            assert h.hexdigest() == sha
        assert target.stat().st_size == initial.st_size and digest(target) == sha
        assert digest(primary) == sha
        for p in paths:
            assert identity(p) == before[p]
        record = dict(paths=[str(p) for p in paths],cold_path=str(target),bytes=initial.st_size,
            sha256=sha,full_original_and_cold_readback_verified=True,
            reclaimed_allocated_bytes=0,source_released=False,
            stage_back='Copy cold_path to each original path plus a temporary suffix, verify sha256, then atomically replace the symlink; originals remain accessible through the stable aliases')
        save(OUT / ('cold-'+sha+'.json'),record)
        for p in paths:
            temp = p.with_name(p.name + f'.cold-link-{os.getpid()}')
            assert not temp.exists()
            temp.symlink_to(target)
            try:
                assert identity(p) == before[p]
                os.replace(temp,p)
            finally:
                temp.unlink(missing_ok=True)
        assert digest(paths[0]) == sha and all(p.resolve() == target for p in paths)
        record.update(source_released=True,stable_original_paths_verified=True,
            reclaimed_allocated_bytes=initial.st_blocks*512,completed_shanghai=now())
        save(OUT / ('cold-'+sha+'.json'),record)
        with mutex:
            states[sha] = record

    snapshot(completed=False)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(archive, paths) for paths in groups.values()]
            while True:
                done,pending = wait(futures,timeout=5)
                errors = [str(f.exception()) for f in done if f.exception() is not None]
                snapshot(completed=False,errors=errors)
                if not pending:
                    for future in futures:
                        future.result()
                    break
        result = snapshot(completed=True)
        save(OUT / 'COLD-ARCHIVE-RESULT.json',result);save(COLD/'RECEIPT.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as error:
        snapshot(completed=False,failed=True,error=repr(error));raise


if __name__ == '__main__':
    main()
