"""Archive the closed Faucet hot remainder, then separately reclaim it.

This is run-local retention, not a change to the active training runtime.
Preparation compares every source byte, restores the cold object on local disk,
and compares every restored byte before reclamation can be requested.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

RUN = 'robocasa-faucet-retention-20261114-r1'
ROOT = Path('/mowenjie/fluffy-bassoon-2')
REL = Path('outputs/robocasa-eight-hour-babysit-20261104-r1/source5/TurnOnSinkFaucet')
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives') / RUN
STAGE = Path('/tmp') / RUN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'reclaim'))
    parser.add_argument('--pin', required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[2]
    assert sys.platform == 'linux'
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == args.pin
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    spec = importlib.util.spec_from_file_location('archive_utils', code / 'outputs/storage-audit-baige-20261104-r1/archive.py')
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    source = ROOT / REL
    assert source.is_dir() and not source.is_symlink()
    assert not arch.users(source), arch.users(source)
    receipt = COLD / 'ARCHIVE.json'
    inventory_path = COLD / 'INVENTORY.json'
    archive = COLD / 'faucet-hot-remainder.tar.zst'
    if args.phase == 'reclaim':
        result = json.loads(receipt.read_text())
        assert result['source_commit'] == args.pin
        assert result['full_cold_restore_compared'] and result['cold_full_sha256_readback']
        assert result['source'] == str(source) and not result['source_reclaimed']
        before = json.loads(inventory_path.read_text())
        assert arch.inventory(source) == before, 'Source changed since verified archive'
        assert arch.digest(archive) == result['sha256']
        assert not arch.users(source), arch.users(source)
        pointer = source.with_name(source.name + '.bos-archive.json')
        quarantine = source.with_name(source.name + '.archiving-' + RUN)
        assert not pointer.exists() and not quarantine.exists()
        arch.save(pointer, result)
        source.rename(quarantine)
        shutil.rmtree(quarantine)
        result.update(source_reclaimed=True, reclaimed_unix=time.time(), free_project_bytes=shutil.disk_usage(ROOT).free)
        arch.save(receipt, result)
        arch.save(pointer, result)
        # Only this run's successfully verified scratch objects are removed.
        shutil.rmtree(STAGE)
        print(json.dumps(result), flush=True)
        return

    COLD.mkdir(exist_ok=False)
    STAGE.mkdir(exist_ok=False)
    for directory in (COLD, STAGE):
        probe = directory / 'WRITE-PROBE'
        probe.write_bytes(b'archive'); assert probe.read_bytes() == b'archive'; probe.unlink()
    plan_path = Path(__file__).with_name('PLAN.json')
    plan = json.loads(plan_path.read_text())
    arch.save(COLD / 'PLAN.json', plan)
    before = arch.inventory(source)
    assert all(not any(x in Path(r[0]).parts for x in ('.secrets', '.ssh', '.aws', '.bce', '.env', '.netrc')) for r in before)
    links = [dict(path=r[0], target=r[10]) for r in before if stat.S_ISLNK(r[1])]
    # Existing cold migrations remain independent, preserved dependencies.
    for link in links:
        target = (ROOT / link['path']).resolve(strict=True)
        assert str(target).startswith('/mnt/bos/adgfyl/strong-plant-archives/'), link
    arch.save(inventory_path, before)
    logical = sum(r[2] for r in before if stat.S_ISREG(r[1]))
    assert shutil.disk_usage(STAGE).free > 3 * logical
    local = STAGE / archive.name
    restored = STAGE / 'restore'
    restored.mkdir()
    started = time.time()
    result = dict(source_commit=args.pin, source=str(source), archive=str(archive), started_unix=started,
                  source_reclaimed=False, source_logical_bytes=logical,
                  source_allocated_bytes=sum(r[6] for r in before)*512, entries=len(before),
                  preserved_cold_symlinks=links, phase='packing', GPU_hours=0)
    arch.save(COLD / 'PROGRESS.json', result)
    with (STAGE / 'archive.log').open('w') as log:
        subprocess.run(['tar', '--sort=name', '--acls', '--xattrs', '--numeric-owner', '-I', 'zstd -1 -T2', '-cf', str(local), '-C', str(ROOT), str(REL)], stdout=log, stderr=log, check=True)
        subprocess.run(['tar', '-I', 'zstd', '-df', str(local), '-C', str(ROOT)], stdout=log, stderr=log, check=True)
        listing = subprocess.check_output(['tar', '-I', 'zstd', '-tf', str(local)], text=True)
        assert sorted(x.rstrip('/') for x in listing.splitlines()) == [r[0] for r in before]
        assert arch.inventory(source) == before
        result['phase'] = 'uploading'
        arch.save(COLD / 'PROGRESS.json', result)
        expected = arch.digest(local)
        with local.open('rb') as src, archive.open('xb') as dst:
            shutil.copyfileobj(src, dst, 16 << 20); dst.flush(); os.fsync(dst.fileno())
        assert archive.stat().st_size == local.stat().st_size
        assert arch.digest(archive) == expected
        result['phase'] = 'restoring cold object'
        arch.save(COLD / 'PROGRESS.json', result)
        subprocess.run(['tar', '--acls', '--xattrs', '-I', 'zstd', '-xf', str(archive), '-C', str(restored)], stdout=log, stderr=log, check=True)
        subprocess.run(['tar', '-I', 'zstd', '-df', str(archive), '-C', str(restored)], stdout=log, stderr=log, check=True)
    assert arch.inventory(source) == before
    assert not arch.users(source), arch.users(source)
    result.update(phase='verified; hot source retained', sha256=expected, archive_bytes=archive.stat().st_size,
                  source_compare=True, exact_member_coverage=True, cold_full_sha256_readback=True,
                  full_cold_restore_compared=True, verified_unix=time.time(),
                  restore_command=f"tar --acls --xattrs -I zstd -xf '{archive}' -C '{ROOT}'")
    arch.save(receipt, result)
    arch.save(COLD / 'PROGRESS.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
