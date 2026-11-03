"""Tier superseded complete checkpoints; always retain the newest full boundary.

Only this campaign's five task directories are eligible. A verified complete
newer boundary must exist before any older local payload is reclaimed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

TASKS = ['OpenDrawer', 'CloseToasterOvenDoor', 'PickPlaceDrawerToCounter',
         'CoffeeSetupMug', 'TurnOffStove']


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            value.update(block)
    return value.hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in root.rglob('*') if p.is_file()}


def verify(root):
    boundary = json.loads((root / 'pilot-boundary.json').read_text())
    assert boundary['completed'] and len(boundary['workers']) == 18
    for row in boundary['workers'] + boundary['actor_checkpoint_files']:
        assert sha(row['path']) == row['sha256'], row['path']
    return boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bos', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    assert args.root.name == 'robocasa-flow-sde-10m-per-task-20261103-r1'
    assert args.root.resolve().parent == Path('/mowenjie/fluffy-bassoon-2/outputs')
    assert str(args.bos).startswith('/mnt/bos/adgfyl/strong-plant-archives/')
    assert not args.record.exists()
    lock = (args.root / '.checkpoint-tier.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.bos.mkdir(parents=True, exist_ok=True)
    stage = args.root / 'tier-staging'
    stage.mkdir(exist_ok=True)
    receipts = []
    for task in TASKS:
        checkpoints = []
        for marker in (args.root / task).glob(f'*/{task}/checkpoints/global_step_*/pilot-boundary.json'):
            cp = marker.parent
            if (cp / 'COLD-STORAGE.json').exists():
                continue
            data = json.loads(marker.read_text())
            if data.get('completed'):
                checkpoints.append((int(data['global_step']), marker.stat().st_mtime_ns, cp))
        if len(checkpoints) < 2:
            continue
        checkpoints.sort()
        latest_step, _, latest = checkpoints[-1]
        verify(latest)
        for step, _, cp in checkpoints[:-1]:
            if step >= latest_step:
                continue
            assert cp.resolve().is_relative_to((args.root / task).resolve())
            assert not cp.is_symlink() and cp != latest
            boundary = verify(cp)
            original = inventory(cp)
            size = sum(row[0] for row in original.values())
            fs = os.statvfs(stage)
            assert fs.f_bavail * fs.f_frsize > (200 << 30) + size
            name = f'{task}--{cp.parents[2].name}--{cp.name}.tar.zst'
            local, dest = stage / name, args.bos / name
            assert not local.exists() and not dest.exists()
            subprocess.run(['tar', '--sort=name', '-I', 'zstd -1 -T2', '-cf', str(local),
                            '-C', str(cp.parent), cp.name], check=True)
            subprocess.run(['tar', '-I', 'zstd', '-df', str(local), '-C', str(cp.parent)], check=True)
            assert inventory(cp) == original
            expected = sha(local)
            with local.open('rb') as source, dest.open('xb') as target:
                shutil.copyfileobj(source, target, 16 << 20)
                target.flush(); os.fsync(target.fileno())
            assert dest.stat().st_size == local.stat().st_size and sha(dest) == expected
            assert inventory(cp) == original
            assert (latest / 'pilot-boundary.json').is_file() and not (latest / 'COLD-STORAGE.json').exists()
            receipt = dict(task=task, source=str(cp), source_bytes=size,
                archive=str(dest), archive_bytes=dest.stat().st_size, sha256=expected,
                newer_full_checkpoint=str(latest), newer_global_step=latest_step,
                source_inventory=original, tar_compare=True, full_BOS_readback=True,
                verified_unix=time.time(), restore=['tar', '-I', 'zstd', '-xf', str(dest), '-C', str(cp.parent)])
            record = args.bos / (name + '.json')
            record.write_text(json.dumps(receipt, indent=2) + '\n')
            # The complete verified object remains recoverable. Keep the small
            # boundary manifest at its original path and an explicit restore receipt.
            shutil.rmtree(cp)
            cp.mkdir()
            (cp / 'pilot-boundary.json').write_text(json.dumps(boundary, indent=2) + '\n')
            (cp / 'COLD-STORAGE.json').write_text(json.dumps(receipt, indent=2) + '\n')
            local.unlink()
            receipts.append(receipt)
            print(json.dumps({k: v for k, v in receipt.items() if k != 'source_inventory'}), flush=True)
    args.record.write_text(json.dumps(dict(completed=True, receipts=receipts,
        latest_full_boundaries_preserved=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
