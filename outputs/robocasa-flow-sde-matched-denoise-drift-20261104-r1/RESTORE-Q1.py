"""Restore only the ten archived Q1 actor batches, verifying their original hashes."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flow-sde-10m-per-task-20261103-r1')
OUT = ROOT.parent / 'robocasa-flow-sde-matched-denoise-drift-20261104-r1'
TASKS = ['PickPlaceDrawerToCounter', 'OpenDrawer', 'CoffeeSetupMug',
         'CloseToasterOvenDoor', 'TurnOffStove']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(rows, completed):
    p = OUT/'RESTORED-Q1.json'
    temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(completed=completed, rows=rows,
        controls=0, SGD=0, GPU_seconds=0, checked_unix=time.time()), indent=2)+'\n')
    temporary.replace(p)


rows = []
for task in TASKS:
    q1 = ROOT/task/('q1-r2' if task == 'OpenDrawer' else 'q1-r1')
    assert json.loads((q1/'clock.json').read_text())['exit_code'] == 0
    receipt_path = ROOT/f'CORPUS-ARCHIVE-{task}-R1-PREPARED.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['completed'] and receipt['full_object_readback']
    assert receipt['all_archive_members_hash_compared']
    archive = Path(receipt['archive'])
    assert archive.stat().st_size == receipt['archive_bytes']
    manifest_path = Path(receipt['manifest'])
    assert sha(manifest_path) == receipt['manifest_sha256']
    inventory = json.loads(manifest_path.read_text())['files']
    targets = {str((q1/f'training-rollouts/iteration-0001-policy-0000-rank-{rank:02d}.pt').relative_to(ROOT))
               for rank in range(2)}
    assert targets <= set(inventory)
    # TIER-CORPUS.py wrote sorted regular PAX members with these metadata.
    # Reconstruct offsets, then check each actual header and payload hash.
    # This avoids streaming the other ~100 GiB of unrelated archived data.
    offset = 0
    for name, item in sorted(inventory.items()):
        header = tarfile.TarInfo(name)
        header.size, header.mode = item['bytes'], item['mode']
        header.mtime = item['mtime_ns']/1e9
        if name in targets:
            target = ROOT/name
            actor = json.loads(target.with_suffix('.json').read_text())
            assert actor['completed'] and actor['producing_policy_iteration'] == 0
            assert actor['sha256'] == item['sha256']
            private = OUT/'restored-q1'/name
            if not target.exists():
                fs = os.statvfs(OUT)
                assert fs.f_bavail*fs.f_frsize > (200 << 30)+item['bytes']
                private.parent.mkdir(parents=True, exist_ok=True)
                with archive.open('rb') as f:
                    f.seek(offset)
                    with tarfile.open(fileobj=f, mode='r:') as tar:
                        member = tar.next()
                        assert member.isfile() and member.name == name and member.size == item['bytes']
                        digest = hashlib.sha256()
                        with private.open('xb') as dest:
                            source = tar.extractfile(member)
                            for b in iter(lambda: source.read(8 << 20), b''):
                                dest.write(b)
                                digest.update(b)
                            dest.flush()
                            os.fsync(dest.fileno())
                        assert private.stat().st_size == item['bytes']
                        assert digest.hexdigest() == item['sha256']
                os.link(private, target)
            assert sha(target) == item['sha256']
            rows.append(dict(task=task, path=str(target), sha256=item['sha256'],
                bytes=item['bytes'], archive=str(archive), offset=offset,
                archive_receipt_sha256=sha(receipt_path), manifest_sha256=sha(manifest_path)))
            save(rows, False)
            print(json.dumps(rows[-1]), flush=True)
        offset += len(header.tobuf(format=tarfile.PAX_FORMAT)) + ((item['bytes']+511)//512)*512
save(rows, len(rows) == 10)
assert len(rows) == 10
