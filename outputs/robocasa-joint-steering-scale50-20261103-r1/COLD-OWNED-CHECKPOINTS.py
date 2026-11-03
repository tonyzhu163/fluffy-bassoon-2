"""Cold-copy only this owner's closed Flow checkpoints, retaining exact bytes.

No model loading, scheduler changes or active-source edits. The final checkpoint
and original raw histories remain on PFS. Existing private BOS is the destination.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
SELECTED = [
    'robocasa-opendrawer-flow-sde-20261102-r2/opendrawer_flow_sde/checkpoints/global_step_0',
    'robocasa-opendrawer-flow-sde-20261102-r3/opendrawer_flow_sde/checkpoints/global_step_1',
    'robocasa-opendrawer-flow-sde-20261102-r3/opendrawer_flow_sde/checkpoints/global_step_6',
    'robocasa-opendrawer-flow-sde-20261102-r4/opendrawer_flow_sde/checkpoints/global_step_7',
    'robocasa-opendrawer-flow-sde-20261102-r4/opendrawer_flow_sde/checkpoints/global_step_12',
]
BOS = Path('/mnt/bos/adgfyl/strong-plant-archives/20261103-joint-steering-owned-flow-checkpoints-r1')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def identity(path):
    s = path.lstat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, mode=s.st_mode)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-hostname', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if socket.gethostname() != a.expected_hostname or os.name != 'posix':
        raise RuntimeError('Require the allocated Linux development hostname')
    if a.output.exists() or BOS.exists():
        raise RuntimeError('Fresh audit identity and destination required')
    if not Path('/mnt/bos/adgfyl').is_mount():
        raise RuntimeError('Require the already configured private BOS mount')
    for relative in SELECTED:
        source = ROOT / relative
        if not source.is_dir() or source.is_symlink() or 'global_step_18' in relative:
            raise RuntimeError('Unexpected owned closed checkpoint source')
    a.output.mkdir(parents=True)
    result = dict(completed=False, hostname=socket.gethostname(), GPU_seconds=0,
        owner='parent01a19b74', terminal_source_job='job-h3d8va4slqg5',
        source_job_terminal_Shanghai='2026-11-02T20:21:39+08:00',
        dependency_check='Current tracked src/scripts/configs/run contracts: selected payload references only in this owner\'s closed Flow receipts; final18/raw histories stay warm.',
        destination=str(BOS), started_unix=time.time(), sources=[],
        source_bytes_preserved=True, private_existing_mount=True,
        restore='With BOS mounted, copy the verified destination tree to a fresh PFS directory, compare all receipt SHA256 values, then atomically replace its PFS symlink with the restored directory.')
    write(a.output/'RESULT.json', result)
    BOS.mkdir()
    probe = BOS/'WRITE-PROBE.txt'
    probe.write_bytes(b'owned closed Flow checkpoint archive\n')
    if probe.read_bytes() != b'owned closed Flow checkpoint archive\n':
        raise RuntimeError('BOS write/read probe failed')
    for index, relative in enumerate(SELECTED):
        source, destination = ROOT/relative, BOS/f'checkpoint-{index:02d}'
        files, directories = {}, []
        for path in sorted(source.rglob('*')):
            if path.is_symlink():
                raise RuntimeError('No implicit external symlink targets may be moved')
            if path.is_file():
                files[str(path.relative_to(source))] = identity(path)
            elif path.is_dir():
                directories.append(str(path.relative_to(source)))
            else:
                raise RuntimeError('Only regular checkpoint files and directories allowed')
        row = dict(source=str(source), destination=str(destination), files=files,
            bytes=sum(s['size'] for s in files.values()), copied=False, readback_verified=False,
            PFS_reclaimed=False, started_unix=time.time())
        result['sources'].append(row)
        write(a.output/'RESULT.json', result)
        if not a.execute:
            continue
        destination.mkdir()
        for directory in directories:
            (destination/directory).mkdir(parents=True, exist_ok=True)
        for relative_file, before in files.items():
            src, dst = source/relative_file, destination/relative_file
            dst.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with src.open('rb') as incoming, dst.open('xb') as outgoing:
                for block in iter(lambda: incoming.read(8*2**20), b''):
                    h.update(block)
                    outgoing.write(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if identity(src) != before:
                raise RuntimeError('Source changed while copying; retain PFS payload')
            before['sha256'] = h.hexdigest()
            if digest(dst) != h.hexdigest() or dst.stat().st_size != before['size']:
                raise RuntimeError('Full BOS SHA256 readback differs; retain PFS payload')
            write(a.output/'RESULT.json', result)
        current = {str(q.relative_to(source)): identity(q) for q in source.rglob('*') if q.is_file()}
        if set(current) != set(files) or any(current[k] != {field:v for field,v in files[k].items() if field!='sha256'} for k in files):
            raise RuntimeError('Source census changed before cutover; retain PFS payload')
        row.update(copied=True, readback_verified=True)
        write(a.output/'RESULT.json', result)
        quarantine = source.with_name(source.name+'.verified-cold-cutover-r1')
        if quarantine.exists():
            raise RuntimeError('Unexpected prior cutover; diagnose without overwrite')
        source.rename(quarantine)
        try:
            source.symlink_to(destination, target_is_directory=True)
            if source.resolve() != destination.resolve() or any((source/k).stat().st_size!=v['size'] for k,v in files.items()):
                raise RuntimeError('Verified cold read-through link failed')
        except BaseException:
            if source.is_symlink():
                source.unlink()
            quarantine.rename(source)
            raise
        # The complete byte-verified private BOS copy is the retained payload.
        # Reclaim only this owner's duplicate PFS copy after a readable link exists.
        shutil.rmtree(quarantine)
        row.update(PFS_reclaimed=True, finished_unix=time.time())
        result['free_PFS_bytes'] = shutil.disk_usage(ROOT).free
        write(a.output/'RESULT.json', result)
    result.update(completed=bool(a.execute), finished_unix=time.time(),
        free_PFS_bytes=shutil.disk_usage(ROOT).free,
        archived_bytes=sum(r['bytes'] for r in result['sources'] if r['PFS_reclaimed']))
    write(a.output/'RESULT.json', result)


if __name__ == '__main__':
    main()
