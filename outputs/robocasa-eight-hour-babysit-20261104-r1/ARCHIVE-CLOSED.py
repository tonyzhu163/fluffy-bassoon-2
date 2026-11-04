"""Move audited, released, owned readout bytes to private cold storage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def main():
    assert sys.platform=='linux'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group')
    parser.add_argument('archive_id')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    assert 1 <= args.workers <= 4
    assert all(re.fullmatch(r'[A-Za-z0-9_-]+',value) for value in (args.group,args.archive_id))
    run=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1')
    source=run/'gpu'/args.group
    cold_root=Path('/mnt/bos/adgfyl/strong-plant-archives/robocasa-eight-hour-babysit-20261104-r1')
    target=cold_root/args.archive_id
    receipts=run/'archives';receipts.mkdir(exist_ok=True)
    lock=(receipts/(args.group+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert source.is_dir() and not source.is_symlink() and not target.exists()
    ledger=json.loads((run/'META-ALLOCATION-LEDGER.json').read_text())
    row=next(row for row in ledger['allocations'] if row.get('output')=='gpu/'+args.group)
    assert row.get('released_shanghai'), 'Scheduler release must be independently verified first'
    audit=json.loads((source/'AUDIT-STATUS.json').read_text())
    assert audit['passed'] and sha(source/'RESULT.json')==audit['group_result_sha256']
    summary=Path(audit['audit_summary']['path'])
    assert sha(summary)==audit['audit_summary']['sha256']
    entries=sorted(source.rglob('*'))
    assert not any(path.is_symlink() for path in entries), 'Never follow shared dependencies into an archive'
    files=[path for path in entries if path.is_file()]
    assert files and not any(path.name=='ARCHIVE-MANIFEST.json' for path in files)
    cold_root.mkdir(parents=True,exist_ok=True)
    fs=os.statvfs(cold_root)
    assert not fs.f_flag & os.ST_RDONLY
    probe=cold_root/(args.archive_id+'.write-probe')
    with probe.open('xb') as stream:
        stream.write(b'owned closed readout archive probe\n'*65536);stream.flush();os.fsync(stream.fileno())
    probe_hash=sha(probe);assert probe_hash==hashlib.sha256(b'owned closed readout archive probe\n'*65536).hexdigest()
    probe.unlink()
    target.mkdir()
    for directory in entries:
        if directory.is_dir():(target/directory.relative_to(source)).mkdir(parents=True,exist_ok=True)
    began=time.time()
    def copy(path):
        relative=path.relative_to(source);destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256();size=0
        with path.open('rb') as incoming,destination.open('xb') as outgoing:
            while True:
                block=incoming.read(8<<20)
                if not block:break
                digest.update(block);outgoing.write(block);size+=len(block)
            outgoing.flush();os.fsync(outgoing.fileno())
        assert destination.stat().st_size==size and sha(destination)==digest.hexdigest()
        return dict(relative=str(relative),bytes=size,sha256=digest.hexdigest())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        manifest=list(pool.map(copy,files))
    assert {path.relative_to(source) for path in source.rglob('*') if path.is_file()}=={Path(row['relative']) for row in manifest}
    assert all(sha(source/row['relative'])==row['sha256'] for row in manifest)
    record=dict(source=str(source),target=str(target),job_id=row['job_id'],scheduler_released_shanghai=row['released_shanghai'],
        group_result_sha256=audit['group_result_sha256'],audit_summary=audit['audit_summary'],files=manifest)
    cold_manifest=target/'ARCHIVE-MANIFEST.json';write(cold_manifest,record)
    quarantine=source.with_name(source.name+'.archive-verified-'+args.archive_id)
    assert not quarantine.exists()
    source.rename(quarantine)
    try:
        source.symlink_to(target,target_is_directory=True)
        assert source.is_symlink() and source.resolve()==target
        assert all(sha(source/item['relative'])==item['sha256'] for item in manifest)
    except BaseException:
        if source.is_symlink():source.unlink()
        quarantine.rename(source)
        raise
    # Only the independently verified duplicate is removed; every original
    # referenced pathname now resolves to the same cold byte stream.
    shutil.rmtree(quarantine)
    receipt=dict(completed=True,source=str(source),target=str(target),job_id=row['job_id'],
        files=len(manifest),bytes=sum(item['bytes'] for item in manifest),
        manifest=dict(path=str(cold_manifest),sha256=sha(cold_manifest)),
        stable_referenced_paths_verified=True,hot_duplicate_removed=True,probe_sha256=probe_hash,
        cold_capacity_reported_bytes=fs.f_bavail*fs.f_frsize,
        quota='No per-project BOS quota interface exposed; actual writes/fsync/reads all verified before hot removal',
        elapsed_seconds=time.time()-began,completed_unix=time.time(),GPU_hours=0)
    write(receipts/(args.archive_id+'.json'),receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
