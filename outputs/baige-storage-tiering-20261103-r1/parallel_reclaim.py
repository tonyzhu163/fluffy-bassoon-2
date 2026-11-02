"""Bounded reclaim of this audit's already verified private quarantine trees."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import stat
import time

ROOT = Path('/mowenjie')
BASE = ROOT / 'fluffy-bassoon-2/outputs/baige-storage-tiering-20261103-r1'
PRODUCER = BASE.parent / 'baige-storage-tiering-20261103-r3'
AUDIT = BASE.parent / 'baige-storage-tiering-20261103-r4'
RUN = PRODUCER.name


def log(s):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), s, flush=True)


def save(path, value):
    with path.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def remove_known_tree(quarantine, rel, rows):
    """Unlink only inventoried members; never follow directory symlinks."""
    files, dirs = [], defaultdict(list)
    for row in rows:
        suffix = Path(row[0]).relative_to(rel)
        assert '..' not in suffix.parts
        target = quarantine / suffix
        if stat.S_ISDIR(row[1]):
            dirs[len(suffix.parts)].append(target)
        else:
            files.append(target)
    def unlink(p):
        try:
            p.unlink()
            return 1
        except FileNotFoundError:
            return 0
    def rmdir(p):
        try:
            p.rmdir()
            return 1
        except FileNotFoundError:
            return 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        file_count = sum(pool.map(unlink, files))
        dir_count = 0
        for depth in sorted(dirs, reverse=True):
            dir_count += sum(pool.map(rmdir, dirs[depth]))
    assert not quarantine.exists() and not quarantine.is_symlink()
    return {'files_unlinked_by_helper': file_count, 'directories_removed_by_helper': dir_count}


def qualify():
    outside = AUDIT / 'qualification-outside'
    outside.mkdir()
    sentinel = outside / 'keep.txt'
    sentinel.write_bytes(b'unchanged external data\n')
    for name in ('fixture-standard', 'fixture-parallel'):
        root = AUDIT / name
        (root/'a/b').mkdir(parents=True)
        for i in range(32):
            (root/'a/b'/str(i)).write_bytes(bytes([i])*17)
        (root/'outside-link').symlink_to(outside, target_is_directory=True)
        os.link(sentinel, root/'external-hardlink')
    standard = AUDIT/'fixture-standard'
    parallel = AUDIT/'fixture-parallel'
    rows=[]; pending=[parallel]
    while pending:
        p=pending.pop(); s=p.lstat()
        rows.append([str(p.relative_to(ROOT)), s.st_mode])
        if stat.S_ISDIR(s.st_mode):
            pending.extend(p.iterdir())
    shutil.rmtree(standard)
    result = remove_known_tree(parallel, parallel.relative_to(ROOT), rows)
    assert not standard.exists() and not parallel.exists()
    assert sentinel.read_bytes()==b'unchanged external data\n'
    save(AUDIT/'QUALIFICATION.json', dict(result, same_result_as_shutil_rmtree=True,
         outside_symlink_target_and_hardlink_preserved=True, filesystem='actual Baige PFS'))


def reclaim(rel):
    source=ROOT/rel
    quarantine=source.with_name(source.name+'.tiering-'+RUN)
    key=rel.replace('/', '__')
    manifest=PRODUCER/(key+'.json')
    item=json.loads(manifest.read_text())
    assert item['source']==str(source)
    assert item['archive_source_compare'] and item['bos_readback_verified']
    assert not source.exists() and not source.is_symlink()
    archive=Path(item['bos_archive'])
    assert archive.parent==Path('/mnt/bos/adgfyl/strong-plant-archives/20261103-rl-checkpoint-space-r1/parallel-r3')
    assert archive.stat().st_size==item['archive_bytes']
    assert archive.with_name(archive.name+'.sha256').read_text().split()[0]==item['sha256']
    rows=json.loads((PRODUCER/(key+'.inventory.json')).read_text())
    root_row=next(r for r in rows if r[0]==rel)
    if quarantine.exists():
        s=quarantine.lstat()
        assert stat.S_ISDIR(s.st_mode) and not quarantine.is_symlink()
        assert (s.st_dev,s.st_ino)==(root_row[7],root_row[8]), 'Quarantine identity changed'
    log('Adopting VERIFIED quarantine reclaim: '+rel)
    result=remove_known_tree(quarantine, Path(rel), rows)
    item['source_moved']=True
    item['moved_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    item['reclaim_owner']=AUDIT.name
    for path in (manifest, archive.parent/manifest.name, source.with_name(source.name+'.bos-archive.json')):
        save(path,item)
    result.update(source=str(source), sha256=item['sha256'], complete=True,
                  finished_at=item['moved_at'], archive_bytes=item['archive_bytes'])
    save(AUDIT/(key+'.json'),result)
    log('VERIFIED parallel reclaim complete: '+rel)
    return result


if __name__=='__main__':
    AUDIT.mkdir(exist_ok=False)
    qualify()
    sources=json.loads((PRODUCER/'PLAN.json').read_text())['sources']
    save(AUDIT/'PLAN.json', {'sources':sources,'max_parallel_trees':2,'unlink_workers_per_tree':16,
         'guard':'Only r3 private quarantine with complete archive/source comparison, complete BOS SHA256 readback, matching object size/sidecar and original directory device/inode. No original source namespace is unlinked.',
         'handoff':'The original serial remover may report ENOENT after this helper adopts reclamation. Preserve that raw report; reconcile against the completed reclaim receipt and the verified archive before finalization.'})
    futures={}; results=[]; failures=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            for rel in sources:
                q=(ROOT/rel).with_name((ROOT/rel).name+'.tiering-'+RUN)
                if rel not in futures and q.is_dir() and not q.is_symlink():
                    futures[rel]=pool.submit(reclaim,rel)
            for rel,future in list(futures.items()):
                if future.done() and not getattr(future,'receipt_written',False):
                    try:
                        results.append(future.result())
                    except Exception as error:
                        failures.append({'source':rel,'error':str(error)})
                        log('RECLAIM NEEDS REVIEW '+rel+': '+str(error))
                    future.receipt_written=True
                    save(AUDIT/'SUMMARY.json',{'completed':results,'failures':failures})
            try:
                producer_done=json.loads((PRODUCER/'SUMMARY.json').read_text()).get('finished_at')
            except (FileNotFoundError,json.JSONDecodeError):
                producer_done=False
            if producer_done and all(f.done() for f in futures.values()):
                break
            time.sleep(2)
    save(AUDIT/'SUMMARY.json',{'completed':results,'failures':failures,
         'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    log('PARALLEL VERIFIED RECLAIM FINISHED')
