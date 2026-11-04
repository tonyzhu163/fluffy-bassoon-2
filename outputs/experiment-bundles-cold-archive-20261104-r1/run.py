"""Cold archive explicitly selected legacy experiment bundles; no active scans."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import stat
import threading
import time

ROOT=Path('/mowenjie')
RUN='experiment-bundles-cold-archive-20261104-r1'
BASE=ROOT/'fluffy-bassoon-2/outputs'/RUN
LIB=ROOT/'fluffy-bassoon-2/outputs/storage-audit-baige-20261104-r1'
spec=importlib.util.spec_from_file_location('qualified_archive',LIB/'fast_archive.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
f.ROOT=ROOT;f.BASE=BASE;f.AUDIT=BASE/'receipts'
f.BOS=Path('/mnt/bos/adgfyl/strong-plant-archives')/RUN
f.STAGE=Path('/tmp')/RUN
f.m.RUN=RUN


def inventory(source):
    pending=queue.Queue();pending.put(source);rows=[];errors=[]
    def worker():
        while True:
            p=pending.get()
            try:
                if p is None:return
                s=p.lstat()
                assert stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode) or stat.S_ISLNK(s.st_mode),p
                rows.append([str(p.relative_to(f.ROOT)),s.st_mode,s.st_size,s.st_mtime_ns,
                             s.st_uid,s.st_gid,s.st_blocks,s.st_dev,s.st_ino,None,
                             os.readlink(p) if stat.S_ISLNK(s.st_mode) else None])
                if stat.S_ISDIR(s.st_mode):
                    for child in p.iterdir():pending.put(child)
            except Exception as e:errors.append([str(p),repr(e)])
            finally:pending.task_done()
    threads=[threading.Thread(target=worker) for _ in range(24)]
    for t in threads:t.start()
    pending.join()
    for _ in threads:pending.put(None)
    for t in threads:t.join()
    assert not errors,errors[:5]
    return sorted(rows)


def reclaim(source,rows):
    quarantine=source.with_name(source.name+'.archiving-'+RUN)
    assert not quarantine.exists()
    original=source.stat();r=next(r for r in rows if r[0]==str(source.relative_to(f.ROOT)))
    assert (original.st_dev,original.st_ino)==(r[7],r[8])
    source.rename(quarantine)
    def target(r):return quarantine/Path(r[0]).relative_to(source.relative_to(f.ROOT))
    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(lambda r:target(r).unlink(),[r for r in rows if not stat.S_ISDIR(r[1])]))
        dirs=[r for r in rows if stat.S_ISDIR(r[1])]
        for depth in sorted({len(Path(r[0]).parts) for r in dirs},reverse=True):
            list(pool.map(lambda r:target(r).rmdir(),[r for r in dirs if len(Path(r[0]).parts)==depth]))
    assert not quarantine.exists()

f.inventory=inventory;f.reclaim=reclaim


def main():
    assert os.path.ismount('/mnt/bos/adgfyl')
    assert (LIB/'FAST-QUALIFICATION.json').is_file()
    f.qualify()
    if '--qualify-only' in __import__('sys').argv:return
    for p in (f.AUDIT,f.BOS,f.STAGE):p.mkdir(exist_ok=False)
    for p in (BASE/'WRITE-PROBE.txt',f.BOS/'WRITE-PROBE.txt'):
        p.write_text(RUN+'\n');assert p.read_text()==RUN+'\n'
    plan=json.loads((BASE/'PLAN.json').read_text())
    candidates=[]
    for family,rs in plan['families'].items():
        for r in rs:
            p=Path(r['source']);assert p.parent in (ROOT/'fluffy-bassoon/outputs',ROOT/'fluffy-bassoon-2/outputs')
            assert any(x in p.name.lower() for x in ('mikasa','dsrl','robotwin','libero'))
            assert not p.name.endswith('.bos-archive.json')
            assert not p.is_symlink()
            assert p.is_dir(),p
            candidates.append(dict(r,family=family))
    candidates.sort(key=lambda r:r['census_allocated_bytes'],reverse=True)
    assert len({r['source'] for r in candidates})==len(candidates)
    before=shutil.disk_usage(ROOT)._asdict();f.m.save(BASE/'SPACE-BEFORE.json',before)
    f.m.save(BASE/'LIBRARY-HASHES.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (LIB/'fast_archive.py',LIB/'archive.py',LIB/'parallel_compare.py')})
    completed=[];preserved=[]
    def one(r):
        if shutil.disk_usage(f.STAGE).free<max(10<<30,int(r['census_allocated_bytes']*1.25)):
            raise RuntimeError('Insufficient private overlay staging capacity; source untouched')
        result=f.archive(str(Path(r['source']).relative_to(ROOT)),workers=4)
        result=dict(result,family=r['family'])
        return result
    def summary(finished=False):
        data={'run_id':RUN,'completed':completed,'preserved':preserved,'finished':finished,
              'requested_sources':len(candidates),'completed_sources':len(completed),
              'archived_source_allocation_bytes':sum(r['source_allocated_bytes'] for r in completed),
              'archive_bytes':sum(r['archive_bytes'] for r in completed),
              'filesystem_before':before,'filesystem_now':shutil.disk_usage(ROOT)._asdict(),
              'updated_shanghai':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'GPU_jobs_launched':0,'running_job_folders_scanned':False}
        f.m.save(BASE/'SUMMARY.json',data)
        return data
    summary()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(one,r):r for r in candidates}
        for future in as_completed(futures):
            r=futures[future]
            try:completed.append(future.result())
            except Exception as e:
                preserved.append({'source':r['source'],'family':r['family'],'error':repr(e)})
                f.m.log(r['source'],'PRESERVED / '+repr(e))
            summary()
    result=summary(True);f.m.save(f.BOS/'SUMMARY.json',result)
    f.m.log(RUN,'FINISHED')

if __name__=='__main__':main()
