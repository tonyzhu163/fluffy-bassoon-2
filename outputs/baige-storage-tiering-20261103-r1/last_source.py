"""Resume the final source after qualified parallel GNU tar comparison."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import time

BASE=Path('/mowenjie/fluffy-bassoon-2/outputs/baige-storage-tiering-20261103-r1')
AUDIT=BASE.parent/'baige-storage-tiering-20261103-r5'

def load(name):
    spec=importlib.util.spec_from_file_location(name,BASE/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__=='__main__':
    p=load('concurrent_inventory_phase')
    c=load('parallel_compare')
    m=p.m
    m.inventory=p.inventory
    rel='code/wam-ttt'
    key=rel.replace('/','__')
    source=m.ROOT/rel
    local=m.STAGE/(key+'.tar.zst')
    dest=m.BOS/local.name
    producer=1944806
    tar_pid=1952081
    process=Path('/proc')/str(producer)
    assert b'concurrent_inventory_phase.py' in (process/'cmdline').read_bytes()
    identity=(process/'stat').read_text().split()[21]
    assert json.loads((AUDIT/'QUALIFICATION.json').read_text())['qualified']
    assert source.is_dir() and not source.is_symlink() and not dest.exists()
    assert not (m.AUDIT/(key+'.json')).exists()
    for candidate in json.loads((m.AUDIT/'PLAN.json').read_text())['sources']:
        if candidate!=rel:
            assert json.loads((m.AUDIT/(candidate.replace('/','__')+'.json')).read_text())['source_moved']
    migration={'source':str(source),'producer_pid':producer,'producer_start_ticks':identity,
               'reason':'Final serial comparison is I/O bound on 515567 entries; use eight exact, nonrecursive GNU tar member partitions after positive and negative qualification.',
               'old_comparison':'Serial comparison is retained until every partition passes; its parent is paused to prevent concurrent upload or reclaim.',
               'all_other_sources_verified_moved':True,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    os.kill(producer,signal.SIGSTOP)
    try:
        m.save(AUDIT/'MIGRATION.json',migration)
        rows=json.loads((m.AUDIT/(key+'.inventory.json')).read_text())
        assert m.inventory(source)==rows,'Source changed since packing'
        m.log('r5 listing complete archive for exact member coverage')
        names=c.archive_names(local)
        assert sorted(names)==[r[0] for r in rows],'Archive/inventory member set mismatch'
        m.save(AUDIT/'COVERAGE.json',{'archive':str(local),'inventory':str(m.AUDIT/(key+'.inventory.json')),
               'members':len(names),'archive_member_set_equals_inventory':True,
               'sorted_member_names_sha256':hashlib.sha256(b''.join(os.fsencode(n)+b'\0' for n in sorted(names))).hexdigest()})
        m.log('r5 eight exact GNU tar comparison partitions started')
        compared=c.run_comparison(local,m.ROOT,names,AUDIT/'actual',8)
        assert compared['all_passed'],'At least one partition reported a source difference'
        m.log('r5 all GNU tar comparison partitions PASSED')
        # Terminate only this audit's redundant comparator, after full replacement
        # comparison succeeds. The producer remains paused until reclaim completes.
        old_tar=Path('/proc')/str(tar_pid)
        if old_tar.exists() and (old_tar/'cmdline').read_bytes():
            assert (old_tar/'stat').read_text().split()[3]==str(producer)
            args=(old_tar/'cmdline').read_bytes().split(b'\0')
            assert b'-df' in args and os.fsencode(local) in args
            os.kill(tar_pid,signal.SIGTERM)
        deadline=time.monotonic()+30
        while m.users(source):
            assert time.monotonic()<deadline, m.users(source)
            time.sleep(1)
        assert m.inventory(source)==rows,'Source changed during comparison'
        h=hashlib.sha256(); copied=0
        with local.open('rb') as inp,dest.open('xb') as out:
            while chunk:=inp.read(16*1024**2):
                out.write(chunk);h.update(chunk);copied+=len(chunk)
            out.flush();os.fsync(out.fileno())
        expected=h.hexdigest()
        assert dest.stat().st_size==copied==local.stat().st_size
        assert m.digest(dest)==expected,'BOS readback mismatch'
        assert m.inventory(source)==rows,'Source changed during object upload'
        assert not m.users(source)
        item=dict(source=str(source),bos_archive=str(dest),archive_bytes=copied,
                  source_allocated_bytes=sum(r[6] for r in rows)*512,
                  source_logical_bytes=sum(r[2] for r in rows if stat.S_ISREG(r[1])),
                  entries=len(rows),sha256=expected,archive_source_compare=True,
                  archive_source_compare_method='Eight exact nonrecursive GNU tar comparison partitions; complete archive member set equals unchanged inventory.',
                  comparison_receipt=str(AUDIT/'actual/RESULT.json'),
                  bos_readback_verified=True,source_moved=False,
                  verified_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  restore_command=f"tar --acls --xattrs -I zstd -xf '{dest}' -C /mowenjie")
        manifest=m.AUDIT/(key+'.json')
        m.save(manifest,item);m.save(m.BOS/manifest.name,item)
        shutil.copyfile(m.AUDIT/(key+'.inventory.json'),m.BOS/(key+'.inventory.json'))
        (m.BOS/(dest.name+'.sha256')).write_text(expected+'  '+dest.name+'\n')
        m.save(source.with_name(source.name+'.bos-archive.json'),item)
        quarantine=source.with_name(source.name+'.tiering-'+m.RUN)
        assert not quarantine.exists()
        os.rename(source,quarantine)
        m.log('r5 final source BOS verified; handed private quarantine to r4')
        receipt=BASE.parent/'baige-storage-tiering-20261103-r4'/(key+'.json')
        while not receipt.exists():
            assert Path('/proc/1947986').exists(),'Reclaim helper exited'
            time.sleep(10)
        assert json.loads(receipt.read_text())['complete']
        assert json.loads(manifest.read_text())['source_moved']
        assert not quarantine.exists()
        local.unlink()
        migration.update(complete=True,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                         comparison_receipt=str(AUDIT/'actual/RESULT.json'),reclaim_receipt=str(receipt))
        m.save(AUDIT/'MIGRATION.json',migration)
        m.log('r5 VERIFIED final source complete')
    finally:
        if process.exists() and (process/'stat').read_text().split()[21]==identity:
            os.kill(producer,signal.SIGCONT)
