"""Cold-copy only closed train07 canonical PTs; verify full staging before release.

Plain file copies preserve the original bytes. Each cold object and every
fully staged file is SHA256 checked. Metadata remains hot and is copied cold.
No torch, model computation, checkpoint/cache move, or active input is used.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

BASE=Path('/mowenjie/fluffy-bassoon-2')
PILOT=BASE/'outputs/robotwin-sidney-rlinf-ppo-20261105-r1/train07'
SOURCE=PILOT/'training-rollouts'
AUDIT=BASE/'outputs/robotwin-sidney-matched-baselines-20261106-r1/train07-archive01'
COLD=Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-sidney-matched-baselines-20261106-r1/train07-archive01')
STAGE=AUDIT/'staging'


def save(path,value):
    tmp=path.with_name(path.name+'.writing')
    with tmp.open('w') as h:
        json.dump(value,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
    tmp.replace(path)


def stamp(path):
    s=path.stat();return [s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev,s.st_uid,s.st_gid]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def copy_checked(source,dest):
    h=hashlib.sha256();count=0
    with source.open('rb') as inp,dest.open('xb') as out:
        for block in iter(lambda:inp.read(8*1024**2),b''):
            out.write(block);h.update(block);count+=len(block)
        out.flush();os.fsync(out.fileno())
    assert dest.stat().st_size==count
    return h.hexdigest(),count


def readers(path):
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:files=list((proc/'fd').iterdir())+[proc/'cwd',proc/'exe']
        except OSError:continue
        for f in files:
            try:
                target=os.readlink(f)
                if target==str(path):result.append(dict(pid=int(proc.name),path=target))
            except OSError:pass
    return result


def main():
    assert os.path.ismount('/mnt/bos/adgfyl')
    assert json.loads((PILOT/'EXIT.json').read_text())['exit_code']==0
    assert json.loads((PILOT/'RESULT.json').read_text())['status']=='complete'
    assert shutil.disk_usage(BASE).free>350*2**30  # other286 + grad16 + qual40 + staging8
    files=sorted(SOURCE.glob('*.pt'));assert len(files)==8
    assert not COLD.exists() and not AUDIT.exists()
    AUDIT.mkdir();STAGE.mkdir();COLD.mkdir(parents=True)
    probe=COLD/'write-probe';probe.write_bytes(b'robotwin-closed-rollout-archive\n')
    assert probe.read_bytes()==b'robotwin-closed-rollout-archive\n'
    protected={str(f):stamp(f) for f in (PILOT/'sidney-native-two/checkpoints').rglob('*') if f.is_file()}
    report=dict(status='running',pid=os.getpid(),started_unix=time.time(),gpu_used=0,
        source=str(SOURCE),cold_root=str(COLD),staging_limit_bytes=8*2**30,
        protected_checkpoint_files=len(protected),completed=[],peak_staging_bytes=0,
        reader_scope='Visible development namespace FD scan plus scheduler-closed own job; owner-coordinated no active consumers. Active grad02 uses qual02, science uses original SFT.',
        original_metadata_retained=True,checkpoints_and_model_caches_retained=True)
    save(AUDIT/'STATUS.json',report)
    for source in files:
        assert not readers(source),readers(source)
        before=stamp(source);assert before[0]<8*2**30
        metadata=source.with_suffix('.json');raw=metadata.read_bytes();ref=json.loads(raw)
        assert ref['completed'] and ref['path']==str(source) and ref['sha256']
        assert not source.is_symlink()
        dest=COLD/source.name;expected,count=copy_checked(source,dest)
        assert expected==ref['sha256'] and count==before[0]
        assert sha(dest)==expected  # full independent cold readback
        cold_meta=COLD/metadata.name;cold_meta.write_bytes(raw)
        assert cold_meta.read_bytes()==raw
        staged=STAGE/source.name;actual,staged_bytes=copy_checked(dest,staged)
        assert staged_bytes==count and actual==expected and sha(staged)==expected
        report['peak_staging_bytes']=max(report['peak_staging_bytes'],staged_bytes)
        assert report['peak_staging_bytes']<8*2**30
        assert stamp(source)==before and sha(source)==expected and metadata.read_bytes()==raw
        assert not readers(source),readers(source)
        assert all(stamp(Path(n))==s for n,s in protected.items())
        item=dict(source=str(source),archive=str(dest),bytes=count,sha256=expected,
            cold_full_readback_sha256_verified=True,complete_staged_copy_sha256_verified=True,
            every_member_verified=True,metadata_sha256=hashlib.sha256(raw).hexdigest(),
            source_released=False,verified_unix=time.time(),
            restore_command=f"cp '{dest}' '{source}'; sha256sum '{source}'")
        pointer=source.with_name(source.name+'.cold.json')
        save(pointer,item);save(AUDIT/(source.name+'.json'),item);save(COLD/(source.name+'.receipt.json'),item)
        # Reclaim only this unchanged, completely staged and cold-readback-verified file.
        source.unlink();staged.unlink();item['source_released']=True
        save(pointer,item);save(AUDIT/(source.name+'.json'),item);save(COLD/(source.name+'.receipt.json'),item)
        report['completed'].append(item);report['free_bytes_now']=shutil.disk_usage(BASE).free
        save(AUDIT/'STATUS.json',report)
    assert all(stamp(Path(n))==s for n,s in protected.items())
    STAGE.rmdir();report.update(status='complete',completed_unix=time.time(),
        released_bytes=sum(r['bytes'] for r in report['completed']),
        every_member_staged_and_SHA256_verified=True,free_bytes_now=shutil.disk_usage(BASE).free)
    save(AUDIT/'STATUS.json',report);save(COLD/'STATUS.json',report)


if __name__=='__main__':main()
