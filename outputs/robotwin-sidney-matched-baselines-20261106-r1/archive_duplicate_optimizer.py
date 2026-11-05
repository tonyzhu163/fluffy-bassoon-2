"""Cold archive the closed opt01 duplicate zero-update DCP; retain all metadata."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

BASE=Path('/mowenjie/fluffy-bassoon-2')
RUN=BASE/'outputs/robotwin-sidney-matched-baselines-20261106-r1'
SOURCE=RUN/'opt01/train/checkpoint0/actor/dcp_checkpoint'
AUDIT=RUN/'opt01-archive01'
COLD=Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-sidney-matched-baselines-20261106-r1/opt01-zero-dcp-archive01')
HELPER=BASE/'sources/robotwin-sidney-matched-20261106-archive01/outputs/robotwin-sidney-matched-baselines-20261106-r1/archive_closed_rollouts.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='2a6ff30fec95ef9e593596dc34d1405168fd013e7d25a9631f86e3d438880919'
spec=importlib.util.spec_from_file_location('verified_rollout_archive',HELPER)
archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)


def main():
    assert os.path.ismount('/mnt/bos/adgfyl')
    assert json.loads((RUN/'opt01/EXIT.json').read_text())['exit_code']==0
    assert json.loads((RUN/'grad02/EXIT.json').read_text())['exit_code']==0
    for rank in range(4):
        value=json.loads((RUN/'grad02/restore/recovery-signatures'/f'actor-{rank:02d}.json').read_text())
        assert value['semantic_identity'] and value['actual']['optimizer_steps']==1
    files=sorted(SOURCE.iterdir());assert len(files)==5
    assert sum(f.suffix=='.distcp' for f in files)==4
    assert all(f.is_file() and not f.is_symlink() for f in files)
    assert shutil.disk_usage(BASE).free>342*2**30
    protected={str(f):archive.stamp(f) for f in (RUN/'opt01').rglob('*')
        if f.is_file() and (f.suffix=='.json' or 'pilot' in f.parts)}
    assert not AUDIT.exists() and not COLD.exists()
    AUDIT.mkdir();stage=AUDIT/'staging';stage.mkdir();COLD.mkdir(parents=True)
    probe=COLD/'write-probe';probe.write_bytes(b'closed-opt01-duplicate-DCP\n')
    assert probe.read_bytes()==b'closed-opt01-duplicate-DCP\n'
    report=dict(status='running',pid=os.getpid(),started_unix=time.time(),gpu_used=0,
        source=str(SOURCE),cold_root=str(COLD),staging_limit_bytes=8*2**30,
        protected_metadata_and_pilot_files=len(protected),completed=[],peak_staging_bytes=0,
        scope='Closed duplicate opt01 zero-update DCP only; qual02/main/grad02/qual03 checkpoints and all model caches stay hot.',
        dependency='No current reader: grad02/fullqual use their own checkpoints, science starts original SFT and final restoration reads its own step40 checkpoint.',
        staging_reservation='Existing8GiB archive allowance migrated from completed train07 archive; no overlap.')
    archive.save(AUDIT/'STATUS.json',report)
    for source in files:
        assert not archive.readers(source)
        before=archive.stamp(source);dest=COLD/source.name
        expected,count=archive.copy_checked(source,dest)
        assert count==before[0] and archive.sha(dest)==expected
        staged=stage/source.name;actual,size=archive.copy_checked(dest,staged)
        assert size==count and actual==expected and archive.sha(staged)==expected
        assert archive.stamp(source)==before and archive.sha(source)==expected
        assert not archive.readers(source)
        assert all(archive.stamp(Path(n))==s for n,s in protected.items())
        report['peak_staging_bytes']=max(report['peak_staging_bytes'],size)
        assert report['peak_staging_bytes']<8*2**30
        item=dict(source=str(source),archive=str(dest),bytes=count,sha256=expected,
            cold_full_readback_sha256_verified=True,complete_staged_copy_sha256_verified=True,
            source_released=False,verified_unix=time.time(),
            restore_command=f"cp '{dest}' '{source}'; sha256sum '{source}'")
        pointer=source.with_name(source.name+'.cold.json')
        archive.save(AUDIT/(source.name+'.json'),item);archive.save(COLD/(source.name+'.receipt.json'),item)
        if source.suffix=='.distcp':
            archive.save(pointer,item);source.unlink();item['source_released']=True;archive.save(pointer,item)
        staged.unlink();report['completed'].append(item)
        archive.save(AUDIT/(source.name+'.json'),item);archive.save(COLD/(source.name+'.receipt.json'),item)
        archive.save(AUDIT/'STATUS.json',report)
    assert all(archive.stamp(Path(n))==s for n,s in protected.items())
    stage.rmdir();report.update(status='complete',completed_unix=time.time(),
        released_bytes=sum(x['bytes'] for x in report['completed'] if x['source_released']),
        all_files_staged_and_SHA256_verified=True,free_bytes_now=shutil.disk_usage(BASE).free)
    archive.save(AUDIT/'STATUS.json',report);archive.save(COLD/'STATUS.json',report)


if __name__=='__main__':main()
