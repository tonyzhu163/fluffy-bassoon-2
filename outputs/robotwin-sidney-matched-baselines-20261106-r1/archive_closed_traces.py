"""Archive only closed train07 simulator traces with full byte restoration proof."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

BASE=Path('/mowenjie/fluffy-bassoon-2')
RUN=BASE/'outputs/robotwin-sidney-matched-baselines-20261106-r1'
PILOT=BASE/'outputs/robotwin-sidney-rlinf-ppo-20261105-r1/train07'
SOURCE=PILOT/'traces'
AUDIT=RUN/'train07-traces-archive01'
COLD=Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-sidney-matched-baselines-20261106-r1/train07-traces-archive01')
ACTIVE=BASE/'sources/robotwin-sidney-matched-20261106-pair02'
HELPER=BASE/'sources/robotwin-sidney-matched-20261106-archive01/outputs/robotwin-sidney-matched-baselines-20261106-r1/archive_closed_rollouts.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='2a6ff30fec95ef9e593596dc34d1405168fd013e7d25a9631f86e3d438880919'
spec=importlib.util.spec_from_file_location('verified_archive',HELPER)
archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)


def main():
    assert os.path.ismount('/mnt/bos/adgfyl')
    assert json.loads((PILOT/'EXIT.json').read_text())['exit_code']==0
    assert json.loads((PILOT/'RESULT.json').read_text())['status']=='complete'
    files=sorted(SOURCE.glob('rank-*/*.pt.gz'))
    assert len(files)==256 and all(f.is_file() and not f.is_symlink() for f in files)
    inventory={str(f):archive.stamp(f) for f in files}
    protected={str(f):archive.stamp(f) for f in (PILOT/'sidney-native-two/checkpoints').rglob('*') if f.is_file()}
    assert len(protected)==36
    # All code is read from the actual immutable scientific source. The native
    # final restore reads its own global_step_2, with no trace replay.
    run=ACTIVE/'outputs/robotwin-sidney-matched-baselines-20261106-r1'
    dependency_files={n:archive.sha(run/n) for n in
        ('launch.sh','gradient_probe.py','paired_entry.py','admission.py','summarize_group_returns.py')}
    assert 'TASK_PROBE_MODE=restore' in (run/'launch.sh').read_text()
    assert 'checkpoints/global_step_2' in (run/'launch.sh').read_text()
    assert not AUDIT.exists() and not COLD.exists()
    AUDIT.mkdir();stage=AUDIT/'staging';stage.mkdir();COLD.mkdir(parents=True)
    report=dict(status='running',pid=os.getpid(),started_unix=time.time(),gpu_used=0,
        source=str(SOURCE),cold_root=str(COLD),source_file_count=len(files),
        staging_limit_bytes=2**30,protected_checkpoint_files=len(protected),
        dependency_source_commit='48fa7ea4db37de7ac499e555670dcb3187343b57',
        dependency_file_sha256=dependency_files,
        reader_scope='Closed successful job-rxih2makcwv4; visible FD scan before each release; owner audit: active arms collect fresh traces and restore only their own trained step2 checkpoint.',
        all_active_science_inputs_and_closed_checkpoints_retained=True,
        original_inventory=inventory,completed=[],peak_staging_bytes=0)
    archive.save(AUDIT/'STATUS.json',report)
    for source in files:
        before=inventory[str(source)];assert before[0]<2**30
        assert archive.stamp(source)==before and not archive.readers(source)
        # Reserve one sub-GiB staging file in addition to every remaining arm cap.
        used=[sum(f.stat().st_blocks*512 for f in (RUN/n).rglob('*') if f.is_file())
            for n in ('ppo02','grpo01')]
        remaining=sum(max(0,135*2**30-n) for n in used)
        assert shutil.disk_usage(BASE).free >= 286*2**30+remaining+before[0]
        relative=source.relative_to(SOURCE);dest=COLD/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        expected,count=archive.copy_checked(source,dest)
        assert count==before[0] and archive.sha(dest)==expected
        staged=stage/'one-trace.pt.gz'
        actual,size=archive.copy_checked(dest,staged)
        assert size==count and actual==expected and archive.sha(staged)==expected
        assert archive.stamp(source)==before and archive.sha(source)==expected
        assert not archive.readers(source)
        assert all(archive.stamp(Path(n))==s for n,s in protected.items())
        assert all(archive.sha(run/n)==v for n,v in dependency_files.items())
        item=dict(source=str(source),archive=str(dest),bytes=count,sha256=expected,
            cold_full_readback_sha256_verified=True,complete_staged_copy_sha256_verified=True,
            source_released=False,verified_unix=time.time(),
            restore_command=f"cp '{dest}' '{source}'; sha256sum '{source}'")
        pointer=source.with_name(source.name+'.cold.json')
        archive.save(pointer,item)
        archive.save(dest.with_name(dest.name+'.receipt.json'),item)
        source.unlink();staged.unlink();item['source_released']=True
        archive.save(pointer,item)
        archive.save(dest.with_name(dest.name+'.receipt.json'),item)
        report['peak_staging_bytes']=max(report['peak_staging_bytes'],size)
        report['completed'].append(item);report['free_bytes_now']=shutil.disk_usage(BASE).free
        archive.save(AUDIT/'STATUS.json',report)
    assert all(archive.stamp(Path(n))==s for n,s in protected.items())
    stage.rmdir()
    report.update(status='complete',completed_unix=time.time(),
        released_bytes=sum(x['bytes'] for x in report['completed']),
        every_file_cold_read_and_fully_staged_SHA256_verified=True,
        free_bytes_now=shutil.disk_usage(BASE).free)
    archive.save(AUDIT/'STATUS.json',report);archive.save(COLD/'STATUS.json',report)


if __name__=='__main__':
    main()
