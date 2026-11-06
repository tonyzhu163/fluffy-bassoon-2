"""Reversibly cold-archive only completed Sidney pair simulator traces.

Every original byte is cold-read and fully restored on local scratch before
release. All canonical learner batches, checkpoints, sampler/RNG sidecars,
models, initial states and endpoint exports stay at their original paths.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import time


BASE = Path('/mowenjie/fluffy-bassoon-2')
PAIR = BASE/'outputs/robotwin-sidney-matched-baselines-20261106-r1'
AUDIT = BASE/'outputs/robotwin-restoration-20261106-r1/traces01'
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/traces01')
STAGE = Path('/tmp/fluffy-bassoon-2-robotwin-restoration-20261107-traces01')


def main():
    code = Path(os.environ['TASK_CODE'])
    sys.path.insert(0,str(code/'outputs/robotwin-sidney-matched-baselines-20261106-r1'))
    from archive_closed_rollouts import copy_checked, readers, save, sha, stamp

    assert sys.platform == 'linux' and os.path.ismount('/mnt/bos/adgfyl')
    assert not AUDIT.exists() and not COLD.exists() and not STAGE.exists()
    assert shutil.disk_usage('/tmp').free > 2*2**30
    readout = BASE/'outputs/robotwin-sidney-endpoint-readout-20261106-r1/readout01'
    assert json.loads((readout/'RESULT.json').read_text())['status'] == 'complete'
    assert json.loads((readout/'EXIT.json').read_text())['exit_code'] == 0
    files, protected = [], {}
    for arm in ('ppo02','grpo01'):
        root = PAIR/arm
        assert json.loads((root/'RESULT.json').read_text())['status'] == 'complete'
        assert json.loads((root/'EXIT.json').read_text())['exit_code'] == 0
        rows = sorted((root/'traces').glob('rank-*/*.pt.gz'))
        assert len(rows) == 256
        assert all(p.is_file() and not p.is_symlink() for p in rows)
        assert max(p.stat().st_size for p in rows) < 2**30
        files.extend(rows)
        for path in root.rglob('*'):
            if path.is_file() and ('checkpoints' in path.parts or path.parent.name == 'training-rollouts'
                                   or 'sampling' in path.parts or 'initial-states' in path.parts):
                protected[str(path)] = stamp(path)
    inventory = {str(path):stamp(path) for path in files}
    assert protected and len(files) == 512
    AUDIT.mkdir(); COLD.mkdir(parents=True); STAGE.mkdir()
    for parent in (COLD, STAGE):
        probe = parent/'write-probe'
        with probe.open('xb') as stream:
            stream.write(b'closed-owned-robotwin-trace-archive\n');stream.flush();os.fsync(stream.fileno())
        assert probe.read_bytes() == b'closed-owned-robotwin-trace-archive\n'
        probe.unlink()
    report = dict(status='running', source_commit=os.environ['TASK_SOURCE'], pid=os.getpid(),
        started_unix=time.time(), gpu_seconds=0, source_file_count=len(files), completed=[],
        cold_root=str(COLD), staging_root=str(STAGE), staging_limit_bytes=2**30,
        original_inventory=inventory, protected_files=len(protected), peak_staging_bytes=0,
        ownership='Closed own PPO02/GRPO01 jobs and completed endpoint readout; no active source consumer. Visible FD check before each release.',
        other_owner_floor_gib=286, other_owner_allowance_changed=False,
        metadata_growth_cap_bytes=64*2**20, raw_shared_staging_bytes=0,
        checkpoint_canonical_sampling_model_and_export_paths_retained=True)
    save(AUDIT/'STATUS.json',report)
    try:
        for source in files:
            before = inventory[str(source)]
            assert stamp(source) == before and not readers(source)
            assert shutil.disk_usage(STAGE).free > before[0]+2**30
            relative = source.relative_to(PAIR); dest = COLD/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            digest, count = copy_checked(source,dest)
            assert count == before[0] and sha(dest) == digest
            staged = STAGE/'one-trace.pt.gz'
            actual, size = copy_checked(dest,staged)
            assert size == count and actual == digest and sha(staged) == digest
            assert stamp(source) == before and sha(source) == digest and not readers(source)
            assert all(stamp(Path(path)) == value for path,value in protected.items())
            item = dict(source=str(source), archive=str(dest), bytes=count, sha256=digest,
                independent_cold_readback=True, complete_local_restore_readback=True,
                source_released=False, verified_unix=time.time(),
                restore_command=f"cp '{dest}' '{source}' && sha256sum '{source}'")
            pointer = source.with_name(source.name+'.cold.json')
            save(pointer,item);save(dest.with_name(dest.name+'.receipt.json'),item)
            source.unlink();staged.unlink();item['source_released'] = True
            save(pointer,item);save(dest.with_name(dest.name+'.receipt.json'),item)
            report['completed'].append(item)
            report['peak_staging_bytes'] = max(report['peak_staging_bytes'],size)
            report['released_bytes'] = sum(row['bytes'] for row in report['completed'])
            report['free_bytes_now'] = shutil.disk_usage(BASE).free
            save(AUDIT/'STATUS.json',report)
            if len(report['completed']) % 16 == 0:
                print(json.dumps(dict(files=len(report['completed']),released_gib=report['released_bytes']/2**30,
                    free_gib=report['free_bytes_now']/2**30)),flush=True)
        assert all(stamp(Path(path)) == value for path,value in protected.items())
        STAGE.rmdir()
        report.update(status='complete',completed_unix=time.time(),all_original_bytes_recoverable=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        save(AUDIT/'STATUS.json',report);save(COLD/'STATUS.json',report)


if __name__ == '__main__': main()
