"""Reversibly archive two closed plain Sidney exports, not their native DCP.

Their completed paired endpoint readout is the last consumer. Preserve every
byte, export/calibration metadata, native checkpoint, sampler/RNG and cache.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import time

BASE = Path('/mowenjie/fluffy-bassoon-2')
RUN = BASE/'outputs/robotwin-restoration-20261106-r1'
EXPORT = BASE/'outputs/robotwin-sidney-endpoint-readout-20261106-r1/export02'
READOUT = BASE/'outputs/robotwin-sidney-endpoint-readout-20261106-r1/readout01'
PAIR = BASE/'outputs/robotwin-sidney-matched-baselines-20261106-r1'
AUDIT = RUN/'exports01'
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/exports01')
STAGE = Path('/tmp/fluffy-bassoon-2-robotwin-restoration-20261107-exports01')


def main():
    code = Path(os.environ['TASK_CODE'])
    sys.path.insert(0,str(code/'outputs/robotwin-sidney-matched-baselines-20261106-r1'))
    from archive_closed_rollouts import copy_checked, readers, save, sha, stamp
    assert sys.platform == 'linux' and os.path.ismount('/mnt/bos/adgfyl')
    assert not AUDIT.exists() and not COLD.exists() and not STAGE.exists()
    assert json.loads((READOUT/'RESULT.json').read_text())['status'] == 'complete'
    assert json.loads((READOUT/'EXIT.json').read_text())['exit_code'] == 0
    assert json.loads((EXPORT/'RESULT.json').read_text())['plain_inference_weights_qualified']
    assert json.loads((EXPORT/'EXIT.json').read_text())['exit_code'] == 0
    files = [EXPORT/arm/'policy.pt' for arm in ('ppo','grpo')]
    inventory, references, protected = {}, {}, {}
    for source in files:
        receipt = json.loads((source.parent/'EXPORT.json').read_text())
        assert receipt['status'] == 'qualified' and receipt['path'] == str(source)
        assert source.is_file() and not source.is_symlink() and not readers(source)
        assert source.stat().st_size == receipt['bytes'] < 8*2**30
        inventory[str(source)],references[str(source)] = stamp(source),receipt
    for arm in ('ppo02','grpo01'):
        root = PAIR/arm
        assert json.loads((root/'RESULT.json').read_text())['status'] == 'complete'
        assert json.loads((root/'EXIT.json').read_text())['exit_code'] == 0
        for path in root.rglob('*'):
            if path.is_file() and ('checkpoints' in path.parts or path.parent.name == 'training-rollouts'
                                  or 'sampling' in path.parts or 'initial-states' in path.parts):
                protected[str(path)] = stamp(path)
    for root in (READOUT, EXPORT):
        for path in root.rglob('*'):
            if path.is_file() and path not in files: protected[str(path)] = stamp(path)
    motor = BASE/'cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1'
    for path in (motor/'model.safetensors',motor/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'):
        protected[str(path)] = stamp(path)
    assert len(protected) >= 1172
    assert shutil.disk_usage('/tmp').free > 9*2**30
    AUDIT.mkdir();COLD.mkdir(parents=True);STAGE.mkdir()
    report = dict(status='running',pid=os.getpid(),source_commit=os.environ['TASK_SOURCE'],
        started_unix=time.time(),gpu_seconds=0,completed=[],original_inventory=inventory,
        cold_root=str(COLD),local_staging_root=str(STAGE),local_staging_limit_bytes=8*2**30,
        protected_files=len(protected),peak_local_staging_bytes=0,released_bytes=0,
        original_other_and_floor_gib=286,other_owner_allowance_changed=False,
        checkpoints_canonical_sampler_rng_calibrations_and_model_caches_retained=True,
        consumer_audit='Export02 and readout01 exited0 and scheduler released; no active own Sidney consumer. Native Motus profiling uses its separate untouched cache. Visible namespace FD scan before release.')
    save(AUDIT/'PROTECTED.json',protected);save(AUDIT/'STATUS.json',report)
    try:
        for source in files:
            before, receipt = inventory[str(source)],references[str(source)]
            assert stamp(source) == before and not readers(source)
            assert shutil.disk_usage(STAGE).free > before[0]+2**30
            dest = COLD/source.relative_to(EXPORT);dest.parent.mkdir(parents=True)
            digest, count = copy_checked(source,dest)
            assert digest == receipt['sha256'] and count == before[0] and sha(dest) == digest
            staged = STAGE/'one-export.pt'
            actual, size = copy_checked(dest,staged)
            assert actual == digest and size == count and sha(staged) == digest
            assert sha(source) == digest and stamp(source) == before and not readers(source)
            assert all(stamp(Path(path)) == value for path,value in protected.items())
            item = dict(source=str(source),archive=str(dest),bytes=count,sha256=digest,
                independent_full_cold_readback=True,complete_local_stageback=True,
                complete_local_stageback_readback=True,source_released=False,verified_unix=time.time(),
                restore_command=f"cp '{dest}' '{source}' && sha256sum '{source}'")
            pointer = source.with_name(source.name+'.cold.json')
            save(pointer,item);save(dest.with_name(dest.name+'.receipt.json'),item)
            source.unlink();staged.unlink();item['source_released'] = True
            save(pointer,item);save(dest.with_name(dest.name+'.receipt.json'),item)
            report['completed'].append(item);report['released_bytes'] += count
            report['peak_local_staging_bytes'] = max(report['peak_local_staging_bytes'],size)
            report['free_gib_now'] = shutil.disk_usage(BASE).free/2**30
            save(AUDIT/'STATUS.json',report)
            print(json.dumps(dict(files=len(report['completed']),released_gib=report['released_bytes']/2**30,
                                  free_gib=report['free_gib_now'])),flush=True)
        assert all(stamp(Path(path)) == value for path,value in protected.items())
        STAGE.rmdir()
        report.update(status='complete',completed_unix=time.time(),all_original_bytes_recoverable=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        save(AUDIT/'STATUS.json',report);save(COLD/'STATUS.json',report)


if __name__ == '__main__': main()
