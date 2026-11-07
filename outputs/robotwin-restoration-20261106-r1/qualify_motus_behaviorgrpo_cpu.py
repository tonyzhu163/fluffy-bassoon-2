"""Final Linux CPU admission of the fresh B32/group8 GRPO control."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch
from hydra import compose, initialize_config_dir
from unittest.mock import patch

from motus_configuration import overrides, validate
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    plan = json.loads((ROOT/'MOTUSBEHAVIORGRPO01-CONTRACT.json').read_bytes())
    for name, expected in plan['source_files'].items():assert sha(ROOT.parents[1]/name)==expected, name
    assert sha(ROOT/'motus_grpo_entry.py') == plan['entry_sha256']
    assert sha(ROOT/'train_motus_behaviorgrpo01.sh') == plan['launcher_sha256']
    subprocess.run(['bash','-n',str(ROOT/'train_motus_behaviorgrpo01.sh')],check=True)
    import motus_grpo_entry
    assert callable(motus_grpo_entry.main)
    import rlinf.config as native
    class Placement:
        def get_world_size(self,role):return 8
    with initialize_config_dir(version_base='1.1',config_dir=str(ROOT.parents[1]/'configs/native_ppo')):
        cfg=compose(config_name='motus_phone_two',overrides=overrides('grpo',optimized_setup=True,
            behavior_matched=True,ppo_group1=False)+['native_pilot.stop_after_iteration=2'])
    summary=validate(cfg)
    with patch.object(native,'Cluster',lambda *a,**k:None),patch.object(native,'HybridComponentPlacement',lambda *a,**k:Placement()):
        native.validate_cfg(cfg)
    checked={}
    for name, status in [('cpumotusgroup1control01','passed'),
            ('motusbehaviorupdate01-readback01','passed_independent_actual_B32_update_checkpoint_readback')]:
        path=BASE/name/'RESULT.json';r=json.loads(path.read_bytes())
        assert r['status']==status and r['gpu_seconds']==0
        checked[name]=sha(path)
    readback=json.loads((BASE/'motusbehaviorupdate01-readback01/RESULT.json').read_bytes())
    assert readback['all8_original_behavior_likelihoods_exact'] and readback['all_owned_Adam_bytes_exact']
    # No checkpoint normalization change is necessary for this corrected path.
    assert all(r['original_guard_failure'] is None and r['normalized_empty_unowned_count']==0
               for r in readback['actors'])
    assert plan['closed_conservative_gpu_seconds']+plan['other_active_reserved_gpu_seconds']+plan['reserved_gpu_seconds']==plan['all_worst_gpu_seconds']<=360000
    assert shutil.disk_usage('/tmp').free>=plan['local_staging_cap_gib']*2**30
    probe=BASE/'behaviorgrpo01-admission-writeprobe'
    with probe.open('xb') as f:f.write(b'fresh-B32-group8-GRPO');f.flush();os.fsync(f.fileno())
    assert probe.read_bytes()==b'fresh-B32-group8-GRPO';probe.unlink()
    for name in ['motusbehaviorgrpo01','motusbehaviorgrpo01-signals']:
        assert not (BASE/name).exists(), name
    output=BASE/'cpumotusbehaviorgrpo01';output.mkdir()
    write(output/'RESULT.json',dict(status='passed',source_commit=os.environ['TASK_SOURCE'],gpu_seconds=0,
        models_or_simulators_constructed=False,entry_imported=True,launcher_bash_checked=True,
        full_native_schema_passed=True,configuration=summary,source_files=plan['source_files'],
        independent_input_result_hashes=checked,production_optimizer_guard_unchanged=True,
        capacity_gpu_count_and_simulator_reset_checked_in_actual_job=True))


if __name__=='__main__':main()
