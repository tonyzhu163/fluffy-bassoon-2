"""Linux core likelihood tests and B32 PPO/GRPO schema qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from motus_configuration import overrides, validate

ROOT = Path(__file__).resolve().parent


def main():
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    import rlinf.config as native
    class Placement:
        def get_world_size(self, role): return 8
    rows = []
    for method in ('ppo','grpo'):
        with initialize_config_dir(version_base='1.1',config_dir=str(ROOT.parents[1]/'configs/native_ppo')):
            cfg = compose(config_name='motus_phone_two',
                          overrides=overrides(method,optimized_setup=True,behavior_matched=True))
        row = validate(cfg)
        with patch.object(native,'Cluster',lambda *a,**k:None), patch.object(native,'HybridComponentPlacement',lambda *a,**k:Placement()):
            native.validate_cfg(cfg)
        assert row['micro_batch'] == 32 and row['accumulation'] == 8 and row['global_batch'] == 2048
        assert cfg.native_pilot.require_behavior_likelihood_match and cfg.native_pilot.common_dev_noise_seed == 31000
        rows.append(row)
    import motus_teacher_entry
    assert callable(motus_teacher_entry.MotusRollout.pilot_common_dev_noise)
    checked = subprocess.run([sys.executable,'-m','pytest','-q',
        str(ROOT.parents[1]/'tests/test_rlinf_openpi_replay.py')],capture_output=True,text=True)
    print(checked.stdout, flush=True)
    if checked.returncode:
        print(checked.stderr,flush=True)
        raise RuntimeError('Core likelihood tests failed')
    assert not torch.cuda.is_initialized()
    output = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/cpumotusbehavior01')
    output.mkdir()
    write(output/'RESULT.json',dict(status='passed',source_commit=os.environ['TASK_SOURCE'],
        gpu_seconds=0,models_or_simulators_constructed=False,core_test_output=checked.stdout,
        configurations=rows,core_likelihood_sha256=sha(ROOT.parents[1]/'src/fluffy_bassoon/meta_rl/rlinf_openpi_replay.py'),
        actual_accumulated_gpu_update_and_common_noise_readback_pending=True))


if __name__ == '__main__': main()
