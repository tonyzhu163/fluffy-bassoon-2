"""Linux CPU preflight for an unchanged-policy likelihood diagnostic."""
import ast
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from hydra import compose, initialize_config_dir
from unittest.mock import patch
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from motus_configuration import overrides, validate

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    plan = json.loads((ROOT/'MOTUSBEHAVIORUPDATE01-CONTRACT.json').read_bytes())
    for name,digest in plan['source_files'].items(): assert sha(ROOT.parents[1]/name) == digest
    assert sha(ROOT/'qualify_motus_behavior_update.py') == plan['entry_sha256']
    assert sha(ROOT/'qualify_motus_behavior_update.sh') == plan['launcher_sha256']
    tree = ast.parse((ROOT/'qualify_motus_behavior_update.py').read_bytes())
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('backward','optimizer_step','step') for n in ast.walk(tree))
    subprocess.run(['bash','-n',str(ROOT/'qualify_motus_behavior_update.sh')],check=True)
    with initialize_config_dir(version_base='1.1', config_dir=str(ROOT.parents[1]/'configs/native_ppo')):
        cfg = compose(config_name='motus_phone_two', overrides=overrides('ppo',optimized_setup=True,behavior_matched=True))
    validate(cfg)
    import rlinf.config as native
    class Placement:
        def get_world_size(self, role): return 8
    with patch.object(native,'Cluster',lambda *a,**k:None), patch.object(native,'HybridComponentPlacement',lambda *a,**k:Placement()):
        native.validate_cfg(cfg)
    import qualify_motus_behavior_update  # Import actual actor subclass; no worker constructed.
    assert qualify_motus_behavior_update.BehaviorUpdateActor is not None
    checkpoint = BASE/'motusoptimized03/checkpoints/global_step_1'
    manifest = json.loads((checkpoint/'MANIFEST.json').read_bytes())
    assert manifest['global_step'] == 1
    started = time.monotonic()
    for name,row in manifest['files'].items():
        path = checkpoint/name
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], name
    for rank in range(8):
        signature = json.loads((checkpoint/f'pilot/motus-actor-signature-{rank:02d}.json').read_bytes())
        assert signature['optimizer_steps'] == 15
    for rank in range(8):
        receipt = json.loads((BASE/f'motusresume01/state/training-rollouts/iteration-0002-policy-0001-rank-{rank:02d}.json').read_bytes())
        path = Path(receipt['path'])
        assert path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256']
        assert receipt['rank'] == rank and receipt['optimizer_steps_before'] == 15 and receipt['producing_policy_iteration'] == 1
    actual = BASE/'motus-learningaudit02/RESULT.json'
    expected = json.loads((ROOT/'MOTUSLEARNINGAUDIT02-COMPLETE.json').read_bytes())
    assert sha(actual) == expected['result_sha256']
    assert json.loads(actual.read_bytes())['status'] == 'completed_actual_batch_learning_signal_audit'
    output = BASE/'cpumotusbehaviorupdate01'; output.mkdir()
    assert not torch.cuda.is_initialized()
    write(output/'RESULT.json', dict(status='passed',source_commit=os.environ['TASK_SOURCE'],
        gpu_seconds=0, models_or_simulators_constructed=False, actual_entry_imported=True,
        native_schema_composed=True, checkpoint1_manifest_sha256=sha(checkpoint/'MANIFEST.json'),
        checkpoint_files=len(manifest['files']), checkpoint_bytes=sum(r['bytes'] for r in manifest['files'].values()),
        all_checkpoint1_files_cold_readback=True, all8_optimizer_counts_fifteen=True,
        actual_batch_audit_hash_verified=True, seconds=time.monotonic()-started,
        actual_gpu_B32_update_checkpoint_roundtrip_pending=True))


if __name__ == '__main__': main()
