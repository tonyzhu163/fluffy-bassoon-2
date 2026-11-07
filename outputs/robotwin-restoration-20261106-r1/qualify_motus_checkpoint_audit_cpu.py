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
    plan = json.loads((ROOT/'MOTUSCHECKPOINTAUDIT01-CONTRACT.json').read_bytes())
    for name,digest in plan['source_files'].items(): assert sha(ROOT.parents[1]/name) == digest
    assert sha(ROOT/'audit_motus_checkpoint.py') == plan['entry_sha256']
    assert sha(ROOT/'audit_motus_checkpoint.sh') == plan['launcher_sha256']
    tree = ast.parse((ROOT/'audit_motus_checkpoint.py').read_bytes())
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('backward','optimizer_step','step') for n in ast.walk(tree))
    subprocess.run(['bash','-n',str(ROOT/'audit_motus_checkpoint.sh')],check=True)
    with initialize_config_dir(version_base='1.1', config_dir=str(ROOT.parents[1]/'configs/native_ppo')):
        cfg = compose(config_name='motus_phone_two', overrides=overrides('ppo',optimized_setup=True))
    validate(cfg)
    import rlinf.config as native
    class Placement:
        def get_world_size(self, role): return 8
    with patch.object(native,'Cluster',lambda *a,**k:None), patch.object(native,'HybridComponentPlacement',lambda *a,**k:Placement()):
        native.validate_cfg(cfg)
    import audit_motus_checkpoint  # Import actual actor subclass; no worker constructed.
    assert audit_motus_checkpoint.CheckpointActor is not None
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
    actual = BASE/'motus-learningaudit02/RESULT.json'
    expected = json.loads((ROOT/'MOTUSLEARNINGAUDIT02-COMPLETE.json').read_bytes())
    assert sha(actual) == expected['result_sha256']
    assert json.loads(actual.read_bytes())['status'] == 'completed_actual_batch_learning_signal_audit'
    output = BASE/'cpumotuscheckpointaudit01'; output.mkdir()
    assert not torch.cuda.is_initialized()
    write(output/'RESULT.json', dict(status='passed',source_commit=os.environ['TASK_SOURCE'],
        gpu_seconds=0, models_or_simulators_constructed=False, actual_entry_imported=True,
        native_schema_composed=True, checkpoint1_manifest_sha256=sha(checkpoint/'MANIFEST.json'),
        checkpoint_files=len(manifest['files']), checkpoint_bytes=sum(r['bytes'] for r in manifest['files'].values()),
        all_checkpoint1_files_cold_readback=True, all8_optimizer_counts_fifteen=True,
        actual_batch_audit_hash_verified=True, seconds=time.monotonic()-started,
        actual_gpu_checkpoint_roundtrip_pending=True))


if __name__ == '__main__': main()
