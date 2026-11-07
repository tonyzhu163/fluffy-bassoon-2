"""Qualify source-group scheduling and the corrected native schemas on Linux."""
import json
import os
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import torch

from case_schedule import CaseSchedule
from motus_configuration import overrides, validate
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    import rlinf.config as native
    class Placement:
        def get_world_size(self, role): return 8
    rows = []
    for method, group1 in [('ppo',True),('ppo',False),('grpo',False)]:
        with initialize_config_dir(version_base='1.1',config_dir=str(ROOT.parents[1]/'configs/native_ppo')):
            cfg = compose(config_name='motus_phone_two',overrides=overrides(method,
                optimized_setup=True,behavior_matched=True,ppo_group1=group1))
        summary = validate(cfg)
        with patch.object(native,'Cluster',lambda *a,**k:None), patch.object(native,'HybridComponentPlacement',lambda *a,**k:Placement()):
            native.validate_cfg(cfg)
        group = 1 if group1 else 8
        assert summary['scene_group_size'] == group
        schedule = CaseSchedule(cfg.native_pilot.case_manifest_path,cfg.native_pilot.case_manifest_sha256)
        for world in (4,8):
            for ordinal in range(8):
                actual = [r for rank in range(world) for r in schedule.rows(ordinal,rank,world)]
                expected = [r for r in schedule.data['groups'] if
                    r['policy_iteration']==ordinal//4 and r['round_index']==ordinal%4]
                assert sorted(r['scene_seed'] for r in actual) == sorted(r['scene_seed'] for r in expected)
                assert len(actual)*group == 256
                for rank in range(world):
                    local = schedule.rows(ordinal,rank,world)
                    assert [r['local_group'] for r in local] == list(range(256//world//group))
                    assert len([r['scene_seed'] for r in local for _ in range(group)]) == 256//world
        summary['all4and8rank_case_partitions_exact'] = True
        rows.append(summary)
    bank = json.loads((ROOT/'MOTUS-PHONE-TRAIN-SEED-BANK.json').read_bytes())
    assert len(bank['success_seeds']) == len(set(bank['success_seeds'])) == 1000
    output = BASE/'cpumotusgroup1control01';output.mkdir()
    write(output/'RESULT.json',dict(status='passed',source_commit=os.environ['TASK_SOURCE'],gpu_seconds=0,
        models_or_simulators_constructed=False,configurations=rows,
        case_schedule_sha256=sha(ROOT.parent/'robotwin-sidney-matched-baselines-20261106-r1/case_schedule.py'),
        group1_manifest_sha256=sha(ROOT/'MOTUS-PPO-GROUP1-CASES.json'),
        actual_simulator_group1_reset_readback_pending=True))


if __name__=='__main__':main()
