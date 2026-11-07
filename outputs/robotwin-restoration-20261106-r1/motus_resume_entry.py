"""Explicit same-world resume of optimized03 checkpoint1 through iteration2."""
import json
import os
from pathlib import Path
import sys

import hydra
import ray
import torch

import motus_teacher_entry as teacher
from motus_optimized_entry import CapacityActor, CapacityRollout, CapacityEnvironment
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write

ROOT = Path(__file__).resolve().parent


@hydra.main(version_base='1.1', config_path=str(ROOT.parents[1]/'configs/native_ppo'), config_name='motus_phone_two')
def main(cfg):
    torch.multiprocessing.set_start_method('spawn', force=True)
    plan = json.loads((ROOT/'MOTUSRESUME01-CONTRACT.json').read_bytes())
    assert plan['status'] == 'admitted' and plan['budget_extension_approved']
    assert plan['closed_conservative_gpu_seconds'] + plan['reserved_gpu_seconds'] <= plan['approved_cap_gpu_seconds']
    assert cfg.runner.resume_dir == str(Path(os.environ['TASK_LOCAL'])/'global_step_1')
    assert sha(Path(cfg.runner.resume_dir)/'MANIFEST.json') == plan['resume_manifest_sha256']
    teacher.validate(cfg)
    assert int(cfg.native_pilot.world_size) == 8 and cfg.native_pilot.stop_after_iteration == 2
    assert cfg.native_pilot.comparison_algorithm == 'ppo'
    assert cfg.native_pilot.defer_scene_setup and cfg.native_pilot.recycle_scene_executor
    cfg = teacher.validate_cfg(cfg)
    cluster = teacher.Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = teacher.HybridComponentPlacement(cfg, cluster)
    groups = []
    try:
        for role, worker in (('actor', CapacityActor), ('rollout', CapacityRollout), ('env', CapacityEnvironment)):
            groups.append(worker.create_group(cfg).launch(cluster, name=cfg[role].group_name,
                placement_strategy=placement.get_strategy(role)))
        runner = teacher.MotusRunner(cfg=cfg, actor=groups[0], rollout=groups[1], env=groups[2], reward=None)
        runner.init_workers()  # Native DCP actor/Adam/scheduler recovery precedes all worker sidecars.
        assert runner.global_step == 1
        runner.run()  # Sidecar recovery checks exact owned actor state, RNG and case cursors before collection.
        output = Path(cfg.native_pilot.output)
        restored = json.loads((output/'operational-restore.json').read_bytes())
        assert restored['global_step'] == 1 and len(restored['workers']) == 24
        assert all(row['restored'] for row in restored['workers'])
        for rank in range(8):
            row = json.loads((output/f'recovery-signatures/actor-{rank:02d}.json').read_bytes())
            assert row['owned_state_exact'] and row['actual']['optimizer_steps'] == 15
    finally:
        primary_failure = sys.exc_info()[0] is not None
        errors = []
        for group in groups:
            try:
                if ray.is_initialized(): group._close()
            except Exception as error:
                errors.append(dict(type=type(error).__name__, error=str(error)))
        ray.shutdown()
        if errors:
            write(Path(os.environ['TASK_OUT'])/'cleanup-errors.json', dict(primary_failure_preserved=primary_failure, errors=errors))
            if not primary_failure: raise RuntimeError('Owned worker cleanup failed')


if __name__ == '__main__':
    main()
