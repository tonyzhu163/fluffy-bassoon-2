"""Two matched native GRPO iterations with the qualified scene lifecycle.

The original Motus workers own all model, sampling, loss and checkpoint work.
This entry adds the full-allocation reserve check missing from actor-only
qualification. A failed iteration retains the previous complete boundary.
"""
import json
import os
from pathlib import Path
import time

import hydra
import ray
import torch

import motus_teacher_entry as teacher
from fluffy_bassoon.meta_rl.rlinf_native_ppo import write


ROOT = Path(__file__).resolve().parent


def require_reserve(worker):
    root = Path(os.environ['TASK_OUT'])
    path = Path(os.environ.get('TASK_RESOURCE_LIVE',str(root/'resources.json')))
    report = json.loads(path.read_bytes())
    fresh = 0 <= time.time()-report['at_unix'] <= 30
    passed = (fresh and len(report['gpus']) == int(worker.cfg.native_pilot.world_size) and report['samples'] > 0
              and report['sampled_reserve_passed']
              and report['host_peak_bytes'] < .9*report['host_limit_bytes'])
    if not passed:
        write(root/f'capacity-rejected-{worker.pilot_role}-{worker._rank:02d}.json',
              dict(status='rejected',fresh_observation=fresh,resource_report=report,
                   latest_complete_checkpoint_must_be_preserved=True))
        raise RuntimeError('Combined GPU/host reserve failed; preserve the prior complete boundary')


class CapacityActor(teacher.MotusActor):
    def train_micro_batch(self, *args, **kwargs):
        require_reserve(self)
        return super().train_micro_batch(*args, **kwargs)


class CapacityRollout(teacher.MotusRollout):
    def _predict_rollout_actions(self, *args, **kwargs):
        require_reserve(self)
        result = super()._predict_rollout_actions(*args, **kwargs)
        require_reserve(self)
        return result


class CapacityEnvironment(teacher.MotusEnvironment):
    def _pilot_deadline(self):
        require_reserve(self)
        return super()._pilot_deadline()


@hydra.main(version_base='1.1',config_path=str(ROOT.parents[1]/'configs/native_ppo'),
            config_name='motus_phone_two')
def main(cfg):
    torch.multiprocessing.set_start_method('spawn',force=True)
    teacher.validate(cfg)
    assert cfg.native_pilot.comparison_algorithm == 'grpo'
    assert not cfg.actor.model.add_value_head and cfg.algorithm.adv_type == 'grpo'
    assert cfg.native_pilot.stop_after_iteration == 2
    assert cfg.native_pilot.defer_scene_setup and cfg.native_pilot.recycle_scene_executor
    assert not cfg.runner.resume_dir, 'This is a fresh SFT lineage; no implicit worker migration'
    cfg = teacher.validate_cfg(cfg)
    cluster = teacher.Cluster(cluster_cfg=cfg.cluster,distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = teacher.HybridComponentPlacement(cfg,cluster)
    groups = []
    try:
        for role,worker in (('actor',CapacityActor),('rollout',CapacityRollout),('env',CapacityEnvironment)):
            group = worker.create_group(cfg).launch(cluster,name=cfg[role].group_name,
                placement_strategy=placement.get_strategy(role))
            groups.append(group)
        runner = teacher.MotusRunner(cfg=cfg,actor=groups[0],rollout=groups[1],env=groups[2],reward=None)
        runner.init_workers()
        runner.run()
    finally:
        primary_failure = __import__('sys').exc_info()[0] is not None
        cleanup_errors = []
        for group in groups:
            try:
                if ray.is_initialized(): group._close()
            except Exception as error:
                cleanup_errors.append(dict(error_type=type(error).__name__,error=str(error)))
        try:
            ray.shutdown()
        finally:
            if cleanup_errors:
                write(Path(os.environ['TASK_OUT'])/'cleanup-errors.json',
                    dict(primary_failure_preserved=primary_failure,errors=cleanup_errors))
        if cleanup_errors and not primary_failure:
            raise RuntimeError('Worker cleanup failed; see cleanup-errors.json')


if __name__ == '__main__':
    main()
