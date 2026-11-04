"""Run one declared comparison arm using the frozen native PPO recipe."""
import os
from pathlib import Path

import hydra
import torch.multiprocessing as mp

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import flow_contract
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo_runtime import FlowEnvironment, FlowRollout, FlowRunner
from comparison_runtime import ComparisonActor


@hydra.main(version_base='1.1', config_path=str(Path(os.environ['TASK_RUNTIME']) /
    'configs/native_ppo'), config_name='robocasa365_pi05_flow_sde_10m')
def main(cfg):
    cfg = validate_cfg(cfg)
    contract = flow_contract(cfg)
    assert contract['iterations'] == 3 and contract['optimizer_steps_per_update'] == 24
    assert contract['task'] == 'PickPlaceDrawerToCounter'
    assert contract['eval_cases'] == 64 and contract['episodes_per_iteration'] == 128
    assert cfg.actor.global_batch_size == 256 and cfg.algorithm.update_epoch == 4
    assert cfg.actor.micro_batch_size == 16 and cfg.actor.optim.lr == 5e-6
    assert cfg.actor.optim.value_lr == 1e-4 and cfg.algorithm.value_clip == .2
    assert not cfg.actor.model.openpi.value_after_vlm and cfg.actor.model.openpi.detach_critic_input
    assert cfg.runner.resume_dir and not cfg.runner.only_eval
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    assert placement.get_world_size('actor') == 2
    assert placement.get_world_size('env') == placement.get_world_size('rollout') == 8
    groups = []
    for cls, name, component in [(ComparisonActor, cfg.actor.group_name, 'actor'),
            (FlowRollout, cfg.rollout.group_name, 'rollout'),
            (FlowEnvironment, cfg.env.group_name, 'env')]:
        groups.append(cls.create_group(cfg).launch(cluster, name=name,
            placement_strategy=placement.get_strategy(component)))
    actor, rollout, env = groups
    runner = FlowRunner(cfg=cfg, actor=actor, rollout=rollout, env=env)
    runner.init_workers()
    runner.run()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
