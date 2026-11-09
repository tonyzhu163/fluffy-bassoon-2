"""Run the native action-expert learner with RoboCasa recovery sidecars."""
import json
from pathlib import Path

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from fluffy_bassoon.robocasa.flow_ppo_runtime import (
    FlowActor, FlowEnvironment, FlowRollout, FlowRunner,
)
from fluffy_bassoon.robocasa.flow_ppo import flow_contract


@hydra.main(version_base='1.1', config_path=str(Path(__file__).resolve().parents[2] /
    'benchmarks/robocasa/configs'), config_name='robocasa365_pi05_flow_sde_10m')
def main(cfg):
    cfg = validate_cfg(cfg)
    assert not cfg.runner.only_eval
    contract = flow_contract(cfg)  # task, horizon, seeds, schedule and action ceiling
    assert cfg.runner.max_steps in (-1, contract['iterations'])
    assert not cfg.runner.use_training_pipeline and not cfg.runner.overlap_env_bootstrap
    assert contract['optimizer_steps_per_update'] >= 1
    assert cfg.actor.model.openpi.train_expert_only and cfg.actor.model.openpi.noise_method == 'flow_sde'
    assert contract['noise_level'] > 0 and cfg.algorithm.gamma == .99
    assert cfg.algorithm.kl_beta == 0, 'KL penalty is off for this campaign; KL is logged only'
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2), flush=True)
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    assert placement.get_world_size('actor') in (2, 8)
    assert placement.get_world_size('env') == placement.get_world_size('rollout') == cfg.native_pilot.world_size
    assert cfg.native_pilot.world_size in (2, 8)
    world = cfg.native_pilot.world_size
    assert contract['train_envs'] % (world * contract['group_size']) == 0 and contract['eval_cases'] % world == 0
    actor = FlowActor.create_group(cfg).launch(cluster, name=cfg.actor.group_name,
        placement_strategy=placement.get_strategy('actor'))
    rollout = FlowRollout.create_group(cfg).launch(cluster, name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy('rollout'))
    env = FlowEnvironment.create_group(cfg).launch(cluster, name=cfg.env.group_name,
        placement_strategy=placement.get_strategy('env'))
    runner = FlowRunner(cfg=cfg, actor=actor, rollout=rollout, env=env)
    runner.init_workers()
    runner.run()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
