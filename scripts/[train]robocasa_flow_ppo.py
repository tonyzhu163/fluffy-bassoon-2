"""Run the native action-expert learner with RoboCasa recovery sidecars."""
import json
from pathlib import Path

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo_runtime import (
    FlowActor, FlowEnvironment, FlowRollout, FlowRunner,
)


@hydra.main(version_base='1.1', config_path=str(Path(__file__).resolve().parents[1] /
    'configs/native_ppo'), config_name='robocasa365_opendrawer_pi05_sim_ppo')
def main(cfg):
    cfg = validate_cfg(cfg)
    assert not cfg.runner.only_eval
    assert cfg.runner.max_epochs == 18 and cfg.runner.max_steps in (-1, 18)
    assert not cfg.runner.use_training_pipeline and not cfg.runner.overlap_env_bootstrap
    assert cfg.env.train.total_num_envs == 32 and cfg.env.eval.total_num_envs == 64
    assert cfg.actor.global_batch_size == 256 and cfg.algorithm.update_epoch == 4
    assert cfg.actor.model.openpi.train_expert_only and cfg.actor.model.openpi.noise_method == 'flow_sde'
    assert cfg.actor.model.openpi.noise_level == .3 and cfg.algorithm.gamma == .99
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2), flush=True)
    cluster = Cluster(cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path)
    placement = HybridComponentPlacement(cfg, cluster)
    assert placement.get_world_size('actor') == placement.get_world_size('env') == 2
    assert placement.get_world_size('rollout') == 2
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
