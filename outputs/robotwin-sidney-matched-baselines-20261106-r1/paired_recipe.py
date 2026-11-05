"""Validate declared method differences around the common Sidney PPO recipe."""
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
from recipe import validate_sidney as validate_shared
from case_schedule import CaseSchedule


def validate_pair(cfg):
    algorithm=str(cfg.native_pilot.comparison_algorithm)
    assert algorithm in ('ppo','grpo')
    assert cfg.algorithm.adv_type==('gae' if algorithm=='ppo' else 'grpo')
    assert cfg.algorithm.loss_type==('actor_critic' if algorithm=='ppo' else 'actor')
    assert bool(cfg.actor.model.add_value_head)==(algorithm=='ppo')
    assert cfg.algorithm.group_size==cfg.env.train.group_size==8
    assert cfg.env.eval.group_size==1 and cfg.rollout.pipeline_stage_num==1
    assert cfg.actor.micro_batch_size==64,f'Unexpected microbatch: {cfg.actor.micro_batch_size}'
    assert int(cfg.native_pilot.world_size)==4,f'Unexpected world size: {cfg.native_pilot.world_size}'
    qualification=bool(cfg.native_pilot.qualification_only)
    assert cfg.native_pilot.maximum_actions==(120000 if qualification else 1000000)
    assert cfg.native_pilot.previous_maximum_actions is None
    assert cfg.native_pilot.memory_strategy=='native_batch_end' and not cfg.native_pilot.trim_after_scene_close
    schedule=CaseSchedule(cfg.native_pilot.case_manifest_path,cfg.native_pilot.case_manifest_sha256)
    bank=Path(cfg.env.train.seeds_path).read_bytes()
    assert hashlib.sha256(bank).hexdigest()==schedule.data['train_bank_sha256']
    assert {r['scene_seed'] for r in schedule.data['groups']}<=set(
        json.loads(bank)['place_phone_stand']['success_seeds'])
    # These are the explicitly admitted method/layout/accounting differences.
    # All remaining shared scientific and motor fields use the original guard.
    common=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
    common.algorithm.adv_type='gae';common.algorithm.loss_type='actor_critic'
    common.algorithm.group_size=1;common.env.train.group_size=1
    common.actor.model.add_value_head=True
    common.native_pilot.maximum_actions=1200000;common.native_pilot.previous_maximum_actions=1000000
    if qualification:
        assert algorithm=='grpo'
        assert cfg.runner.max_steps==cfg.runner.max_epochs==1
        assert cfg.env.train.rollout_epoch==cfg.algorithm.update_epoch==1
        common.runner.max_steps=common.runner.max_epochs=2
        common.env.train.rollout_epoch=4;common.algorithm.update_epoch=5
    result=validate_shared(common)
    return dict(result,comparison_algorithm=algorithm,group_size=8,
        value_head=algorithm=='ppo',maximum_actions=cfg.native_pilot.maximum_actions,
        qualification_only=qualification,
        maximum_aggregate_gpu_seconds=(23 if qualification else 46)*3600,case_manifest_sha256=schedule.sha256,
        unique_scene_groups_per_iteration=128,replicas_per_scene=8,
        source_case_selection='immutable common existing-TRAIN-bank subset',
        original_initializer_reproduction=False,paper_numeric_reproduction=False)
