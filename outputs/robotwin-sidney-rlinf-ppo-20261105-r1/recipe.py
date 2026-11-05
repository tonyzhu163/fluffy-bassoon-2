"""Static metadata contract for the Sidney native PPO pilot."""
from omegaconf import OmegaConf
from fluffy_bassoon.rl.sidney import CONFIG_NAME,REVISION
from fluffy_bassoon.meta_rl.rlinf_native_ppo import validate_recipe


def validate_sidney(cfg):
    assert cfg.actor.model.openpi.config_name==CONFIG_NAME
    assert cfg.native_pilot.initializer_revision==REVISION
    assert cfg.actor.model.precision is None
    assert not cfg.actor.model.openpi_data.adapt_to_pi
    assert not cfg.actor.model.openpi_data.extra_delta_transform
    assert cfg.runner.val_check_interval==-1
    for mode in ('train','eval'):
        assert cfg.env[mode].task_config.task_name=='place_phone_stand'
        assert cfg.env[mode].task_config.camera.head_camera_type=='Large_D435'
        assert cfg.env[mode].task_config.camera.wrist_camera_type=='Large_D435'
    micro=int(cfg.actor.micro_batch_size);world=int(cfg.native_pilot.world_size)
    assert micro in (8,16,32,64) and 2048%(micro*world)==0
    copy=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
    for mode in ('train','eval'):copy.env[mode].task_config.task_name='adjust_bottle'
    copy.actor.model.openpi.config_name='pi05_aloha_robotwin'
    copy.actor.micro_batch_size=32
    result=validate_recipe(copy)
    return dict(result,source_task='place_phone_stand',initializer='Sidney broad multi-task SFT',
        initializer_revision=REVISION,original_initializer_reproduction=False,
        micro_batch_size=micro,accumulation_steps=2048//(micro*world),
        nominal_before_after_actions=0,nominal_all_in_actions=819200,
        maximum_aggregate_gpu_seconds=50400,
        precision='native mixed BF16; entire vision and norms FP32',
        baseline_reference='published Sidney SFT; no rebenchmark panel',
        migration='H50 retained; K10 native reference to VLRLA K5/.3 Flow-SDE PPO')

