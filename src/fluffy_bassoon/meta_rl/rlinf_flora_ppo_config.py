"""Task44 recipe and finite placement schedules; imports no model or simulator."""
from .rlinf_flora_ppo import INITIALIZER_PARAMS, INITIALIZER_RECEIPT, RLINF

NORM_SHA='b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84'


def evaluation_schedule(world, phase):
    """Padding is explicit, unscored and physically charged; never extra cases."""
    if world not in (1,2,4,8):raise ValueError('Explicit1/2/4/8 worker placement required')
    if phase=='smoke':
        if world!=1:raise ValueError('Single smoke episode uses one rollout/env worker')
        cases=[dict(nominal_state=0,draw=0,counted=True)]
    elif phase=='baseline':
        cases=[dict(nominal_state=i%40,draw=i//40,counted=True) for i in range(64)]
    elif phase=='DEV':
        cases=[dict(nominal_state=40+i%10,draw=i//10,counted=True) for i in range(50)]
    else:raise ValueError('Unknown evaluation phase')
    # Same56-slot DEV vector for both requested4/8-worker profiles; padding
    # never contributes a scored case and every action remains charged.
    padding=(-len(cases))%(8 if world in (4,8) else world)
    cases+= [dict(cases[i],counted=False) for i in range(padding)]
    return [dict(row,slot=i) for i,row in enumerate(cases)]


def validate_config(cfg):
    from omegaconf import OmegaConf
    expected={
        'runner.use_training_pipeline':False,'runner.overlap_env_bootstrap':False,
        'runner.enable_decoupled_mode':False,'algorithm.adv_type':'gae',
        'algorithm.loss_type':'actor_critic','algorithm.update_epoch':1,
        'algorithm.loss_agg_func':'token-mean','algorithm.gamma':.99,'algorithm.gae_lambda':.95,
        'algorithm.normalize_advantages':True,'algorithm.clip_ratio_high':.2,
        'algorithm.clip_ratio_low':.2,'algorithm.clip_ratio_c':3.,'algorithm.value_clip':.2,
        'algorithm.huber_delta':10.,'algorithm.entropy_bonus':0,'algorithm.kl_beta':0.,
        'algorithm.reward_type':'chunk_level','algorithm.logprob_type':'chunk_level',
        'actor.global_batch_size':2048,'actor.seed':42,'actor.optim.lr':5e-6,
        'actor.optim.value_lr':1e-4,'actor.optim.adam_beta1':.9,'actor.optim.adam_beta2':.95,
        'actor.optim.adam_eps':1e-8,'actor.optim.weight_decay':.01,'actor.optim.clip_grad':1.,
        'actor.optim.critic_warmup_steps':0,'actor.model.pi05':True,
        'actor.model.num_action_chunks':5,'actor.model.action_dim':7,'actor.model.num_steps':3,
        'actor.model.add_value_head':True,'actor.model.openpi.config_name':'pi05_libero',
        'actor.model.openpi.train_expert_only':True,'actor.model.openpi.value_after_vlm':True,
        'actor.model.openpi.model_action_dim':32,'actor.model.openpi.max_token_len':200,
        'actor.model.openpi.num_images_in_input':2,'actor.model.openpi.noise_method':'flow_sde',
        'actor.model.openpi.noise_level':.5,'reward.use_reward_model':False,
        'env.train.total_num_envs':32,'env.train.rollout_epoch':4,
        'flora_ppo.rlinf_commit':RLINF,'flora_ppo.initializer_receipt':INITIALIZER_RECEIPT,
        'flora_ppo.initializer_params':INITIALIZER_PARAMS,'flora_ppo.normalization_sha256':NORM_SHA,
    }
    for mode in ('train','eval'):
        for field,value in dict(env_type='libero',task_suite_name='libero_90',task_id_filter=[44],
                max_episode_steps=400,max_steps_per_rollout_epoch=400,num_steps_wait=15,
                auto_reset=False,ignore_terminations=True,skip_intermediate_renders=False).items():
            expected[f'env.{mode}.{field}']=value
        expected[f'env.{mode}.video_cfg.save_video']=False # Our per-env raw recorder replaces annotated grids.
        expected[f'env.{mode}.init_params.control_freq']=20
    for key,value in expected.items():
        if OmegaConf.select(cfg,key)!=value:raise ValueError(f'Matched published PPO contract differs: {key}')
    world=int(cfg.flora_ppo.world_size);micro=int(cfg.actor.micro_batch_size)
    if world not in (1,2,4,8) or micro<1 or 2048%(micro*world):raise ValueError('Preserve effective global2048 batch')
    placement=cfg.cluster.component_placement
    if len(placement)!=1 or list(placement.values())!=[f'0-{world-1}']:
        raise ValueError('Explicit common actor/env/rollout placement required')
    if cfg.flora_ppo.arm not in ('S','F') or cfg.flora_ppo.phase not in ('smoke','baseline','train'):
        raise ValueError('Concrete comparison phase and arm required')
    schedule=evaluation_schedule(world,'DEV' if cfg.flora_ppo.phase=='train' else cfg.flora_ppo.phase)
    if cfg.env.eval.total_num_envs!=len(schedule) or cfg.env.eval.rollout_epoch!=1:
        raise ValueError('Evaluation vector must bind exact scored schedule plus explicit padding')
    if cfg.runner.only_eval!=(cfg.flora_ppo.phase!='train'):raise ValueError('No optimizer initialization for smoke/baseline')
    if not 1<=int(cfg.runner.max_steps)<=13:raise ValueError('No automatic extension beyond13')
    return dict(world=world,accumulation_steps=2048//(micro*world),
        train_episodes=128,eval_scored=sum(r['counted'] for r in schedule),
        eval_padding=sum(not r['counted'] for r in schedule),
        estimated_iteration_actions=(128+len(schedule))*415,scored_schedule=schedule)
