"""RoboCasa365 contract for the existing matched terminal-reward PPO path.

Only scalar configuration/case routing lives here. Model, PPO and simulator
implementations remain the pinned native ones; this module imports none of them.
"""
from copy import deepcopy

RLINF='d80cf00e95e545f579e33432e7d7f14e86bb39d7'
ROBOCASA='3443cdf3dd20f9ae8ed41adba442059b506891d0'
ROBOSUITE='5ce6643f3092639d08f7b0f90ed1c6a84f50552c'
REPOSITORY='robocasa/robocasa365_checkpoints'
REVISION='c484448aba1a9b60a04c9b0ca117241518ea69f3'
PREFIX='pi05_pretrain_human300/multitask_learning/75000'
TASKS={'OpenDrawer':(750,'atomic_seen','atomic',320),
       'ScrubCuttingBoard':(1200,'composite_seen','composite',512),
       'CloseFridge':(900,'atomic_seen','atomic',None)}
DEV_STEPS=(2,4,6,8,10)
# Prospective seed banks. Native cases are constructor(seed) -> FIRST reset,
# not a LIBERO state index or a seed applied to an already-running simulator.
# The same native split/layout distribution is used for TRAIN and held-out ID.
BANK_SEEDS={'headroom':110100,'train':210100,'DEV':310100,'ID':410100,'OOD':510100}
ACTION_SEEDS={'headroom':610100,'train':710100,'DEV':810100,'ID':910100,'OOD':1010100}


def protocol(task):
    if task not in TASKS:raise ValueError('Only the two predeclared candidates and fixed OOD task are supported')
    return dict(benchmark='robocasa365',task=task,horizon=TASKS[task][0],
        camera_order=['robot0_agentview_left'],train_episodes=64,baseline_count=8)


def validate_protocol(value,training=False):
    if not isinstance(value,dict) or value!=protocol(value.get('task')):
        raise ValueError('Exact native task/horizon/video protocol required')
    if training and value['task']=='CloseFridge':raise ValueError('OOD cannot enter PPO/calibration')
    return value


def require_calibration_permission(calibration, expected_protocol, *, calibration_sha256,
                                   authorization=None):
    """Keep a failed measurement immutable when a human permits continuation."""
    import math
    validate_protocol(expected_protocol, training=True)
    if (calibration.get('completed') is not True or calibration.get('protocol')!=expected_protocol
            or calibration.get('count')!=8 or not .05<=calibration.get('initial_success_rate',-1)<=.70
            or calibration.get('AUROC') is None or not math.isfinite(calibration['AUROC'])):
        raise ValueError('Same completed native headroom/both-class calibration required')
    if calibration.get('go') is True:return 'original_calibration_gate_passed'
    if (not isinstance(authorization,dict) or authorization.get('allow_training_below_auc_gate') is not True
            or authorization.get('calibration_sha256')!=calibration_sha256
            or authorization.get('protocol')!=expected_protocol
            or authorization.get('measured_AUROC')!=calibration['AUROC']
            or not authorization.get('direct_human_instruction')):
        raise ValueError('Failed calibration requires an explicit human decision bound to these measured bytes')
    return 'human_authorized_below_original_AUROC_gate'


def case_bank(task,phase,*,iteration=0,round_index=0):
    """Fixed identities, disjoint by purpose; DEV is never final ID or OOD."""
    validate_protocol(protocol(task))
    if phase in ('baseline','smoke'):bank='headroom'
    elif phase in ('initial_DEV','DEV'):bank='DEV'
    elif phase in ('before_ID','ID'):bank='ID'
    elif phase in ('before_OOD','OOD'):bank='OOD'
    elif phase=='train':bank='train'
    else:raise ValueError('Unknown native episode bank')
    if (bank=='OOD')!=(task=='CloseFridge'):raise ValueError('Fixed OOD task is evaluation only')
    if bank=='train':
        if not 1<=iteration<=10 or not 0<=round_index<4:raise ValueError('Ten bounded four-round PPO batches')
        offset=((iteration-1)*4+round_index)*16;count=16
    else:offset=0;count=8
    return [dict(bank=bank,task=task,split='target',native_seed=BANK_SEEDS[bank]+offset+i,
        reset_ordinal=0,slot=i,counted=True,
        # Layout/style/objects come from the UNMODIFIED native target split
        # and are saved from actual ep_meta alongside initial RGB/state hashes.
        layout_source='native_target_constructor_seed_first_reset') for i in range(count)]


def evaluation_task(training_task,phase):
    return 'CloseFridge' if phase in ('before_OOD','OOD') else training_task


def private_action_seed(phase,rank):
    bank=('train' if phase=='train' else case_bank('CloseFridge' if 'OOD' in phase else 'OpenDrawer',phase)[0]['bank'])
    return ACTION_SEEDS[bank]+int(rank)


def validate_config(cfg):
    from omegaconf import OmegaConf
    task=str(cfg.flora_ppo.task);horizon,_,_,global_batch=TASKS.get(task,(0,None,None,None))
    if not global_batch:raise ValueError('PPO task must be frozen OpenDrawer or ScrubCuttingBoard')
    expected={
        'runner.use_training_pipeline':False,'runner.overlap_env_bootstrap':False,
        'runner.enable_decoupled_mode':False,'algorithm.adv_type':'gae',
        'algorithm.loss_type':'actor_critic','algorithm.update_epoch':4,
        'algorithm.loss_agg_func':'token-mean','algorithm.gamma':.99,'algorithm.gae_lambda':.95,
        'algorithm.normalize_advantages':True,'algorithm.clip_ratio_high':.2,
        'algorithm.clip_ratio_low':.2,'algorithm.clip_ratio_c':3.,'algorithm.value_clip':.2,
        'algorithm.huber_delta':10.,'algorithm.entropy_bonus':0,'algorithm.kl_beta':0.,
        'algorithm.reward_type':'chunk_level','algorithm.logprob_type':'chunk_level',
        'actor.global_batch_size':global_batch,'actor.seed':42,'actor.optim.lr':5e-6,
        'actor.optim.value_lr':1e-4,'actor.optim.adam_beta1':.9,'actor.optim.adam_beta2':.95,
        'actor.optim.adam_eps':1e-8,'actor.optim.weight_decay':.01,'actor.optim.clip_grad':1.,
        'actor.optim.critic_warmup_steps':0,'actor.model.pi05':True,
        'actor.model.num_action_chunks':50,'actor.model.action_dim':12,'actor.model.num_steps':10,
        'actor.model.add_value_head':True,'actor.model.openpi.config_name':'pi05_robocasa365_pretrain_human300',
        'actor.model.openpi.train_expert_only':True,'actor.model.openpi.value_after_vlm':True,
        'actor.model.openpi.model_action_dim':32,'actor.model.openpi.action_horizon':50,
        'actor.model.openpi.action_chunk':50,'actor.model.openpi.max_token_len':200,
        'actor.model.openpi.num_images_in_input':3,'actor.model.openpi.discrete_state_input':True,
        'actor.model.openpi.noise_method':'flow_sde','actor.model.openpi.noise_level':.5,
        'actor.model.openpi_data.use_quantile_norm':False,'actor.model.openpi_data.pad_before_normalization':True,
        'actor.model.openpi_data.extra_delta_transform':False,
        'reward.use_reward_model':False,'env.train.total_num_envs':16,
        'env.train.rollout_epoch':4,
        'env.eval.total_num_envs':1 if cfg.flora_ppo.phase=='baseline' else 8,
        'env.eval.rollout_epoch':8 if cfg.flora_ppo.phase=='baseline' else 1,
        'flora_ppo.rlinf_commit':RLINF,'flora_ppo.benchmark':'robocasa365',
    }
    for mode in ('train','eval'):
        for field,value in dict(env_type='robocasa365',auto_reset=False,ignore_terminations=True,
                episode_horizon_source='task_horizon',max_episode_steps=horizon,
                max_steps_per_rollout_epoch=horizon,split='target',group_size=1,
                task_filter=[task],task_soup=TASKS[task][1],task_mode=TASKS[task][2]).items():
            expected[f'env.{mode}.{field}']=value
        expected[f'env.{mode}.video_cfg.save_video']=False
        for field,value in dict(env_action_dim=12,gripper_index=6,base_mode_index=11,
                binarize_gripper_control=True,binarize_base_mode_control=True,disable_base_control=False).items():
            expected[f'env.{mode}.action_space.{field}']=value
    for key,value in expected.items():
        if OmegaConf.select(cfg,key)!=value:raise ValueError(f'Native matched PPO differs: {key}')
    world=int(cfg.flora_ppo.world_size);micro=int(cfg.actor.micro_batch_size)
    if world not in (1,2,4,8) or micro<1 or global_batch%(micro*world):
        raise ValueError('Execution microbatch must divide the task global batch on every rank')
    placement=cfg.cluster.component_placement
    if len(placement)!=1 or list(placement.values())!=[f'0-{world-1}']:raise ValueError('One common actor/env/rollout rank layout')
    if cfg.flora_ppo.arm not in ('S','F') or cfg.flora_ppo.phase not in ('baseline','train'):
        raise ValueError('Explicit matched phase/arm required')
    if cfg.flora_ppo.phase=='baseline' and world!=1:raise ValueError('Native case0 then remaining7 in one bounded GPU worker')
    if cfg.runner.only_eval!=(cfg.flora_ppo.phase!='train') or not 1<=int(cfg.runner.max_steps)<=10:
        raise ValueError('No optimizer for headroom; at most ten training iterations')
    samples=64*(horizon//50)
    if samples%global_batch or samples//global_batch!=3:raise ValueError('No dropped chunks/partial global batch')
    return dict(world=world,accumulation_steps=global_batch//(micro*world),train_episodes=64,
        protocol=protocol(task),eval_scored=8,eval_padding=0,dev_steps=list(DEV_STEPS),
        estimated_iteration_actions=64*(horizon+128)+8*(horizon+128),
        final_reserve_actions=8*(horizon+128)+8*(900+128),
        scored_schedule=case_bank(task,'DEV'))


def initializer_files(receipt,model_directory):
    """Bind original Orbax -> existing converter output; never another SFT."""
    from pathlib import Path
    root=Path(model_directory).resolve()
    if (receipt.get('completed') is not True or receipt.get('repository_id')!=REPOSITORY
            or receipt.get('revision')!=REVISION or receipt.get('prefix')!=PREFIX
            or Path(receipt['model_directory']).resolve()!=root
            or receipt.get('model_action_dim')!=32 or receipt.get('action_horizon')!=50
            or receipt.get('max_token_len')!=200 or receipt.get('conversion_mode')!='jax_to_openpi'
            or receipt.get('backbone_missing_keys')!=[] or receipt.get('unexpected_keys')!=[]):
        raise ValueError('Actual strict converted original Human30075000 initializer receipt required')
    files=deepcopy(receipt['files'])
    paths=[Path(r['path']).resolve() for r in files]
    if not files or len(set(paths))!=len(paths) or not all(p.is_relative_to(root) for p in paths):
        raise ValueError('Every converted model file must be uniquely bound inside its directory')
    if root/'model.safetensors' not in paths or root/'robocasa365/norm_stats.json' not in paths:
        raise ValueError('Converted weights and exact original native norm must both be bound')
    return files


def paired_stopping(own,peer):
    """Both arms use the same completed DEV panels; final/OOD are excluded."""
    for rows in (own,peer):
        if len(rows)<1:raise ValueError('Initial same-DEV baseline required')
        for row in rows:
            if (row.get('phase') not in ('initial_DEV','DEV') or row.get('episodes')!=8
                    or type(row.get('successes')) is not int or not 0<=row['successes']<=8):
                raise ValueError('Only complete native DEV evidence may stop PPO')
    if [r['iteration'] for r in own]!=[r['iteration'] for r in peer]:raise ValueError('Wait for matching peer DEV boundary')
    latest=own[-1]['iteration']
    if latest not in (0,*DEV_STEPS):raise ValueError('DEV only at the prospective cadence')
    ceiling=own[-1]['successes']==peer[-1]['successes']==8
    flat=latest>=6 and all(max(r['successes'] for r in rows[1:])<=rows[0]['successes'] for rows in (own,peer))
    return dict(stop=bool(latest>=10 or ceiling or flat),
        reason='fixed10' if latest>=10 else ('both_DEV8of8' if ceiling else ('both_no_DEV_improvement_after6' if flat else 'continue')),
        interpretation='Exploratory paired resource heuristic, not statistical convergence')


def arm_stopping(rows):
    """Same DEV rule per independent arm; never wait for the other job."""
    if not rows or rows[0].get('phase')!='initial_DEV' or rows[0].get('iteration')!=0:
        raise ValueError('Fixed initial DEV evidence required')
    for index,row in enumerate(rows):
        if (row.get('phase')!=('initial_DEV' if index==0 else 'DEV') or row.get('episodes')!=8
                or type(row.get('successes')) is not int or not 0<=row['successes']<=8
                or row.get('iteration')!=(0,*DEV_STEPS)[index]):
            raise ValueError('Only the ordered complete fixed DEV panels may stop an arm')
    latest=rows[-1]['iteration'];ceiling=rows[-1]['successes']==8
    flat=latest>=6 and max(row['successes'] for row in rows[1:])<=rows[0]['successes']
    return dict(stop=bool(latest>=10 or ceiling or flat),
        reason='fixed10' if latest>=10 else ('DEV8of8' if ceiling else ('no_DEV_improvement_after6' if flat else 'continue')),
        interpretation='Independent-arm exploratory stopping; compare shared prefixes and report time censoring')
