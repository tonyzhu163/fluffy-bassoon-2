"""RoboCasa365 contract for the existing matched terminal-reward PPO path.

Only scalar configuration/case routing lives here. Model, PPO and simulator
implementations remain the pinned native ones; this module imports none of them.
"""
from copy import deepcopy
from .robocasa_video import REWARD_CAMERAS

RLINF='d80cf00e95e545f579e33432e7d7f14e86bb39d7'
ROBOCASA='3443cdf3dd20f9ae8ed41adba442059b506891d0'
ROBOSUITE='5ce6643f3092639d08f7b0f90ed1c6a84f50552c'
REPOSITORY='robocasa/robocasa365_checkpoints'
REVISION='c484448aba1a9b60a04c9b0ca117241518ea69f3'
PREFIX='pi05_pretrain_human300/multitask_learning/75000'
TASKS={'OpenDrawer':(750,'atomic_seen','atomic',320),
       'ScrubCuttingBoard':(1200,'composite_seen','composite',512),
       'CloseFridge':(900,'atomic_seen','atomic',None),
       'PickPlaceCounterToCabinet':(750,'atomic_seen','atomic',None),
       'PickPlaceDrawerToCounter':(750,'atomic_seen','atomic',None),
       'TurnOffStove':(750,'atomic_seen','atomic',None),
       'CoffeeSetupMug':(600,'atomic_seen','atomic',None),
       'OpenCabinet':(1050,'atomic_seen','atomic',None),
       'TurnOnMicrowave':(450,'atomic_seen','atomic',None),
       'TurnOnSinkFaucet':(600,'atomic_seen','atomic',None),
       'OpenStandMixerHead':(450,'atomic_seen','atomic',None)}
PPO_TASKS=('OpenDrawer','PickPlaceCounterToCabinet','PickPlaceDrawerToCounter',
           'TurnOffStove','CoffeeSetupMug')
OOD_TASKS=('CloseFridge','OpenCabinet','TurnOnMicrowave','TurnOnSinkFaucet','OpenStandMixerHead')
DEV_STEPS=(2,4,6,8,10)
# Prospective seed banks. Native cases are constructor(seed) -> FIRST reset,
# not a LIBERO state index or a seed applied to an already-running simulator.
# The same native split/layout distribution is used for TRAIN and held-out ID.
BANK_SEEDS={'headroom':110100,'train':210100,'DEV':310100,'ID':410100,'OOD':510100}
ACTION_SEEDS={'headroom':610100,'train':710100,'DEV':810100,'ID':910100,'OOD':1010100}


def protocol(task, *, camera_order=None, baseline_count=8):
    if task not in TASKS:raise ValueError('Only prospectively bound native tasks are supported')
    if baseline_count not in (8,50):raise ValueError('Bind the original pilot8 or new per-task50 contract')
    cameras=list(REWARD_CAMERAS) if camera_order is None else list(camera_order)
    if cameras not in (list(REWARD_CAMERAS), ['robot0_agentview_left']):
        raise ValueError('Only current three-view or archived left-only reward contracts are supported')
    return dict(benchmark='robocasa365',task=task,horizon=TASKS[task][0],
        camera_order=cameras,train_episodes=64,baseline_count=baseline_count)


def validate_protocol(value,training=False):
    if not isinstance(value,dict) or value!=protocol(value.get('task'),camera_order=value.get('camera_order'),
                                                  baseline_count=value.get('baseline_count')):
        raise ValueError('Exact native task/horizon/video protocol required')
    if training and value['task'] in OOD_TASKS:raise ValueError('OOD cannot enter PPO/calibration')
    return value


def require_calibration_permission(calibration, expected_protocol, *, calibration_sha256,
                                   authorization=None):
    """Keep a failed measurement immutable when a human permits continuation."""
    import math
    validate_protocol(expected_protocol, training=True)
    rate=calibration.get('initial_success_rate',-1)
    auc=calibration.get('AUROC')
    if (calibration.get('completed') is not True or calibration.get('protocol')!=expected_protocol
            or calibration.get('count')!=expected_protocol['baseline_count']
            or isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 0<rate<1
            or isinstance(auc,bool) or not isinstance(auc,(int,float)) or not math.isfinite(auc) or not 0<=auc<=1):
        raise ValueError('Same completed native headroom/both-class calibration required')
    native_pass=.05<=rate<=.70
    auc_pass=auc>=.7
    if calibration.get('go') is not bool(native_pass and auc_pass):
        raise ValueError('Keep the original measured calibration gate unchanged')
    if native_pass and auc_pass:return 'original_calibration_gate_passed'
    if (not isinstance(authorization,dict)
            or authorization.get('calibration_sha256')!=calibration_sha256
            or authorization.get('protocol')!=expected_protocol
            or authorization.get('measured_AUROC')!=auc
            or not authorization.get('direct_human_instruction')):
        raise ValueError('Failed calibration requires an explicit human decision bound to these measured bytes')
    if not auc_pass and authorization.get('allow_training_below_auc_gate') is not True:
        raise ValueError('An AUROC exception requires its own explicit human decision')
    if not native_pass:
        if (authorization.get('allow_training_outside_native_headroom_gate') is not True
                or authorization.get('measured_initial_success_rate')!=rate):
            raise ValueError('A native headroom exception requires its own measured-rate human decision')
        return 'human_authorized_native_headroom_exception' if auc_pass else 'human_authorized_native_and_AUROC_exceptions'
    return 'human_authorized_below_original_AUROC_gate'


def case_bank(task,phase,*,iteration=0,round_index=0,count=None,offset=0):
    """Fixed identities, disjoint by purpose; DEV is never final ID or OOD."""
    validate_protocol(protocol(task))
    if phase in ('baseline','smoke'):bank='headroom'
    elif phase in ('initial_DEV','DEV'):bank='DEV'
    elif phase in ('before_ID','ID'):bank='ID'
    elif phase in ('before_OOD','OOD'):bank='OOD'
    elif phase=='train':bank='train'
    else:raise ValueError('Unknown native episode bank')
    if (bank=='OOD')!=(task in OOD_TASKS):raise ValueError('Fixed OOD tasks are evaluation only')
    if bank=='train':
        if count is not None or offset:raise ValueError('Training bank size/offset is fixed by its prospective round')
        if not 1<=iteration<=10 or not 0<=round_index<4:raise ValueError('Ten bounded four-round PPO batches')
        offset=((iteration-1)*4+round_index)*16;count=16
    else:
        count=8 if count is None else count
        if type(count) is not int or type(offset) is not int or count<1 or offset<0 or count+offset>50:
            raise ValueError('Fixed held-out/qualification banks contain at most50 native cases')
    task_offset=(PPO_TASKS.index(task) if task in PPO_TASKS else OOD_TASKS.index(task)
                 if task in OOD_TASKS else 0)*10000
    return [dict(bank=bank,task=task,split='target',native_seed=BANK_SEEDS[bank]+task_offset+offset+i,
        reset_ordinal=0,slot=offset+i if bank!='train' else i,counted=True,
        # Layout/style/objects come from the UNMODIFIED native target split
        # and are saved from actual ep_meta alongside initial RGB/state hashes.
        layout_source='native_target_constructor_seed_first_reset') for i in range(count)]


def evaluation_task(training_task,phase):
    return 'CloseFridge' if phase in ('before_OOD','OOD') else training_task


def joint_case_bank(phase,*,iteration=0):
    """Interleave complete episodes so each ten-episode rank sees all five tasks."""
    tasks=OOD_TASKS if phase in ('before_OOD','OOD') else PPO_TASKS
    if phase=='train':
        banks=[case_bank(task,phase,iteration=iteration,round_index=0) for task in tasks]
    else:
        count=8 if phase in ('initial_DEV','DEV') else 50
        if phase not in ('initial_DEV','DEV','before_ID','ID','before_OOD','OOD'):
            raise ValueError('Joint calibration is measured in the separate fixed per-task banks')
        banks=[case_bank(task,phase,count=count) for task in tasks]
    return [bank[slot] for slot in range(len(banks[0])) for bank in banks]


def joint_episode_protocols(episodes,cases):
    """Bind all80 native episode identities/horizons to their producing cases."""
    from collections import Counter
    if (len(episodes)!=80 or len(cases)!=80 or len({e['id'] for e in episodes})!=80
            or Counter(c['task'] for c in cases)!=Counter({task:16 for task in PPO_TASKS})):
        raise ValueError('One balanced eighty-episode on-policy bank required')
    result={}
    for index,(episode,case) in enumerate(zip(episodes,cases)):
        if (episode['rank']!=index//10 or episode['round']!=index%10 or episode['local_env']!=0
                or episode['horizon']!=TASKS[case['task']][0]):
            raise ValueError('Producing native rank/case/full-horizon order differs')
        result[episode['id']]=protocol(case['task'],baseline_count=50)
    return result


def joint_arm_stopping(rows):
    """DEV-only per-task heuristic; final ID/OOD never enter the decision."""
    if not rows or len(rows)>1+len(DEV_STEPS) or rows[0].get('phase')!='initial_DEV':
        raise ValueError('Fixed initial per-task DEV evidence required')
    for index,row in enumerate(rows):
        panels=row.get('per_task',[])
        if (row.get('phase')!=('initial_DEV' if index==0 else 'DEV')
                or row.get('iteration')!=(0,*DEV_STEPS)[index]
                or row.get('episodes')!=40 or [p.get('task') for p in panels]!=list(PPO_TASKS)
                or any(p.get('episodes')!=8 or type(p.get('successes')) is not int
                       or not 0<=p['successes']<=8 for p in panels)
                or row.get('successes')!=sum(p['successes'] for p in panels)):
            raise ValueError('Only ordered complete fixed DEV panels for all five tasks may stop PPO')
    latest=rows[-1]['iteration']
    ceiling=all(p['successes']==8 for p in rows[-1]['per_task'])
    flat=latest>=6 and all(max(row['per_task'][i]['successes'] for row in rows[1:])
                          <=rows[0]['per_task'][i]['successes'] for i in range(5))
    return dict(stop=bool(latest>=10 or ceiling or flat),
        reason='fixed10' if latest>=10 else 'all_tasks_DEV8of8' if ceiling else
               'no_task_DEV_improvement_after6' if flat else 'continue',
        interpretation='Independent-arm descriptive per-task resource heuristic, not statistical convergence')


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
def rollout_migration_contract(current, migration, *, checkpoint_sha256,
                               kind='eight_gpu_rollout_only'):
    """Validate logical JSON equality, then retain typed checkpoint fields.

    JSON turns bound tuples into lists. Reconstructing from the current typed
    contract after exact logical comparison keeps strict Torch recovery intact.
    Only the source pin may change; every scientific scalar must still match.
    """
    from copy import deepcopy
    from .rlinf_flora_ppo import digest
    original = migration["original_contract"]
    if (kind not in ('eight_gpu_rollout_only', 'eight_gpu_evaluation_only')
            or migration["kind"] != kind
            or migration["checkpoint_sha256"] != checkpoint_sha256
            or migration["to_source_commit"] != current["source_commit"]
            or original["source_commit"] != migration["from_source_commit"]
            or digest({key: value for key, value in original.items() if key != "source_commit"})
               != digest({key: value for key, value in current.items() if key != "source_commit"})):
        raise ValueError("Only an explicit complete-state rollout execution migration is allowed")
    result = deepcopy(current)
    result["source_commit"] = original["source_commit"]
    return result
