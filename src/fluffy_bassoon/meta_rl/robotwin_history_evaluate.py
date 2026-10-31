"""Frozen native RoboTwin evaluation adapter; caller owns qualified runtime/ledger.

This module never constructs a simulator, allocates hardware, trains a policy,
or creates a campaign. The execution owner supplies the reviewed environment
and its existing CampaignBudget. The same adapter supports injected CPU fakes.
"""
import hashlib
from dataclasses import asdict
import inspect
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .robotwin_history import FrozenNativeHistoryController, load_policy
from .robotwin_history_corpus import _interface, load_teacher, parameter_digest, source_recovery
from .vla_history import file_sha256


TASKS = ('adjust_bottle', 'move_stapler_pad')
MODELS = ('ad-full', 'ad-static', 'dpt-full', 'dpt-static')
CONDITIONS = ('native', 'native-zero', 'sac5-mean', *MODELS, 'ad-full-reset', 'dpt-full-reset')
TRAIN_PIN = '1aeea4f9fcc307bb147def92374bb81a3e2cb50e'
TEACHER_SHA = 'f2fb39571cc3d0e809dfdc454f7f6df7f48e585fc972fd4c2fb8c83d36e63a42'
SOURCE_TEACHER_SHA = '59588f07be18d4a1339af0c7536c62ef735a713f3c933d7e372fb3f4f4fa7e90'
SOURCE_COORDINATOR_SHA = '20c512080f3012452343a0c7d8aff67ee03966add4497d61256854e52d5d611f'
SOURCE_TASK_LIMITS = dict(handover_mic=600, pick_diverse_bottles=400,
    place_a2b_left=400, place_object_stand=400, place_phone_stand=400, scan_object=500)
SOURCE_CONDITIONS = ('native', 'sac100-mean', 'sac100-sample')
SOURCE_FORMAT = 'robotwin_source_teacher_evaluation_v1'
AD500_FORMAT = 'robotwin_source100_ad500_readout_v1'
AD500_MODELS = ('ad-full', 'ad-static')
AD500_CONDITIONS = ('native', 'sac100-frozen', 'ad-full-k0', 'ad-static-k0',
    'ad-full-k1', 'ad-full-k1-reset')


def source_case_bank():
    return {task:[3_000_000_000+i*10_000+e*997 for e in range(4)]
            for i,task in enumerate(SOURCE_TASK_LIMITS)}


def trial_profile(plan):
    """Explicit profiles share one accounting/runtime/trial implementation."""
    validate_plan(plan)
    source = plan['format'] == SOURCE_FORMAT
    if plan['format'] == AD500_FORMAT:
        return dict(tasks=TASKS,conditions=AD500_CONDITIONS,episodes=8,phase='target',
            max_actions=51200,max_gpu_seconds=5400,teacher_condition=('sac100-frozen',),
            teacher_sha=SOURCE_TEACHER_SHA,source=False,ad500=True)
    return dict(tasks=tuple(SOURCE_TASK_LIMITS) if source else TASKS,
        conditions=SOURCE_CONDITIONS if source else CONDITIONS,
        episodes=4 if source else 8, phase='teacher_qualification' if source else 'target',
        max_actions=32400 if source else 57600, max_gpu_seconds=3600 if source else 5400,
        teacher_condition=('sac100-mean','sac100-sample') if source else ('sac5-mean',),
        teacher_sha=SOURCE_TEACHER_SHA if source else TEACHER_SHA, source=source,ad500=False)


def support_case_bank():
    return {task:[32_261_030+i*997 for i in range(8)] for task in TASKS}


def action_seed_bank(*, support=False):
    base=80_261_030 if support else 70_261_030
    return {task:[base+t*100_000+i*997 for i in range(8)] for t,task in enumerate(TASKS)}


def ad500_model_key(condition):
    return 'ad-static' if condition == 'ad-static-k0' else 'ad-full'


def episode_schedule(plan, condition, task):
    """Eight isolated queries, each optionally preceded by its own support."""
    if plan['format'] != AD500_FORMAT:
        return [dict(seed=seed,clock=i) for i,seed in enumerate(plan['cases'][task])]
    rows=[]
    for query,seed in enumerate(plan['cases'][task]):
        if '-k1' in condition:
            rows.append(dict(seed=plan['support_cases'][task][query],clock=0,role='support',
                query_index=query,action_seed=plan['support_action_seeds'][task][query],new_trial=True))
        rows.append(dict(seed=seed,clock=int('-k1' in condition),role='query',query_index=query,
            action_seed=plan['query_action_seeds'][task][query],new_trial='-k1' not in condition))
    return rows


def unit_episode_count(plan, condition):
    return 16 if plan['format']==AD500_FORMAT and '-k1' in condition else plan['episodes_per_trial']


def validate_ad500_plan(plan):
    if (plan.get('cases')!=case_bank() or plan.get('support_cases')!=support_case_bank()
            or plan.get('query_action_seeds')!=action_seed_bank()
            or plan.get('support_action_seeds')!=action_seed_bank(support=True)
            or plan.get('conditions')!=list(AD500_CONDITIONS) or plan.get('episodes_per_trial')!=8
            or plan.get('episode_limit')!=400 or plan.get('max_actions')!=51200
            or plan.get('max_episodes')!=128 or plan.get('max_gpu_seconds')!=5400
            or plan.get('student_action')!='deterministic_mean'
            or plan.get('sample_convention')!='independent_declared_per_query_action_seed'
            or plan.get('paid_sample_reuse') is not None
            or plan.get('teacher',{}).get('sha256')!=SOURCE_TEACHER_SHA
            or set(plan.get('models',{}))!=set(AD500_MODELS)):
        raise ValueError('Require the exact source100 AD500 isolated K0/K1 readout; no implicit reuse')
    def hex_digest(value, n):
        return isinstance(value,str) and len(value)==n and all(c in '0123456789abcdef' for c in value)
    if (not hex_digest(plan.get('fit_commit'),40)
            or any(not hex_digest(plan.get(key),64) for key in ('train_index_sha256','validation_index_sha256'))
            or any(not isinstance(model,dict) or set(model)!={'path','sha256'}
                or not Path(model.get('path','')).is_absolute()
                or Path(model['path']).name!='candidate-000500.pt'
                or not hex_digest(model.get('sha256'),64) for model in plan['models'].values())):
        raise ValueError('Bind actual source-only candidate-000500.pt artifacts, fit commit and final100 indexes first')
    interface=plan['interface']
    if (interface.get('observation_dim')!=2062 or interface.get('noise_shape')!=[32,32]
            or interface.get('action_limit')!=1.5
            or any(interface['task_episode_limits'].get(t)!=400 for t in TASKS)
            or not plan.get('frozen_motor_sha256') or not plan.get('native_backend_sha256')
            or not plan.get('runtime_contract')):
        raise ValueError('AD500 native motor/features/horizons differ')
    return plan


def validate_source_teacher_plan(plan):
    if (plan.get('cases') != source_case_bank() or plan.get('conditions') != list(SOURCE_CONDITIONS)
            or plan.get('episodes_per_trial') != 4 or plan.get('episode_limit') != 600
            or plan.get('max_actions') != 32400 or plan.get('max_episodes') != 72
            or plan.get('max_gpu_seconds') != 3600 or plan.get('paid_sample_reuse') is not None
            or plan.get('teacher',{}).get('sha256') != SOURCE_TEACHER_SHA
            or plan.get('source100_coordinator',{}).get('sha256') != SOURCE_COORDINATOR_SHA
            or plan.get('sample_convention') != 'restore_fixed_source_private_torch_rng_per_task_K4'
            or plan.get('native_noise_seed_base') != 27261030 or plan.get('models')):
        raise ValueError('Require the exact source100 native/mean/sample K4 plan; no implicit paid-sample reuse')
    interface = plan['interface']
    if (interface.get('observation_dim') != 2062 or interface.get('noise_shape') != [32,32]
            or interface.get('action_limit') != 1.5
            or any(interface['task_episode_limits'].get(t) != n for t,n in SOURCE_TASK_LIMITS.items())
            or not plan.get('frozen_motor_sha256') or not plan.get('native_backend_sha256')
            or not plan.get('runtime_contract')):
        raise ValueError('Source teacher native motor/features/horizons differ')
    return plan


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def case_bank(seed=20261030):
    if type(seed) is not int or seed != 20261030:
        raise ValueError('Use the already declared development-target case bank')
    return {task:[seed+2_000_000+i*997 for i in range(8)] for task in TASKS}


def validate_plan(plan):
    """The fixed development screen or exact source100 teacher profile."""
    if plan.get('format') == SOURCE_FORMAT:
        return validate_source_teacher_plan(plan)
    if plan.get('format') == AD500_FORMAT:
        return validate_ad500_plan(plan)
    if (plan.get('format') != 'robotwin_native_history_evaluation_v1'
            or plan.get('cases') != case_bank() or plan.get('conditions') != list(CONDITIONS)
            or plan.get('episodes_per_trial') != 8 or plan.get('episode_limit') != 400
            or plan.get('max_actions') != 57600 or plan.get('max_episodes') != 144
            or plan.get('student_action') != 'deterministic_mean'
            or plan.get('teacher', {}).get('sha256') != TEACHER_SHA
            or set(plan.get('models', {})) != set(MODELS)):
        raise ValueError('Require the exact fixed-endpoint paired development panel')
    for name, model in plan['models'].items():
        if (Path(model['path']).name != 'candidate-000100.pt'
                or len(model.get('sha256', '')) != 64):
            raise ValueError(f'{name} requires its actual checksum-pinned 100-update endpoint')
    interface = plan['interface']
    if (interface.get('observation_dim') != 2062 or interface.get('noise_shape') != [32,32]
            or interface.get('action_limit') != 1.5
            or any(interface['task_episode_limits'].get(task) != 400 for task in TASKS)
            or not plan.get('frozen_motor_sha256') or not plan.get('native_backend_sha256')
            or not plan.get('runtime_contract')):
        raise ValueError('Native motor/feature/noise/horizon interface differs')
    return plan


def assert_runtime(env, plan):
    """Require the qualified native backend, including raw capture and limits."""
    expected = plan['interface']
    runtime = env.backend.contract
    if (file_sha256(inspect.getfile(type(env))) != plan['native_backend_sha256']
            or {key:value for key,value in runtime.items() if key!='actual_render_pci'} != plan['runtime_contract']
            or getattr(env.backend,'final_observation_only',False) is not True):
        raise ValueError('Require the pinned qualified native backend and final-camera runtime')
    actual = dict(runtime={key:runtime.get(key) for key in expected['runtime']},
        model_path=runtime.get('model_path'), task_episode_limits=runtime.get('task_episode_limits'),
        episode_limit=env.default_episode_limit, observation_dim=env.observation_dim,
        noise_shape=list(env.noise_shape), action_limit=1.5)
    limits = SOURCE_TASK_LIMITS if plan['format'] == SOURCE_FORMAT else dict.fromkeys(TASKS,400)
    if (actual != expected or env.native_horizon != 32 or getattr(env,'reset_policy_action_bound',0) != 0
            or getattr(env,'corpus_writer',None) is None
            or any(env.episode_limit_for_task(task) != limit for task,limit in limits.items())):
        raise ValueError('Require the exact source-bound native runtime and raw corpus recording')
    if env.frozen_fingerprint() != plan['frozen_motor_sha256']:
        raise ValueError('Frozen motor fingerprint differs')


class StudentAgent:
    def __init__(self, policy, manifest, interface, *, erase_each_episode, deterministic):
        self.controller = FrozenNativeHistoryController(policy,manifest,interface=interface,
            erase_each_episode=erase_each_episode)
        self.policy = policy
        self.deterministic = deterministic

    def reset_trial(self, seed): self.controller.reset_trial(seed=seed)
    def reseed_action_rng(self, seed): self.controller.reseed_action_rng(seed)
    def begin_episode(self, index, limit): self.controller.begin_episode(episode_index=index,episode_limit=limit)
    def act(self, features): return self.controller.act(features,deterministic=self.deterministic)
    def observe(self, noise, reward, done, steps):
        self.controller.observe(noise,reward,done,executed_steps=steps)


class ReferenceAgent:
    """Unclipped native Gaussian, zero initial noise, or physical SAC mean."""
    def __init__(self, policy=None, *, zero=False):
        self.policy = policy
        self.condition = 'native-zero' if zero else 'native' if policy is None else 'sac5-mean'
    def reset_trial(self, seed): self.rng = np.random.default_rng(seed)
    def reseed_action_rng(self, seed): self.rng = np.random.default_rng(seed)
    def begin_episode(self, index, limit): pass
    @torch.no_grad()
    def act(self, features):
        if self.policy is None:
            if self.condition=='native-zero': return np.zeros((32,32),np.float32)
            return self.rng.normal(size=(32,32)).astype(np.float32)
        weight = next(self.policy.parameters())
        value,_ = self.policy(torch.as_tensor(features[None],device=weight.device),
            generator=None,deterministic=True)
        return value[0].cpu().numpy().reshape(32,32).copy()
    def observe(self, noise, reward, done, steps): pass


class SourceTeacherAgent(ReferenceAgent):
    """The source driver's per-task frozen SAC clone, without replay/optimizers."""
    def __init__(self, policy, interaction_state, *, stochastic):
        super().__init__(policy)
        if (not isinstance(interaction_state,torch.Tensor) or interaction_state.dtype != torch.uint8
                or interaction_state.ndim != 1 or not interaction_state.numel()):
            raise ValueError('Require the fixed source private interaction-generator state')
        self.initial_state = interaction_state.cpu().clone()
        self.stochastic = stochastic
        self.condition = 'sac100-sample' if stochastic else 'sac100-mean'
        self.generator = None

    def reset_trial(self, seed):
        # e825 source validation creates a fresh clone from trained[method]
        # PER task, preserving its saved private RNG across that task's K4.
        # Case/task seeds never reseed this generator; optimizer RNG is unused.
        if self.stochastic:
            self.generator = torch.Generator(device=next(self.policy.parameters()).device)
            self.generator.set_state(self.initial_state.clone())

    @torch.no_grad()
    def act(self, features):
        if self.stochastic and self.generator is None:
            raise RuntimeError('Restore the fixed private source RNG before sampling')
        weight = next(self.policy.parameters())
        value,_ = self.policy(torch.as_tensor(features[None],device=weight.device),
            generator=self.generator,deterministic=not self.stochastic)
        return value[0].cpu().numpy().astype(np.float32,copy=True).reshape(32,32)

    def sampling_receipt(self):
        final = self.generator.get_state().cpu() if self.generator is not None else self.initial_state
        return dict(convention='restore_fixed_source_private_torch_rng_per_task_K4',
            deterministic=not self.stochastic,
            initial_private_rng_sha256=parameter_digest({'interaction_rng':self.initial_state}),
            final_private_rng_sha256=parameter_digest({'interaction_rng':final}),
            source_checkpoint_sha256=SOURCE_TEACHER_SHA,optimizer_rng_used=False)


class QuerySACAgent(ReferenceAgent):
    """Frozen source100 actor with fresh declared query streams, not source K4 RNG."""
    condition = 'sac100-frozen'

    def __init__(self, policy):
        super().__init__(policy)
        self.condition='sac100-frozen'; self.generator=None

    def reset_trial(self, seed): self.reseed_action_rng(seed)

    def reseed_action_rng(self, seed):
        self.generator=torch.Generator(device=next(self.policy.parameters()).device).manual_seed(seed)

    @torch.no_grad()
    def act(self, features):
        if self.generator is None: raise RuntimeError('Declare the query action seed before sampling')
        weight=next(self.policy.parameters())
        value,_=self.policy(torch.as_tensor(features[None],device=weight.device),
            generator=self.generator,deterministic=False)
        return value[0].cpu().numpy().astype(np.float32,copy=True).reshape(32,32)


def load_source_teacher(plan, *, device):
    source = source_recovery(plan['teacher']['path'],SOURCE_TEACHER_SHA,'sac')
    if (_interface(source['contract']) != plan['interface'] or source['source_iteration'] != 100
            or source['trials'] != 600 or source['state']['source_steps'] != 1003790
            or source['learner']['updates'] != 638500):
        raise ValueError('Require the fixed source100 stage actor and original source counters')
    policy,_ = load_teacher(source,device=device)
    return source,policy


def load_ad500_agent(plan, condition, *, device):
    if condition not in AD500_CONDITIONS: raise ValueError('Unknown AD500 condition')
    if condition=='native': return ReferenceAgent()
    if condition=='sac100-frozen':
        _,policy=load_source_teacher(plan,device=device)
        return QuerySACAgent(policy)
    key=ad500_model_key(condition); spec=plan['models'][key]; mode=key.split('-')[1]
    policy,manifest=load_policy(spec['path'],spec['sha256'],device=device)
    contract=manifest['contract']
    expected_config=dict(observation_dim=2062,noise_dim=1024,width=128,layers=2,heads=4,
        context_length=300,initial_std=.5,gate_bias=2.,action_limit=1.5,train_episodes=4)
    expected_budget=dict(updates=500,batch_size=8,max_window=128,eval_every=25,validation_batches=4,
        milestones=[25,100,500],max_seconds_per_invocation=600.)
    if (manifest.get('update')!=500 or manifest.get('mode')!=mode or contract.get('mode')!=mode
            or manifest.get('interface')!=plan['interface'] or contract.get('interface')!=plan['interface']
            or contract.get('source_commit')!=plan['fit_commit'] or contract.get('objective')!='ad'
            or contract.get('validation_objective')!='behavior_cloning' or contract.get('seed')!=0
            or contract.get('config')!=expected_config or asdict(policy.config)!=expected_config
            or contract.get('budget')!=expected_budget
            or contract.get('optimizer')!=dict(name='AdamW',learning_rate=3e-4,weight_decay=0.,max_grad_norm=1.)
            or contract.get('normalization')!='2062-feature mean/std fitted on source training only; fixed for validation/deployment'
            or any(contract.get(k)!=plan[k] for k in ('train_index_sha256','validation_index_sha256'))):
        raise ValueError('AD500 endpoint differs from the fixed source-only recipe')
    cases={}
    for partition,objective,count,actions in (('train','ad',600,1003790),('validation','behavior_cloning',6,10180)):
        metadata=contract[partition+'_metadata']; counts=metadata.get('counts',{})
        if (metadata.get('completed') is not True or metadata.get('partition')!=partition
                or metadata.get('selected_objective')!=objective or set(metadata.get('datasets',{}))!={objective}
                or metadata.get('interface')!=plan['interface'] or metadata.get('source_method')!='sac'
                or metadata.get('source_recovery_sha256')!=SOURCE_TEACHER_SHA
                or metadata.get('teacher_recovery_sha256') is not None or metadata.get('teacher_method') is not None
                or metadata.get('index_sha256')!=plan[partition+'_index_sha256']
                or metadata.get('validation_iteration')!=(100 if partition=='validation' else None)
                or counts.get('complete_K4_trials')!=count or counts.get('episodes')!=4*count
                or counts.get('selected_actions')!=actions or counts.get('committed_source_actions')!=1003790
                or len(metadata.get('source_trials',[]))!=count or len(metadata.get('case_ids',[]))!=count):
            raise ValueError('AD500 requires complete teacher-free SAC100 AD/BC metadata')
        cases[partition]=set(metadata['case_ids'])
    if (cases['train'] & cases['validation']
            or manifest.get('unique_training_decisions')!=contract['train_metadata']['counts']['decisions']):
        raise ValueError('AD500 source/validation split or training coverage differs')
    agent=StudentAgent(policy,manifest,plan['interface'],erase_each_episode=condition.endswith('-reset'),
        deterministic=True)
    agent.condition=condition; agent.checkpoint_sha256=spec['sha256']
    return agent


def load_agent(plan, condition, *, device='cuda'):
    validate_plan(plan)
    if plan['format'] == AD500_FORMAT:
        return load_ad500_agent(plan,condition,device=device)
    if plan['format'] == SOURCE_FORMAT:
        if condition not in SOURCE_CONDITIONS: raise ValueError('Unknown source teacher condition')
        if condition == 'native': return ReferenceAgent()
        source,policy = load_source_teacher(plan,device=device)
        return SourceTeacherAgent(policy,source['learner']['torch_rng'],stochastic=condition=='sac100-sample')
    if condition not in CONDITIONS: raise ValueError('Unknown predeclared condition')
    if condition == 'native': return ReferenceAgent()
    if condition == 'native-zero': return ReferenceAgent(zero=True)
    if condition == 'sac5-mean':
        source = source_recovery(plan['teacher']['path'],TEACHER_SHA,'sac')
        if _interface(source['contract']) != plan['interface']:
            raise ValueError('SAC5 teacher interface differs')
        policy,_ = load_teacher(source,device=device)
        return ReferenceAgent(policy)
    key = condition.removesuffix('-reset')
    spec = plan['models'][key]
    policy,manifest = load_policy(spec['path'],spec['sha256'],device=device)
    contract = manifest['contract']
    objective,mode = key.split('-')
    expected_config = dict(observation_dim=2062,noise_dim=1024,width=128,layers=2,heads=4,
        context_length=300,initial_std=.5,gate_bias=2.,action_limit=1.5,train_episodes=4)
    if (manifest.get('update') != 100 or manifest.get('mode') != mode
            or manifest.get('interface') != plan['interface'] or contract.get('source_commit') != TRAIN_PIN
            or contract.get('objective') != objective or contract.get('seed') != 0
            or contract.get('config') != expected_config or contract['budget']['updates'] != 100
            or contract.get('train_index_sha256') != plan['train_index_sha256']
            or contract.get('validation_index_sha256') != plan['validation_index_sha256']
            or contract['train_metadata']['teacher_recovery_sha256'] != TEACHER_SHA):
        raise ValueError('Portable endpoint differs from the fixed source-only first-fit contract')
    agent = StudentAgent(policy,manifest,plan['interface'],erase_each_episode=condition.endswith('-reset'),
        deterministic=plan['student_action']=='deterministic_mean')
    agent.condition = condition
    agent.checkpoint_sha256 = spec['sha256']
    return agent


def admit_plan(plan):
    """Validate every endpoint/teacher on allocated CPU before motor creation."""
    validate_plan(plan)
    if plan['format'] == AD500_FORMAT:
        checked=[]; normalization=None
        for condition in ('ad-full-k0','ad-static-k0','sac100-frozen'):
            agent=load_agent(plan,condition,device='cpu')
            checked.append(dict(condition=condition,parameter_sha256=parameter_digest(agent.policy.state_dict())))
            if condition!='sac100-frozen':
                actual=parameter_digest({k:getattr(agent.policy,k) for k in ('observation_mean','observation_std')})
                if normalization is not None and actual!=normalization:
                    raise ValueError('AD full/static source normalization differs')
                normalization=actual
            del agent
        return dict(plan_sha256=canonical_sha(plan),checked=checked,model_endpoint=500,
            source_teacher_sha256=SOURCE_TEACHER_SHA,normalization_sha256=normalization,
            source_or_target_SGD_updates=0,additional_robot_actions=0,paid_sample_reuse=False,
            status='CPU artifact admission only; limited two-task readout, not the full adaptation protocol')
    if plan['format'] == SOURCE_FORMAT:
        source,policy = load_source_teacher(plan,device='cpu')
        closed = source_recovery(plan['source100_coordinator']['path'],SOURCE_COORDINATOR_SHA,'sac')
        if (parameter_digest(source['learner']['actor']) != parameter_digest(closed['learner']['actor'])
                or any(source['state'][key] != closed['state'][key] for key in ('source_steps','task_steps'))
                or source['learner']['updates'] != closed['learner']['updates']
                or source['trials'] != closed['trials']
                or not torch.equal(source['learner']['torch_rng'],closed['learner']['torch_rng'])):
            raise ValueError('Stage100 teacher differs from the closed source actor/counters')
        agent = SourceTeacherAgent(policy,source['learner']['torch_rng'],stochastic=True)
        return dict(plan_sha256=canonical_sha(plan),source_sha256=SOURCE_COORDINATOR_SHA,
            teacher_sha256=SOURCE_TEACHER_SHA,actor_parameter_sha256=parameter_digest(policy.state_dict()),
            sample_initial_private_rng_sha256=agent.sampling_receipt()['initial_private_rng_sha256'],
            source_iteration=100,source_K4=600,source_actions=1003790,additional_robot_actions=0,
            CUDA_generator_restored=False,teacher_competence_measured=False,
            status='CPU actor/source admission only; resource/native runtime/teacher outcome remain separate')
    checked=[]
    for condition in (*MODELS,'sac5-mean'):
        agent=load_agent(plan,condition,device='cpu')
        checked.append(dict(condition=condition,parameter_sha256=parameter_digest(agent.policy.state_dict())))
        del agent
    return dict(plan_sha256=canonical_sha(plan),checked=checked,
        model_endpoint=100,student_action='deterministic_mean',additional_robot_actions=0,
        status='model admission only; GPU/renderer/native runtime qualification remains separate')


def _atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent,os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def renderer_records(env):
    root=os.environ.get('FB2_ROBOTWIN_RENDER_RECORDS')
    if root is None:
        output=getattr(env.backend,'output_dir',None)
        if output is None: raise ValueError('Require the qualified renderer record directory')
        root=Path(output)/'renderer-bindings'
    return {str(path):file_sha256(path) for path in sorted(Path(root).glob('renderer-*.json'))}


def case_identity(case):
    return {key:case[key] for key in ('task','requested_seed','actual_seed','observation_sha256','features_sha256')}


def verify_ad500_result(result, plan, condition, task, *, native, support, admission):
    """Recheck the complete isolated-query topology on fresh or resumed units."""
    schedule=episode_schedule(plan,condition,task); rows=result['episodes']; context=result['context']
    if (admission is None or admission.get('plan_sha256')!=canonical_sha(plan)
            or admission.get('model_endpoint')!=500 or result.get('source_iteration')!=100
            or result.get('source_or_target_SGD_updates')!=0 or result.get('paid_sample_reuse') is not False
            or result.get('query_feedback_carried_to_later_trials') is not False
            or result.get('offline_weights_unchanged') is not True or context.get('phase')!='target'
            or context.get('erase_each_episode') is not condition.endswith('-reset')
            or len(rows)!=len(schedule)):
        raise ValueError('AD500 result lacks its exact frozen isolated-query admission')
    query=[]; supports=[]
    for i,(row,entry,case) in enumerate(zip(rows,schedule,result['reset_cases'],strict=True)):
        if (row.get('role')!=entry['role'] or row.get('episode_clock')!=entry['clock']
                or row.get('query_index')!=entry['query_index'] or row.get('trial_index')!=entry['query_index']
                or row.get('action_seed')!=entry['action_seed'] or row.get('requested_seed')!=entry['seed']
                or row.get('task')!=task or row.get('episode')!=i+1 or case_identity(row)!=case_identity(case)
                or len(row.get('proposal_sha256',''))!=64):
            raise ValueError('AD500 support/query order, clocks, action stream or cases differ')
        bank=query if entry['role']=='query' else supports
        bank.append(dict(case,episode=entry['query_index']+1))
    if (query!=result.get('query_reset_cases') or supports!=result.get('support_reset_cases')
            or result.get('query_episodes')!=[i+1 for i,row in enumerate(rows) if row['role']=='query']
            or result.get('support_episodes')!=[i+1 for i,row in enumerate(rows) if row['role']=='support']
            or result.get('query_successes')!=[row['success'] for row in rows if row['role']=='query']
            or len({r['actual_seed'] for r in query})!=8
            or (supports and len({r['actual_seed'] for r in supports})!=8)
            or {r['actual_seed'] for r in query} & {r['actual_seed'] for r in supports}):
        raise ValueError('AD500 accepted support/query banks are incomplete or overlap')
    if condition!='native' and (native is None or query!=native['reset_cases']):
        raise ValueError('AD500 queries do not match the completed native cases')
    if condition=='ad-full-k1-reset' and (support is None or supports!=support['support_reset_cases']
            or [r for r in rows if r['role']=='support']!=[r for r in support['episodes'] if r['role']=='support']):
        raise ValueError('AD500 same-weight supports differ from the full K1 condition')
    if condition!='native':
        key='sac100-frozen' if condition=='sac100-frozen' else ad500_model_key(condition)+'-k0'
        checked={r['condition']:r['parameter_sha256'] for r in admission['checked']}
        checkpoint=SOURCE_TEACHER_SHA if condition=='sac100-frozen' else plan['models'][ad500_model_key(condition)]['sha256']
        if (context.get('policy_checkpoint_sha256')!=checkpoint or context.get('policy_sha256')!=checked.get(key)
                or result.get('model_endpoint')!=(None if condition=='sac100-frozen' else 500)
                or context.get('action_convention')!=('per_query_rng_stochastic_physical1.5*tanh(sample)'
                    if condition=='sac100-frozen' else 'deterministic_mean')):
            raise ValueError('AD500 actor identity or frozen action convention differs')


def run_trial(env, budget, agent, plan, *, condition, task, output, run_id, references=None,
              support_references=None):
    """Execute one fresh fixed-profile unit with pre-dispatch reservations.

    Native runs first to establish accepted reset cases. Every other condition
    must match its exact accepted seed, raw observation and feature hashes.
    Failure leaves real costs/raw intents intact and never retries a case.
    Caller owns GPU/CUDA qualification, process lifetime, wall time and ledger.
    """
    spec = trial_profile(plan)
    count = unit_episode_count(plan,condition)
    schedule=episode_schedule(plan,condition,task)
    limit = SOURCE_TASK_LIMITS[task] if spec['source'] and task in SOURCE_TASK_LIMITS else 400
    allowance = count*limit
    if (condition not in spec['conditions'] or task not in spec['tasks'] or (condition != 'native' and references is None)
            or getattr(agent,'condition',None) != condition):
        raise ValueError('Run native first and use its paired accepted reset cases')
    if references is not None and (references.get('completed') is not True or references.get('task') != task
            or references.get('context',{}).get('method') != 'native'
            or references['context'].get('evaluation_plan_sha256') != canonical_sha(plan)
            or len(references.get('reset_cases',[])) != (8 if spec['ad500'] else count)):
        label='query-bank' if spec['ad500'] else 'K4' if spec['source'] else 'K8'
        raise ValueError('Need the completed native '+label+' receipt bound to this exact plan')
    if spec['ad500'] and condition=='ad-full-k1-reset' and (support_references is None
            or support_references.get('completed') is not True or support_references.get('task')!=task
            or support_references.get('context',{}).get('method')!='ad-full-k1'
            or support_references['context'].get('evaluation_plan_sha256')!=canonical_sha(plan)
            or len(support_references.get('support_reset_cases',[]))!=8):
        raise ValueError('Require the completed same-weight K1 support receipt before the reset lesion')
    header=getattr(budget,'header',{})
    if (header.get('total_actions') != plan['max_actions'] or header.get('source_caps') != {}
            or header.get('phase_caps') != {spec['phase']:plan['max_actions']}
            or header.get('contract',{}).get('evaluation_plan_sha256') != canonical_sha(plan)):
        raise ValueError('Require the separately admitted evaluation ledger bound to this exact plan')
    assert_runtime(env,plan)
    scenes_before=renderer_records(env)
    if budget.remaining(phase=spec['phase'],method=condition) < allowance:
        raise RuntimeError('Reserve a complete trial allowance before beginning this trial')
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    trial_id = f'{run_id}:{condition}:{task}:'+('isolated-query-bank' if spec['ad500'] else f'K{count}')
    policy = agent.policy
    before = None if policy is None else parameter_digest(policy.state_dict())
    identity = 'native_zero_initial_noise' if condition=='native-zero' else 'native_standard_gaussian' if condition == 'native' else (
        spec['teacher_sha'] if condition in spec['teacher_condition'] else
        plan['models'][ad500_model_key(condition) if spec['ad500'] else condition.removesuffix('-reset')]['sha256'])
    if condition not in ('native','native-zero',*spec['teacher_condition']) and agent.checkpoint_sha256 != identity:
        raise ValueError('Steering agent checkpoint identity differs from the planned condition')
    checkpoint_path = (None if condition in ('native','native-zero') else plan['teacher']['path']
        if condition in spec['teacher_condition'] else
        plan['models'][ad500_model_key(condition) if spec['ad500'] else condition.removesuffix('-reset')]['path'])
    context = dict(method=condition,phase=spec['phase'],trial_id=trial_id,policy_version=identity,
        policy_checkpoint=checkpoint_path,policy_checkpoint_sha256=None if condition in ('native','native-zero') else identity,
        policy_sha256=before,
        evaluation_plan_sha256=canonical_sha(plan),offline_weights_frozen=True,
        action_convention=('native_standard_gaussian_unclipped' if condition=='native' else
            'native_zero_initial_noise' if condition=='native-zero' else
            'physical1.5*tanh(mean)' if condition in ('sac5-mean','sac100-mean') else
            'source_private_rng_stochastic_physical1.5*tanh(sample)' if condition=='sac100-sample' else
            'per_query_rng_stochastic_physical1.5*tanh(sample)' if condition=='sac100-frozen' else plan['student_action']),
        erase_each_episode=condition.endswith('-reset'))
    env.set_recording_context(context)
    # Model/control identities do not perturb their shared action random stream.
    policy_seed = 20261030+7_000_000+spec['tasks'].index(task)*100_000
    if not spec['ad500']: agent.reset_trial(policy_seed)
    episodes=[]; reset_cases=[]; decisions=0; started=time.monotonic()
    _atomic(output/'intent.json',dict(context=context,seeds=[row['seed'] for row in schedule],max_actions=allowance))
    support_cases=[]; query_cases=[]; history_decisions=0
    for index,entry in enumerate(schedule):
        seed=entry['seed']
        if spec['ad500']:
            if entry['new_trial']:
                agent.reset_trial(entry['action_seed']); history_decisions=0
            # This reseeds ONLY actions. In particular it must not clear K1 support memory.
            agent.reseed_action_rng(entry['action_seed'])
            env.set_recording_context(dict(trial_id=f'{trial_id}:query:{entry["query_index"]}',
                adaptation_role=entry['role'],query_index=entry['query_index'],
                action_seed=entry['action_seed'],episode_clock=entry['clock']))
        features=env.reset(task,seed)
        reset=dict(env.last_reset_info)
        required=('actual_seed','observation_sha256','features_sha256')
        if (env.episode_limit!=limit or reset.get('task')!=task or reset.get('requested_seed')!=seed
                or reset.get('reset_policy_actions')!=0 or type(reset.get('actual_seed')) is not int
                or any(not reset.get(key) for key in required[1:])):
            raise ValueError('Missing actual zero-action reset/case provenance')
        case=dict(task=task,episode=index+1,requested_seed=seed,
            **{key:reset[key] for key in required})
        _atomic(output/f'reset-{index+1:02d}.json',case)
        if not spec['ad500']:
            if references is not None and case != references['reset_cases'][index]:
                raise ValueError('Accepted native reset case differs; no policy action issued for this case')
        else:
            query=entry['query_index']; role=entry['role']
            if role=='query' and references is not None and case_identity(case)!=case_identity(references['reset_cases'][query]):
                raise ValueError('Accepted native reset case differs; no policy action issued for this case')
            if role=='support' and support_references is not None and case_identity(case)!=case_identity(support_references['support_reset_cases'][query]):
                raise ValueError('Accepted K1 support case differs; no policy action issued for this case')
            existing=query_cases if role=='query' else support_cases
            if any(c['actual_seed']==case['actual_seed'] for c in existing):
                raise ValueError('Accepted readout cases must remain distinct within each bank')
            if role=='support' and any(c['actual_seed']==case['actual_seed'] for c in references['reset_cases']):
                raise ValueError('Accepted support cases must be disjoint from every query')
            existing.append(dict(case,episode=query+1))
        reset_cases.append(case); agent.begin_episode(entry['clock'],limit)
        steps=0; chunks=0; reward_sum=0.; success=False; done=False
        proposals=hashlib.sha256()
        while not done:
            noise=agent.act(features)
            if (noise.shape!=(32,32) or noise.dtype!=np.float32 or not np.isfinite(noise).all()
                    or (condition!='native' and (np.abs(noise)>1.5+1e-6).any())):
                raise ValueError('Invalid physical action proposal')
            requested=min(32,limit-steps)
            decision_id=f'{trial_id}:episode:{index+1}:chunk:{chunks}'
            reservation=budget.reserve(decision_id,requested,phase=spec['phase'],method=condition,task=task,
                episode=f'{trial_id}:{index+1}',chunk=chunks,policy_version=identity)
            if not reservation.execute or reservation.granted!=requested:
                raise RuntimeError('Refuse replay or a partial physical-action reservation')
            env.set_recording_context(dict(campaign_decision_id=decision_id,
                campaign_reserved_actions=requested))
            env.record_decision_metadata(dict(episode_index=entry['clock'],chunk_index=chunks,
                episode_limit=limit,primitive_step=steps,history_index=history_decisions if spec['ad500'] else decisions,
                offline_weights_frozen=True))
            proposals.update(noise.tobytes(order='C'))
            following,reward,end,trunc,info=env.step(noise)
            actual=info.get('primitive_steps')
            if type(actual) is not int or not 0<=actual<=requested:
                raise RuntimeError('Native runtime did not return a valid action duration; reservation retained')
            budget.resolve(decision_id,actual)
            done=bool(end or trunc)
            if (info.get('budget_truncated') or not math.isfinite(float(reward))
                    or (actual==0 and not done) or (not done and steps+actual>=limit)):
                raise RuntimeError('Invalid terminal/horizon/accounting result')
            agent.observe(noise,float(reward),done,actual)
            steps+=actual; chunks+=1; decisions+=1; history_decisions+=1; reward_sum+=float(reward)
            success|=bool(info.get('success',False)); features=following
        row=dict(case,success=success,return_=reward_sum,primitive_steps=steps,chunks=chunks)
        if spec['ad500']:
            row.update(role=entry['role'],query_index=entry['query_index'],episode_clock=entry['clock'],
                action_seed=entry['action_seed'],trial_index=entry['query_index'],proposal_sha256=proposals.hexdigest())
            if entry['role']=='support' and support_references is not None:
                prior=[r for r in support_references['episodes'] if r['role']=='support'][entry['query_index']]
                if row!=prior: raise ValueError('Same-weight support rollout differs; stop before its paired query')
        _atomic(output/f'episode-{index+1:02d}.json',row); episodes.append(row)
    if policy is not None and parameter_digest(policy.state_dict())!=before:
        raise RuntimeError('Frozen steering weights changed')
    if env.frozen_fingerprint()!=plan['frozen_motor_sha256']:
        raise RuntimeError('Frozen motor changed')
    scenes_after=renderer_records(env)
    result=dict(completed=True,context=context,task=task,reset_cases=reset_cases,episodes=episodes,
        actual_actions=sum(row['primitive_steps'] for row in episodes),motor_queries=decisions,
        renderer_records=scenes_after,scene_creations=len(set(scenes_after)-set(scenes_before)),
        elapsed_seconds=time.monotonic()-started,offline_weights_unchanged=True,
        reported_episodes={} if spec['ad500'] else
            {str(k):episodes[k-1]['success'] for k in ((1,2,4) if spec['source'] else (1,2,4,8))},
        support_episodes=[] if spec['source'] else [1,2,3,4],query_episodes=[] if spec['source'] else [5,6,7,8],
        status=('one source-only teacher K4; no target fitting or expert claim' if spec['source'] else
                'one paired development K8 trial; no target fitting or ICRL conclusion'))
    if spec['source']:
        result.update(source_iteration=100,source_teacher_sha256=SOURCE_TEACHER_SHA,
            source_or_target_SGD_updates=0,paid_sample_reuse=False,
            sampling=None if condition=='native' else agent.sampling_receipt())
    if spec['ad500']:
        result.update(query_reset_cases=query_cases,support_reset_cases=support_cases,
            support_episodes=[i+1 for i,row in enumerate(episodes) if row['role']=='support'],
            query_episodes=[i+1 for i,row in enumerate(episodes) if row['role']=='query'],
            query_successes=[row['success'] for row in episodes if row['role']=='query'],
            source_iteration=100,model_endpoint=None if condition in ('native','sac100-frozen') else 500,
            source_or_target_SGD_updates=0,paid_sample_reuse=False,
            query_feedback_carried_to_later_trials=False,
            reset_lesion='cache/previous noise/reward/carry flag cleared; query clock stays 1'
                if condition.endswith('-reset') else None,
            status='limited two-development-task K0/K1 readout; full 0/2/4/8 six-task protocol remains open')
    _atomic(output/'result.json',result)
    return result
