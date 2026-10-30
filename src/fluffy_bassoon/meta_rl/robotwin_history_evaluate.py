"""Frozen native RoboTwin evaluation adapter; caller owns qualified runtime/ledger.

This module never constructs a simulator, allocates hardware, trains a policy,
or creates a campaign. The execution owner supplies the reviewed environment
and its existing CampaignBudget. The same adapter supports injected CPU fakes.
"""
import hashlib
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


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def case_bank(seed=20261030):
    if type(seed) is not int or seed != 20261030:
        raise ValueError('Use the already declared development-target case bank')
    return {task:[seed+2_000_000+i*997 for i in range(8)] for task in TASKS}


def validate_plan(plan):
    """A fixed nine-condition screen; no target-based endpoint selection."""
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
    if (actual != expected or env.native_horizon != 32 or getattr(env,'reset_policy_action_bound',0) != 0
            or getattr(env,'corpus_writer',None) is None
            or any(env.episode_limit_for_task(task) != 400 for task in TASKS)):
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


def load_agent(plan, condition, *, device='cuda'):
    validate_plan(plan)
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


def run_trial(env, budget, agent, plan, *, condition, task, output, run_id, references=None):
    """Execute one fresh K8 unit with durable pre-dispatch physical reservations.

    Native runs first to establish accepted reset cases. Every other condition
    must match its exact accepted seed, raw observation and feature hashes.
    Failure leaves real costs/raw intents intact and never retries a case.
    Caller owns GPU/CUDA qualification, process lifetime, wall time and ledger.
    """
    validate_plan(plan)
    if (condition not in CONDITIONS or task not in TASKS or (condition != 'native' and references is None)
            or getattr(agent,'condition',None) != condition):
        raise ValueError('Run native first and use its paired accepted reset cases')
    if references is not None and (references.get('completed') is not True or references.get('task') != task
            or references.get('context',{}).get('method') != 'native'
            or references['context'].get('evaluation_plan_sha256') != canonical_sha(plan)
            or len(references.get('reset_cases',[])) != 8):
        raise ValueError('Need the completed native K8 receipt bound to this exact plan')
    assert_runtime(env,plan)
    scenes_before=renderer_records(env)
    if budget.remaining(phase='target',method=condition) < 3200:
        raise RuntimeError('Reserve a complete K8 allowance before beginning this trial')
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    trial_id = f'{run_id}:{condition}:{task}:K8'
    policy = agent.policy
    before = None if policy is None else parameter_digest(policy.state_dict())
    identity = 'native_zero_initial_noise' if condition=='native-zero' else 'native_standard_gaussian' if condition == 'native' else (
        TEACHER_SHA if condition == 'sac5-mean' else plan['models'][condition.removesuffix('-reset')]['sha256'])
    if condition not in ('native','native-zero','sac5-mean') and agent.checkpoint_sha256 != identity:
        raise ValueError('Steering agent checkpoint identity differs from the planned condition')
    checkpoint_path = (None if condition in ('native','native-zero') else plan['teacher']['path']
        if condition=='sac5-mean' else plan['models'][condition.removesuffix('-reset')]['path'])
    context = dict(method=condition,phase='target',trial_id=trial_id,policy_version=identity,
        policy_checkpoint=checkpoint_path,policy_checkpoint_sha256=None if condition in ('native','native-zero') else identity,
        policy_sha256=before,
        evaluation_plan_sha256=canonical_sha(plan),offline_weights_frozen=True,
        action_convention=('native_standard_gaussian_unclipped' if condition=='native' else
            'native_zero_initial_noise' if condition=='native-zero' else
            'physical1.5*tanh(mean)' if condition=='sac5-mean' else plan['student_action']),
        erase_each_episode=condition.endswith('-reset'))
    env.set_recording_context(context)
    # Model/control identities do not perturb their shared action random stream.
    policy_seed = 20261030+7_000_000+TASKS.index(task)*100_000
    agent.reset_trial(policy_seed)
    episodes=[]; reset_cases=[]; decisions=0; started=time.monotonic()
    _atomic(output/'intent.json',dict(context=context,seeds=plan['cases'][task],max_actions=3200))
    for index,seed in enumerate(plan['cases'][task]):
        features=env.reset(task,seed)
        reset=dict(env.last_reset_info)
        required=('actual_seed','observation_sha256','features_sha256')
        if (env.episode_limit!=400 or reset.get('task')!=task or reset.get('requested_seed')!=seed
                or reset.get('reset_policy_actions')!=0 or type(reset.get('actual_seed')) is not int
                or any(not reset.get(key) for key in required[1:])):
            raise ValueError('Missing actual zero-action reset/case provenance')
        case=dict(task=task,episode=index+1,requested_seed=seed,
            **{key:reset[key] for key in required})
        _atomic(output/f'reset-{index+1:02d}.json',case)
        if references is not None and case != references['reset_cases'][index]:
            raise ValueError('Accepted native reset case differs; no policy action issued for this case')
        reset_cases.append(case); agent.begin_episode(index,400)
        steps=0; chunks=0; reward_sum=0.; success=False; done=False
        while not done:
            noise=agent.act(features)
            if (noise.shape!=(32,32) or noise.dtype!=np.float32 or not np.isfinite(noise).all()
                    or (condition!='native' and (np.abs(noise)>1.5+1e-6).any())):
                raise ValueError('Invalid physical action proposal')
            requested=min(32,400-steps)
            decision_id=f'{trial_id}:episode:{index+1}:chunk:{chunks}'
            reservation=budget.reserve(decision_id,requested,phase='target',method=condition,task=task,
                episode=f'{trial_id}:{index+1}',chunk=chunks,policy_version=identity)
            if not reservation.execute or reservation.granted!=requested:
                raise RuntimeError('Refuse replay or a partial physical-action reservation')
            env.set_recording_context(dict(campaign_decision_id=decision_id,
                campaign_reserved_actions=requested))
            env.record_decision_metadata(dict(episode_index=index,chunk_index=chunks,
                episode_limit=400,primitive_step=steps,history_index=decisions,
                offline_weights_frozen=True))
            following,reward,end,trunc,info=env.step(noise)
            actual=info.get('primitive_steps')
            if type(actual) is not int or not 0<=actual<=requested:
                raise RuntimeError('Native runtime did not return a valid action duration; reservation retained')
            budget.resolve(decision_id,actual)
            done=bool(end or trunc)
            if (info.get('budget_truncated') or not math.isfinite(float(reward))
                    or (actual==0 and not done) or (not done and steps+actual>=400)):
                raise RuntimeError('Invalid terminal/horizon/accounting result')
            agent.observe(noise,float(reward),done,actual)
            steps+=actual; chunks+=1; decisions+=1; reward_sum+=float(reward)
            success|=bool(info.get('success',False)); features=following
        row=dict(case,success=success,return_=reward_sum,primitive_steps=steps,chunks=chunks)
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
        reported_episodes={str(k):episodes[k-1]['success'] for k in (1,2,4,8)},
        support_episodes=[1,2,3,4],query_episodes=[5,6,7,8],
        status='one paired development K8 trial; no target fitting or ICRL conclusion')
    _atomic(output/'result.json',result)
    return result
