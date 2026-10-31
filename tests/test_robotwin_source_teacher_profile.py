"""Necessary source-profile CPU fakes; no simulator, GPU or physical actions.

Run on allocated Linux. Production checkpoint admission is a separate CPU
call to admit_plan on the exact immutable plan, never a teacher outcome.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_robotwin_history_evaluate import Env
from test_robotwin_history_live import arguments
from fluffy_bassoon.meta_rl import robotwin_history_evaluate as evaluate
from fluffy_bassoon.meta_rl import robotwin_history_live as live
from fluffy_bassoon.meta_rl.vla_history import file_sha256
from fluffy_bassoon.meta_rl.vla_noise_sac import SACConfig, SquashedGaussianActor


PLAN = live.ROOT/'outputs/robotwin-source-teacher-profile-20261031-r1/plan.json'


def configuration():
    value=json.loads(PLAN.read_text())
    value['native_backend_sha256']=file_sha256(__file__)  # This injected CPU fake only.
    return value


class SourceBudget:
    def __init__(self,plan):
        self.events=[];self.actual=0
        self.header=dict(total_actions=32400,source_caps={},phase_caps=dict(teacher_qualification=32400),
            contract=dict(evaluation_plan_sha256=evaluate.canonical_sha(plan)))
    def remaining(self,**kwargs):return 32400-self.actual
    def reserve(self,decision,requested,**kwargs):
        assert kwargs['phase']=='teacher_qualification'
        self.events.append(('reserve',decision,requested));return type('Reservation',(),dict(execute=True,granted=requested))()
    def resolve(self,decision,actual):self.events.append(('resolve',decision,actual));self.actual+=actual


class SourceEnv(Env):
    def episode_limit_for_task(self,task):return evaluate.SOURCE_TASK_LIMITS[task]
    def reset(self,task,seed):
        value=super().reset(task,seed);self.episode_limit=self.episode_limit_for_task(task)
        return value


def tiny_actor():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        return SquashedGaussianActor(2062,1024,SACConfig(hidden_dims=(8,8)),1.5).eval().requires_grad_(False)


def initial_rng():return torch.Generator(device='cpu').manual_seed(731).get_state()


def source_agent(*,sample=False):
    return evaluate.SourceTeacherAgent(tiny_actor(),initial_rng(),stochastic=sample)


def admission(plan):
    return dict(plan_sha256=evaluate.canonical_sha(plan),teacher_sha256=evaluate.SOURCE_TEACHER_SHA,
        actor_parameter_sha256='a'*64,sample_initial_private_rng_sha256='b'*64)


def test_exact_source_plan_hash_and_default_development_profile_are_distinct(tmp_path):
    source=live.read_source_teacher_plan(PLAN,live.SOURCE_PLAN_SHA)
    assert evaluate.trial_profile(source)['episodes']==4
    assert live.unit_order(source)[:6]==[('native',task) for task in evaluate.SOURCE_TASK_LIMITS]
    assert sum(evaluate.SOURCE_TASK_LIMITS.values())*4*3==32400
    assert live.parser().get_default('profile')=='fixed100'
    with pytest.raises(ValueError,match='immutable fixed-panel'):
        live.read_fixed_plan(PLAN,live.SOURCE_PLAN_SHA)
    changed=tmp_path/'changed.json';changed.write_text(PLAN.read_text()+' ')
    with pytest.raises(ValueError,match='immutable source100'):
        live.read_source_teacher_plan(changed,live.sha(changed))


@pytest.mark.parametrize('change',['target','case','K8','teacher','condition','reuse','source_cap'])
def test_source_profile_rejects_scope_and_unproved_sample_reuse(change):
    plan=configuration()
    if change=='target':plan['cases']['adjust_bottle']=plan['cases'].pop('handover_mic')
    elif change=='case':plan['cases']['handover_mic'][0]+=1
    elif change=='K8':plan['episodes_per_trial']=8
    elif change=='teacher':plan['teacher']['sha256']=evaluate.TEACHER_SHA
    elif change=='condition':plan['conditions'].append('sac_online')
    elif change=='reuse':plan['paid_sample_reuse']=dict(episodes=24)
    else:plan['max_actions']=1194400
    with pytest.raises(ValueError,match='exact source100'):
        evaluate.validate_plan(plan)


def test_sample_restores_saved_private_rng_per_task_and_continues_between_episodes():
    agent=source_agent(sample=True);features=np.zeros(2062,np.float32)
    expected=torch.Generator(device='cpu');expected.set_state(initial_rng())
    with torch.no_grad():
        first,_=agent.policy(torch.as_tensor(features[None]),generator=expected,deterministic=False)
        second,_=agent.policy(torch.as_tensor(features[None]),generator=expected,deterministic=False)
    with pytest.raises(RuntimeError,match='fixed private source RNG'):agent.act(features)
    agent.reset_trial(123)
    torch.manual_seed(999999)  # Global draws and reset-case seeds must not select actions.
    np.testing.assert_array_equal(agent.act(features).ravel(),first.numpy()[0])
    agent.begin_episode(1,600)
    np.testing.assert_array_equal(agent.act(features).ravel(),second.numpy()[0])
    assert agent.sampling_receipt()['initial_private_rng_sha256']!=agent.sampling_receipt()['final_private_rng_sha256']
    agent.reset_trial(987654321)  # Next task reloads the SAME source state, not a task seed.
    np.testing.assert_array_equal(agent.act(features).ravel(),first.numpy()[0])
    assert not agent.sampling_receipt()['optimizer_rng_used']


def test_mean_is_deterministic_physical_tanh_and_does_not_consume_private_rng():
    agent=source_agent();features=np.ones(2062,np.float32)
    agent.reset_trial(123)
    with torch.no_grad():expected,_=agent.policy(torch.as_tensor(features[None]),generator=None,deterministic=True)
    np.testing.assert_array_equal(agent.act(features).ravel(),expected.numpy()[0])
    assert np.abs(agent.act(features)).max()<=1.5
    proof=agent.sampling_receipt()
    assert proof['deterministic'] and proof['initial_private_rng_sha256']==proof['final_private_rng_sha256']


@pytest.mark.parametrize('task,limit,last',[('handover_mic',600,24),('scan_object',500,20),('place_a2b_left',400,16)])
def test_same_trial_loop_keeps_complete_k4_native_horizons_and_pre_dispatch_reservations(tmp_path,task,limit,last):
    plan=configuration();budget=SourceBudget(plan);env=SourceEnv(plan,budget,horizon=limit)
    value=evaluate.run_trial(env,budget,evaluate.ReferenceAgent(),plan,condition='native',task=task,
        output=tmp_path/task,run_id='cpu-fake')
    assert value['actual_actions']==4*limit and len(value['episodes'])==4
    assert sum(row[2]==last for row in budget.events if row[0]=='reserve')==4
    assert value['context']['phase']=='teacher_qualification' and value['source_or_target_SGD_updates']==0
    assert not value['paid_sample_reuse'] and value['support_episodes']==value['query_episodes']==[]
    assert max(np.abs(a).max() for a in env.actions)>1.5  # Native Gaussian remains unclipped.


def test_source_qualification_rejects_original_source_or_target_ledgers_before_reset(tmp_path):
    plan=configuration();budget=SourceBudget(plan);budget.header['phase_caps']={'target':32400}
    env=SourceEnv(plan,budget)
    with pytest.raises(ValueError,match='separately admitted evaluation ledger'):
        evaluate.run_trial(env,budget,evaluate.ReferenceAgent(),plan,condition='native',task='handover_mic',
            output=tmp_path/'bad-ledger',run_id='fake')
    assert not env.resets and not budget.events


def test_changed_accepted_source_case_stops_before_mean_action_or_reservation(tmp_path):
    plan=configuration();budget=SourceBudget(plan);env=SourceEnv(plan,budget)
    native=evaluate.run_trial(env,budget,evaluate.ReferenceAgent(),plan,condition='native',task='handover_mic',
        output=tmp_path/'native',run_id='cpu-fake')
    changed=SourceEnv(plan,budget,drift=True);before=len(budget.events)
    with pytest.raises(ValueError,match='Accepted native reset case differs'):
        evaluate.run_trial(changed,budget,source_agent(),plan,condition='sac100-mean',task='handover_mic',
            output=tmp_path/'mean',run_id='cpu-fake',references=native)
    assert not changed.actions and len(budget.events)==before


def fake_source(plan):
    policy=tiny_actor()
    source=dict(contract={},source_iteration=100,trials=600,state=dict(source_steps=1003790,task_steps={'source':1003790}),
        learner=dict(updates=638500,actor=deepcopy(policy.state_dict()),torch_rng=initial_rng()))
    return source,policy


@pytest.mark.parametrize('change',['none','actor','actions','task_actions','updates','trials','rng'])
def test_cpu_source_admission_matches_actual_closed_actor_and_counters_without_restoring_cuda_rng(monkeypatch,change):
    plan=configuration();source,policy=fake_source(plan);closed=deepcopy(source)
    if change=='actor':closed['learner']['actor'][next(iter(closed['learner']['actor']))]+=1
    elif change=='actions':closed['state']['source_steps']-=1
    elif change=='task_actions':closed['state']['task_steps']['source']-=1
    elif change=='updates':closed['learner']['updates']-=1
    elif change=='trials':closed['trials']-=1
    elif change=='rng':closed['learner']['torch_rng'][0]^=1
    monkeypatch.setattr(evaluate,'load_source_teacher',lambda *a,**k:(source,policy))
    monkeypatch.setattr(evaluate,'source_recovery',lambda *a,**k:closed)
    if change!='none':
        with pytest.raises(ValueError,match='closed source actor/counters'):evaluate.admit_plan(plan)
    else:
        result=evaluate.admit_plan(plan)
        assert result['teacher_sha256']==evaluate.SOURCE_TEACHER_SHA and not result['CUDA_generator_restored']
        assert not result['teacher_competence_measured'] and result['additional_robot_actions']==0


class SourceWorker:
    def __init__(self,plan,args):self.plan=plan;self.args=args;self.actual=0;self.calls=[]
    def audit(self):return dict(actual_actions=self.actual,charged_actions=self.actual,pending_actions=0,uncertain_actions=0)
    def __call__(self,index,condition,task,unit,native,timeout):
        assert timeout>0;self.calls.append((condition,task))
        cases=[dict(task=task,episode=e+1,requested_seed=s,actual_seed=s,
            observation_sha256='c'*64,features_sha256='d'*64) for e,s in enumerate(self.plan['cases'][task])]
        proof=admission(self.plan)
        context=dict(method=condition,phase='teacher_qualification',evaluation_plan_sha256=evaluate.canonical_sha(self.plan),
            policy_checkpoint_sha256=None if condition=='native' else evaluate.SOURCE_TEACHER_SHA,
            policy_sha256=None if condition=='native' else proof['actor_parameter_sha256'])
        sampling=dict(source_checkpoint_sha256=evaluate.SOURCE_TEACHER_SHA,
            convention='restore_fixed_source_private_torch_rng_per_task_K4',
            initial_private_rng_sha256=proof['sample_initial_private_rng_sha256'],
            deterministic=condition=='sac100-mean',optimizer_rng_used=False)
        value=dict(completed=True,task=task,context=context,reset_cases=cases,
            episodes=[dict(c,primitive_steps=1) for c in cases],actual_actions=4,
            source_iteration=100,source_teacher_sha256=evaluate.SOURCE_TEACHER_SHA,
            source_or_target_SGD_updates=0,paid_sample_reuse=False,sampling=sampling)
        (unit/'trial').mkdir();live.write(unit/'trial/result.json',value)
        receipt=dict(completed=True,condition=condition,task=task,backend_sha256=live.BACKEND_SHA,
            parity=dict(passed=True),result_sha256=live.sha(unit/'trial/result.json'))
        p=unit/'unit-receipt.json';live.write(p,receipt);self.actual+=4
        return dict(condition=condition,task=task,receipt_path=str(p),receipt_sha256=live.sha(p))


def source_arguments(tmp_path,*,create=True):
    args=arguments(tmp_path,create=create)
    args.profile='source-teacher100';args.plan_sha=live.SOURCE_PLAN_SHA;args.wall_seconds=3500.
    args.condition='native';args.task='handover_mic';args.unit_index=0
    return args


def test_existing_parent_dispatches_exact18_source_k4_units_with_native_first(tmp_path):
    plan=configuration();args=source_arguments(tmp_path);worker=SourceWorker(plan,args)
    result=live.run_units(args,plan,worker=worker,audit=worker.audit,source_admission=admission(plan))
    assert result['status']=='complete' and worker.calls==live.unit_order(plan) and worker.actual==72
    assert worker.calls[:6]==[('native',t) for t in evaluate.SOURCE_TASK_LIMITS]
    assert not any(t in ('adjust_bottle','move_stapler_pad') for _,t in worker.calls)


@pytest.mark.parametrize('change',['actor','rng','deterministic','reuse'])
def test_parent_rejects_rehashed_wrong_actor_rng_or_implicit_paid_reuse(tmp_path,change):
    plan=configuration();args=source_arguments(tmp_path);worker=SourceWorker(plan,args);natives={}
    unit=tmp_path/'native';unit.mkdir();r=worker(0,'native','handover_mic',unit,None,100)
    live.verify_unit(r,plan,('native','handover_mic'),natives,source_admission=admission(plan))
    unit=tmp_path/'sample';unit.mkdir();r=worker(12,'sac100-sample','handover_mic',unit,natives['handover_mic'],100)
    path=unit/'trial/result.json';value=json.loads(path.read_text())
    if change=='actor':value['context']['policy_sha256']='f'*64
    elif change=='rng':value['sampling']['initial_private_rng_sha256']='f'*64
    elif change=='deterministic':value['sampling']['deterministic']=True
    else:value['paid_sample_reuse']=True
    live.write(path,value,replace=True);receipt=unit/'unit-receipt.json';row=json.loads(receipt.read_text())
    row['result_sha256']=live.sha(path);live.write(receipt,row,replace=True);r['receipt_sha256']=live.sha(receipt)
    with pytest.raises(ValueError,match='Source teacher'):
        live.verify_unit(r,plan,('sac100-sample','handover_mic'),natives,source_admission=admission(plan))


def test_source_child_requires_full_k4_allowance_before_agent_or_runtime(tmp_path):
    plan=configuration();args=source_arguments(tmp_path,create=False);calls=[]
    with live.ledger(args,plan,'prior-fake-cost') as budget:
        budget.reserve('prior',30001,phase='teacher_qualification',method='native',task='handover_mic',
            episode=0,chunk=0,policy_version='fake')
        budget.resolve('prior',30001)
    with pytest.raises(RuntimeError,match='complete K4 allowance'):
        live.run_child(args,plan,prepare=lambda *a:calls.append('runtime'),agent_loader=lambda *a,**k:calls.append('agent'))
    assert not calls


def test_child_command_carries_source_profile_without_changing_parent_death_identity(tmp_path):
    args=source_arguments(tmp_path)
    command=live.child_command(args,0,'native','handover_mic',tmp_path/'unit',None)
    assert command[command.index('--profile')+1]=='source-teacher100'
    assert '--parent-pid' in command and '--parent-start-ticks' in command
