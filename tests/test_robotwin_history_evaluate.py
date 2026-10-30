"""Prepared evaluation tests: run on allocated Linux, never a simulator import."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.robotwin_history import NativeHistoryConfig,NativeHistoryPolicy,export_policy
from fluffy_bassoon.meta_rl.robotwin_history_evaluate import (
    CONDITIONS,MODELS,TEACHER_SHA,TRAIN_PIN,ReferenceAgent,StudentAgent,assert_runtime,
    case_bank,load_agent,run_trial,validate_plan,
)
from fluffy_bassoon.meta_rl.vla_history import file_sha256


def plan():
    runtime=dict(norm_stats_sha256='a'*64,rlinf_commit='b'*40,robotwin_commit='c'*40,
        features='public2062',action_transform='native',task_config_name='demo_clean',
        appearance_randomization=False)
    interface=dict(runtime=runtime,model_path='/fixed/motus',
        task_episode_limits=dict(adjust_bottle=400,move_stapler_pad=400,handover_mic=600),
        episode_limit=600,observation_dim=2062,noise_shape=[32,32],action_limit=1.5)
    full=dict(runtime,model_path=interface['model_path'],task_episode_limits=interface['task_episode_limits'])
    return dict(format='robotwin_native_history_evaluation_v1',cases=case_bank(),conditions=list(CONDITIONS),
        episodes_per_trial=8,episode_limit=400,max_actions=57600,max_episodes=144,
        student_action='deterministic_mean',teacher=dict(path='/fixed/sac5.pt',sha256=TEACHER_SHA),
        models={name:dict(path=f'/fixed/{name}/candidate-000100.pt',sha256='d'*64) for name in MODELS},
        interface=interface,runtime_contract=full,native_backend_sha256=file_sha256(__file__),
        frozen_motor_sha256='e'*64,train_index_sha256='f'*64,validation_index_sha256='1'*64)


class Budget:
    def __init__(self,deny=False): self.events=[]; self.actual=0; self.deny=deny
    def remaining(self,**kwargs): return 57600-self.actual
    def reserve(self,decision,requested,**kwargs):
        self.events.append(('reserve',decision,requested))
        return SimpleNamespace(execute=not self.deny,granted=requested)
    def resolve(self,decision,actual): self.events.append(('resolve',decision,actual)); self.actual+=actual


class Env:
    observation_dim=2062; noise_shape=(32,32); native_horizon=32; default_episode_limit=600
    reset_policy_action_bound=0; corpus_writer=True
    def __init__(self,configuration,budget,*,horizon=1,drift=False,fail=False):
        self.backend=SimpleNamespace(contract=dict(configuration['runtime_contract'],actual_render_pci='allocation'),
            final_observation_only=True,output_dir=Path(__file__).parent/'not-a-live-renderer')
        self.configuration=configuration; self.budget=budget; self.horizon=horizon; self.drift=drift; self.fail=fail
        self.actions=[]; self.resets=[]
    def frozen_fingerprint(self): return self.configuration['frozen_motor_sha256']
    def episode_limit_for_task(self,task): return 400
    def set_recording_context(self,context): pass
    def record_decision_metadata(self,metadata): pass
    def reset(self,task,seed):
        self.steps=0; self.episode_limit=400; self.resets.append((task,seed))
        self.last_reset_info=dict(task=task,requested_seed=seed,actual_seed=seed+int(self.drift),
            observation_sha256=f'raw-{task}-{seed}',features_sha256=f'features-{task}-{seed}',reset_policy_actions=0)
        return np.zeros(2062,np.float32)
    def step(self,noise):
        assert self.budget.events[-1][0]=='reserve' # Precedes motor dispatch.
        if self.fail: raise RuntimeError('native simulator fault after reservation')
        self.actions.append(noise.copy()); actual=min(32,self.horizon-self.steps); self.steps+=actual
        return np.zeros(2062,np.float32),0.,self.steps==self.horizon,False,dict(
            primitive_steps=actual,budget_truncated=False,success=False)


def test_exact_panel_and_fixed_endpoint_are_required():
    configuration=plan(); validate_plan(configuration)
    assert len(CONDITIONS)==9 and len(configuration['cases'])==2
    assert len(CONDITIONS)*len(configuration['cases'])*8==144
    assert 144*400==57600
    configuration['max_actions']=51200
    with pytest.raises(ValueError,match='fixed-endpoint'): validate_plan(configuration)
    configuration=plan(); configuration['models']['ad-full']['path']='/fixed/candidate.pt'
    with pytest.raises(ValueError,match='100-update endpoint'): validate_plan(configuration)


def test_native_gaussian_is_unclipped_and_zero_control_is_properly_paired(tmp_path):
    configuration=plan(); budget=Budget(); env=Env(configuration,budget)
    native=run_trial(env,budget,ReferenceAgent(),configuration,condition='native',task='adjust_bottle',
        output=tmp_path/'native',run_id='test')
    assert native['actual_actions']==8 and max(np.abs(action).max() for action in env.actions)>1.5
    env=Env(configuration,budget)
    zero=run_trial(env,budget,ReferenceAgent(zero=True),configuration,condition='native-zero',task='adjust_bottle',
        output=tmp_path/'zero',run_id='test',references=native)
    assert zero['reset_cases']==native['reset_cases'] and all(not action.any() for action in env.actions)
    assert budget.actual==16 and zero['reported_episodes']==dict.fromkeys(('1','2','4','8'),False)


def test_full_native_horizon_reserves_the_last16_actions_and_never_cuts_k8(tmp_path):
    configuration=plan(); budget=Budget(); env=Env(configuration,budget,horizon=400)
    result=run_trial(env,budget,ReferenceAgent(),configuration,condition='native',task='move_stapler_pad',
        output=tmp_path/'full',run_id='test')
    assert result['actual_actions']==3200 and result['motor_queries']==104
    assert sum(row[2]==16 for row in budget.events if row[0]=='reserve')==8


def test_different_actual_case_is_rejected_before_policy_motor_query(tmp_path):
    configuration=plan(); budget=Budget(); env=Env(configuration,budget)
    native=run_trial(env,budget,ReferenceAgent(),configuration,condition='native',task='adjust_bottle',
        output=tmp_path/'native',run_id='test')
    changed=Env(configuration,budget,drift=True); before=len(budget.events)
    with pytest.raises(ValueError,match='Accepted native reset case differs'):
        run_trial(changed,budget,ReferenceAgent(zero=True),configuration,condition='native-zero',task='adjust_bottle',
            output=tmp_path/'changed',run_id='test',references=native)
    assert not changed.actions and len(budget.events)==before


@pytest.mark.parametrize('deny,fail',[(True,False),(False,True)])
def test_replay_refusal_and_native_failure_never_resolve_an_unexecuted_intent(tmp_path,deny,fail):
    configuration=plan(); budget=Budget(deny=deny); env=Env(configuration,budget,fail=fail)
    with pytest.raises(RuntimeError):
        run_trial(env,budget,ReferenceAgent(),configuration,condition='native',task='adjust_bottle',
            output=tmp_path/'fault',run_id='test')
    assert len(budget.events)==1 and budget.events[0][0]=='reserve'
    assert not (tmp_path/'fault/result.json').exists() and (tmp_path/'fault/intent.json').exists()


def test_runtime_byte_identity_and_final_camera_contract_are_required():
    configuration=plan(); env=Env(configuration,Budget()); assert_runtime(env,configuration)
    with pytest.raises(ValueError,match='pinned qualified native backend'):
        assert_runtime(env,dict(configuration,native_backend_sha256='0'*64))
    env.backend.final_observation_only=False
    with pytest.raises(ValueError,match='pinned qualified native backend'): assert_runtime(env,configuration)


def test_prior_or_unpaired_native_receipts_cannot_authorize_target_actions(tmp_path):
    configuration=plan(); budget=Budget(); env=Env(configuration,budget)
    with pytest.raises(ValueError,match='completed native K8 receipt'):
        run_trial(env,budget,ReferenceAgent(zero=True),configuration,condition='native-zero',
            task='adjust_bottle',output=tmp_path/'unpaired',run_id='test',
            references=dict(completed=True,task='adjust_bottle',context=dict(method='native',
                evaluation_plan_sha256='old-native-bank'),reset_cases=[{}]*8))
    assert not env.resets and not budget.events


def test_portable_loader_binds_actual_fixed100_endpoint_and_same_weight_reset(tmp_path):
    configuration=plan(); policy=NativeHistoryPolicy()
    contract=dict(source_commit=TRAIN_PIN,objective='ad',seed=0,config=asdict(policy.config),budget=dict(updates=100),
        train_index_sha256=configuration['train_index_sha256'],validation_index_sha256=configuration['validation_index_sha256'],
        train_metadata=dict(teacher_recovery_sha256=TEACHER_SHA))
    manifest=dict(update=100,mode='full',interface=configuration['interface'],contract=contract)
    path=tmp_path/'candidate-000100.pt'; export_policy(path,policy,manifest)
    configuration['models']['ad-full']=dict(path=str(path),sha256=file_sha256(path))
    agent=load_agent(configuration,'ad-full-reset',device='cpu')
    assert agent.controller.erase_each_episode and agent.deterministic
    manifest['update']=25; export_policy(path,policy,manifest)
    configuration['models']['ad-full']['sha256']=file_sha256(path)
    with pytest.raises(ValueError,match='fixed source-only first-fit contract'):
        load_agent(configuration,'ad-full-reset',device='cpu')


def test_zero_action_terminal_transition_is_preserved_without_inventing_a_step(tmp_path):
    configuration=plan(); budget=Budget(); env=Env(configuration,budget,horizon=0)
    native=run_trial(env,budget,ReferenceAgent(),configuration,condition='native',task='adjust_bottle',
        output=tmp_path/'native',run_id='test')
    policy=NativeHistoryPolicy(NativeHistoryConfig(width=8,layers=1,heads=2,context_length=8))
    agent=StudentAgent(policy,dict(mode='full',interface=configuration['interface']),configuration['interface'],
        erase_each_episode=False,deterministic=True)
    agent.condition='ad-full'; agent.checkpoint_sha256=configuration['models']['ad-full']['sha256']
    agent.reset_trial(0); agent.begin_episode(0,400); action=agent.act(np.zeros(2062,np.float32))
    with pytest.raises(ValueError,match='valid native duration'): agent.observe(action,0.,False,0)
    agent.observe(action,0.,True,0)
    result=run_trial(Env(configuration,budget,horizon=0),budget,agent,configuration,
        condition='ad-full',task='adjust_bottle',output=tmp_path/'student',run_id='test',references=native)
    assert result['actual_actions']==0 and result['motor_queries']==8
    assert agent.controller.episode_index==7 and agent.controller.primitive_step==0
