"""Allocated-Linux CPU fakes for isolated source100 AD500 readout semantics.

No simulator, production ledger, allocation, fitting or physical actions.
Actual fit artifact admission remains separate from these synthetic fixtures.
"""
from contextlib import contextmanager
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_robotwin_history_evaluate import Env
from test_robotwin_history_live import arguments
from test_robotwin_source_teacher_profile import tiny_actor
from fluffy_bassoon.meta_rl import robotwin_history_evaluate as ev
from fluffy_bassoon.meta_rl import robotwin_history_live as live
from fluffy_bassoon.meta_rl.robotwin_history import NativeHistoryConfig
from fluffy_bassoon.meta_rl.vla_history import file_sha256


def plan():
    p=json.loads(live.AD500_TEMPLATE.read_text())
    p.update(fit_commit='a'*40,train_index_sha256='b'*64,validation_index_sha256='c'*64,
        models={k:dict(path=f'/synthetic/{k}/candidate-000500.pt',sha256=('d' if k=='ad-full' else 'e')*64)
            for k in ev.AD500_MODELS})
    p['native_backend_sha256']=file_sha256(__file__)
    return p


class Budget:
    def __init__(self,p):
        self.actual=0; self.events=[]
        self.header=dict(total_actions=51200,source_caps={},phase_caps=dict(target=51200),
            contract=dict(evaluation_plan_sha256=ev.canonical_sha(p)))
    def remaining(self,**kwargs): return 51200-self.actual
    def reserve(self,decision,requested,**kwargs):
        assert kwargs['phase']=='target'
        self.events.append(('reserve',decision,requested));return SimpleNamespace(execute=True,granted=requested)
    def resolve(self,decision,actual):self.events.append(('resolve',decision,actual));self.actual+=actual
    def snapshot(self):return dict(actual_actions=self.actual,charged_actions=self.actual,
        pending_actions=0,uncertain_actions=0,unresolved_decisions=[])


class ReadoutEnv(Env):
    def set_recording_context(self,context):
        if not hasattr(self,'context'):self.context={}
        self.context.update(context)
    def step(self,noise):
        features,_,end,trunc,info=super().step(noise)
        return features,(1. if self.context['adaptation_role']=='support' else 99.),end,trunc,info


class TokenPolicy(torch.nn.Module):
    """Expose feedback, cache and clocks without constructing a learner."""
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1));self.inputs=[];self.states=[]
        self.config=NativeHistoryConfig()  # Selector fake; no production model is constructed.
        self.register_buffer('observation_mean',torch.zeros(2062))
        self.register_buffer('observation_std',torch.ones(2062))
    def step(self,token,state,**kwargs):
        self.inputs.append(token.clone());self.states.append(state)
        value=.01+token[:,3086:3087]*.01+token[:,-2:-1]*.01
        return value.expand(-1,1024),None,torch.ones(1) if state is None else state+1


def student(p,condition):
    key=ev.ad500_model_key(condition)
    a=ev.StudentAgent(TokenPolicy(),dict(mode=key.split('-')[1],interface=p['interface']),p['interface'],
        erase_each_episode=condition.endswith('-reset'),deterministic=True)
    a.condition=condition;a.checkpoint_sha256=p['models'][key]['sha256'];return a


def agent(p,condition):
    if condition=='native':return ev.ReferenceAgent()
    if condition=='sac100-frozen':return ev.QuerySACAgent(tiny_actor())
    return student(p,condition)


def admission(p):
    return dict(plan_sha256=ev.canonical_sha(p),model_endpoint=500,
        checked=[dict(condition=c,parameter_sha256=ev.parameter_digest(agent(p,c).policy.state_dict()))
            for c in ('ad-full-k0','ad-static-k0','sac100-frozen')])


def run(tmp_path,p,b,condition,*,native=None,support=None,horizon=1,env=None,a=None):
    env=env or ReadoutEnv(p,b,horizon=horizon)
    a=a or agent(p,condition)
    return ev.run_trial(env,b,a,p,condition=condition,task='adjust_bottle',
        output=tmp_path/condition,run_id='CPU-FAKE',references=native,support_references=support)


def test_prepared_template_is_pinned_nonexecutable_and_only_artifact_slots_can_bind(tmp_path):
    assert live.sha(live.AD500_TEMPLATE)==live.AD500_TEMPLATE_SHA
    with pytest.raises(ValueError,match='actual post-fit'):live.read_ad500_plan(live.AD500_TEMPLATE,live.AD500_TEMPLATE_SHA)
    p=plan();p['native_backend_sha256']=live.BACKEND_SHA
    path=tmp_path/'bound.json';live.write(path,p)
    assert live.read_ad500_plan(path,live.sha(path))==p
    p['max_actions']+=1;live.write(path,p,replace=True)
    with pytest.raises(ValueError,match='topology'):live.read_ad500_plan(path,live.sha(path))
    assert live.parser().get_default('profile')=='fixed100'
    with pytest.raises(ValueError,match='immutable fixed-panel'):live.read_fixed_plan(path,live.sha(path))


@pytest.mark.parametrize('key',['cases','support_cases','query_action_seeds','support_action_seeds','max_actions','paid_sample_reuse'])
def test_readout_rejects_unreviewed_cases_streams_costs_and_reuse(key):
    p=plan()
    if key.endswith('cases') or key.endswith('seeds'):p[key]['adjust_bottle'][0]+=1
    elif key=='paid_sample_reuse':p[key]=dict(episodes=16)
    else:p[key]+=400
    with pytest.raises(ValueError,match='exact source100 AD500'):ev.validate_plan(p)


def test_action_rng_reseed_preserves_support_state_feedback_and_clock():
    p=plan();a=student(p,'ad-full-k1');c=a.controller
    a.reset_trial(11);a.begin_episode(0,400)
    with pytest.raises(ValueError,match='between completed episodes'):a.reseed_action_rng(12)
    noise=a.act(np.zeros(2062,np.float32));a.observe(noise,7.,True,1)
    state=c.state;previous=c.previous_noise.copy();a.reseed_action_rng(12345)
    assert c.state is state and c.previous_reward==7. and c.episode_index==0 and c.primitive_step==1
    np.testing.assert_array_equal(c.previous_noise,previous)
    assert torch.equal(c.generator.get_state(),torch.Generator().manual_seed(12345).get_state())
    a.begin_episode(1,400);assert c.state is state and c.episode_index==1


def test_k0_has_eight_fresh_trials_no_prior_query_feedback_and_unclipped_native(tmp_path):
    p=plan();b=Budget(p);native=run(tmp_path,p,b,'native')
    a=student(p,'ad-full-k0');value=run(tmp_path,p,b,'ad-full-k0',native=native,a=a)
    assert [r['episode_clock'] for r in value['episodes']]==[0]*8
    assert all(state is None for state in a.policy.states)
    assert all(not token[:,2062:].any() for token in a.policy.inputs)
    assert value['query_reset_cases']==native['reset_cases'] and value['actual_actions']==8
    ev.verify_ad500_result(value,p,'ad-full-k0','adjust_bottle',native=native,support=None,admission=admission(p))
    env=ReadoutEnv(p,b);run(tmp_path/'another',p,b,'native',env=env)
    assert max(np.abs(n).max() for n in env.actions)>1.5


def test_k1_retains_failed_supports_resets_between_pairs_and_sameweight_lesion_keeps_clock1(tmp_path):
    p=plan();b=Budget(p);native=run(tmp_path,p,b,'native')
    full=student(p,'ad-full-k1');f=run(tmp_path,p,b,'ad-full-k1',native=native,a=full)
    reset=student(p,'ad-full-k1-reset');r=run(tmp_path,p,b,'ad-full-k1-reset',native=native,support=f,a=reset)
    assert len(f['support_episodes'])==len(f['query_episodes'])==8
    assert not any(row['success'] for row in f['episodes'] if row['role']=='support')
    for i in range(8):
        s,q=2*i,2*i+1
        assert full.policy.states[s] is reset.policy.states[s] is None
        assert full.policy.states[q] is not None and reset.policy.states[q] is None
        assert full.policy.inputs[q][0,3086]==1 and reset.policy.inputs[q][0,3086]==0
        assert full.policy.inputs[q][0,3087]==1 and reset.policy.inputs[q][0,3087]==0
        torch.testing.assert_close(full.policy.inputs[q][:,-2:],reset.policy.inputs[q][:,-2:],rtol=0,atol=0)
        assert f['episodes'][q]['episode_clock']==r['episodes'][q]['episode_clock']==1
    assert f['actual_actions']==r['actual_actions']==16 and b.actual==40
    assert [row for row in f['episodes'] if row['role']=='support']==[row for row in r['episodes'] if row['role']=='support']
    ev.verify_ad500_result(r,p,'ad-full-k1-reset','adjust_bottle',native=native,support=f,admission=admission(p))


def test_query_sac_streams_are_independent_per_query_and_never_restore_source_k4_rng():
    p=plan();a=agent(p,'sac100-frozen');features=np.zeros(2062,np.float32)
    with pytest.raises(RuntimeError,match='query action seed'):a.act(features)
    seed=p['query_action_seeds']['adjust_bottle'][2];a.reset_trial(seed)
    first=a.act(features);a.act(features);torch.manual_seed(9999)
    a.reseed_action_rng(seed);np.testing.assert_array_equal(a.act(features),first)
    assert a.generator.initial_seed()==seed
    a.reseed_action_rng(seed+997);assert not np.array_equal(a.act(features),first)
    assert np.abs(first).max()<=1.5


@pytest.mark.parametrize('kind',['query_drift','support_overlap','support_drift','wrong_ledger'])
def test_pairing_and_ledger_faults_stop_before_any_new_action(tmp_path,kind):
    p=plan();b=Budget(p);native=run(tmp_path,p,b,'native')
    full=run(tmp_path,p,b,'ad-full-k1',native=native) if kind=='support_drift' else None
    env=ReadoutEnv(p,b,drift=kind in ('query_drift','support_drift'))
    if kind=='support_overlap':
        original=env.reset
        def reset(task,seed):
            value=original(task,seed);env.last_reset_info['actual_seed']=native['reset_cases'][0]['actual_seed'];return value
        env.reset=reset
    if kind=='wrong_ledger':b.header['source_caps']={'sac':1100000}
    condition='ad-full-k1-reset' if full else 'ad-full-k1' if kind=='support_overlap' else 'ad-full-k0'
    before=len(b.events)
    with pytest.raises(ValueError):run(tmp_path,p,b,condition,native=native,support=full,env=env)
    assert not env.actions and len(b.events)==before


def test_native_k0_and_k1_units_keep_400step_horizons_and_last16_reservations(tmp_path):
    p=plan();b=Budget(p);native=run(tmp_path,p,b,'native',horizon=400)
    assert native['actual_actions']==3200
    f=run(tmp_path,p,b,'ad-full-k1',native=native,horizon=400)
    assert f['actual_actions']==6400 and f['motor_queries']==16*13
    assert sum(row[2]==16 for row in b.events if row[0]=='reserve')==24


def portable_manifest(p,mode):
    metadata={}
    for part,obj,n,actions in (('train','ad',600,1003790),('validation','behavior_cloning',6,10180)):
        metadata[part+'_metadata']=dict(completed=True,partition=part,selected_objective=obj,datasets={obj:{}},
            interface=p['interface'],source_method='sac',source_recovery_sha256=ev.SOURCE_TEACHER_SHA,
            teacher_recovery_sha256=None,teacher_method=None,index_sha256=p[part+'_index_sha256'],
            validation_iteration=100 if part=='validation' else None,source_trials=[{} for _ in range(n)],
            case_ids=[f'{part}-{i}' for i in range(n)],counts=dict(complete_K4_trials=n,episodes=4*n,
                selected_actions=actions,committed_source_actions=1003790,decisions=4000 if part=='train' else 40))
    contract=dict(source_commit=p['fit_commit'],mode=mode,interface=p['interface'],objective='ad',seed=0,
        validation_objective='behavior_cloning',config=asdict(NativeHistoryConfig()),
        train_index_sha256=p['train_index_sha256'],validation_index_sha256=p['validation_index_sha256'],
        budget=dict(updates=500,batch_size=8,max_window=128,eval_every=25,validation_batches=4,
            milestones=[25,100,500],max_seconds_per_invocation=600.),
        optimizer=dict(name='AdamW',learning_rate=3e-4,weight_decay=0.,max_grad_norm=1.),
        normalization='2062-feature mean/std fitted on source training only; fixed for validation/deployment',**metadata)
    return dict(update=500,mode=mode,interface=p['interface'],contract=contract,unique_training_decisions=4000)


@pytest.mark.parametrize('fault',[None,'update100','teacher','source65','objective','index','budget','best_file','model_config'])
def test_ad500_selector_requires_exact_source100_recipe_and_actual500_endpoint(monkeypatch,fault):
    p=plan();m=portable_manifest(p,'full')
    if fault=='update100':m['update']=100
    elif fault=='teacher':m['contract']['train_metadata']['teacher_recovery_sha256']=ev.SOURCE_TEACHER_SHA
    elif fault=='source65':m['contract']['train_metadata']['counts']['complete_K4_trials']=390
    elif fault=='objective':m['contract']['objective']='dpt'
    elif fault=='index':m['contract']['train_index_sha256']='f'*64
    elif fault=='budget':m['contract']['budget']['updates']=100
    elif fault=='best_file':p['models']['ad-full']['path']='/synthetic/candidate.pt'
    calls=[]
    def loader(path,digest,*,device):
        calls.append((path,digest,device));policy=TokenPolicy()
        if fault=='model_config':policy.config=NativeHistoryConfig(width=8,layers=1,heads=2,context_length=8)
        return policy,m
    monkeypatch.setattr(ev,'load_policy',loader)
    if fault:
        with pytest.raises(ValueError):ev.load_agent(p,'ad-full-k1-reset',device='cpu')
        if fault=='best_file':assert not calls
    else:
        a=ev.load_agent(p,'ad-full-k1-reset',device='cpu')
        assert a.controller.erase_each_episode and a.deterministic
        assert calls==[(p['models']['ad-full']['path'],p['models']['ad-full']['sha256'],'cpu')]


def test_k1_full_unit_allowance_is_checked_before_agent_or_runtime(tmp_path,monkeypatch):
    p=plan();args=arguments(tmp_path);args.condition='ad-full-k1';args.unit_index=8
    args.native_result=tmp_path/'native.json';live.write(args.native_result,dict(synthetic=True))
    args.native_result_sha=live.sha(args.native_result);b=Budget(p);b.actual=51200-6399
    @contextmanager
    def ledger(*unused):yield b
    monkeypatch.setattr(live,'ledger',ledger)
    def forbidden(*unused,**kwargs):raise AssertionError('Agent/runtime must not load')
    with pytest.raises(RuntimeError,match='complete K0/K1 unit allowance'):
        live.run_child(args,p,prepare=forbidden,agent_loader=forbidden)


def test_child_keeps_profile_parent_identity_and_both_native_and_support_hashes(tmp_path,monkeypatch):
    args=arguments(tmp_path);args.profile='ad500-source100'
    monkeypatch.setattr(live,'process_start_ticks',lambda pid:123456)
    native=({},'/synthetic/native.json','a'*64,({},'/synthetic/support.json','b'*64))
    command=live.child_command(args,10,'ad-full-k1-reset','adjust_bottle',tmp_path/'unit',native)
    assert command[command.index('--profile')+1]=='ad500-source100'
    assert command[command.index('--support-result')+1]==native[3][1]
    assert command[command.index('--support-result-sha')+1]==native[3][2]
    assert command[command.index('--native-result-sha')+1]==native[2]
    assert command[command.index('--parent-start-ticks')+1]=='123456'


@pytest.mark.parametrize('drift',[False,True])
def test_cpu_admission_requires_equal_source_normalization_and_opens_no_cuda(monkeypatch,drift):
    p=plan();calls=[]
    def loader(configuration,condition,*,device):
        calls.append(device);a=agent(configuration,condition)
        if drift and condition=='ad-static-k0':a.policy.observation_mean.add_(1.)
        return a
    monkeypatch.setattr(ev,'load_agent',loader)
    if drift:
        with pytest.raises(ValueError,match='normalization differs'):ev.admit_plan(p)
    else:
        value=ev.admit_plan(p)
        assert value['model_endpoint']==500 and value['additional_robot_actions']==0 and len(value['checked'])==3
    assert set(calls)=={'cpu'}


def test_failed_native_step_retains_reserved_intent_and_never_returns_completed_result(tmp_path):
    p=plan();b=Budget(p);env=ReadoutEnv(p,b,fail=True)
    with pytest.raises(RuntimeError,match='fault after reservation'):run(tmp_path,p,b,'native',env=env)
    assert len(b.events)==1 and b.events[0][0]=='reserve'
    assert (tmp_path/'native/intent.json').exists() and not (tmp_path/'native/result.json').exists()


class Worker:
    def __init__(self,p,args):self.p=p;self.args=args;self.b=Budget(p);self.calls=[]
    def __call__(self,index,condition,task,unit,native,timeout):
        self.calls.append((condition,task));env=ReadoutEnv(self.p,self.b)
        value=ev.run_trial(env,self.b,agent(self.p,condition),self.p,condition=condition,task=task,
            output=unit/'trial',run_id='CPU-FAKE',references=None if native is None else native[0],
            support_references=native[3][0] if native is not None and len(native)==4 else None)
        receipt=unit/'unit-receipt.json';live.write(receipt,dict(completed=True,condition=condition,task=task,
            backend_sha256=live.BACKEND_SHA,parity=dict(passed=True),result_sha256=live.sha(unit/'trial/result.json')))
        return dict(receipt_path=str(receipt),receipt_sha256=live.sha(receipt))


def test_twelve_fresh_workers_run_128_isolated_episodes_and_resume_only_complete_units(tmp_path,monkeypatch):
    p=plan();args=arguments(tmp_path/'first');args.plan_sha='f'*64;args.wall_seconds=600.
    assert len(live.unit_order(p))==12
    assert sum(ev.unit_episode_count(p,c)*400 for c,t in live.unit_order(p))==51200
    worker=Worker(p,args);clock=[0.];monkeypatch.setattr(live.time,'monotonic',lambda:clock[0])
    def bounded(*values):
        record=worker(*values)
        if len(worker.calls)==10:clock[0]=500.
        return record
    summary=live.run_units(args,p,worker=bounded,audit=worker.b.snapshot,started=0.,source_admission=admission(p))
    assert summary['status']=='wall_boundary' and summary['completed_unit_count']==10 and worker.b.actual==96
    resumed=arguments(tmp_path/'second');resumed.plan_sha=args.plan_sha;resumed.prior_gpu_seconds=500.
    resumed.resume_manifest=args.output/'summary.json';resumed.resume_sha=live.sha(resumed.resume_manifest);clock[0]=0.
    final=live.run_units(resumed,p,worker=worker,audit=worker.b.snapshot,started=0.,source_admission=admission(p))
    assert final['status']=='complete' and len(worker.calls)==12 and worker.b.actual==128
    assert worker.calls[-2:]==[('ad-full-k1-reset',t) for t in ev.TASKS]


@pytest.mark.parametrize('field',['episode_clock','action_seed','trial_index'])
def test_resumed_result_cannot_relabel_clocks_streams_or_trial_isolation(tmp_path,field):
    p=plan();b=Budget(p);native=run(tmp_path,p,b,'native')
    value=run(tmp_path,p,b,'ad-full-k0',native=native);value['episodes'][1][field]+=1
    with pytest.raises(ValueError,match='order, clocks, action stream'):
        ev.verify_ad500_result(value,p,'ad-full-k0','adjust_bottle',native=native,support=None,admission=admission(p))
