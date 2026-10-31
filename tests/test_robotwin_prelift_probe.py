"""Targeted synthetic tests; run on allocated Linux CPU, never macOS."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.checkpoint import rng_state,restore_rng
from fluffy_bassoon.meta_rl.robotwin_prelift_probe import (
    CONDITIONS,PreliftHead,PreliftControl,probe_schedule,probe_request,observe_native_plans,
    run_pair,assert_restored_rng,preserve_global_rng,compare_pair,
)
from fluffy_bassoon.meta_rl.robotwin_prelift_probe_run import validate_head_kind,validate_fork_recovery,RESULT_SHA,SUFFIX_RESULT_SHA,FROZEN
from fluffy_bassoon.meta_rl.robotwin_event_schedule import scheduled_request


def source04():
    x=np.zeros((140,14),np.float32);x[:,0]=np.arange(140)/1000;x[:,[6,13]]=1
    x[56:70,6]=np.linspace(.99,0,14);x[70:120,6]=0;x[120:134,6]=np.linspace(.01,1,14)
    return x


def test_one_inserted_probe_preserves_every_source_frame_and_native_horizon():
    x=source04();plan=probe_schedule(x)
    assert len(plan['entries'])==8 and plan['entries'][3]['kind']=='probe'
    assert plan['full_slots_with_probe']==256 and plan['minimum_slots_to_issue_every_original_frame']==230
    chunks=[]
    for entry in plan['entries']:
        if entry['kind']=='source':
            request,_,new=scheduled_request(x,plan['source_schedule'],entry['source_chunk']);chunks.append(request[new])
    assert np.array_equal(np.concatenate(chunks),x)


@pytest.mark.parametrize('mutation',['length','phase','close'])
def test_reject_changed_fixed04_reference(mutation):
    x=source04()
    if mutation=='length':x=x[:-1]
    if mutation=='phase':x[56,6]=1
    if mutation=='close':x[69,6]=.01
    with pytest.raises(ValueError):probe_schedule(x)


def test_matched_hold_changes_only_one_coordinate_and_keeps_signed_bits():
    current=np.arange(14,dtype=np.float32)/20;current[0]=-0.;current[6]=.4723618;current[13]=1
    placebo=probe_request(current,0.,'hold_current');close=probe_request(current,0.,'close_endpoint')
    indices=[i for i in range(14) if i!=6]
    assert np.array_equal(placebo[:,indices].view(np.int32),close[:,indices].view(np.int32))
    assert (placebo[:,6]==current[6]).all() and (close[:,6]==0).all()
    assert placebo.shape==close.shape==(32,14)


def test_head_issues_current_hold_and_then_exact_original_next_chunk():
    x=source04();head=PreliftHead();head.targets=torch.from_numpy(x);head.plan=probe_schedule(x)
    head.condition='close_endpoint';head.execution_index=3
    target=probe_request(np.ones(14,np.float32),0.,'close_endpoint');head.probe_target=torch.from_numpy(target)
    base=torch.zeros(1,32,14);features=torch.zeros(1,2062)
    assert np.array_equal(head(base,features).actions[0].numpy(),target)
    head.execution_index=4
    expected=scheduled_request(x,head.plan['source_schedule'],3)[0]
    assert np.array_equal(head(base,features).actions[0].numpy(),expected)
    assert not head(base,features).clipped.any()


class Planner:
    def __init__(self):self.calls=0;self.result=(np.zeros(8),np.zeros((8,6)),np.zeros((8,6)),np.zeros((8,6)),.032)
    def TOPP(self,*args,**kwargs):self.calls+=1;return self.result


class Task:
    def __init__(self):self.robot=SimpleNamespace(left_mplib_planner=Planner(),right_mplib_planner=Planner());self.samples=0
    def downsample_trajectory(self,position,velocity):self.samples+=1;return position,velocity


def test_optional_topp_capture_calls_each_original_once_returns_identity_and_restores():
    task=Task();before=deepcopy(np.random.get_state())
    original=task.robot.left_mplib_planner.TOPP.__func__
    with observe_native_plans(task) as record:
        for side in ('left','right'):
            planner=getattr(task.robot,side+'_mplib_planner')
            result=planner.TOPP(np.zeros((2,6)),1/250)
            assert result is planner.result
            returned=task.downsample_trajectory(result[1],result[2])
            assert returned[0] is result[1]
    assert record['available'] and [r['output_position_steps'] for r in record['downsample']]==[8,8]
    assert task.robot.left_mplib_planner.calls==task.robot.right_mplib_planner.calls==1 and task.samples==2
    assert task.robot.left_mplib_planner.TOPP.__func__ is original
    assert 'TOPP' not in task.robot.left_mplib_planner.__dict__ and 'downsample_trajectory' not in task.__dict__
    after=np.random.get_state()
    assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]


def test_topp_exception_is_preserved_without_retry():
    class Broken(Planner):
        def TOPP(self,*args,**kwargs):self.calls+=1;raise RuntimeError('native sentinel')
    task=Task();task.robot.left_mplib_planner=Broken()
    with observe_native_plans(task) as record:
        with pytest.raises(RuntimeError,match='native sentinel'):task.robot.left_mplib_planner.TOPP(None)
    assert task.robot.left_mplib_planner.calls==1 and record['topp'][0]['status']=='raised'


def test_unshadowable_planner_marks_optional_capture_unavailable_without_calling():
    class Slots:
        __slots__=()
        def TOPP(self,*args):raise AssertionError('Observer must not invoke native planner')
    task=Task();task.robot.left_mplib_planner=Slots()
    with observe_native_plans(task) as record:assert record['available'] is False
    assert record['unavailable'] and 'TOPP' not in task.robot.right_mplib_planner.__dict__


def test_pair_runs_complete_negative_control_then_treatment_and_forks_existing_rng():
    old=rng_state()
    try:
        expected=rng_state();draws=[];qualified=[]
        def episode(spec,*,qualify):
            qualified.append(qualify);draws.append((float(np.random.random()),torch.rand(1).clone()))
            return {'success':False,'condition':spec['condition']}
        result=run_pair([dict(id=str(i),condition=c) for i,c in enumerate(CONDITIONS)],episode,expected)
        assert len(result)==2 and qualified==[True,False]
        assert draws[0][0]==draws[1][0] and torch.equal(draws[0][1],draws[1][1])
        restore_rng(expected);assert_restored_rng(expected)
    finally:restore_rng(old)


def test_pair_integrity_exception_stops_without_automatic_second_episode():
    calls=[]
    def episode(spec,*,qualify):calls.append(spec['condition']);raise RuntimeError('integrity sentinel')
    with pytest.raises(RuntimeError,match='integrity sentinel'):
        run_pair([dict(id=str(i),condition=c) for i,c in enumerate(CONDITIONS)],episode,{},restore=lambda s:None,verify=lambda s:None)
    assert calls==['hold_current']


def test_observational_qualification_rng_restores_even_on_exception():
    expected=rng_state()
    with pytest.raises(RuntimeError,match='qualification sentinel'):
        with preserve_global_rng():
            np.random.random();torch.rand(1);raise RuntimeError('qualification sentinel')
    assert_restored_rng(expected)


def pair_records(successes=(False,True)):
    records=[]
    for index,condition in enumerate(CONDITIONS):
        target=probe_request(np.full(14,.47,np.float32),0.,condition)
        records.append(dict(spec=dict(condition=condition,id=condition),complete=True,success=successes[index],
            probe_target14=target.tolist(),prelude_signatures=[{'exact':i} for i in range(3)],
            episode_manifest_ref={'path':condition,'sha256':'f'*64}))
    return records


@pytest.mark.parametrize('successes,selected',[
    ((False,False),None),((True,False),'hold_current'),((False,True),'close_endpoint'),((True,True),'close_endpoint')])
def test_all_counterfactual_outcomes_retained_and_nomination_priority_fixed(successes,selected):
    records=pair_records(successes);original=deepcopy(records)
    comparison,candidate=compare_pair(records,['close_endpoint','hold_current'])
    assert records==original and comparison['mechanism_interpretation_eligible']
    assert (None if candidate is None else candidate['condition'])==selected
    if candidate is not None:assert candidate['admitted'] is False


@pytest.mark.parametrize('falsifier',['degenerate','different_prelude','different_other_parameter','before_probe','incomplete'])
def test_success_cannot_bypass_mechanism_falsifiers_or_closed_pair(falsifier):
    records=pair_records((True,True))
    if falsifier=='degenerate':records[0]['probe_target14']=deepcopy(records[1]['probe_target14'])
    if falsifier=='different_prelude':records[1]['prelude_signatures'][0]['exact']='different'
    if falsifier=='different_other_parameter':records[1]['probe_target14'][0][0]+=.01
    if falsifier=='before_probe':records[0]['probe_target14']=None
    if falsifier=='incomplete':
        records[1]['complete']=False
        with pytest.raises(ValueError):compare_pair(records,['close_endpoint','hold_current'])
        return
    comparison,candidate=compare_pair(records,['close_endpoint','hold_current'])
    assert comparison['mechanism_interpretation_eligible'] is False
    assert candidate['admitted'] is False


def test_controller_consumes_added_chunk_without_advancing_source_cursor(tmp_path,monkeypatch):
    import fluffy_bassoon.meta_rl.robotwin_event_replay as base_module
    monkeypatch.setattr(base_module,'physical_audit',lambda task:{'synthetic':True})
    class Head:
        def config(self):return {'synthetic':True}
    class Env:
        native_horizon=32;active=False;elapsed_steps=0;contract={}
        def __init__(self):
            self.backend=SimpleNamespace(model=SimpleNamespace(device=torch.device('cpu')),
                _kv_cache=((torch.zeros(1,1,1,2),torch.zeros(1,1,1,2)),),env=SimpleNamespace(venv=SimpleNamespace(envs=[SimpleNamespace(task=Task())])),
                _raw_obs={'states':np.ones((1,14),np.float32)})
            self.requests=[]
        def reset(self,*args,**kwargs):
            self.active=True;self.elapsed_steps=0;self.last_reset_info={'corpus_episode_id':'probe-episode'}
            return np.zeros(2062,np.float32)
        def record_decision_metadata(self,value):self.metadata=value
        def step(self,noise):
            assert noise.shape==(32,32)
            request=self.backend.action_residual(torch.zeros(1,32,14),torch.zeros(1,2062)).actions[0].detach().numpy()
            self.requests.append(request.copy());self.elapsed_steps+=32
            current=request[-1].copy()
            if len(self.requests)==3:current[6]=.4723618
            self.backend._raw_obs['states']=current[None]
            relative=f'episodes/probe-episode/chunks/{len(self.requests):06d}.json';path=tmp_path/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(observation={'content_sha256':'a'},next_observation={'content_sha256':'b'},
                decision_arrays={'arrays':{'issued_motor_joint_targets':{'sha256':'c'},'sampled_noise':{'sha256':'n'}}},
                decision_metadata={'capacity_motor_rng_before':{'synthetic':'before'},'capacity_motor_rng_after':{'synthetic':'after'}},
                native_result={'arrays':{'arrays':{'valid_action_mask':{'sha256':'d'}}}})))
            return np.zeros(2062,np.float32),0,False,False,dict(primitive_steps=32,issued_primitive_steps=32,corpus_chunk=relative)
    env=Env();control=PreliftControl(env,{'kv':Head(),'action':Head()},SimpleNamespace(root=tmp_path,register_runtime=lambda c:None),
        {'kv_layers':1,'kv_heads':1,'head_dim':2})
    x=source04();control.select_probe(x,{'sha256':'f'*64},'close_endpoint');control.set_coefficients('action',[0.]*4)
    control.reset('place_phone_stand',4,motor_seed=20261205)
    for _ in range(3):control.step([0.]*4)
    assert control.cursor['source_cursor']==70 and len(control.prelude)==3
    control.step([0.]*4)
    assert control.cursor['source_cursor']==70 and control.cursor['physical_slots']==128 and control.cursor['probe_submitted']
    assert (env.requests[3][:,6]==0).all() and control.last_audit['native_plan_observation']['available']
    control.step([0.]*4)
    expected=scheduled_request(x,control.probe_plan['source_schedule'],3)[0]
    assert np.array_equal(env.requests[4],expected) and control.cursor['source_cursor']==102


@pytest.mark.parametrize('mutation',['running','failed','wrongcases','fit','pending'])
def test_only_actual_closed_suffix_head_can_replace_historical_accounting(mutation):
    head=dict(source_commit='aefd5563b128cb7ea8e017d4664e1df0ba22a160',
        format='original_a1_event_fixed_suffix_diagnostic_v1',status='complete_diagnostic_coverage',
        predecessor_result={'sha256':RESULT_SHA},records=[{'spec':{'teacher_id':f'phone-{i:02d}'},'complete':True} for i in range(5,12)],
        cache_created=False,fit_updates=0,source_student_admitted=False,all_twelve_teacher_gate_passed=False,frozen_after=FROZEN,pending=None)
    validate_head_kind(head,SUFFIX_RESULT_SHA)
    if mutation=='running':head['status']='running'
    if mutation=='failed':head['status']='blocked_failure'
    if mutation=='wrongcases':head['records'][0]['spec']['teacher_id']='phone-04'
    if mutation=='fit':head['fit_updates']=1
    if mutation=='pending':head['pending']={'active':True}
    with pytest.raises(ValueError):validate_head_kind(head,SUFFIX_RESULT_SHA)


@pytest.mark.parametrize('mutation',['old_control','missing_private_rng','pending','fit'])
def test_acting_rng_fork_is_latest_closed_recovery_not_historical_failure(mutation):
    control=dict(mode='action',elapsed_steps=193,motor_rng={'latest':'private'},source_cursor={'closed':True})
    head=dict(records=[{'primitive_steps':193}],closed_control_recoveries=[control])
    saved=dict(state=head,fits={},fit_recovery_refs={},control=deepcopy(control),
        global_rng={key:'latest retained' for key in ('python','numpy','torch','cuda')})
    validate_fork_recovery(saved,head)
    if mutation=='old_control':saved['control']['elapsed_steps']=400
    if mutation=='missing_private_rng':saved['control']['motor_rng']=None
    if mutation=='pending':saved['control']['source_cursor']['closed']=False
    if mutation=='fit':saved['fits']={'unadmitted':'fit'}
    with pytest.raises(ValueError):validate_fork_recovery(saved,head)
