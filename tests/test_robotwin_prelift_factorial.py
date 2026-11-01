"""Prospective dose/grip delta only; execute on allocated Linux CPU."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_robotwin_prelift_probe import source04,Task
from fluffy_bassoon.meta_rl.checkpoint import rng_state,restore_rng
from fluffy_bassoon.meta_rl.robotwin_event_schedule import scheduled_request
from fluffy_bassoon.meta_rl.robotwin_prelift_probe import (
    FACTORIAL_FORMAT,FACTORIAL_CASES,FACTORIAL_PRIORITY,PreliftControl,probe_schedule,
    probe_request,run_factorial,compare_factorial,sha_json,
)
from fluffy_bassoon.meta_rl.robotwin_prelift_probe_run import factorial_specs,validate_probe_allocation
from fluffy_bassoon.meta_rl import robotwin_prelift_probe_run as driver


@pytest.mark.parametrize('dose,slots,minimum',[(1,256,230),(5,384,358)])
def test_both_doses_keep_all_original_frames_and_fit_native400(dose,slots,minimum):
    targets=source04();plan=probe_schedule(targets,dose,protocol=FACTORIAL_FORMAT)
    source=[e for e in plan['entries'] if e['kind']=='source'];probes=[e for e in plan['entries'] if e['kind']=='probe']
    assert len(source)==7 and len(probes)==dose and plan['full_slots_with_probe']==slots
    assert plan['minimum_slots_to_issue_every_original_frame']==minimum<=400
    assert [e['probe_repetition_index'] for e in probes]==list(range(dose))
    frames=[]
    for entry in source:
        request,_,new=scheduled_request(targets,plan['source_schedule'],entry['source_chunk']);frames.append(request[new])
    assert np.array_equal(np.concatenate(frames),targets)


@pytest.mark.parametrize('dose',[0,2,4,8,True])
def test_only_predeclared_integer_doses_admitted(dose):
    with pytest.raises(ValueError):probe_schedule(source04(),dose,protocol=FACTORIAL_FORMAT)


@pytest.mark.parametrize('condition',['hold_current','close_endpoint'])
def test_five_holds_capture_anchor_once_preserve_source_cursor_and_close_naturally(tmp_path,monkeypatch,condition):
    import fluffy_bassoon.meta_rl.robotwin_event_replay as base
    monkeypatch.setattr(base,'physical_audit',lambda task:{'synthetic':True})
    class Head:
        def config(self):return {'synthetic':True}
    class Env:
        native_horizon=32;active=False;elapsed_steps=0;contract={}
        def __init__(self):
            self.backend=SimpleNamespace(model=SimpleNamespace(device=torch.device('cpu')),
                _kv_cache=((torch.zeros(1,1,1,2),torch.zeros(1,1,1,2)),),
                env=SimpleNamespace(venv=SimpleNamespace(envs=[SimpleNamespace(task=Task())])),
                _raw_obs={'states':np.ones((1,14),np.float32)})
            self.requests=[];self.metadata=[]
        def reset(self,*args,**kwargs):
            self.active=True;self.elapsed_steps=0;self.last_reset_info={'corpus_episode_id':'factorial'}
            return np.zeros(2062,np.float32)
        def record_decision_metadata(self,value):self.metadata.append(deepcopy(value))
        def step(self,noise):
            request=self.backend.action_residual(torch.zeros(1,32,14),torch.zeros(1,2062)).actions[0].detach().numpy()
            self.requests.append(request.copy());n=min(32,400-self.elapsed_steps);self.elapsed_steps+=n
            current=request[-1].copy()
            if len(self.requests)==3:current[6]=.321608
            if 4<=len(self.requests)<=8:current[0]+=.05  # Must NOT recapture this changing anchor.
            self.backend._raw_obs['states']=current[None];self.active=self.elapsed_steps<400
            relative=f'episodes/factorial/chunks/{len(self.requests):06d}.json';p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(dict(observation={'content_sha256':'a'},next_observation={'content_sha256':'b'},
                decision_arrays={'arrays':{'issued_motor_joint_targets':{'sha256':'c'},'sampled_noise':{'sha256':'n'}}},
                decision_metadata={'capacity_motor_rng_before':{'synthetic':'before'},'capacity_motor_rng_after':{'synthetic':'after'}},
                native_result={'arrays':{'arrays':{'valid_action_mask':{'sha256':'d'}}}})))
            return np.zeros(2062,np.float32),0,False,not self.active,dict(primitive_steps=n,issued_primitive_steps=n,corpus_chunk=relative)
    env=Env();control=PreliftControl(env,{'kv':Head(),'action':Head()},SimpleNamespace(root=tmp_path,register_runtime=lambda c:None),
        {'kv_layers':1,'kv_heads':1,'head_dim':2})
    targets=source04();control.select_probe(targets,{'sha256':'f'*64},condition,probe_holds=5,protocol=FACTORIAL_FORMAT)
    control.set_coefficients('action',[0.]*4);control.reset('place_phone_stand',4,motor_seed=20261205)
    for _ in range(3):control.step([0.]*4)
    assert control.cursor['source_cursor']==70 and control.cursor['physical_slots']==96
    for _ in range(5):
        control.step([0.]*4);assert control.cursor['source_cursor']==70
    assert control.cursor['physical_slots']==256 and len(control.prelude)==3
    assert all(np.array_equal(x,env.requests[3]) for x in env.requests[3:8])
    assert (env.requests[3][:,6]==(0 if condition=='close_endpoint' else np.float32(.321608))).all()
    assert [m['prelift_probe']['entry']['probe_repetition_index'] for m in env.metadata[3:8]]==list(range(5))
    control.step([0.]*4);assert np.array_equal(env.requests[8],scheduled_request(targets,control.probe_plan['source_schedule'],3)[0])
    while control.active:control.step([0.]*4)
    assert control.elapsed_steps==400 and control.cursor['source_cursor']==140 and control.cursor['closed']
    assert len(env.requests)==13 and control.cursor['chunk_index']==13


def case_specs():
    return [dict(id=f'dose{n}-{c}',case_key=f'dose{n}-{c}',probe_holds=n,condition=c) for n,c in FACTORIAL_CASES]


def test_all_four_fresh_forks_run_even_after_success_and_qualify_only_once():
    old=rng_state()
    try:
        fork=rng_state();draws=[];flags=[]
        def episode(spec,*,qualify):
            draws.append((float(np.random.random()),torch.rand(1).clone()));flags.append(qualify)
            return dict(complete=True,success=True,spec=spec)
        records=run_factorial(case_specs(),episode,fork)
        assert len(records)==4 and flags==[True,False,False,False]
        assert all(a==draws[0][0] and torch.equal(b,draws[0][1]) for a,b in draws)
    finally:restore_rng(old)


def test_integrity_failure_halts_without_reordering_or_retry():
    calls=[]
    def episode(spec,*,qualify):
        calls.append(spec['case_key'])
        if len(calls)==2:raise RuntimeError('native integrity')
        return dict(complete=True,success=False,spec=spec)
    with pytest.raises(RuntimeError,match='native integrity'):
        run_factorial(case_specs(),episode,{},restore=lambda x:None,verify=lambda x:None)
    assert calls==['dose1-hold_current','dose1-close_endpoint']


def records(successes):
    return [dict(spec=s,complete=True,success=success,prelude_signatures=[{'exact':i} for i in range(3)],
        probe_target14=probe_request(np.full(14,.32,np.float32),0.,s['condition']).tolist(),
        episode_manifest_ref={'path':s['id'],'sha256':'a'*64}) for s,success in zip(case_specs(),successes)]


@pytest.mark.parametrize('successes,selected',[
    ((False,False,False,False),None),((True,True,True,True),'dose1-close_endpoint'),
    ((True,False,True,True),'dose1-hold_current'),((False,False,True,True),'dose5-close_endpoint'),
    ((False,False,True,False),'dose5-hold_current')])
def test_after_four_priority_smallest_dose_then_close_retains_every_outcome(successes,selected):
    values=records(successes);original=deepcopy(values)
    comparison,candidate=compare_factorial(values,list(FACTORIAL_PRIORITY))
    assert values==original and comparison['recipe_comparison_eligible']
    assert (None if candidate is None else candidate['case_key'])==selected
    if candidate:assert candidate['admitted'] is False


def test_cross_dose_prelude_difference_or_missing_closed_case_prevents_recipe_claim():
    values=records((False,False,True,True));values[2]['prelude_signatures'][0]['exact']='different'
    comparison,_=compare_factorial(values,list(FACTORIAL_PRIORITY));assert not comparison['recipe_comparison_eligible']
    values[3]['complete']=False
    with pytest.raises(ValueError):compare_factorial(values,list(FACTORIAL_PRIORITY))


@pytest.mark.parametrize('mutation',['overspend','actions','refund','stale_prior','cleanup','old_mode'])
def test_prospective900_clock_does_not_refill_spent336_or_reuse_B(mutation):
    profile={'format':FACTORIAL_FORMAT};allocation=dict(mode=FACTORIAL_FORMAT,run_id='fresh',capacity_action_bound=1600,
        reserved_gpu_seconds=900,scheduler_start_unix=100.,driver_deadline_unix=940.,prior_a1_gpu_seconds=2988,
        retained_previous_pair_gpu_seconds=336)
    validate_probe_allocation(allocation,profile,now=101.,run_id='fresh')
    if mutation=='overspend':allocation['reserved_gpu_seconds']=901
    if mutation=='actions':allocation['capacity_action_bound']=1601
    if mutation=='refund':allocation['retained_previous_pair_gpu_seconds']=0
    if mutation=='stale_prior':allocation['prior_a1_gpu_seconds']=2652
    if mutation=='cleanup':allocation['driver_deadline_unix']=941.
    if mutation=='old_mode':allocation['mode']='robotwin_phone04_prelift_closure_pair_v1'
    with pytest.raises(ValueError):validate_probe_allocation(allocation,profile,now=101.,run_id='fresh')


@pytest.mark.parametrize('mutation',['order','dose','hash','budget'])
def test_four_explicit_case_profiles_and_hashes_cannot_silently_change(mutation):
    path=Path(__file__).resolve().parents[1]/'outputs/archive/robotwin-prelift-dose-grip-20261101-r1/PROFILE.json'
    profile=json.loads(path.read_text());row=dict(id='phone-04',canonical_index=4,reset_seed=4,instruction='Take the black phone to the green phone rack.')
    result=factorial_specs(profile,row);assert [(s['probe_holds'],s['condition']) for s in result]==list(FACTORIAL_CASES)
    if mutation=='order':profile['case_profiles'].reverse()
    if mutation=='dose':profile['case_profiles'][0]['profile']['probe_holds']=5
    if mutation=='hash':profile['case_profiles'][0]['sha256']='0'*64
    if mutation=='budget':profile['maximum_gpu_seconds']=1200
    with pytest.raises(ValueError):factorial_specs(profile,row)


@pytest.fixture(scope='module')
def closed_ledger_fixture(tmp_path_factory):
    """Synthetic metadata only; never open the actual campaign or any ledger class."""
    phase=tmp_path_factory.mktemp('factorial-closed');root=phase/'ledger';events=root/'events';events.mkdir(parents=True)
    for i in range(1483):(events/f'{i:010d}.json').write_text('{}')
    (events/'0000000000.json').write_text(json.dumps({'header':{'synthetic':'original'}}))
    (events/'0000001482.json').write_text(json.dumps({'sha256':driver.PAIR_TIP}))
    return phase,root


def current_predecessor_fixture(closed_ledger_fixture):
    phase,root=closed_ledger_fixture;ledger={'root':str(root),'synthetic':'same original'}
    old=dict(actual_actions=22322,charged_actions=22322,journal_events=1483,journal_sha256=driver.PAIR_TIP,
        pending_actions=0,uncertain_actions=0,unresolved_decisions={},runs={},header={'synthetic':'original'})
    head=dict(actions=old,action_ledger=ledger,format=driver.FORMAT,source_commit='374ee8958f7cabcd54d4401057a604c6cc6ae51a',
        status='complete_diagnostic_pair',pending=None,cache_created=False,fit_updates=0,source_student_admitted=False,
        all_twelve_teacher_gate_passed=False,frozen_after=driver.FROZEN,bank_candidate_04=None,
        records=[dict(spec={'condition':c},complete=True,success=False,primitive_steps=400) for c in ['hold_current','close_endpoint']])
    terminal=dict(format='frozen_pi05_prelift_terminal_independent_v1',job_id='job-i5dem442uskn',scheduler_status='Succeeded',
        source_commit=head['source_commit'],exit_code=0,result={'sha256':driver.PAIR_RESULT_SHA},full_recovery={'sha256':driver.PAIR_RECOVERY_SHA},
        canonical_snapshot=old,canonical_ledger_contract=ledger,cumulative_phase_GPU_seconds=2988,whole_pair_GPU_seconds=336)
    context=dict(format='robotwin_prelift_factorial_execution_context_v1',accounting_result={'sha256':driver.PAIR_RESULT_SHA},
        accounting_terminal={'sha256':driver.PAIR_TERMINAL_SHA},accounting_recovery={'sha256':driver.PAIR_RECOVERY_SHA},
        prior_a1_gpu_seconds=2988,prior_phase_gpu_seconds=2988,retained_previous_pair_gpu_seconds=336)
    allocation=dict(action_ledger=ledger,phase_output_root=str(phase),prior_a1_gpu_seconds=2988,
        prior_phase_gpu_seconds=2988,retained_previous_pair_gpu_seconds=336)
    return context,head,allocation,terminal


def test_current_closed_E_metadata_admitted_without_constructing_or_mutating_ledger(closed_ledger_fixture,monkeypatch):
    context,head,allocation,terminal=current_predecessor_fixture(closed_ledger_fixture)
    monkeypatch.setattr(driver,'bound_json',lambda ref:terminal if ref['sha256']==driver.PAIR_TERMINAL_SHA else head)
    driver.validate_factorial_predecessor(context,head,allocation,run_id='fresh-factorial')
    assert len(list((closed_ledger_fixture[1]/'events').iterdir()))==1483


@pytest.mark.parametrize('mutation',['old_B','refund','pending','incomplete','old_cost'])
def test_old_B_or_edited_closed_E_cannot_replace_latest_fork(closed_ledger_fixture,monkeypatch,mutation):
    context,head,allocation,terminal=current_predecessor_fixture(closed_ledger_fixture)
    monkeypatch.setattr(driver,'bound_json',lambda ref:terminal if ref['sha256']==driver.PAIR_TERMINAL_SHA else head)
    if mutation=='old_B':context['accounting_result']['sha256']=driver.SUFFIX_RESULT_SHA
    if mutation=='refund':context['retained_previous_pair_gpu_seconds']=0
    if mutation=='pending':head['pending']={'live':True}
    if mutation=='incomplete':head['records'][-1]['complete']=False
    if mutation=='old_cost':allocation['prior_a1_gpu_seconds']=2652
    with pytest.raises(ValueError):driver.validate_factorial_predecessor(context,head,allocation,run_id='fresh-factorial')
