"""Allocated-CPU synthetic recovery/contract tests, never real fit qualification."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import robotwin_reactive_curve as curve
from fluffy_bassoon.meta_rl.warmstart import atomic_save


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__();self.offset=torch.nn.Parameter(torch.zeros(()))
    def config(self):return dict(kind='synthetic_recovery_fixture')
    def forward(self,base,features):return SimpleNamespace(actions=base+self.offset)


class Motor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(()),requires_grad=False)
        self.eval();self.device=torch.device('cpu')
    def _sample_actions_from_prefix_cache(self,observation,mask,cache,noise):return noise*0
    def decode_actions(self,targets,state):return targets[...,:14]


class Backend:
    def __init__(self):
        self.model=Motor();self.env=None;self.contract=dict(norm_stats_sha256=curve.NORM_SHA)
    def _check_frozen(self):assert not self.model.weight.requires_grad and self.model.weight.grad is None
    def frozen_fingerprint(self):return curve.bank_api.FROZEN


def records(tmp_path):
    rows=[]
    for i,(split,n,target) in enumerate([('train',2,1.),('train',6,2.),('validation',3,1.5)]):
        data=dict(state=torch.zeros(1,14),features=torch.zeros(1,2062),mask=torch.ones(1,1,dtype=torch.bool),
            cache=((torch.zeros(1,1,1,256),torch.zeros(1,1,1,256)),),
            target32=torch.zeros(1,32,32),teacher14=torch.full((1,32,14),target),
            base14=torch.zeros(1,32,14),valid=(torch.arange(32)<n)[None])
        p=tmp_path/f'row-{i}.pt';sha=atomic_save(p,dict(data=data,proposal_noise=torch.zeros(32,32)))
        rows.append(dict(path=str(p),sha256=sha,split=split,valid_actions=n))
    return rows


def small_config():
    return dict(seed=17,updates=3,global_batch=2,learning_rate=.01,milestones=[0,1,3])


def recovery(output):return json.loads((output/'progress.json').read_text())['recovery']


@pytest.mark.parametrize('key,value',[
    ('updates',100),('milestones',[0,100,500]),('primary_endpoint',100),
    ('selection','best_validation'),('global_batch',4),('wide_contract_sha256','a'*64)])
def test_public_profile_refuses_endpoint_selection_or_schedule_changes(key,value):
    p=deepcopy(curve.PROFILE);p[key]=value
    with pytest.raises(ValueError):curve.validate_profile(p)


@pytest.mark.parametrize('fault',['partial','ineligible','resource_claim','missing_id','changed_split','duplicate'])
def test_fixed12_gate_refuses_partial_or_corrupt_join(monkeypatch,fault):
    monkeypatch.setattr(curve.bank_api,'validate_bank',lambda bank,complete:None)
    d=dict(format='robotwin_fixed12_complete_cache_v1',complete=True,fixed12_fit_eligible=True,
        resource_fit_admitted=False,bank={},records=[dict(context=dict(teacher_id=k,split=v),anchor=0,
        valid_actions=32,native_actual_actions=7) for k,v in curve.PROFILE['whole_split'].items()])
    if fault=='partial':d['complete']=False
    if fault=='ineligible':d['fixed12_fit_eligible']=False
    if fault=='resource_claim':d['resource_fit_admitted']=True
    if fault=='missing_id':d['records'].pop(4)
    if fault=='changed_split':d['records'][0]['context']['split']='validation'
    if fault=='duplicate':d['records'].append(deepcopy(d['records'][0]))
    with pytest.raises(ValueError):curve.fit_records(d)


def test_whole_case_mapping_keeps_full_plan_and_native_cost_separate(monkeypatch):
    monkeypatch.setattr(curve.bank_api,'validate_bank',lambda bank,complete:None)
    source=[dict(context=dict(teacher_id=k,split=v),anchor=0,valid_actions=32,native_actual_actions=7)
            for k,v in curve.PROFILE['whole_split'].items()]
    out=curve.fit_records(dict(format='robotwin_fixed12_complete_cache_v1',complete=True,
        fixed12_fit_eligible=True,resource_fit_admitted=False,bank={},records=list(reversed(source))))
    assert [r['context']['teacher_id'] for r in out]==list(curve.PROFILE['whole_split'])
    assert sum(r['split']=='train' for r in out)==8 and sum(r['split']=='validation' for r in out)==4
    assert all(r['valid_actions']==32 and r['native_actual_actions']==7 for r in out)
    assert all('split' not in r for r in source)


def test_synthetic_action_curve_has_fixed_final_not_best_validation(tmp_path):
    rows=records(tmp_path);backend=Backend();head=TinyHead()
    state=curve._fit('action',head,backend,rows,tmp_path/'fit',guard=lambda:None,
                     config=small_config(),contract={'fixture':True})
    assert state['iteration']==3 and state['complete']
    assert set(state['milestones'])=={'0','1','3'}
    assert state['primary_endpoint']==state['milestones']['3']['policy']
    assert 'best' not in state and head.offset.item()!=0
    for mark in state['milestones'].values():
        assert mark['diagnostics']['train']['valid_submitted_slots']==8
        assert mark['diagnostics']['validation']['valid_submitted_slots']==3
        assert curve.file_sha(mark['policy']['path'])==mark['policy']['sha256']
    saved=torch.load(recovery(tmp_path/'fit')['path'],weights_only=False)
    assert saved['optimizer']['state'] and saved['state']['iteration']==3
    assert set(saved['global_rng'])=={'python','numpy','torch','cuda'}


@pytest.mark.parametrize('kind',['action','kv'])
def test_resume_is_exact_complete_update_with_optimizer_sampler_and_rng(tmp_path,monkeypatch,kind):
    rows=records(tmp_path);backend=Backend();config=small_config()
    # KV recovery fixture exercises private NumPy/Torch corruption streams;
    # it does not stand in for a real frozen suffix or physical fit.
    if kind=='kv':
        def loss(model,head,batch,*,rng,generator):
            target=float(rng.random())+torch.rand((),generator=generator)
            return (head.offset-target).square()
        monkeypatch.setattr(curve,'reactive_flow_loss',loss)
        monkeypatch.setattr(curve,'evaluate',lambda *a,**k:{'MSE':0.,'valid_submitted_slots':1})
    full=tmp_path/'full';curve._fit(kind,TinyHead(),backend,rows,full,guard=lambda:None,config=config,contract={})
    partial=tmp_path/'partial'
    def stop_after_two():
        if (partial/'progress.json').exists() and json.loads((partial/'progress.json').read_text())['iteration']==2:
            raise TimeoutError('synthetic allocated-slice boundary')
    with pytest.raises(TimeoutError):
        curve._fit(kind,TinyHead(),backend,rows,partial,guard=stop_after_two,config=config,contract={})
    old=recovery(partial);assert old['iteration']==2
    resumed=tmp_path/'resumed'
    curve._fit(kind,TinyHead(),backend,rows,resumed,guard=lambda:None,config=config,contract={},resume=old)
    a=torch.load(recovery(full)['path'],weights_only=False);b=torch.load(recovery(resumed)['path'],weights_only=False)
    assert torch.equal(a['policy']['offset'],b['policy']['offset'])
    assert a['sampler_numpy']==b['sampler_numpy'] and torch.equal(a['noise_torch'],b['noise_torch'])
    assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
    for key in a['optimizer']['state'][0]:assert torch.equal(a['optimizer']['state'][0][key],b['optimizer']['state'][0][key])
    assert curve.file_sha(old['path'])==old['sha256']


@pytest.mark.parametrize('fault',['hash','contract','milestone'])
def test_resume_refuses_changed_recovery_or_retained_milestone(tmp_path,fault):
    rows=records(tmp_path);backend=Backend();out=tmp_path/'partial'
    def stop():
        if (out/'progress.json').exists():raise TimeoutError()
    with pytest.raises(TimeoutError):curve._fit('action',TinyHead(),backend,rows,out,guard=stop,config=small_config(),contract={})
    ref=recovery(out);contract={}
    if fault=='hash':ref['sha256']='0'*64
    if fault=='contract':contract={'changed':True}
    if fault=='milestone':
        saved=torch.load(ref['path'],weights_only=False)
        Path(saved['state']['milestones']['0']['policy']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        curve._fit('action',TinyHead(),backend,rows,tmp_path/'resume',guard=lambda:None,
                   config=small_config(),contract=contract,resume=ref)


def test_interrupted_recovery_publication_preserves_previous_generation(tmp_path,monkeypatch):
    head=TinyHead();optimizer=torch.optim.AdamW(head.parameters());rng=np.random.default_rng(1)
    generator=torch.Generator().manual_seed(2);state=dict(iteration=0,milestones={})
    old=curve.publish(tmp_path,state,head,optimizer,rng,generator)
    old_pointer=(tmp_path/'progress.json').read_bytes()
    monkeypatch.setattr(curve,'_atomic_json',lambda *a,**k:(_ for _ in ()).throw(OSError('publish window')))
    state['iteration']=1
    with pytest.raises(OSError):curve.publish(tmp_path,state,head,optimizer,rng,generator)
    assert (tmp_path/'progress.json').read_bytes()==old_pointer
    assert curve.file_sha(old['path'])==old['sha256'] and len(list((tmp_path/'recoveries').glob('*.pt')))==2


def test_actual_parity_checks_recorded_noise_and_raw_base_without_rng_draw(tmp_path):
    rows=records(tmp_path);before=torch.get_rng_state().clone();backend=Backend()
    result=curve.parity_check('action',TinyHead(),backend,rows,guard=lambda:None)
    assert result==dict(actual_cached_rows=3,all_native_base_bits_equal=True,new_robot_actions=0)
    assert torch.equal(before,torch.get_rng_state()) and backend.env is None
    packet=torch.load(rows[0]['path'],weights_only=False);packet['data']['base14'][0,0,0]=1.
    rows[0]['sha256']=atomic_save(Path(rows[0]['path']),packet)
    with pytest.raises(ValueError):curve.parity_check('action',TinyHead(),backend,rows,guard=lambda:None)
    assert torch.equal(before,torch.get_rng_state())


def ticket(tmp_path):
    t=dict(format=curve.FORMAT,admitted=True,source_commit='a'*40,request_sha256='b'*64,kind='action',
        hostname=curve.socket.gethostname(),gpu_count=1,private_motor_context=True,maximum_robot_actions=0,
        output=str(tmp_path/'reactive-curve-v1'/'action'/'fit'),phase_output_root=str(tmp_path),clock_contract={'phase':'prospective_explicit_continuation'},resource_attempt='attempt',
        driver_deadline_unix=250,authority_receipt={'path':'authority','sha256':'c'*64},
        resource_plan_receipt={'path':'plan','sha256':'d'*64})
    c=dict(contract=t['clock_contract'],worker_seconds=400,attempts=[dict(path='attempt',started_unix=100,
        reserved_seconds=300,status='started')])
    return t,c


@pytest.mark.parametrize('fault',['not_admitted','wrong_host','expired','changed_clock','missing_attempt','over_cap',
                                  'scope','no_plan','eight_hours'])
def test_parent_admission_rejects_stale_or_unfunded_work(tmp_path,fault):
    t,c=ticket(tmp_path)
    if fault=='not_admitted':t['admitted']=False
    if fault=='wrong_host':t['hostname']='somewhere'
    if fault=='expired':t['driver_deadline_unix']=100
    if fault=='changed_clock':c['contract']={}
    if fault=='missing_attempt':c['attempts']=[]
    if fault=='over_cap':c['worker_seconds']=200
    if fault=='scope':t['maximum_robot_actions']=1
    if fault=='no_plan':t['resource_plan_receipt']=None
    if fault=='eight_hours':c['attempts'][0]['reserved_seconds']=8*3600
    with pytest.raises(ValueError):curve.validate_ticket(t,c,source='a'*40,request_sha='b'*64,kind='action',output=tmp_path/'reactive-curve-v1'/'action'/'fit',now=150)


def test_parent_existing_reservation_is_read_only(tmp_path):
    t,c=ticket(tmp_path);before=deepcopy(c)
    curve.validate_ticket(t,c,source='a'*40,request_sha='b'*64,kind='action',output=tmp_path/'reactive-curve-v1'/'action'/'fit',now=150)
    assert c==before


@pytest.mark.parametrize('fault',['shared_context','physics','train_mode','motor_grad','fingerprint','normalization'])
def test_public_fit_refuses_unqualified_or_shared_motor_before_cache_or_output(tmp_path,monkeypatch,fault):
    b=Backend()
    if fault=='shared_context':b._reactive_curve_active=True
    if fault=='physics':b.env=object()
    if fault=='train_mode':b.model.train()
    if fault=='motor_grad':b.model.weight.requires_grad_(True)
    if fault=='fingerprint':b.frozen_fingerprint=lambda:'bad'
    if fault=='normalization':b.contract['norm_stats_sha256']='bad'
    monkeypatch.setattr(curve.bank_api,'seal_full_cache',lambda *a,**k:pytest.fail('must refuse before data work'))
    with pytest.raises(ValueError):
        curve.fit_curve('action',b,{},[],{},tmp_path/'never-created',guard=lambda:None,deadline=1e20,
                        contract=dict(profile=curve.PROFILE))
    assert not (tmp_path/'never-created').exists()


def test_public_fit_requires_real_full_cache_seal_before_constructing_head(tmp_path,monkeypatch):
    def reject(*a,**k):raise ValueError('No actual admitted full12 bank')
    monkeypatch.setattr(curve.bank_api,'seal_full_cache',reject)
    monkeypatch.setattr(curve,'make_head',lambda *a,**k:pytest.fail('no head without actual data gate'))
    with pytest.raises(ValueError,match='full12'):
        curve.fit_curve('action',Backend(),{},[],{},tmp_path/'never-created',guard=lambda:None,deadline=1e20,
                        contract=dict(profile=curve.PROFILE))
    assert not (tmp_path/'never-created').exists()


def test_loaded_runtime_outside_declared_git_root_is_rejected(tmp_path):
    with pytest.raises(ValueError):curve.validate_runtime_blobs({},tmp_path)


def test_two_backend_wrappers_cannot_share_one_motor_during_fit(tmp_path):
    b=Backend();identity=id(b.model);curve._MOTOR_CONTEXTS.add(identity)
    try:
        with pytest.raises(ValueError,match='already fitting'):
            curve.fit_curve('kv',b,{},[],{},tmp_path/'fit',guard=lambda:None,deadline=1e20,contract={})
    finally:curve._MOTOR_CONTEXTS.remove(identity)
