"""Allocated-CPU deltas only: failed04 purpose, original plans, reuse and gate."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import robotwin_teacher_bank as core
from fluffy_bassoon.meta_rl import robotwin_command_diagnostic_bank as bank
from fluffy_bassoon.meta_rl import robotwin_command_diagnostic_curve as curve
from fluffy_bassoon.meta_rl import robotwin_reactive_curve as old_curve
from fluffy_bassoon.meta_rl.warmstart import atomic_save


def diagnostic_bank():
    entries=[];outcomes={}
    for i,identity in enumerate(core.IDS):
        ref=dict(path='/synthetic/'+identity,sha256=str(i%10)*64)
        row=dict(complete=True,success=(i!=4),primitive_steps=400 if i==4 else 193,chunks=13 if i==4 else 7,
            spec=dict(id='event-teacher-'+identity),episode_manifest_ref=ref)
        entry=dict(id=identity,split='train' if i<8 else 'validation',status='closed_command_diagnostic',
            record=row,plan_convention='original_event_h32_v1',producer_result=ref,producer_commit='a'*40)
        entries.append(entry);outcomes[identity]=dict(success=row['success'],primitive_steps=row['primitive_steps'],
            chunks=row['chunks'],record_id=row['spec']['id'],producer_result=ref,producer_commit='a'*40,episode_manifest_ref=ref)
    return dict(format=bank.FORMAT,diagnostic_purpose=bank.PURPOSE,source_eligible=False,fixed12_fit_eligible=False,
        source_student_admitted=False,diagnostic12_data_eligible=True,entries=entries,outcome_evidence=outcomes,
        retained_producer_results={'event':{'whole_original_failure_and_controls':True}},retained_cost_snapshots={'original':'unchanged'})


@pytest.mark.parametrize('fault',['positive04','source_promotion','student_promotion','missing04','subset','F4_labels','not_closed'])
def test_diagnostic_identity_never_promotes_failed04_or_reduces_fixed12(fault):
    value=diagnostic_bank()
    if fault=='positive04':value['entries'][4]['record']['success']=True
    if fault=='source_promotion':value['fixed12_fit_eligible']=True
    if fault=='student_promotion':value['source_student_admitted']=True
    if fault=='missing04':value['entries'].pop(4)
    if fault=='subset':value['entries']=value['entries'][:8]
    if fault=='F4_labels':value['entries'][4]['plan_convention']='prelift_factorial_explicit_anchor_v2'
    if fault=='not_closed':value['entries'][4]['record']['complete']=False
    with pytest.raises(ValueError):bank.validate_bank(value,complete=True)


def test_complete_diagnostic_cache_is_not_a_positive_bank_or_old_curve():
    value=diagnostic_bank();cache=dict(format=bank.COMPLETE_FORMAT,complete=True,bank=value,
        resource_fit_admitted=False,**bank.seal_annotations(value))
    bank.validate_complete_cache(cache)
    assert bank.validate_bank(value)==core.IDS
    with pytest.raises(ValueError):old_curve.fit_records(cache)
    with pytest.raises(ValueError):core.validate_bank(value,complete=True)
    with pytest.raises(ValueError):old_curve.validate_profile(curve.PROFILE)
    with pytest.raises(ValueError):curve.validate_profile(old_curve.PROFILE)
    assert cache['outcome_evidence']['phone-04']['success'] is False


def test_assembly_selects_original_failed04_keeps_whole_producer_and_costs(monkeypatch):
    original=diagnostic_bank();partial=deepcopy(original);partial['entries'][4]=dict(id='phone-04',split='train',status='PENDING04',record=None)
    row=deepcopy(original['entries'][4]['record']);row['spec'].update(stage='event_aligned_teacher_execution',method='event_aligned_teacher')
    event={'records':[row,dict(spec={'id':'later-F4-success'},success=True)]}
    demos={'episodes':[{'id':i} for i in core.IDS],'actual_whole_files':{'phone-04':{'path':'original04','sha256':'f'*64}}}
    class Audit:
        def json(self,ref):return demos if ref['path']=='demos' else event
    monkeypatch.setattr(core,'assemble_bank',lambda *a,**k:deepcopy(partial))
    monkeypatch.setattr(core,'_validate_record_spec',lambda actual,episode,require_success:None)
    audited=[];monkeypatch.setattr(core,'_closed_manifest',lambda actual,audit:audited.append(actual['success']))
    result=bank.assemble_bank({'path':'demos'},{},{'event':{'path':'event'}},audit=Audit())
    assert audited==[False] and result['entries'][4]['record']==row
    assert result['entries'][4]['producer_commit']==core.PRODUCERS['event'][0]
    assert result['retained_producer_results']==partial['retained_producer_results']
    assert result['retained_cost_snapshots']==partial['retained_cost_snapshots']
    assert result['source_eligible'] is False and result['fixed12_fit_eligible'] is False
    assert partial['entries'][4]['record'] is None


def test_cache_dispatch_encodes_only04_and_explicitly_keeps_failure(monkeypatch,tmp_path):
    seen={}
    def cache(*args,**kwargs):seen.update(kwargs);return {'prepared':True}
    monkeypatch.setattr(core,'cache_bank',cache)
    assert bank.cache_bank({},object(),tmp_path,deadline=1e20,guard=lambda:None,model_qualification_ref={})=={'prepared':True}
    assert seen['_only_ids']==['phone-04'] and seen['_require_success'] is False and seen['_api'] is bank


def source_targets():
    x=np.zeros((140,14),np.float32);x[:,0]=np.arange(140)/100
    x[:56,6]=1.;x[56:70,6]=np.linspace(1,0,14);x[120:134,6]=np.linspace(0,1,14);x[134:,6]=1.
    return x


def tail_packet():
    targets=source_targets();full=np.repeat(targets[-1][None],32,axis=0)
    meta=dict(teacher_reference={'sha256':'a'*64},source_cursor_before=140,physical_slots_before=384,chunk_index=12)
    chunk=dict(decision_metadata=dict(event_schedule=meta,frozen_pi05_residual=dict(
        actions={'numeric_array':'full'},base_actions={'numeric_array':'base'})),
        native_result=dict(executed_primitive_steps=16,success=False),learning_result=dict(terminated=False,truncated=True))
    decision=dict(full=full[None],base=np.zeros((1,32,14),np.float32),sampled_noise=np.zeros((32,32),np.float32),
        issued_motor_joint_targets=full.copy())
    return targets,chunk,decision,dict(valid_action_mask=(np.arange(32)<16).astype(np.uint8))


def test_failed04_tail_full_known_plan_and_native16_cost_are_separate():
    targets,chunk,decision,native=tail_packet()
    result=bank.captured_plan(chunk,decision,native,targets=targets,source={'sha256':'a'*64},
        convention='original_event_h32_v1',profile_sha=None,cursor=dict(source=140,physical=384,logical_chunk=12,execution=12))
    assert result['submitted'].sum()==32 and result['executed'].sum()==16 and not result['new'].any()
    assert result['following']['source']==140 and result['following']['physical']==400
    assert np.array_equal(result['target'],decision['full'][0]) and chunk['native_result']['success'] is False


@pytest.mark.parametrize('fault',['future_source','physical_cursor','proposal','mask'])
def test_failed04_source_or_tail_tamper_is_refused(fault):
    targets,chunk,decision,native=tail_packet()
    if fault=='future_source':chunk['decision_metadata']['event_schedule']['source_cursor_before']=139
    if fault=='physical_cursor':chunk['decision_metadata']['event_schedule']['physical_slots_before']=383
    if fault=='proposal':decision['full'][0,0,0]+=1
    if fault=='mask':native['valid_action_mask'][20]=1
    with pytest.raises(ValueError):bank.captured_plan(chunk,decision,native,targets=targets,source={'sha256':'a'*64},
        convention='original_event_h32_v1',profile_sha=None,cursor=dict(source=140,physical=384,logical_chunk=12,execution=12))


def test_explicit_new_curve_dispatch_preserves_contract_and_common_optimizer(monkeypatch,tmp_path):
    seen={};contract={'profile':curve.PROFILE,'complete_cache_sha256':'c'*64}
    def fit(*args,**kwargs):seen.update(kwargs);return 'common-loop'
    monkeypatch.setattr(old_curve,'fit_curve',fit)
    assert curve.fit_curve('kv',object(),{},[],{},tmp_path,guard=lambda:None,deadline=1e20,contract=contract)=='common-loop'
    assert seen['contract'] is contract and seen['_api'] is bank and seen['_profile']==curve.PROFILE
    assert seen['_validate'] is curve.validate_profile
    assert curve.PROFILE['updates']==5000 and curve.PROFILE['milestones']==[0,100,500,2000,5000]


def test_geometry_is_raw_command_floor_not_success_and_preserves_rng(tmp_path):
    root=Path(__file__).resolve().parents[1]
    configuration=json.loads((root/'outputs/robotwin-wide-action-head-20261101-r1/CONFIG.json').read_text())
    rows=[]
    for split in ('train','validation'):
        p=tmp_path/(split+'.pt')
        data=dict(base14=torch.zeros(1,32,14),teacher14=torch.full((1,32,14),4.),valid=torch.ones(1,32,dtype=torch.bool),
            state=torch.zeros(1,14),features=torch.zeros(1,2062),target32=torch.zeros(1,32,32),
            mask=torch.ones(1,1,dtype=torch.bool),cache=((torch.zeros(1,1,1,256),torch.zeros(1,1,1,256)),))
        rows.append(dict(path=str(p),sha256=atomic_save(p,{'data':data}),split=split))
    before=torch.get_rng_state().clone();result=bank.geometric_floor(rows,configuration,guard=lambda:None)
    expected=(12*2.**2+2*(4-float(np.float32(1.1)))**2)/14
    assert result['native_actuator_clip_applied'] is False
    assert result['partitions']['train']['attainable_command_MSE_lower_bound']==pytest.approx(expected)
    assert torch.equal(before,torch.get_rng_state())


@pytest.mark.parametrize('fault',['overcap','old_unlisted','no_storage','wrong_purpose'])
def test_diagnostic_resource_stage_accounts_prior_attempts_no_reset(fault):
    row=dict(path='current',lineage='fixed12-command-diagnostic/cache04',lineage_seconds=600,reserved_seconds=400)
    clock=dict(contract={'actual':'clock'},attempts=[row]);t=dict(resource_attempt='current',clock_root='root')
    plan=dict(diagnostic_purpose=bank.PURPOSE,source_eligible=False,admitted=True,stage_gpu_limits={'cache04':600},
        clock_contract=clock['contract'],clock_root='root',prior_cost_receipts=[{'bound':'old'}],storage_admission={'actual':True})
    if fault=='overcap':clock['attempts'].insert(0,dict(path='old',lineage=row['lineage'],lineage_seconds=600,reserved_seconds=300,charged_seconds=300))
    if fault=='old_unlisted':row['lineage_seconds']=1200
    if fault=='no_storage':plan['storage_admission']=None
    if fault=='wrong_purpose':plan['diagnostic_purpose']='successful-teacher'
    with pytest.raises(ValueError):curve.validate_resource(t,clock,operation='cache04',plan=plan)


@pytest.mark.parametrize('missing04',[False,True])
def test_explicit_diagnostic_curve_still_requires_complete12_cache_identity(missing04):
    value=diagnostic_bank();rows=[dict(context=dict(teacher_id=e['id'],split=e['split']),anchor=0,
        valid_actions=32,native_actual_actions=16 if e['id']=='phone-04' else 1) for e in value['entries']]
    if missing04:rows.pop(4)
    cache=dict(format=bank.COMPLETE_FORMAT,complete=True,bank=value,records=rows,resource_fit_admitted=False,
               **bank.seal_annotations(value))
    if missing04:
        with pytest.raises(ValueError,match='Missing fixed12'):old_curve.fit_records(cache,_api=bank)
    else:
        out=old_curve.fit_records(cache,_api=bank)
        assert len(out)==12 and out[4]['native_actual_actions']==16 and out[4]['valid_actions']==32
        assert cache['fixed12_fit_eligible'] is False and cache['source_eligible'] is False
