"""Fixed-bank/captured-plan CPU candidates; allocated Linux only."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import robotwin_teacher_bank as module
from fluffy_bassoon.meta_rl.robotwin_event_schedule import build_schedule,scheduled_request
from fluffy_bassoon.meta_rl.robotwin_demo_capacity import file_sha
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter


class MetadataAudit:
    def __init__(self,values):self.values=values
    def json(self,ref):return deepcopy(self.values[ref['path']])


def record(i,success=True,condition=None):
    spec=dict(id='event-teacher-phone-'+f'{i:02d}',teacher_id=f'phone-{i:02d}',case=i,
        task='place_phone_stand',horizon=400,reset_seed=[0,1,2,3,4,6,9,10,11,14,16,18][i],
        expected_teacher_seed=[0,1,2,3,4,6,9,10,11,14,16,18][i],motor_seed=20261201+i,
        paired_instruction=[f'original text{i}'],arm='action',coefficients=[0.]*4,
        stage='event_aligned_teacher_execution',method='event_aligned_teacher')
    if condition:
        spec.update(id='prelift04-'+condition,condition=condition,stage='prelift_closure_diagnostic',method='prelift_'+condition)
    return dict(spec=spec,complete=True,success=success,primitive_steps=193 if success else 400,chunks=7,
        reset_info=dict(actual_seed=spec['reset_seed'],instruction=spec['paired_instruction']),
        episode_manifest_ref=dict(root='/canonical/'+str(i),path='episodes/actual/manifest.json',sha256='a'*64))


def metadata_fixture(monkeypatch):
    monkeypatch.setattr(module,'validate_demo_manifest',deepcopy)
    demos=dict(episodes=[dict(id=f'phone-{i:02d}',canonical_index=i,split='train' if i<8 else 'validation',
        reset_seed=record(i)['spec']['reset_seed'],instruction=f'original text{i}') for i in range(12)],
        motor_sha256='a'*64,normalization_sha256='b'*64,
        actual_whole_files={f'phone-{i:02d}':dict(path='/source/'+str(i),sha256='c'*64) for i in range(12)})
    interface=dict(motor_sha256='a'*64,normalization_sha256='b'*64)
    actions=dict(actual_actions=20171,charged_actions=20171,pending_actions=0,uncertain_actions=0,unresolved_decisions={})
    event=dict(source_commit=module.PRODUCERS['event'][0],demos_sha256=module.DEMOS_SHA,
        frozen_motor_sha256=module.FROZEN,actions=actions,records=[record(i,i!=4) for i in range(5)])
    suffix=dict(source_commit=module.PRODUCERS['suffix'][0],frozen_after=module.FROZEN,
        actions=actions,plan=dict(predecessor_result_sha256=module.PRODUCERS['event'][1]),records=[record(i) for i in range(5,12)])
    profile=dict(format=module.E_FORMAT,demos_sha256=module.DEMOS_SHA,conditions=['hold_current','close_endpoint'],
        bank_candidate_priority=['close_endpoint','hold_current'],whole_split={r['id']:r['split'] for r in demos['episodes']},
        historical_result=dict(sha256=module.PRODUCERS['event'][1]),maximum_complete_episodes=2,teacher_id='phone-04')
    eref=dict(path='profile',sha256=module.E_PROFILE_SHA)
    e=dict(source_commit=module.E_SOURCE,format=module.E_FORMAT,profile_sha256=eref['sha256'],plan=deepcopy(profile),
        status='complete_diagnostic_pair',frozen_after=module.FROZEN,actions=actions,
        records=[record(4,False,'hold_current'),record(4,True,'close_endpoint')])
    binding=dict(format='robotwin_fixed12_accepted04_profile_v1',profile=eref,result=dict(path='e',sha256='f'*64),source_commit=module.E_SOURCE)
    values=dict(demos=demos,interface=interface,event=event,suffix=suffix,profile=profile,e=e,binding=binding)
    refs={name:dict(path=name,sha256=module.PRODUCERS[name][1]) for name in ('event','suffix')}
    args=(dict(path='demos',sha256=module.DEMOS_SHA),dict(path='interface',sha256=module.INTERFACE_SHA),refs)
    return args,values,dict(path='binding',sha256='e'*64)


def test_eleven_successes_preserve_pending04_and_never_fit(monkeypatch):
    args,values,_=metadata_fixture(monkeypatch)
    bank=module.assemble_bank(*args,audit=MetadataAudit(values))
    assert len(module.validate_bank(bank))==11 and bank['entries'][4]['status']=='PENDING04'
    assert bank['fixed12_fit_eligible'] is False
    with pytest.raises(ValueError):module.validate_bank(bank,complete=True)
    assert bank['retained_producer_results']==args[2]
    assert bank['retained_cost_snapshots']['event']==values['event']['actions']  # Costs copied, never summed/refunded.


def test_full12_requires_both_outcomes_and_predeclared_close_first_selector(monkeypatch):
    args,values,binding=metadata_fixture(monkeypatch)
    values['e']['records'][0]['success']=True
    bank=module.assemble_bank(*args,audit=MetadataAudit(values),accepted04_ref=binding)
    assert module.validate_bank(bank,complete=True)==module.IDS
    assert bank['entries'][4]['record']['spec']['condition']=='close_endpoint'
    values['e']['records'][1]['success']=False
    bank=module.assemble_bank(*args,audit=MetadataAudit(values),accepted04_ref=binding)
    assert bank['entries'][4]['record']['spec']['condition']=='hold_current'


@pytest.mark.parametrize('fault',['failed04','missing04','duplicate04','task','seed','text','split','motor','norm','profile','source','priority','pending','foreign_result'])
def test_foreign_or_unclosed_bank_and_filtered04_are_rejected(monkeypatch,fault):
    args,v,binding=metadata_fixture(monkeypatch)
    if fault=='failed04':v['e']['records'][1]['success']=False
    if fault=='missing04':v['e']['records'].pop()
    if fault=='duplicate04':v['e']['records'][1]=deepcopy(v['e']['records'][0])
    if fault=='task':v['suffix']['records'][0]['spec']['task']='easier'
    if fault=='seed':v['suffix']['records'][0]['spec']['reset_seed']+=1
    if fault=='text':v['suffix']['records'][0]['spec']['paired_instruction']=['different']
    if fault=='split':v['demos']['episodes'][8]['split']='train'
    if fault=='motor':v['interface']['motor_sha256']='c'*64
    if fault=='norm':v['interface']['normalization_sha256']='c'*64
    if fault=='profile':v['e']['profile_sha256']='wrong'
    if fault=='source':v['e']['source_commit']='0'*40
    if fault=='priority':v['profile']['bank_candidate_priority'].reverse()
    if fault=='pending':v['e']['pending']={'unresolved':'intent'}
    if fault=='foreign_result':args[2]['suffix']['sha256']='0'*64
    with pytest.raises(ValueError):module.assemble_bank(*args,audit=MetadataAudit(v),accepted04_ref=binding)


def targets04():
    target=np.zeros((140,14),np.float32);target[:,0]=np.arange(140)/1000
    target[:,6]=1.;target[56:70,6]=np.linspace(1,0,15)[1:];target[70:120,6]=0.
    target[120:134,6]=np.linspace(0,1,15)[1:];return target


def chunk_fixture(target,*,metadata,closed=False,actual=32,issued=32):
    d=dict(actions=target[None].copy(),base=np.zeros((1,32,14),np.float32),
        sampled_noise=np.zeros((32,32),np.float32),issued_motor_joint_targets=target[:issued].copy())
    c=dict(decision_metadata=dict(frozen_pi05_residual=dict(actions=dict(numeric_array='actions'),base_actions=dict(numeric_array='base')),
        **metadata),native_result=dict(executed_primitive_steps=actual),learning_result=dict(terminated=closed,truncated=False))
    n=dict(valid_action_mask=np.arange(issued)<actual)
    return c,d,n


def test_inserted_hold_uses_actual_current_plan_and_next_source_logical_cursor():
    target=targets04();current=np.arange(14,dtype=np.float32)/10;current[6]=.42
    hold=np.repeat(current[None],32,axis=0);hold[:,6]=0.
    metadata=dict(prelift_probe=dict(format=module.E_FORMAT,condition='close_endpoint',profile=dict(sha256='d'*64),
        entry=dict(kind='probe',start=70,end=70,source_chunk=None),execution_index=3,source_cursor_before=70,
        physical_slots_before=96,teacher_reference=dict(sha256='a'*64),
        probe_anchor_public14=dict(numeric_array='anchor'),explicit_probe_target=dict(numeric_array='explicit')))
    c,d,n=chunk_fixture(hold,metadata=metadata);d.update(anchor=current,explicit=hold,_canonical_current_state14=current)
    plan=module.captured_plan(c,d,n,targets=target,source=dict(sha256='a'*64),convention='prelift_profile_v1',
        profile_sha='d'*64,cursor=dict(source=70,physical=96,logical_chunk=3,execution=3),condition='close_endpoint')
    assert (plan['indices']==-1).all() and not plan['new'].any() and plan['executed'].sum()==32
    assert plan['following']==dict(source=70,physical=128,logical_chunk=3,execution=4)
    # Physical call4 resumes original source chunk3, not scheduled_request(4).
    request,_,_=scheduled_request(target,build_schedule(target),3)
    metadata['prelift_probe'].update(entry=dict(kind='source',source_chunk=3,start=70,end=102),execution_index=4,physical_slots_before=128)
    c,d,n=chunk_fixture(request,metadata=metadata,closed=True,actual=1)
    plan=module.captured_plan(c,d,n,targets=target,source=dict(sha256='a'*64),convention='prelift_profile_v1',
        profile_sha='d'*64,cursor=plan['following'],condition='close_endpoint')
    assert plan['indices'][0]==70 and plan['submitted'].sum()==32 and plan['executed'].sum()==1
    assert plan['following']['source']==71


@pytest.mark.parametrize('fault',['other_arm','lookahead','wrong_condition','wrong_profile','native_tail','clipped_label'])
def test_inserted_hold_provenance_and_native_cost_cannot_be_fabricated(fault):
    target=targets04();current=np.arange(14,dtype=np.float32)/10;hold=np.repeat(current[None],32,axis=0);hold[:,6]=0.
    meta=dict(prelift_probe=dict(format=module.E_FORMAT,condition='close_endpoint',profile=dict(sha256='d'*64),
        entry=dict(kind='probe',start=70,end=70,source_chunk=None),execution_index=3,source_cursor_before=70,
        physical_slots_before=96,teacher_reference=dict(sha256='a'*64),
        probe_anchor_public14=dict(numeric_array='anchor'),explicit_probe_target=dict(numeric_array='explicit')))
    c,d,n=chunk_fixture(hold,metadata=meta,closed=True,actual=1);d.update(anchor=current,explicit=hold.copy(),_canonical_current_state14=current)
    if fault=='other_arm':d['explicit'][0,0]+=.1
    if fault=='lookahead':d['anchor']=target[70].copy()
    if fault=='wrong_condition':c['decision_metadata']['prelift_probe']['condition']='hold_current'
    if fault=='wrong_profile':c['decision_metadata']['prelift_probe']['profile']['sha256']='0'*64
    if fault=='native_tail':n['valid_action_mask'][1]=True
    if fault=='clipped_label':d['actions'][0,:,13]=np.clip(d['actions'][0,:,13],0,1)
    with pytest.raises(ValueError):module.captured_plan(c,d,n,targets=target,source=dict(sha256='a'*64),
        convention='prelift_profile_v1',profile_sha='d'*64,cursor=dict(source=70,physical=96,logical_chunk=3,execution=3),condition='close_endpoint')


class TransformModel:
    device=torch.device('cpu')
    def input_transform(self,batch,transpose=False):
        return dict(actions=torch.nn.functional.pad(torch.as_tensor(batch['actions']),(0,18)),
                    state=torch.as_tensor(batch['observation/state']))
    def _actions_to_device(self,value):return value
    def _observation_dict_to_device(self,value):return SimpleNamespace(state=value['state'])
    def decode_actions(self,value,state):return value[...,:14]


def canonical_fixture(tmp_path,monkeypatch):
    target=np.zeros((4,14),np.float32);target[:,0]=np.arange(4)/100;target[:,6]=1.01
    source_path=tmp_path/'source.h5';source_path.write_bytes(b'whole original source commands only')
    source=dict(path=str(source_path),sha256=file_sha(source_path))
    monkeypatch.setattr(module,'read_episode',lambda *a:(target.copy(),target.copy()))
    episode=dict(id='phone-00',split='train',instruction='original')
    spec=record(0)['spec'];spec['paired_instruction']=['original']
    writer=RobotwinCorpusWriter(tmp_path/'corpus',dict(source_commit=module.PRODUCERS['event'][0],
        motor_sha256='a'*64,normalization_sha256='b'*64,dataset_sha256=module.DEMOS_SHA))
    writer.set_context(dict(phase='a1',method='event_aligned_teacher',task='place_phone_stand',trial_id=spec['id'],policy_version='original'))
    raw=dict(main_images=np.arange(12,dtype=np.uint8).reshape(1,2,2,3),wrist_images=np.zeros((1,2,2,2,3),np.uint8),
        states=np.zeros((1,14),np.float32),task_descriptions=['original'])
    public=np.zeros(2062,np.float32);info=dict(actual_seed=0,instruction=['original'])
    identity=writer.begin_episode(raw,public,info);info['corpus_episode_id']=identity
    request,_,_=scheduled_request(target,build_schedule(target),0)
    noise=np.arange(1024,dtype=np.float32).reshape(32,32)/1000;base=noise[:,:14]*.01
    writer.begin_decision(noise,request,dict(capacity_native_noise=noise.copy(),event_schedule=dict(chunk_index=0,source_cursor_before=0,physical_slots_before=0,
        teacher_reference=source),frozen_pi05_residual=dict(actions=request[None],base_actions=base[None])))
    writer.complete_native(next_raw=raw,next_features=public,raw_rewards=np.zeros(32),raw_terminations=np.zeros(32),
        raw_truncations=np.zeros(32),valid_action_mask=np.arange(32)<3,reward=1,terminated=True,truncated=False,
        executed_steps=3,success=True,duration_seconds=.1)
    writer.finalize_learning_flags(reward=1,terminated=True,truncated=False,budget_truncated=False,episode_primitive_steps=3)
    path=writer.root/'episodes'/identity/'manifest.json'
    row=dict(spec=spec,reset_info=info,complete=True,success=True,chunks=1,primitive_steps=3,
        episode_manifest_ref=dict(root=str(writer.root),path=str(path.relative_to(writer.root)),sha256=file_sha(path)))
    entry=dict(id='phone-00',split='train',record=row,original_source=source,producer_result=dict(path='/sealed',sha256='a'*64),
        producer_commit=module.PRODUCERS['event'][0],profile=dict(kind='original_event_h32_v1'),plan_convention='original_event_h32_v1')
    backend=SimpleNamespace(model=TransformModel(),_prepared=SimpleNamespace(state=torch.zeros(1,14)),
        _kv_cache=((torch.zeros(1,1,1,2),torch.zeros(1,1,1,2)),),_prefix_mask=torch.ones(1,1,dtype=torch.bool))
    backend._encode=lambda actual:public.copy();draws=[]
    def decode(actual):draws.append(actual.copy());return actual[:,:14]*.01
    backend.decode=decode
    return entry,episode,dict(motor_sha256='a'*64,normalization_sha256='b'*64,schema={}),backend,draws,noise


def test_real_canonical_writer_cache_preserves_full_plan_native_masks_and_padding(tmp_path,monkeypatch):
    entry,episode,demos,backend,draws,noise=canonical_fixture(tmp_path,monkeypatch)
    rows=module._cache_episode(entry,episode,demos,backend,tmp_path/'cache',deadline=time.time()+60,guard=lambda:None)
    assert len(rows)==1 and np.array_equal(draws[0],noise)
    saved=torch.load(rows[0]['path'],map_location='cpu',weights_only=False)
    assert saved['submitted_plan_mask'].sum()==32 and saved['native_executed_mask'].sum()==3
    assert saved['data']['target32'].shape==(1,32,32) and not saved['data']['target32'][...,14:].any()
    assert saved['full_known_proposed_plan14'][0,6]>1  # No policy/native clipping of label.
    assert rows[0]['submitted_hold_slots']==28 and rows[0]['native_valid_hold_slots']==0
    assert set(saved['data'])=={'state','features','cache','mask','target32','teacher14','base14','valid'}


def test_captured_native_base_must_replay_from_exact_noise(tmp_path,monkeypatch):
    entry,episode,demos,backend,_,_=canonical_fixture(tmp_path,monkeypatch)
    backend.decode=lambda _:np.ones((32,14),np.float32)
    with pytest.raises(ValueError,match='Gaussian/base14'):
        module._cache_episode(entry,episode,demos,backend,tmp_path/'cache',deadline=time.time()+60,guard=lambda:None)


def test_full_cache_seal_refuses_pending04_before_reading_any_cache(monkeypatch):
    args,values,_=metadata_fixture(monkeypatch)
    bank=module.assemble_bank(*args,audit=MetadataAudit(values))
    monkeypatch.setattr(module,'_recheck_bank',lambda value,deadline:value)
    with pytest.raises(ValueError,match='All12'):
        module.seal_full_cache(bank,[],deadline=time.time()+60)


def complete_cache_fixture(tmp_path,monkeypatch):
    args,values,binding=metadata_fixture(monkeypatch)
    bank=module.assemble_bank(*args,audit=MetadataAudit(values),accepted04_ref=binding)
    monkeypatch.setattr(module,'_recheck_bank',lambda value,deadline:value)
    rows=[]
    for entry in bank['entries']:
        entry['record'].update(chunks=1,primitive_steps=3)
        context=dict(teacher_id=entry['id'],split=entry['split'],producer_result=entry['producer_result'],
            producer_commit=entry['producer_commit'],profile=entry['profile'],record_id=entry['record']['spec']['id'],
            episode_manifest_ref=entry['record']['episode_manifest_ref'],original_source=entry['original_source'])
        submitted=torch.ones(32,dtype=torch.bool);executed=torch.arange(32)<3;target=torch.zeros(32,14)
        payload=dict(format='robotwin_captured_plan_cache_v1',context=context,submitted_plan_mask=submitted,
            native_executed_mask=executed,full_known_proposed_plan14=target,
            data=dict(valid=submitted[None],teacher14=target[None].clone(),target32=torch.zeros(1,32,32),
                base14=torch.zeros(1,32,14),features=torch.zeros(1,2062),state=torch.zeros(1,14),
                cache=(),mask=torch.ones(1,1,dtype=torch.bool)))
        path=tmp_path/(entry['id']+'.pt');torch.save(payload,path)
        rows.append(dict(path=str(path),sha256=file_sha(path),context=context,anchor=0,
                         valid_actions=32,native_actual_actions=3))
    receipt=tmp_path/'cache.json'
    value=dict(format='robotwin_fixed12_cache_receipt_v1',fit_eligible=False,records=rows)
    receipt.write_text(json.dumps(value))
    return bank,receipt,value


def test_full_twelve_cache_join_is_data_eligibility_not_resource_fit_admission(tmp_path,monkeypatch):
    bank,path,_=complete_cache_fixture(tmp_path,monkeypatch)
    result=module.seal_full_cache(bank,[dict(path=str(path),sha256=file_sha(path))],deadline=time.time()+60)
    assert result['complete'] is True and result['fixed12_fit_eligible'] is True
    assert result['resource_fit_admitted'] is False and len(result['records'])==12


@pytest.mark.parametrize('fault',['missing04','duplicate04','foreign_profile','native_tail','changed_label','future_input'])
def test_full_cache_join_rejects_missing_duplicate_or_corrupted_provenance_masks_labels(tmp_path,monkeypatch,fault):
    bank,path,value=complete_cache_fixture(tmp_path,monkeypatch)
    if fault=='missing04':value['records'].pop(4)
    elif fault=='duplicate04':value['records'].append(deepcopy(value['records'][4]))
    elif fault=='foreign_profile':value['records'][4]['context']['profile']['sha256']='0'*64
    else:
        row=value['records'][4];saved=torch.load(row['path'],weights_only=False)
        if fault=='native_tail':saved['native_executed_mask'][5]=True
        if fault=='changed_label':saved['data']['teacher14'][0,0,0]=1.
        if fault=='future_input':saved['data']['condition']=torch.zeros(1)
        torch.save(saved,row['path']);row['sha256']=file_sha(row['path'])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.seal_full_cache(bank,[dict(path=str(path),sha256=file_sha(path))],deadline=time.time()+60)
