"""Synthetic CPU qualification only; run on allocated Linux, never macOS."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.robotwin_demo_capacity import (
    file_sha,prepare_demo_manifest,validate_demo_manifest,read_episode,
    whole_episode_chunks,transformed_target,masked_mse,reactive_flow_loss,
)
from fluffy_bassoon.meta_rl.robotwin_reactive_fit import (
    CapacityActionResidual,ReplayTeacherHead,fit_reactive_head,batch_from_cache,
)
from fluffy_bassoon.meta_rl.robotwin_reactive_capacity import (
    build_plan,summarize,CONFIG,FORMAT,validate_capacity_allocation,
)
from fluffy_bassoon.meta_rl.warmstart import atomic_save
from fluffy_bassoon.rl.pi05_residual import ReactiveKVResidual


ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'outputs/robotwin-reactive-capacity-20261031-r1/AUTHOR-CONVENTION.json'
INVENTORY=ROOT/'outputs/frozen-pi05-endtoend-20261031-r1/phone-stage0-cpu-artifacts/inventory.json'


def manifest(tmp_path):
    author=json.loads(EVIDENCE.read_text());path=tmp_path/'author.json'
    path.write_text(json.dumps(author))
    result=prepare_demo_manifest(json.loads(INVENTORY.read_text()),file_sha(INVENTORY),author,
        dict(path=str(path),sha256=file_sha(path)),motor_sha='a'*64,norm_sha='b'*64)
    return result,path


def test_actual_metadata_composition_keeps_absent_flags_and_fixed_whole_split(tmp_path):
    result,_=manifest(tmp_path)
    validate_demo_manifest(result)
    assert [r['reset_seed'] for r in result['episodes']]==[0,1,2,3,4,6,9,10,11,14,16,18]
    assert [r['split'] for r in result['episodes']]==['train']*8+['validation']*4
    assert all(r['explicit_success'] is None for r in result['episodes'])
    assert 'pending' in result['readiness']


@pytest.mark.parametrize('mutation',['flag','order','offset','views','inventory','tamper'])
def test_fail_closed_demo_identity_and_author_binding(tmp_path,mutation):
    result,path=manifest(tmp_path)
    if mutation=='flag':result['episodes'][0]['explicit_success']=True
    if mutation=='order':result['episodes'][0]['canonical_index']=9
    if mutation=='offset':result['schema']['action_offset']=1
    if mutation=='views':result['schema']['camera_order'].reverse()
    if mutation=='inventory':result['source_inventory_sha256']='c'*64
    if mutation=='tamper':path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError):validate_demo_manifest(result)


def test_tail_padding_never_loses_or_reweights_actual_targets():
    targets=np.arange(35*14,dtype=np.float32).reshape(35,14)
    windows=list(whole_episode_chunks(targets))
    assert [int(v.sum()) for _,_,v in windows]==[32,3]
    assert np.array_equal(np.concatenate([x[v] for _,x,v in windows]),targets)
    assert np.array_equal(windows[-1][1][3:],np.repeat(targets[-1:],29,0))


def test_hdf5_reader_rejects_extra_or_truncated_timeline(tmp_path):
    result,_=manifest(tmp_path);row=result['episodes'][0];schema=result['schema']
    path=tmp_path/'demo.h5'
    with h5py.File(path,'w') as f:
        f.create_dataset(schema['state'],data=np.zeros((row['qualified_action_count']-1,14)))
        for key in schema['images']:f.create_dataset(key,data=np.zeros(row['qualified_action_count']-1))
    with pytest.raises(ValueError,match='length'):read_episode(path,row,schema)


class TransformModel:
    def input_transform(self,batch,transpose):
        assert transpose is False and batch['prompt']==['original instruction']
        actions=torch.as_tensor(batch['actions']).clone()
        actions[...,:6]-=torch.as_tensor(batch['observation/state'])[...,None,:6]
        self.last=deepcopy(batch)
        return dict(actions=torch.nn.functional.pad(actions*2,(0,18)),state=torch.as_tensor(batch['observation/state']))
    def _actions_to_device(self,value):return value
    def _observation_dict_to_device(self,value):return SimpleNamespace(state=value['state'])
    def decode_actions(self,actions,state):
        result=actions[...,:14]/2;result[...,:6]+=state[...,None,:6];return result


def test_target_conversion_preserves_prompt_actions_and_physical_roundtrip():
    model=TransformModel();target=np.arange(448,dtype=np.float32).reshape(32,14)/100
    raw=dict(main_images=np.zeros((1,2,2,3),np.uint8),wrist_images=np.zeros((1,2,2,2,3),np.uint8),
             states=np.ones((1,14),np.float32),task_descriptions=['original instruction'])
    _,converted,receipt=transformed_target(model,raw,target)
    assert converted.shape==(1,32,32) and receipt['max_abs_roundtrip_error']<1e-5
    assert np.array_equal(model.last['actions'][0],target)
    model.decode_actions=lambda actions,state:torch.ones(1,32,14)
    with pytest.raises(ValueError,match='roundtrip'):transformed_target(model,raw,target)


def action_head():
    return CapacityActionResidual(public_dim=2,arm_joint_bound=.1,gripper_bound=1.,width=4,
                                  joint_lower=[-1]*6+[0]+[-1]*6+[0],joint_upper=[1]*14).eval()


def test_action_head_preserves_raw_grippers_and_zero_signed_bits():
    head=action_head();base=torch.zeros(1,3,14);base[...,0]=-0.;base[...,6]=-.0101426;base[...,13]=1.0090805
    with torch.no_grad():result=head(base,torch.zeros(1,2))
    assert torch.equal(result.actions.view(torch.int32),base.view(torch.int32))
    with torch.no_grad():head.output.bias.fill_(5)
    result=head(base,torch.zeros(1,2))
    assert torch.equal(result.actions[...,[6,13]],base[...,[6,13]])
    assert (result.requested_delta[...,[6,13]]==0).all()


def test_action_head_uses_only_arm_headroom_without_repairing_base():
    head=action_head();base=torch.zeros(1,2,14);base[...,0]=.99;base[...,6]=-0.1;base[...,13]=1.1
    with torch.no_grad():head.output.bias.fill_(5)
    result=head(base,torch.zeros(1,2))
    assert (result.actions[...,0]==1).all() and result.clipped[...,0].all()
    assert torch.equal(result.actions[...,[6,13]],base[...,[6,13]])
    base[...,0]=1.01
    with pytest.raises(ValueError,match='ARM'):head(base,torch.zeros(1,2))


def test_teacher_replay_is_explicit_unbounded_teacher_not_native_clamp():
    replay=ReplayTeacherHead();replay.targets=torch.arange(3*14,dtype=torch.float32).reshape(3,14)
    replay.offset=2;base=torch.zeros(1,32,14)
    result=replay(base,torch.zeros(1,2062))
    assert torch.equal(result.actions, replay.targets[-1:].expand(32,-1)[None])
    assert replay.config()['kind']=='qualified_source_demo_command_replay_v1'


class TinyMotor(torch.nn.Module):
    device=torch.device('cpu')
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.arange(1,33,dtype=torch.float32)/32,requires_grad=False)
        self.eval()
    def run_suffix(self,observation,noisy,time,cache,mask):
        context=cache[0][1].sum((1,2,3))[:,None,None]
        return noisy*.2+context*self.weight[None,None]
    def velocity_from_suffix(self,value):return value


def cache_records(tmp_path):
    rows=[]
    for i,split in enumerate(['train','train','validation']):
        data=dict(state=torch.zeros(1,14),features=torch.tensor([[.2,.3]]),
            cache=((torch.ones(1,2,1,2),torch.ones(1,2,1,2)),),mask=torch.ones(1,2,dtype=torch.bool),
            target32=torch.zeros(1,32,32),teacher14=torch.zeros(1,32,14),base14=torch.zeros(1,32,14),
            valid=torch.tensor([[True]*3+[False]*29]))
        data['teacher14'][...,[0,1,2,3,4,5,7,8,9,10,11,12]]=.05
        path=tmp_path/f'{i}.pt';digest=atomic_save(path,dict(data=data))
        rows.append(dict(path=str(path),sha256=digest,split=split,valid_actions=3))
    return rows


def test_masked_loss_excludes_only_tail_and_flow_reports_active_vs_padding(tmp_path):
    values=torch.ones(1,32,32);target=torch.zeros_like(values);valid=torch.tensor([[True]*3+[False]*29])
    values[:,3:]=1000
    assert masked_mse(values,target,valid)==1
    rows=cache_records(tmp_path);data=torch.load(rows[0]['path'],weights_only=False)['data']
    motor=TinyMotor();head=ReactiveKVResidual(public_dim=2,kv_heads=1,head_dim=2,rank=2)
    loss,details=reactive_flow_loss(motor,head,data,rng=np.random.default_rng(2),
        generator=torch.Generator().manual_seed(2),diagnostics=True)
    assert set(details)=={'physical14_MSE','padded18_MSE'}
    assert float(loss.detach())==pytest.approx((details['physical14_MSE']*14+details['padded18_MSE']*18)/32)
    loss.backward()
    assert head.output_bases[0].weight.grad.abs().sum()>0 and motor.weight.grad is None
    with torch.no_grad(),pytest.raises(RuntimeError,match='autograd'):
        reactive_flow_loss(motor,head,data,rng=np.random.default_rng(2),generator=torch.Generator().manual_seed(2))


@pytest.mark.parametrize('kind',['kv','action'])
def test_finite_small_fit_keeps_motor_and_complete_optimizer_private_rng_recovery(tmp_path,kind):
    rows=cache_records(tmp_path);motor=TinyMotor();before=motor.weight.clone()
    head=ReactiveKVResidual(public_dim=2,kv_heads=1,head_dim=2,rank=2) if kind=='kv' else action_head()
    parameters=[p.detach().clone() for p in head.parameters()]
    backend=SimpleNamespace(model=motor,_check_frozen=lambda:None)
    config=dict(CONFIG,updates=3,validate_every=1,learning_rate=.01)
    result=fit_reactive_head(kind,head,backend,rows,tmp_path/kind,guard=lambda:None,config=config,contract={'synthetic':True})
    saved=torch.load(tmp_path/kind/'latest.pt',weights_only=False)
    assert result['complete'] and result['iteration']==3
    assert set(saved)>={'optimizer','policy','sampler_numpy','noise_torch','global_rng'}
    assert all(int(v['step'])==3 for v in saved['optimizer']['state'].values())
    assert torch.equal(before,motor.weight) and motor.weight.grad is None
    assert any(not torch.equal(a,b) for a,b in zip(parameters,head.parameters()))
    assert file_sha(tmp_path/kind/'best.pt')==result['best']['sha256']
    assert result['fit_diagnostics']['full_class_rejection_authorized'] is False


def test_native_paired_plan_charges_all_teachers_and_arms_without_locked_panel(tmp_path):
    demos,_=manifest(tmp_path);original={}
    for case in range(8):
        info=dict(actual_seed=case,observation_sha256=str(case)*64,features_sha256='f'*64,instruction=['text'])
        original[str(case)]=dict(spec=dict(case=case,task='place_phone_stand',stage='development',reset_seed=case),
            complete=True,reset_info=info,reset_identity={k:info[k] for k in ('actual_seed','observation_sha256','features_sha256')})
    plan=build_plan(original,demos)
    assert plan['maximum_new_actions']==400*(1+12+12)==10000<=12409
    assert len(plan['development'])==12 and len(plan['teacher_replays'])==12
    for case in range(4):
        rows=[r for r in plan['development'] if r['case']==case]
        assert {r['arm'] for r in rows}=={'native','kv','action'}
        assert len({r['motor_seed'] for r in rows})==1
    records=[dict(spec=s,success=s['arm']=='action',primitive_steps=100) for s in plan['development']]
    result=summarize(records)
    assert result['paired']['action']==dict(wins=4,losses=0)
    original.pop('7')
    with pytest.raises(ValueError,match='entire'):build_plan(original,demos)


def test_cache_tamper_cannot_supply_training_targets(tmp_path):
    records=cache_records(tmp_path)
    path=Path(records[0]['path']);path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):batch_from_cache(records,[0],torch.device('cpu'),guard=lambda:None)


def test_boundary_stop_preserves_last_finite_optimizer_sampler_and_policy(tmp_path):
    records=cache_records(tmp_path);head=action_head();motor=TinyMotor()
    output=tmp_path/'fit'
    def guard():
        path=output/'latest.pt'
        if path.exists() and torch.load(path,weights_only=False)['state']['iteration']==1:
            raise TimeoutError('original allocation clock')
    with pytest.raises(TimeoutError):
        fit_reactive_head('action',head,SimpleNamespace(model=motor,_check_frozen=lambda:None),
            records,output,guard=guard,config=dict(CONFIG,updates=3,validate_every=1),contract={})
    saved=torch.load(output/'latest.pt',weights_only=False)
    assert saved['state']['iteration']==1
    assert all(int(v['step'])==1 for v in saved['optimizer']['state'].values())
    assert all(torch.isfinite(v).all() for v in saved['policy'].values())


@pytest.mark.parametrize('changed',['budget','clock','identity','config'])
def test_capacity_admission_cannot_replenish_residual_or_restart_clock(changed):
    allocation=dict(mode=FORMAT,prior_a1_gpu_seconds=1770,reserved_gpu_seconds=5430,
        capacity_action_bound=10000,capacity_fit_config=deepcopy(CONFIG),run_id='new-unit',
        scheduler_start_unix=1000,driver_deadline_unix=6300)
    validate_capacity_allocation(allocation,now=1010,run_id='new-unit')
    if changed=='budget':allocation['reserved_gpu_seconds']=7200
    if changed=='clock':allocation['driver_deadline_unix']=6500
    if changed=='identity':allocation['run_id']='old-unit'
    if changed=='config':allocation['capacity_fit_config']['updates']=500
    with pytest.raises(ValueError):validate_capacity_allocation(allocation,now=1010,run_id='new-unit')
