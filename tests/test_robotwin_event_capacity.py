"""Meaningful synthetic qualification; execute on allocated Linux CPU only."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.robotwin_event_schedule import (
    build_schedule,scheduled_request,advance_cursor,validate_teacher_bank,
)
from fluffy_bassoon.meta_rl.robotwin_event_replay import EventReplayHead,EventEpisodeControl,physical_audit,cache_rollouts
from fluffy_bassoon.meta_rl.robotwin_event_admission import (
    FORMAT,validate_allocation,validate_recovery,build_event_plan,
)
from fluffy_bassoon.meta_rl.robotwin_reactive_capacity import CONFIG
from fluffy_bassoon.meta_rl.robotwin_reactive_fit import CapacityActionResidual
from fluffy_bassoon.meta_rl.robotwin_demo_capacity import file_sha
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter


def targets(n=80):
    value=np.zeros((n,14),np.float32);value[:,0]=np.arange(n,dtype=np.float32)/1000
    return value


def test_phase_edges_group_changing_samples_and_retain_every_source_frame():
    value=targets();value[10:20,6]=np.linspace(.1,1,10);value[20:,6]=1
    value[40:50,13]=np.linspace(-.1,-1,10);value[50:,13]=-1
    schedule=build_schedule(value)
    assert schedule['boundaries']==[0,10,20,40,50,80]
    assert [(c['start'],c['end']) for c in schedule['chunks']]==[(0,10),(10,20),(20,40),(40,50),(50,80)]
    reconstructed=[]
    for i,c in enumerate(schedule['chunks']):
        request,indices,new=scheduled_request(value,schedule,i)
        assert request.shape==(32,14) and new.sum()==c['new_frames']
        assert np.array_equal(request[~new],np.repeat(value[c['end']-1:c['end']],32-int(new.sum()),0))
        reconstructed.append(request[new])
    assert np.array_equal(np.concatenate(reconstructed),value)
    assert schedule['full_schedule_slots']==160 and schedule['full_schedule_added_holds']==80


def test_long_phase_splits_only_at_h32_and_reversal_starts_a_new_phase():
    value=targets(72);value[1:41,6]=np.arange(1,41)/40;value[41:,6]=np.linspace(.99,0,31)
    schedule=build_schedule(value)
    assert schedule['boundaries']==[0,1,41,72]
    assert [c['new_frames'] for c in schedule['chunks']]==[1,32,8,31]


def test_small_gripper_noise_and_stationary_arm_do_not_create_extra_phase_boundaries():
    value=targets(36);value[:,6]=np.arange(36)*1e-7
    schedule=build_schedule(value)
    assert schedule['boundaries']==[0,36] and len(schedule['chunks'])==2
    assert schedule['minimum_slots_to_issue_every_original_frame']==36


@pytest.mark.parametrize('bad',[np.zeros((0,14),np.float32),np.zeros((4,13),np.float32),
    np.zeros((4,14),np.int32),np.full((4,14),np.nan,np.float32)])
def test_invalid_reference_rejected(bad):
    with pytest.raises(ValueError):build_schedule(bad)


def test_overlong_schedule_retains_all_frames_and_reports_rejection_without_filtering():
    value=targets(50);value[:,6]=np.arange(50)%2
    schedule=build_schedule(value)
    assert schedule['admissible'] is False and schedule['reference_frames']==50
    assert sum(c['new_frames'] for c in schedule['chunks'])==50
    assert schedule['minimum_slots_to_issue_every_original_frame']>400


def test_physical_slots_source_cursor_and_terminal_unexecuted_plan_are_distinct():
    schedule=build_schedule(targets(4))
    terminal=advance_cursor(schedule,0,0,0,actual=3,issued=32,closed=True)
    assert terminal['physical_slots']==3 and terminal['source_cursor']==3
    assert terminal['native_valid_hold_slots']==0 and terminal['terminal_unexecuted_slots']==29
    complete=advance_cursor(schedule,0,0,0,actual=32,issued=32,closed=False)
    assert complete['source_cursor']==4 and complete['native_valid_hold_slots']==28
    tail=advance_cursor(schedule,1,32,4,actual=32,issued=32,closed=False)
    assert tail['physical_slots']==64 and tail['source_cursor']==4 and tail['native_valid_reference_slots']==0
    with pytest.raises(ValueError):advance_cursor(schedule,0,0,0,actual=4,issued=4,closed=False)
    with pytest.raises(ValueError):advance_cursor(schedule,0,0,1,actual=32,issued=32,closed=False)


def test_event_head_uses_declared_full_plan_not_public_hidden_cursor_or_base_clamping():
    value=targets(4);value[:,6]=1.1
    schedule=build_schedule(value);head=EventReplayHead();head.targets=torch.from_numpy(value)
    head.schedule=schedule
    base=torch.full((1,32,14),-2.)
    expected=scheduled_request(value,schedule,0)[0]
    for public in (torch.zeros(1,2062),torch.ones(1,2062)):
        result=head(base,public)
        assert np.array_equal(result.actions[0].numpy(),expected) and not result.clipped.any()
    head.chunk_index=9
    assert torch.equal(head(base,torch.zeros(1,2062)).actions[0],head.targets[-1:].expand(32,-1))


def demo_bank():
    demos=dict(episodes=[dict(id=f'phone-{i:02d}',split='train' if i<8 else 'validation',
        reset_seed=i,instruction=f'fixed instruction {i}',canonical_index=i) for i in range(12)],schema={})
    rows=[dict(spec=dict(teacher_id=e['id'],stage='event_aligned_teacher_execution',
        reset_seed=e['reset_seed'],paired_instruction=[e['instruction']]),complete=True,success=True)
          for e in demos['episodes']]
    return demos,rows


@pytest.mark.parametrize('mutation',['missing02','duplicate','failed02','control','changedseed','changedinstruction'])
def test_all12_gate_rejects_filtering_control_and_changed_case(mutation):
    demos,rows=demo_bank()
    if mutation=='missing02':rows.pop(2)
    if mutation=='duplicate':rows[2]=deepcopy(rows[0])
    if mutation=='failed02':rows[2]['success']=False
    if mutation=='control':rows[2]['spec']['stage']='event_fixed_control'
    if mutation=='changedseed':rows[2]['spec']['reset_seed']=500
    if mutation=='changedinstruction':rows[2]['spec']['paired_instruction']=['changed']
    with pytest.raises(ValueError):validate_teacher_bank(rows,demos)


def test_exact_resource_residual_and_no_new_grant():
    allocation=dict(mode=FORMAT,prior_a1_gpu_seconds=2036,prior_phase_gpu_seconds=2036,
        reserved_gpu_seconds=5164,capacity_action_bound=10000,capacity_fit_config=CONFIG,
        run_id='fresh',scheduler_start_unix=100,driver_deadline_unix=5100)
    validate_allocation(allocation,now=110,run_id='fresh')
    for key,value in [('reserved_gpu_seconds',5165),('prior_a1_gpu_seconds',1770),
        ('capacity_action_bound',11401),('driver_deadline_unix',6000),('run_id','old')]:
        bad=dict(allocation,**{key:value})
        with pytest.raises(ValueError):validate_allocation(bad,now=110,run_id='fresh')


def test_fixed25_episode_plan_keeps_all12_and_control_before_aligned02():
    demos,_=demo_bank();old={}
    for case in range(8):
        info=dict(actual_seed=case,observation_sha256='a'*64,features_sha256='b'*64,instruction=['fixed original'])
        old[str(case)]=dict(spec=dict(task='place_phone_stand',case=case,stage='development',reset_seed=case),
            complete=True,reset_info=info,reset_identity={k:info[k] for k in ('actual_seed','observation_sha256','features_sha256')})
    plan=build_event_plan(old,demos)
    assert plan['legacy_control']['teacher_id']=='phone-02'
    assert plan['teacher_order']==['phone-02','phone-00','phone-01']+[f'phone-{i:02d}' for i in range(3,12)]
    assert len(plan['development'])==12 and 1+len(plan['teacher_replays'])+len(plan['development'])==25
    assert plan['maximum_new_actions']==10000 and plan['maximum_new_GPU_seconds']==5164
    assert list(plan['whole_split'].values())==['train']*8+['validation']*4


def test_failed_closed_recovery_retains_private_and_global_rng_and_zero_fits():
    prior=dict(status='blocked_failure')
    saved=dict(state=prior,fits={},fit_recovery_refs={},control=dict(mode='action',elapsed_steps=400,
        motor_rng={'private':'retained'}),global_rng={k:'retained' for k in ('python','numpy','torch','cuda')})
    validate_recovery(saved,prior)
    for key,value in [('fits',{'kv':'unadmitted'}),('control',dict(mode='action',elapsed_steps=200,motor_rng={})),
        ('global_rng',{'numpy':'only'}),('state',{'status':'complete'})]:
        with pytest.raises(ValueError):validate_recovery(dict(saved,**{key:value}),prior)


def task_fixture():
    class Joint:
        def __init__(self,name,target):self.name,self.target=name,target
        def get_name(self):return self.name
        def get_drive_target(self):return np.array([self.target])
    class Entity:
        def get_qpos(self):return np.array([.1,.02])
        def get_active_joints(self):return [Joint('arm',.2),Joint('grip',.03)]
    class Actor:
        def get_pose(self):return SimpleNamespace(p=np.array([1.,2.,3.]),q=np.array([1.,0.,0.,0.]))
    robot=SimpleNamespace(left_entity=Entity(),right_entity=Entity(),left_gripper_val=.8,right_gripper_val=.9,
        left_gripper=[(Joint('left_finger',.04),1,0)],right_gripper=[(Joint('right_finger',.05),1,0)])
    return SimpleNamespace(robot=robot,phone=Actor(),stand=Actor(),phone_id=0,stand_id=2,
        get_info=lambda:{'{A}':'phone0'},now_obs={'observation':{'head_camera':{'intrinsic_cv':np.eye(3),'rgb':'not accessed'}}},
        get_obs=lambda:pytest.fail('Extra observation forbidden'),scene=SimpleNamespace(step=lambda:pytest.fail('Extra physics forbidden')))


def test_physical_audit_reads_measured_qpos_nominal_and_drive_without_render_or_rng_change():
    task=task_fixture();before=deepcopy(np.random.get_state())
    audit=physical_audit(task)
    assert audit['actors']['phone']['value']['variant_id']==0
    assert audit['articulations']['left']['value']['qpos']==[.1,.02]
    assert audit['articulations']['left']['value']['drive_targets']==[.2,.03]
    assert audit['grippers']['left']['value']['nominal']==.8
    assert 'rgb' not in audit['already_available_camera_calibration']['head_camera']
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    del task.phone
    assert physical_audit(task)['actors']['phone']['available'] is False


def test_control_uses_unchanged_full_h32_env_step_and_separate_source_cursor(tmp_path):
    class Head:
        def config(self):return {'synthetic':True}
    class Env:
        native_horizon=32;active=False;elapsed_steps=0;contract={}
        def __init__(self):
            self.backend=SimpleNamespace(model=SimpleNamespace(device=torch.device('cpu')),
                _kv_cache=((torch.zeros(1,1,1,2),torch.zeros(1,1,1,2)),),
                env=SimpleNamespace(venv=SimpleNamespace(envs=[SimpleNamespace(task=task_fixture())])))
        def reset(self,*args,**kwargs):
            self.active=True;self.elapsed_steps=0;self.last_reset_info={'corpus_episode_id':'synthetic-episode'}
            return np.zeros(2062,np.float32)
        def record_decision_metadata(self,value):self.metadata=value
        def step(self,noise):
            assert noise.shape==(32,32) and self.metadata['event_schedule']['source_cursor_before']==0
            self.elapsed_steps=32;self.active=False
            path=tmp_path/'episodes/synthetic-episode/chunks/000001.json'
            path.parent.mkdir(parents=True);path.write_text('{}')
            return np.zeros(2062,np.float32),1,True,False,dict(primitive_steps=32,issued_primitive_steps=32,
                corpus_chunk=str(path.relative_to(tmp_path)))
    writer=SimpleNamespace(root=tmp_path,register_runtime=lambda contract:None)
    env=Env();control=EventEpisodeControl(env,{'kv':Head(),'action':Head()},writer,
        dict(kv_layers=1,kv_heads=1,head_dim=2))
    control.select_teacher(targets(4),{'sha256':'a'*64},aligned=True)
    control.set_coefficients('action',[0.]*4);control.reset('place_phone_stand',0,motor_seed=1)
    control.step([0.]*4)
    assert env.elapsed_steps==32 and control.cursor['source_cursor']==4
    assert control.cursor['native_valid_hold_slots']==28
    assert control.snapshot()['motor_rng'] is not None
    ref=control.publish_chunk_audit(tmp_path/'audits')
    assert file_sha(ref['path'])==ref['sha256']
    reset_ref=control.publish_reset_audit(tmp_path/'reset-audits')
    assert json.loads(Path(reset_ref['path']).read_text())['physical']['actors']['phone']['available']


class CacheModel:
    device=torch.device('cpu')
    def input_transform(self,batch,transpose=False):
        return dict(actions=torch.nn.functional.pad(torch.as_tensor(batch['actions']),(0,18)),state=torch.as_tensor(batch['observation/state']))
    def _actions_to_device(self,value):return value
    def _observation_dict_to_device(self,value):return SimpleNamespace(state=value['state'])
    def decode_actions(self,actions,state):return actions[...,:14]


def canonical_bank(tmp_path,monkeypatch):
    import fluffy_bassoon.meta_rl.robotwin_event_replay as module
    demos,rows=demo_bank();value=targets(4);paths={}
    monkeypatch.setattr(module,'read_episode',lambda *args:(value.copy(),value.copy()))
    writer=RobotwinCorpusWriter(tmp_path/'corpus',{'synthetic':True})
    raw=dict(main_images=np.zeros((1,2,2,3),np.uint8),wrist_images=np.zeros((1,2,2,2,3),np.uint8),
             states=np.zeros((1,14),np.float32),task_descriptions=['fixed'])
    features=np.array([.2,.3],np.float32)
    for episode,row in zip(demos['episodes'],rows):
        source=tmp_path/(episode['id']+'.ref');source.write_text('immutable original commands')
        paths[episode['id']]=dict(path=str(source),sha256=file_sha(source))
        row['spec']['id']='event-teacher-'+episode['id']
        writer.set_context(dict(phase='a1',method='event_aligned_teacher',task='place_phone_stand',
            trial_id=row['spec']['id'],policy_version=row['spec']['id']))
        raw['task_descriptions']=[episode['instruction']];reset_info=dict(actual_seed=episode['reset_seed'])
        identity=writer.begin_episode(raw,features,reset_info)
        row['reset_info']=dict(reset_info,corpus_episode_id=identity)
        plan=scheduled_request(value,build_schedule(value),0)[0]
        writer.begin_decision(np.zeros((32,32),np.float32),plan,dict(event_schedule=dict(
            chunk_index=0,source_cursor_before=0,physical_slots_before=0,teacher_reference=paths[episode['id']]),
            frozen_pi05_residual=dict(base_actions=np.zeros((1,32,14),np.float32),actions=plan[None])))
        writer.complete_native(next_raw=raw,next_features=features,raw_rewards=np.zeros(32),raw_terminations=np.zeros(32),
            raw_truncations=np.zeros(32),valid_action_mask=np.arange(32)<3,reward=1,terminated=True,truncated=False,
            executed_steps=3,success=True,duration_seconds=.1)
        writer.finalize_learning_flags(reward=1,terminated=True,truncated=False,budget_truncated=False,episode_primitive_steps=3)
        path=writer.root/'episodes'/identity/'manifest.json'
        row.update(chunks=1,primitive_steps=3,episode_manifest_ref=dict(root=str(writer.root),
            path=str(path.relative_to(writer.root)),sha256=file_sha(path)))
    backend=SimpleNamespace(model=CacheModel(),_kv_cache=((torch.zeros(1,1,1,2),torch.zeros(1,1,1,2)),),
        _prefix_mask=torch.ones(1,1,dtype=torch.bool))
    def encode(raw):backend._prepared=SimpleNamespace(state=torch.as_tensor(raw['states']));return features.copy()
    backend._encode=encode;backend.decode=lambda noise:np.zeros((32,14),np.float32)
    head=CapacityActionResidual(public_dim=2,width=4,joint_lower=[-1]*14,joint_upper=[1]*14,
        arm_joint_bound=.1,gripper_bound=1.).eval()
    with torch.no_grad():head.bounds[[6,13]]=0
    return demos,paths,rows,backend,head


def test_actual_canonical_cache_full_submitted_plan_supervision_separate_from_native_cost(tmp_path,monkeypatch):
    demos,paths,rows,backend,head=canonical_bank(tmp_path,monkeypatch)
    cached=cache_rollouts(backend,demos,paths,rows,tmp_path/'cache',guard=lambda:None,
        deadline=time.time()+60,action_head=head)
    assert len(cached)==12 and [r['split'] for r in cached]==['train']*8+['validation']*4
    assert all(r['valid_actions']==32 and r['native_actual_actions']==3 for r in cached)
    saved=torch.load(cached[0]['path'],weights_only=False)
    assert saved['data']['valid'].sum()==32 and saved['native_executed_mask'].sum()==3
    assert saved['submitted_plan_mask'].sum()==32 and saved['native_valid_hold_slots']==0
    assert saved['submitted_hold_slots']==28 and saved['full_known_proposed_plan14'].shape==(32,14)
    assert set(saved['data'])=={'state','features','cache','mask','target32','teacher14','base14','valid'}


def test_cache_rejects_different_same_observation_native_proposal(tmp_path,monkeypatch):
    demos,paths,rows,backend,head=canonical_bank(tmp_path,monkeypatch)
    backend.decode=lambda noise:np.ones((32,14),np.float32)
    with pytest.raises(ValueError,match='proposal does not reproduce'):
        cache_rollouts(backend,demos,paths,rows,tmp_path/'cache',guard=lambda:None,deadline=time.time()+60,action_head=head)


def test_cache_fail_before_reading_if_any_aligned_teacher_failed(tmp_path):
    demos,rows=demo_bank();rows[2]['success']=False
    with pytest.raises(ValueError):cache_rollouts(None,demos,{},rows,tmp_path/'cache',guard=lambda:None,deadline=time.time()+60,action_head=None)
    assert not (tmp_path/'cache').exists()
