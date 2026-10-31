"""Synthetic CPU tests; execute only inside the parent's allocated Linux gate."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
import pytest

from fluffy_bassoon.meta_rl.robotwin_replay_diagnosis import (
    Audit,IDS,PRODUCER,compare_commands,drive_delivery,jump_metrics,panel,inspect_replays,
    gripper_events,compression_geometry,
    validate_reset_identity,
)
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_cursor_terminal_mask_and_final_hold_are_distinct():
    targets=np.arange(129*14,dtype=np.float32).reshape(129,14)
    expected=targets[np.minimum(128+np.arange(32),128)]
    mask=np.arange(32)<8
    report,reference,_=compare_commands(targets,expected,mask,128,terminal=True)
    assert report['valid_reference_exact'] and report['cursor_after']==136
    assert report['reference_indices']==[128]*32
    assert report['terminal_last_action_may_be_partial']
    assert np.array_equal(reference,expected)


@pytest.mark.parametrize('change',['nonprefix','nonbinary','wrongshape'])
def test_bad_mask_or_action_layout_is_rejected(change):
    target=np.zeros((3,14),np.float32);issued=np.zeros((2,14),np.float32);mask=np.ones(2)
    if change=='nonprefix':mask=np.array([0,1])
    if change=='nonbinary':mask=np.array([2,1])
    if change=='wrongshape':issued=np.zeros((2,13))
    with pytest.raises(ValueError):compare_commands(target,issued,mask,0,terminal=False)


def test_every_valid_request_is_checked_but_invalid_padding_is_separate():
    targets=np.arange(40*14,dtype=np.float32).reshape(40,14)/100
    issued=targets[:32].copy();issued[20,0]+=1
    report,_,_=compare_commands(targets,issued,np.arange(32)<8,0,terminal=True)
    assert report['valid_reference_exact'] and not report['all_issued_reference_exact']
    issued[3,8]+=.25
    report,_,_=compare_commands(targets,issued,np.arange(32)<8,0,terminal=True)
    assert not report['valid_reference_exact'] and report['right_arm_error']==pytest.approx(.25)
    shifted=targets[1:33]
    report,_,_=compare_commands(targets,shifted,np.ones(32),0,terminal=False)
    assert not report['valid_reference_exact']


def test_delivery_comparison_is_nominal_and_preserves_requests_even_when_clipping():
    request=np.zeros(14,np.float32);request[0]=.75;request[8]=.5;request[6]=-.01;request[13]=1.01
    before=request.copy();actual=request.copy();actual[0]=.5;actual[8]=.25;actual[6]=0;actual[13]=1
    report=drive_delivery(actual[None],request,terminal=True)
    assert report['left_arm_max_abs']==.25 and report['right_arm_max_abs']==.25
    assert report['gripper_max_abs']==0 and report['terminal_partial_possible']
    assert 'not measured' in report['meaning'] and np.array_equal(before,request)


def test_adjacent_reference_jumps_keep_arms_and_grippers_separate():
    targets=np.zeros((3,14),np.float32);targets[1:,0]=.1;targets[2:,8]=.3;targets[2:,6]=1
    result=jump_metrics(targets)
    assert result['left_arm']['max_L2']==pytest.approx(.1)
    assert result['right_arm']['max_L2']==pytest.approx(.3)
    assert result['grippers']['max_L2']==1


def test_stationary_arm_gripper_transitions_are_preserved_as_ranges():
    target=np.zeros((5,14),np.float32);target[:,6]=[1,.8,.6,.6,1]
    target[3:,0]=.1
    result=gripper_events(target)['left']
    assert result['stationary_arm_closing']==[[1,2]]
    assert result['stationary_arm_opening']==[[4,4]]
    assert result['gripper_opening']==[[4,4]]


def test_compression_geometry_never_draws_native_random_jitter():
    start=np.zeros(14,np.float32);issued=np.zeros((4,14),np.float32);issued[2:,0]=.1
    before=np.random.get_state();report=compression_geometry(start,issued);after=np.random.get_state()
    assert report['left']['retained_command_slots']==[2]
    assert report['right']['stationary_jitter_branch'] is True
    assert all(np.array_equal(a,b) for a,b in zip(before,after))


def test_audit_rejects_tamper_escape_and_clock_extension(tmp_path):
    path=tmp_path/'metadata.json';path.write_text('{}');audit=Audit(time.time()+60)
    with pytest.raises(ValueError,match='SHA'):audit.bound(path,'0'*64)
    with pytest.raises(ValueError,match='escapes'):audit.nested(tmp_path,dict(path='../other',sha256='0'*64))
    with pytest.raises(TimeoutError):Audit(time.time()-1).bound(path,sha(path))


def test_panel_preserves_native_rgb_for_png_display(tmp_path):
    color=np.array([17,103,231],np.uint8);image=np.broadcast_to(color,(4,4,3)).copy()
    raw=dict(main_images=image[None],wrist_images=np.stack([image,image])[None])
    path=tmp_path/'comparison.png';metrics=panel(raw,raw,'synthetic',path)
    decoded=cv2.imread(str(path))
    assert np.array_equal(decoded[40,40],color[::-1])
    assert all(x['mean_absolute_pixel_difference']==0 for x in metrics)


def test_actual_writer_reset_annotation_is_the_only_allowed_difference(tmp_path):
    writer=RobotwinCorpusWriter(tmp_path/'corpus',{'synthetic':True})
    writer.set_context(dict(phase='a1',method='capacity_teacher',trial_id=IDS[2],policy_version='fixed'))
    image=np.zeros((1,2,2,3),np.uint8)
    raw=dict(main_images=image,wrist_images=np.zeros((1,2,2,2,3),np.uint8),
             states=np.zeros((1,14),np.float32),task_descriptions=['fixed'])
    info=dict(actual_seed=2,requested_seed=2,instruction=['fixed'],
        observation_sha256='a'*64,features_sha256='b'*64,
        instruction_binding={'effective_observation_sha256':'a'*64})
    episode=writer.begin_episode(raw,np.zeros(2062,np.float32),info)
    # Exact producer order: snapshot first, append backend annotation afterward.
    info['corpus_episode_id']=episode
    base=writer.root/'episodes'/episode
    reset=json.loads((base/'reset.json').read_text());manifest=json.loads((base/'manifest.json').read_text())
    assert 'corpus_episode_id' not in reset['reset_info']
    validate_reset_identity(reset,info,manifest)
    for key,value in [('corpus_episode_id','foreign'),('actual_seed',3),('instruction',['changed']),
                      ('observation_sha256','c'*64),('features_sha256','d'*64),
                      ('instruction_binding',{'effective_observation_sha256':'e'*64})]:
        altered=deepcopy(info);altered[key]=value
        with pytest.raises(ValueError):validate_reset_identity(reset,altered,manifest)
    altered=deepcopy(reset);altered['episode_id']='foreign'
    with pytest.raises(ValueError):validate_reset_identity(altered,info,manifest)


def test_full_closed_three_episode_audit_uses_existing_capture_schema(tmp_path):
    base=tmp_path/'old'/'experiment';base.mkdir(parents=True)
    writer=RobotwinCorpusWriter(base/'corpus',{'synthetic':True})
    schema=dict(state='joint_action/vector',actions='joint_action/vector',action_offset=0,
        image_encoding='jpeg',image_decoder='opencv_imdecode_color_as_native_channels',
        images=['observation/head_camera/rgb','observation/left_camera/rgb','observation/right_camera/rgb'])
    demos=dict(schema=schema,episodes=[],actual_whole_files={});records=[];resets={}
    color=np.broadcast_to(np.array([17,103,231],np.uint8),(4,4,3)).copy()
    ok,jpeg=cv2.imencode('.jpg',color);assert ok
    for ordinal,(n,total,success) in enumerate([(131,104,True),(130,104,True),(129,400,False)]):
        targets=np.zeros((n,14),np.float32);targets[:,0]=np.arange(n)/100;targets[:,[6,13]]=1
        path=tmp_path/f'demo{ordinal}.h5'
        with h5py.File(path,'w') as f:
            f.create_dataset(schema['state'],data=targets)
            for key in schema['images']:f.create_dataset(key,data=np.asarray([jpeg.tobytes()]*n,dtype=f'S{len(jpeg)}'))
        row=dict(id=f'phone-{ordinal:02d}',reset_seed=ordinal,instruction='fixed',qualified_action_count=n,
            qualified_state_count=n,scene_metadata={'synthetic':True})
        demos['episodes'].append(row);ref=dict(path=str(path),sha256=sha(path));demos['actual_whole_files'][row['id']]=ref
        raw=lambda state:dict(main_images=color[None],wrist_images=np.stack([color,color])[None],
                              states=state[None],task_descriptions=['fixed'])
        info=dict(actual_seed=ordinal,instruction=['fixed'])
        writer.set_context(dict(phase='a1',method='capacity_teacher',trial_id=IDS[ordinal],policy_version='fixed'))
        episode=writer.begin_episode(raw(targets[0]),np.zeros(2062,np.float32),info)
        info['corpus_episode_id']=episode
        rp=writer.root/'episodes'/episode/'reset.json';resets[IDS[ordinal]]=dict(path=str(rp),sha256=sha(rp))
        refs=[];cursor=0
        while cursor<total:
            issued_count=min(32,400-cursor);count=min(issued_count,total-cursor)
            issued=targets[np.minimum(cursor+np.arange(issued_count),n-1)]
            writer.begin_decision(np.zeros((32,32),np.float32),issued,
                {'frozen_pi05_residual':{'configuration':{'kind':'qualified_source_demo_command_replay_v1','reference':ref}}})
            end=cursor+count==total;terminated=success and end;truncated=not success and end
            relative=writer.complete_native(next_raw=raw(issued[count-1]),next_features=np.zeros(2062,np.float32),
                raw_rewards=np.zeros(issued_count),raw_terminations=np.zeros(issued_count),raw_truncations=np.zeros(issued_count),
                valid_action_mask=np.arange(issued_count)<count,reward=float(terminated),terminated=terminated,
                truncated=truncated,executed_steps=count,success=terminated,duration_seconds=.01)
            writer.finalize_learning_flags(reward=float(terminated),terminated=terminated,truncated=truncated,
                budget_truncated=False,episode_primitive_steps=cursor+count)
            refs.append(dict(root=str(writer.root),path=relative,sha256=sha(writer.root/relative)));cursor+=count
        mp=writer.root/'episodes'/episode/'manifest.json'
        records.append(dict(spec={'id':IDS[ordinal]},success=success,primitive_steps=total,chunks=len(refs),
            corpus_refs=refs,reset_info=info,episode_manifest_ref=dict(root=str(writer.root),
                path=str(mp.relative_to(writer.root)),sha256=sha(mp))))
    result=dict(source_commit=PRODUCER,stage='teacher_executability',status='blocked_failure',pending=None,
        actions=dict(actual_actions=18599,pending_actions=0,uncertain_actions=0),records=records)
    report=inspect_replays(result,base/'result.json',demos,tmp_path/'inspection',Audit(time.time()+60),reset_refs=resets)
    assert report['status']=='audit_complete' and report['actual_valid_commands_compared']==608
    assert report['all_valid_reference_exact'] and report['new_robot_actions']==report['model_forwards']==report['SGD']==0
    assert [x['success'] for x in report['episodes']]==[True,True,False]
    assert report['scene_identity_proven'] is False
    assert len(list((tmp_path/'inspection').glob('*.png')))==12
    with (tmp_path/'inspection'/'valid-command-comparisons.csv').open() as stream:assert len(list(csv.DictReader(stream)))==608
