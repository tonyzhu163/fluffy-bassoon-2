"""Synthetic request/physical metadata tests; allocated Linux CPU only."""
from copy import deepcopy
import numpy as np
import pytest

from fluffy_bassoon.meta_rl.robotwin_event_diagnosis import (
    PRODUCER,compare_plan,summarize_physical,physical_change,compare_source_scene,
    validate_closed_metadata,json_sha,
)
from fluffy_bassoon.meta_rl.robotwin_event_schedule import build_schedule,scheduled_request


def reference():
    target=np.zeros((40,14),np.float32);target[:,0]=np.arange(40)/100
    target[5:10,6]=np.linspace(.2,1,5);target[10:,6]=1
    return target


@pytest.mark.parametrize('aligned',[False,True])
def test_full_plan_and_submitted_mask_are_distinct_from_terminal_native_prefix(aligned):
    target=reference()
    expected=scheduled_request(target,build_schedule(target),0)[0] if aligned else target[:32]
    result=compare_plan(target,dict(proposed=expected,issued=expected),np.arange(32)<1,
        ordinal=0,physical_before=0,aligned=aligned)
    assert result['full_proposed_reference_exact'] and result['submitted_reference_exact']
    assert result['proposed_slots']==32 and result['submitted_slots']==32
    assert result['native_valid_slots']==1 and result['terminal_unexecuted_submitted_slots']==31
    bad=expected.copy();bad[-1,0]+=1
    result=compare_plan(target,dict(proposed=bad,issued=bad),np.arange(32)<1,
        ordinal=0,physical_before=0,aligned=aligned)
    assert not result['full_proposed_reference_exact']  # unexecuted plan tail is still checked


def test_natural_horizon_submitted16_and_final_reference_hold_remain_explicit():
    target=reference();expected=np.repeat(target[-1:],32,0)
    result=compare_plan(target,dict(proposed=expected,issued=expected[:16]),np.ones(16),
        ordinal=12,physical_before=384,aligned=True)
    assert result['submitted_slots']==16 and result['native_valid_slots']==16
    assert result['submitted_declared_hold_slots']==16 and result['submitted_reference_slots']==0


def test_nonprefix_native_mask_rejected():
    expected=reference()[:32];mask=np.zeros(32);mask[2]=1
    with pytest.raises(ValueError,match='prefix'):
        compare_plan(reference(),dict(proposed=expected,issued=expected),mask,ordinal=0,physical_before=0,aligned=False)


def physical():
    return dict(task_info=dict(available=True,value={'{A}':'phone2','{B}':'stand1'}),
        actors={key:dict(available=True,value=dict(variant_id=i,p=[0.,0.,.74],q=[1.,0.,0.,0.])) for i,key in enumerate(('phone','stand'))},
        articulations={side:dict(available=True,value=dict(joint_names=['arm','finger'],qpos=[.2,.03],drive_targets=[.4,.045]))
                       for side in ('left','right')},
        grippers={side:dict(available=True,value=dict(nominal=1.,named_drive_targets=[dict(name='finger',target=.045)]))
                  for side in ('left','right')},
        already_available_camera_calibration={'head_camera':{'intrinsic_cv':np.eye(3).tolist()}},
        conventions={'qpos':'measured native named coordinates'},native_counters={'take_action_cnt':32})


def test_measured_gripper_position_is_not_nominal_or_drive_and_names_are_retained():
    result=summarize_physical(physical());grip=result['articulations']['left']['gripper_joints'][0]
    assert grip['name']=='finger' and grip['measured_qpos']==.03 and grip['drive_target']==.045
    assert result['grippers']['left']['value']['nominal']==1.
    assert grip['measured_minus_drive']==pytest.approx(-.015)
    value=physical();value['articulations']['left']={'available':False,'reason':'not captured'}
    assert summarize_physical(value)['articulations']['left']['available'] is False


def test_mismatched_articulation_names_or_gripper_mapping_fail_closed():
    value=physical();value['articulations']['left']['value']['joint_names']=['arm','arm']
    with pytest.raises(ValueError):summarize_physical(value)
    value=physical();value['grippers']['left']['value']['named_drive_targets'][0]['name']='foreign'
    with pytest.raises(ValueError):summarize_physical(value)


def test_world_pose_delta_and_quaternion_sign_do_not_reconstruct_source_physics_time():
    a=physical();b=deepcopy(a);b['actors']['phone']['value']['p'][2]+=.1
    b['actors']['phone']['value']['q']=[-1.,0.,0.,0.]
    b['articulations']['left']['value']['qpos'][1]=.04
    delta=physical_change(a,b)
    assert delta['actors']['phone']['world_translation_L2']==pytest.approx(.1)
    assert delta['actors']['phone']['quaternion_component_delta_up_to_sign']==0
    assert delta['articulations']['left']['named_measured_qpos_delta']['finger']==pytest.approx(.01)


def test_source_variant_and_cached_camera_exact_comparison_with_explicit_missing_physics():
    value=physical();row={'scene_metadata':{'info':{'{A}':'phone2','{B}':'stand1'}}}
    calibration={'observation/head_camera/intrinsic_cv':np.eye(3).tolist(),
                 'observation/left_camera/extrinsic_cv':np.zeros((3,4)).tolist()}
    result=compare_source_scene(row,calibration,value)
    assert all(x['exact'] for x in result['variant_and_arm_tag_checks'].values())
    assert result['reset_camera_calibration_checks'][0]['exact'] is True
    assert result['reset_camera_calibration_checks'][1]['available'] is False
    assert result['source_measured_articulation_qpos_available'] is False
    assert result['synchronized_source_physics_timestamps_available'] is False


def closed():
    names=['event-control-fixed-phone-02','event-teacher-phone-02','event-teacher-phone-00',
           'event-teacher-phone-01','event-teacher-phone-03','event-teacher-phone-04']
    records=[dict(spec={'id':name},complete=True,success=i not in (0,5),primitive_steps=400 if i in (0,5) else 193)
             for i,name in enumerate(names)]
    actions=dict(actual_actions=20171,charged_actions=20171,pending_actions=0,uncertain_actions=0,unresolved_decisions={})
    result=dict(source_commit=PRODUCER,status='blocked_failure',stage='teacher_executability',pending=None,
        failure=dict(type='RuntimeError',message='Fixed aligned teacher failed; preserve all demos and stop before fitting'),
        records=records,actions=actions)
    terminal=dict(format='original_a1_event_capacity_terminal_receipt_v1',job_id='job-2rkimbvypowd',
        scheduler_status='Failed',source_commit=PRODUCER,exit_code=1,charged_gpu_seconds=298,cumulative_a1_gpu_seconds=2334,
        original_twelve_demos_retained=True,all_twelve_teacher_gate_passed=False,cache_created=False,kv_updates=0,action_updates=0,
        unrun_aligned_teacher_suffix=[f'phone-{i:02d}' for i in range(5,12)],canonical_costs={k:v for k,v in actions.items() if k!='unresolved_decisions'})
    closeout=dict(exit_code=1,error=None,source_commit=PRODUCER,charged_gpu_seconds=298,attempt=dict(status='finished',slice_overrun=False))
    return result,terminal,closeout


@pytest.mark.parametrize('mutation',['fit','drop04','pending','refund','changed04','suffix'])
def test_exact_terminal_failure_and_retained_cost_and_unrun_suffix_are_required(mutation):
    result,terminal,closeout=closed();validate_closed_metadata(result,terminal,closeout)
    if mutation=='fit':terminal['kv_updates']=1
    if mutation=='drop04':result['records'].pop()
    if mutation=='pending':result['actions']['uncertain_actions']=16
    if mutation=='refund':terminal['charged_gpu_seconds']=270
    if mutation=='changed04':result['records'][-1]['success']=True
    if mutation=='suffix':terminal['unrun_aligned_teacher_suffix'].pop()
    with pytest.raises(ValueError):validate_closed_metadata(result,terminal,closeout)


def test_rng_state_is_an_opaque_order_independent_hash_not_replayed_or_seeded():
    a={'kind':'MT19937','keys':[1,2],'position':20}
    assert json_sha(a)==json_sha(dict(reversed(list(a.items()))))
    b=deepcopy(a);b['position']=21
    assert json_sha(a)!=json_sha(b)
