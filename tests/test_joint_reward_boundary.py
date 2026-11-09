"""Numerical terminal-label and eligibility boundaries for one joint policy."""
from copy import deepcopy

import pytest

from fluffy_bassoon.robocasa.flora_ppo import flora_jobs, parse_outputs
from fluffy_bassoon.robocasa.ppo import (
    PPO_TASKS, TASKS, joint_arm_stopping, joint_episode_protocols, protocol,
    require_calibration_permission,
)
from fluffy_bassoon.robocasa.residual import install_terminal_labels
from fluffy_bassoon.robocasa.video import REWARD_CAMERAS


def bank():
    cases=[dict(task=task) for _ in range(16) for task in PPO_TASKS]
    episodes=[];outputs=[]
    scores=dict(zip(PPO_TASKS,(5,2,3,1,4)))
    for index,case in enumerate(cases):
        key=f'joint-{index:03d}';task=case['task'];horizon=TASKS[task][0]
        episode=dict(id=key,task='native instruction '+task,camera_order=list(REWARD_CAMERAS),
            info_on_video=False,extra_info_on_video=[],raw_rgb_only=True,
            success_termination_disabled=True,horizon=horizon,completed=True,
            frame_clock=dict(frame_count=horizon+1,native_fps=20.,encoded_fps=20.,resampled=False,
                physical_steps=horizon,initial_frame_included=True,timestamp_sha256='b'*64),
            rank=index//10,round=index%10,local_env=0,
            videos=[dict(camera=c,path=f'/allocated/{key}/{c}.mp4',sha256='a'*64) for c in REWARD_CAMERAS])
        episodes.append(episode)
        value=scores[task]
        outputs.append(dict(id=key,task=episode['task'],expected_score=float(value),
            probabilities={str(i):float(i==value) for i in range(1,6)},progress=(value-1)/4))
    return cases,episodes,outputs


def test_mixed_native_protocols_preserve_all_eighty_official_terminal_grades():
    cases,episodes,outputs=bank()
    protocols=joint_episode_protocols(episodes,cases)
    jobs=flora_jobs(episodes,episode_protocols=protocols)
    assert len(jobs)==80 and all(set(j)=={'id','task','videos'} for j in jobs)
    labels=parse_outputs(episodes,outputs,episode_protocols=protocols)
    records=[]
    for episode in episodes:
        records.extend(dict(reward=0.,done=i==episode['horizon']//50-1)
                       for i in range(episode['horizon']//50))
    labeled=install_terminal_labels(records,episodes,labels,arm='F')
    assert len(labeled)==960 and all(r['reward']==0 for r in records)
    assert [r['reward'] for r in labeled if r['done']]==[labels[e['id']] for e in episodes]
    assert all(r['reward']==0 for r in labeled if not r['done'])
    with pytest.raises(ValueError):flora_jobs(episodes,protocol('OpenDrawer'),episode_protocols=protocols)
    with pytest.raises(ValueError):parse_outputs(episodes,outputs[:-1],episode_protocols=protocols)
    short=deepcopy(protocols);short.pop(episodes[-1]['id'])
    with pytest.raises(ValueError):flora_jobs(episodes,episode_protocols=short)
    wrong=deepcopy(protocols);wrong[episodes[4]['id']]['horizon']=750
    with pytest.raises(ValueError):parse_outputs(episodes,outputs,episode_protocols=wrong)
    wrong_case=deepcopy(cases);wrong_case[0]['task']='CloseFridge'
    with pytest.raises(ValueError):joint_episode_protocols(episodes,wrong_case)


def test_native_and_auroc_exceptions_are_separate_and_never_turn_failed_gate_true():
    proto=protocol('PickPlaceCounterToCabinet',baseline_count=50)
    gate=dict(completed=True,protocol=proto,count=50,initial_success_rate=.74,AUROC=.8,go=False)
    checksum='a'*64
    # Synthetic test authorization only; no execution authority is produced.
    auth=dict(protocol=proto,calibration_sha256=checksum,measured_AUROC=.8,
        direct_human_instruction='synthetic unit fixture, not execution authorization',
        allow_training_below_auc_gate=True)
    with pytest.raises(ValueError):require_calibration_permission(gate,proto,calibration_sha256=checksum)
    with pytest.raises(ValueError):require_calibration_permission(gate,proto,calibration_sha256=checksum,authorization=auth)
    auth.update(allow_training_outside_native_headroom_gate=True,measured_initial_success_rate=.72)
    with pytest.raises(ValueError):require_calibration_permission(gate,proto,calibration_sha256=checksum,authorization=auth)
    auth['measured_initial_success_rate']=.74;auth.pop('allow_training_below_auc_gate')
    assert require_calibration_permission(gate,proto,calibration_sha256=checksum,authorization=auth)=='human_authorized_native_headroom_exception'
    assert gate['go'] is False and gate['initial_success_rate']==.74
    low=deepcopy(gate);low['AUROC']=.6;auth['measured_AUROC']=.6
    with pytest.raises(ValueError):require_calibration_permission(low,proto,calibration_sha256=checksum,authorization=auth)
    auth['allow_training_below_auc_gate']=True
    assert require_calibration_permission(low,proto,calibration_sha256=checksum,authorization=auth)=='human_authorized_native_and_AUROC_exceptions'
    forged=deepcopy(gate);forged['go']=True
    with pytest.raises(ValueError):require_calibration_permission(forged,proto,calibration_sha256=checksum,authorization=auth)
    legacy=protocol('OpenDrawer',camera_order=['robot0_agentview_left'])
    old=dict(completed=True,protocol=legacy,count=8,initial_success_rate=3/8,AUROC=2/3,go=False)
    old_auth=dict(protocol=legacy,calibration_sha256=checksum,measured_AUROC=2/3,
        direct_human_instruction='synthetic old exception fixture',allow_training_below_auc_gate=True)
    assert require_calibration_permission(old,legacy,calibration_sha256=checksum,authorization=old_auth)=='human_authorized_below_original_AUROC_gate'


def test_joint_dev_stopping_cannot_hide_an_improving_task_or_use_final_panels():
    def row(iteration,counts):
        return dict(phase='initial_DEV' if iteration==0 else 'DEV',iteration=iteration,
            episodes=40,successes=sum(counts),
            per_task=[dict(task=t,episodes=8,successes=n) for t,n in zip(PPO_TASKS,counts)])
    rows=[row(i,[4]*5) for i in (0,2,4,6)]
    assert joint_arm_stopping(rows)['stop'] is True
    # Same aggregate as before: one task improved while another declined.
    rows[-1]=row(6,[5,3,4,4,4])
    assert joint_arm_stopping(rows)['stop'] is False
    final=deepcopy(rows);final[-1]['phase']='ID'
    with pytest.raises(ValueError):joint_arm_stopping(final)
    partial=deepcopy(rows);partial[-1]['per_task'].pop()
    with pytest.raises(ValueError):joint_arm_stopping(partial)
    assert joint_arm_stopping([row(0,[4]*5),row(2,[8]*5)])['reason']=='all_tasks_DEV8of8'
