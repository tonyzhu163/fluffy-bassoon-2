"""Focused allocated-Linux CPU tests; no model/scorer/simulator constructed."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.rlinf_flora_ppo import (
    FLORA_CODE,FLORA_MODEL,TASK,baseline_validation,checked_labels,commit_labels,
    digest,flora_jobs,freeze_batch,parse_outputs,ready_rewards,relabel_full_horizon,
    sha,terminal_command,initializer_manifest,INITIALIZER_REPO,INITIALIZER_REVISION,
    INITIALIZER_WEIGHT_SHA,INITIALIZER_WEIGHT_BYTES,NORM_SHA,
)
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_config import evaluation_schedule,validate_config
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_env import strip_training_truth
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_monitor import ProgressLog,stopping_decision,whole_iteration_admission


def episode(i=0):
    return dict(id=f'episode-{i}',task=TASK,camera_order=['agentview'],info_on_video=False,
        extra_info_on_video=[],raw_rgb_only=True,success_termination_disabled=True,horizon=400,completed=True,
        video=dict(path=f'/qualified/video-{i}.mp4',sha256='a'*64),
        frame_clock=dict(frame_count=401,native_fps=20.,encoded_fps=20.,resampled=False,
            physical_steps=400,initial_frame_included=True,timestamp_sha256='b'*64),
        rank=0,round=i//32,local_env=i%32)


def published_manifest(tmp_path):
    return dict(completed=True,repository_id=INITIALIZER_REPO,revision=INITIALIZER_REVISION,
        model_directory=str(tmp_path),normalization_sha256=NORM_SHA,model_action_dim=32,action_horizon=10,
        files=[dict(path=str(tmp_path/'model.safetensors'),sha256=INITIALIZER_WEIGHT_SHA,bytes=INITIALIZER_WEIGHT_BYTES),
            dict(path=str(tmp_path/'physical-intelligence/libero/norm_stats.json'),sha256=NORM_SHA,bytes=4507)])


def test_initializer_accepts_published40_torch_manifest(tmp_path):
    receipt=published_manifest(tmp_path)
    assert initializer_manifest(receipt,tmp_path)==receipt['files']
    assert INITIALIZER_WEIGHT_BYTES==7473091464 and INITIALIZER_REVISION.startswith('45ccfcc4')
    assert 'source_parameter_signature' not in receipt # No fabricated C565/JAX equivalence.


@pytest.mark.parametrize('kind',['repository','revision','old_norm','weight_sha','weight_size','horizon','old_JAX_receipt','escaped_file'])
def test_initializer_refuses_other_release_or_old_c565_abi(tmp_path,kind):
    receipt=published_manifest(tmp_path)
    if kind=='repository':receipt['repository_id']='RLinf/RLinf-Pi05-LIBERO130'
    elif kind=='revision':receipt['revision']='main'
    elif kind=='old_norm':receipt['normalization_sha256']='b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84'
    elif kind=='weight_sha':receipt['files'][0]['sha256']='c565'+'0'*60
    elif kind=='weight_size':receipt['files'][0]['bytes']+=1
    elif kind=='horizon':receipt['action_horizon']=32
    elif kind=='old_JAX_receipt':receipt.pop('repository_id');receipt['source_parameter_signature']='c565'
    else:receipt['files'][0]['path']=str(tmp_path.parent/'outside.safetensors')
    with pytest.raises(ValueError):initializer_manifest(receipt,tmp_path)


def output(i=0,score=3):
    return dict(id=f'episode-{i}',task=TASK,score=score,expected_score=float(score),progress=(score-1)/4,
        probabilities={str(k):float(k==score) for k in range(1,6)},rationale='',duration_s=20.05,
        sampled_frames=64,output=f'<score>{score}</score>')


def frozen(tmp_path,arm='F'):
    return freeze_batch(tmp_path/'batch.json',arm=arm,iteration=1,episodes=[episode(i) for i in range(128)],
        rollout_ref=dict(producer_iteration=0),recovery_refs=[dict(role=role,complete=True,optimizer_sampler_RNG=True)
            for role in ('actor','rollout','env')],rank_pending_receipts={'0':dict(path='/rank0.json',sha256='c'*64)},
        profile_sha256='d'*64,policy_sha256='e'*64)


def test_baseline_is64_paired_no_threshold_fit():
    rows=[dict(id=str(i),expected_score=4. if i%2 else 2.,simulator_final_success=bool(i%2)) for i in range(64)]
    r=baseline_validation(rows)
    assert r['go'] and r['AUROC']==1 and r['initial_success_rate']==.5
    assert 'threshold' not in r and 'cutoff' not in r and r['policy_optimizer_updates']==0
    rows[0]['id']=rows[1]['id']
    with pytest.raises(ValueError):baseline_validation(rows)


@pytest.mark.parametrize('successes',[0,1,2,45,64])
def test_baseline_headroom_gate(successes):
    rows=[dict(id=str(i),expected_score=4. if i<successes else 2.,simulator_final_success=i<successes) for i in range(64)]
    assert not baseline_validation(rows)['go']


def test_auc_failure_does_not_fit_or_flip_labels():
    rows=[dict(id=str(i),expected_score=2. if i%2 else 4.,simulator_final_success=bool(i%2)) for i in range(64)]
    r=baseline_validation(rows)
    assert r['AUROC']==0 and not r['go']


@pytest.mark.parametrize('score',[1,2,3,4,5])
def test_terminal_grade_preserves_partial_credit(score):
    assert parse_outputs([episode()],[output(score=score)])=={'episode-0':(score-1)/4}


@pytest.mark.parametrize('field,value',[('expected_score',float('nan')),('expected_score',6),
    ('progress',.9),('task','wrong'),('id','stale')])
def test_bad_terminal_outputs_hold_batch(field,value):
    row=output();row[field]=value
    with pytest.raises(ValueError):parse_outputs([episode()],[row])


@pytest.mark.parametrize('mutation', ['missing','duplicate','bad_probability'])
def test_all128_labels_or_none(mutation):
    episodes=[episode(i) for i in range(128)];outputs=[output(i) for i in range(128)]
    if mutation=='missing':outputs.pop()
    elif mutation=='duplicate':outputs[-1]=outputs[0]
    else:outputs[0]['probabilities']['3']=.5
    with pytest.raises(ValueError):parse_outputs(episodes,outputs)


@pytest.mark.parametrize('key,value',[('info_on_video',True),('raw_rgb_only',False),('completed',False),
    ('success_termination_disabled',False),('camera_order',['agentview','wrist']),('horizon',399)])
def test_scorer_only_sees_clean_complete_native_video(key,value):
    e=episode();e[key]=value
    with pytest.raises(ValueError):flora_jobs([e])


def test_truth_or_annotations_are_not_prompt_fields():
    e=episode();e['simulator_final_success']=True
    with pytest.raises(ValueError):flora_jobs([e])
    assert set(flora_jobs([episode()])[0])=={'id','task','videos'}
    assert strip_training_truth(dict(success=True,episode={'success':True},valid_action_mask='actual'))=={'valid_action_mask':'actual'}


def test_resampling_or_default_fps_is_not_native_clock():
    e=episode();e['frame_clock']['encoded_fps']=30
    with pytest.raises(ValueError):flora_jobs([e])


def test_local_scorer_command_has_no_ignored_revision_or_overwrite():
    command=terminal_command('/flora/python','/pinned/model','/jobs','/output')
    assert '--revision' not in command and '--overwrite' not in command
    assert command[command.index('--gpu-memory-utilization')+1]=='0.78'
    assert command[command.index('--max-model-len')+1]=='16384'


def test_reward_adapter_zeros_everything_except_final_valid_slot():
    raw=np.ones((80,2,5),np.float32);mask=np.ones_like(raw,dtype=bool)
    result=relabel_full_horizon(raw,mask,['a','b'],dict(a=.25,b=.75))
    assert np.array_equal(raw,np.ones_like(raw))
    assert np.count_nonzero(result)==2 and result[-1,0,-1]==.25 and result[-1,1,-1]==.75


@pytest.mark.parametrize('kind',['integer','mask','horizon','missing','out_of_range'])
def test_no_integer_truncation_or_fake_full_horizon(kind):
    raw=np.zeros((80,2,5),np.float32);mask=np.ones_like(raw,dtype=bool);labels=dict(a=.25,b=.5)
    if kind=='integer':raw=raw.astype(np.int64)
    elif kind=='mask':mask[40,1,2]=False
    elif kind=='horizon':raw=raw[:-1];mask=mask[:-1]
    elif kind=='missing':labels.pop('b')
    else:labels['a']=1.1
    with pytest.raises(ValueError):relabel_full_horizon(raw,mask,['a','b'],labels)


def test_ready_identity_and_truth_isolation_before_gae(tmp_path):
    batch=frozen(tmp_path);labels={e['id']:.25 for e in batch['episodes']}
    receipt=dict(completed=True,input_batch_sha256=digest(batch),code_commit=FLORA_CODE,model_revision=FLORA_MODEL)
    ready=commit_labels(tmp_path/'ready.json',batch,labels,label_receipt=receipt,baseline_validation_sha256='f'*64)
    raw=np.zeros((80,128,5),np.float32);mask=np.ones_like(raw,dtype=bool);ids=[e['id'] for e in batch['episodes']]
    assert ready_rewards(batch,ready,raw,mask,ids).sum()==32
    raw[0,0,0]=1
    with pytest.raises(ValueError,match='Strip native'):ready_rewards(batch,ready,raw,mask,ids)
    raw[:]=0;ready['iteration']=2
    with pytest.raises(ValueError):ready_rewards(batch,ready,raw,mask,ids)


def test_partial_or_replayed_label_commit_refused(tmp_path):
    batch=frozen(tmp_path)
    with pytest.raises(ValueError):commit_labels(tmp_path/'ready.json',batch,{},label_receipt={},baseline_validation_sha256=None)
    assert not (tmp_path/'ready.json').exists()
    assert json.loads((tmp_path/'batch.json').read_text())['status']=='WAITING_LABELS'


def test_pending_resume_binds_actual_model_bytes_not_counters(tmp_path):
    from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import validate_pending_checkpoint
    weight=tmp_path/'actor.pt';weight.write_bytes(b'producing actor+Adam')
    files=[dict(path=str(weight),sha256=sha(weight))]
    checkpoint=tmp_path/'boundary.json';checkpoint.write_text(json.dumps(dict(model_optimizer_files=files)))
    ref=dict(path=str(checkpoint),sha256=sha(checkpoint))
    batch=dict(rollout=dict(checkpoint=ref),policy_sha256=digest(files))
    validate_pending_checkpoint(batch,ref)
    with pytest.raises(ValueError,match='different preceding'):validate_pending_checkpoint(batch,dict(ref,path='/other/boundary.json'))
    weight.write_bytes(b'other policy with same iteration/Adam counts')
    with pytest.raises(ValueError,match='bytes changed'):validate_pending_checkpoint(batch,ref)


def test_real_runtime_symbols_import_without_constructing_workers():
    from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import actor_class,rollout_class
    from fluffy_bassoon.meta_rl.rlinf_flora_ppo_env import environment_class
    from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runner import MatchedRunner
    assert actor_class().__name__=='MatchedFloraActor'
    assert rollout_class().__name__=='MatchedRollout'
    assert environment_class().__name__=='MatchedEnvironment'
    assert hasattr(MatchedRunner,'_label_barrier')


def test_S_reward_hook_matches_actual_official_gae_and_holds_before_labels(tmp_path):
    import torch
    import rlinf.algorithms
    from rlinf.algorithms.registry import calculate_adv_and_returns
    from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import TerminalRewardActorMixin
    class AuthorCall:
        def compute_advantages_and_returns(self):
            self.gae_calls+=1
            return calculate_adv_and_returns(task_type='embodied',adv_type='gae',
                rewards=self.rollout_batch['rewards'],dones=self.rollout_batch['dones'],
                values=self.rollout_batch['prev_values'],prev_logprobs=None,teacher_logprobs=None,
                num_action_chunks=5,gamma=.99,gae_lambda=.95,group_size=1,reward_type='chunk_level',
                loss_mask=None,loss_mask_sum=None,advantage_mode=None)
        def run_training(self):raise AssertionError('No optimizer in this CPU contract test')
    class Actor(TerminalRewardActorMixin,AuthorCall):pass
    actor=Actor();actor._rank=0;actor.version=0;actor.optimizer_steps=0;actor.gae_calls=0
    actor.cfg=SimpleNamespace(flora_ppo=SimpleNamespace(exchange_root=str(tmp_path),arm='S'))
    actor.rollout_batch=dict(rewards=torch.zeros(80,128,5),dones=torch.zeros(81,128,5,dtype=torch.bool),
        truncations=torch.zeros(81,128,5,dtype=torch.bool),terminations=torch.zeros(81,128,5,dtype=torch.bool),
        prev_values=torch.zeros(81,128,1))
    actor.rollout_batch['truncations'][-1,:,-1]=True;actor.rollout_batch['dones'][-1,:,-1]=True
    with pytest.raises(ValueError):actor.compute_advantages_and_returns()
    with pytest.raises(ValueError):actor.run_training()
    assert actor.gae_calls==0
    pending=actor.capture_pending_for_labels()
    root=actor._reward_directory()
    batch=freeze_batch(root/'batch.json',arm='S',iteration=1,episodes=[episode(i) for i in range(128)],
        rollout_ref=dict(producer_iteration=0),recovery_refs=[dict(role=r,complete=True,optimizer_sampler_RNG=True)
            for r in ('actor','rollout','env')],rank_pending_receipts={'0':pending},profile_sha256='a'*64,policy_sha256='b'*64)
    labels={f'episode-{i}':i%2 for i in range(128)}
    commit_labels(root/'ready.json',batch,labels,label_receipt=dict(completed=True,input_batch_sha256=digest(batch)),
        baseline_validation_sha256='c'*64)
    original={k:v.clone() for k,v in actor.rollout_batch.items()}
    actor.install_reward_refs(dict(path=str(root/'batch.json'),sha256=sha(root/'batch.json')),
        dict(path=str(root/'ready.json'),sha256=sha(root/'ready.json')))
    actual=actor.compute_advantages_and_returns()
    native=AuthorCall();native.gae_calls=0;native.rollout_batch=original
    native.rollout_batch['rewards'][-1,:,-1]=torch.tensor([i%2 for i in range(128)],dtype=torch.float32)
    reference=native.compute_advantages_and_returns()
    assert actor.gae_calls==1
    for key in ('advantages','returns'):assert torch.equal(actual[key],reference[key])
    for key in ('dones','prev_values','terminations','truncations'):
        assert torch.equal(actor.rollout_batch[key],original[key])


def test_simulator_labels_stay_binary(tmp_path):
    batch=frozen(tmp_path,'S')
    with pytest.raises(ValueError):commit_labels(tmp_path/'ready.json',batch,{e['id']:.5 for e in batch['episodes']},
        label_receipt={},baseline_validation_sha256=None)


@pytest.mark.parametrize('world,padding',[(1,0),(2,0),(4,6),(8,6)])
def test_eval_slots_match_exact50_scored_cases(world,padding):
    rows=evaluation_schedule(world,'DEV');scored=[r for r in rows if r['counted']]
    assert len(rows)==50+padding and len(rows)%world==0
    assert {(r['nominal_state'],r['draw']) for r in scored}=={(s,d) for s in range(40,50) for d in range(5)}
    assert all(0<=r['nominal_state']<40 for r in evaluation_schedule(world,'baseline'))


def evals(values):
    return [dict(metric_source='simulator_DEV_final_success',episodes=50,successes=n,unique_nominal_states=10,
        full_boundary_checkpoint=True,checkpoint_sha256='a'*64,health='healthy',iteration=i+1) for i,n in enumerate(values)]


def test_ceiling_can_stop_before6():
    assert stopping_decision(evals([48,49,48]),baseline_success_rate=.2)['reason']=='ceiling_48of50_three_consecutive'


def test_baseline_included_in_best_and_futility():
    assert stopping_decision(evals([10]*6),baseline_success_rate=.2)['reason']=='futility'
    assert stopping_decision(evals([20]*6),baseline_success_rate=.2)['reason']=='plateau'
    assert not stopping_decision(evals([10,10,10,11,11,12]),baseline_success_rate=.2)['stop']


@pytest.mark.parametrize('field,value',[('health','stale'),('metric_source','Flora'),
    ('full_boundary_checkpoint',False),('episodes',49),('iteration',9)])
def test_failure_or_incomplete_eval_is_not_convergence(field,value):
    rows=evals([10]*6);rows[-1][field]=value
    with pytest.raises(ValueError):stopping_decision(rows,baseline_success_rate=.2)


def test_logs_append_atomic_resume_and_readable_success(tmp_path,capsys):
    clock=lambda:1000.
    first=ProgressLog(tmp_path,'F',contract_sha256='pin',clock=clock)
    first.emit(phase='evaluation',physical_actions=500,evaluation_successes='29/50',stop_reason='continuing')
    second=ProgressLog(tmp_path,'F',contract_sha256='pin',clock=clock)
    second.emit(event='heartbeat')
    assert json.loads((tmp_path/'status.json').read_text())['physical_actions']==500
    assert len((tmp_path/'metrics.jsonl').read_text().splitlines())==2
    assert '29/50' in capsys.readouterr().out and not (tmp_path/'status.tmp').exists()
    with pytest.raises(ValueError):ProgressLog(tmp_path,'F',contract_sha256='other')


def test_new_human_authority_has_no_old10h_gate():
    assert whole_iteration_admission(charged_actions=100,projected_actions=500,
        charged_gpu_seconds=40000,projected_gpu_seconds=5000)
    assert not whole_iteration_admission(charged_actions=999000,projected_actions=2000,
        charged_gpu_seconds=0,projected_gpu_seconds=0)


@pytest.mark.parametrize('world',[1,2,4,8])
def test_compose_actual_official_config(monkeypatch,tmp_path,world):
    import os
    from hydra import compose,initialize_config_dir
    assert Path(os.environ['EMBODIED_PATH']).is_dir(),'Allocated CPU gate must stage actual c605 config checkout'
    phase='train';schedule=evaluation_schedule(world,'DEV')
    for key,value in dict(ARM='S',PHASE=phase,WORLD=world,PLACEMENT=f'0-{world-1}',EVAL_SLOTS=len(schedule),
        ONLY_EVAL='false',MODEL='/published/RLinf40',NORM='/published/RLinf40/physical-intelligence/libero/norm_stats.json',OUTPUT=str(tmp_path),
        EXCHANGE=str(tmp_path/'exchange'),MONITORING=str(tmp_path/'monitor'),ACCOUNTING=str(tmp_path/'actions'),
        PROFILE=str(tmp_path/'profile.json'),PROFILE_SHA='a'*64,ATTEMPT='cpu').items():monkeypatch.setenv('FLORA_PPO_'+key,str(value))
    code=Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base='1.1',config_dir=str(code/'configs/native_ppo')):
        cfg=compose(config_name='libero44_flora')
    result=validate_config(cfg)
    assert result['accumulation_steps']==2048//(32*world)
    assert cfg.actor.model.openpi.action_horizon==10 and cfg.actor.model.openpi_data.extra_delta_transform is False
    assert result['estimated_iteration_actions']==(128+len(schedule))*415
    cfg.algorithm.update_epoch=5
    with pytest.raises(ValueError):validate_config(cfg)
