"""Native source-only paired labels; run on allocated Linux CPU resources."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.robotwin_history_corpus import (SCHEMA, TASKS, _digest_array,
    export_corpus, load_teacher, native_tokens, read_source, source_recovery, teacher_labels)
from fluffy_bassoon.meta_rl.vla_history import file_sha256
from fluffy_bassoon.meta_rl.vla_noise_ppo import PPOConfig, RecurrentNoisePPO, policy_inputs
from fluffy_bassoon.meta_rl.vla_noise_sac import NoiseSAC, SACConfig


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def array_ref(root, relative, **arrays):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    return dict(path=relative, bytes=path.stat().st_size, sha256=file_sha256(path),
        arrays={key: dict(dtype=str(value.dtype), shape=list(value.shape), sha256=_digest_array(value))
                for key, value in arrays.items()})


def recovery(root, method, *, updates=2, trials=2, reset=False):
    runtime = dict(norm_stats_sha256='c' * 64, rlinf_commit='d' * 40, robotwin_commit='e' * 40,
        features='masked prefix2048 plus normalized proprio14', action_transform='native ALOHA',
        task_config_name='demo_clean', appearance_randomization=False)
    contract = dict(source_commit='a' * 40, source_tasks=list(TASKS), observation_dim=2062,
        noise_shape=[32, 32], native_horizon=32, action_limit=1.5, train_episodes=4,
        model_path='/fixed/motus', episode_limit=600, task_episode_limits={task:600 if task=='handover_mic'
            else 500 if task=='scan_object' else 400 for task in TASKS}, runtime=runtime,
        source_iterations=dict(count=100, trials_per_iteration=6))
    if method == 'sac':
        learner = NoiseSAC(2062, 1024, device='cpu', seed=3, config=SACConfig(hidden_dims=(8,),
            batch_size=2, replay_capacity=8), action_limit=1.5)
    else:
        learner = RecurrentNoisePPO(2062, (32,32), device='cpu', seed=3,
            config=PPOConfig(width=8,layers=1,heads=2,memory_length=8,episode_limit=600,
                reset_each_episode=reset or method=='ppo_reset'))
    learner.updates = updates
    payload = dict(contract=contract, method=method, learner=learner.state_dict(),
        schedule=dict(tasks=list(TASKS), trials=trials), source_steps=trials*4*32,
        task_steps={task:trials*4*32 if task==TASKS[0] else 0 for task in TASKS},
        boundary='completed whole-trial source iteration')
    path=root/f'{method}-recovery.pt'
    torch.save(payload,path)
    return path, payload


def fixture(tmp_path, method='ppo', *, fixed=False):
    source_path, source = recovery(tmp_path,method)
    teacher_path, teacher = recovery(tmp_path,'sac',updates=3)
    root = tmp_path/'corpus'
    root.mkdir()
    write_json(root/'contract.json',dict(schema=SCHEMA,source_commit='a'*40,source_tasks=list(TASKS),model_path='/fixed/motus'))
    # Actual db01840 schema: corpus retains allocation identity, checkpoint omits it.
    write_json(root/'runtime-contract.json',dict(source['contract']['runtime'],actual_render_pci='0000:a3:00.0'))
    for trial in (1,2):
        version_update = 0 if fixed else trial-1
        policy_path=root/'policies'/f'{version_update}.pt'
        policy_path.parent.mkdir(exist_ok=True)
        state={'weight':torch.tensor([float(version_update)])}
        parameter_digest=hashlib.sha256()
        for name,value in sorted(state.items()):
            parameter_digest.update(name.encode()+str(value.dtype).encode()+str(tuple(value.shape)).encode())
            parameter_digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        parameter_sha=parameter_digest.hexdigest()
        if not policy_path.exists():
            torch.save(dict(kind='sac_actor' if method=='sac' else 'ppo_actor_value',
                state_dict=state,parameter_sha256=parameter_sha),policy_path)
        context=dict(phase='source',method=method,trial_id=f'source:{method}:{trial}',
            policy_version=f'{method}:update:{version_update}:{parameter_sha}',
            policy_sha256=parameter_sha,policy_checkpoint=str(policy_path),
            policy_checkpoint_sha256=file_sha256(policy_path))
        previous_noise,previous_reward=np.zeros(1024,np.float32),0.
        for episode in range(4):
            episode_id=f'trial{trial}-ep{episode}'
            episode_dir=root/'episodes'/episode_id
            frame=array_ref(root,f'frames/{episode_id}-0.npz',features=np.zeros(2062,np.float32))
            write_json(episode_dir/'reset.json',dict(schema=SCHEMA,episode_id=episode_id,context=context,
                reset_info=dict(task=TASKS[0],requested_seed=episode,actual_seed=episode),initial_observation=frame))
            if method=='ppo_reset':
                previous_noise,previous_reward=np.zeros(1024,np.float32),0.
            refs=[]
            for chunk in range(2):
                noise=np.full((32,32),.15*trial+.01*chunk,np.float32)
                done=chunk==1
                duration=1 if done else 31
                metadata=dict(episode_index=episode,chunk_index=chunk,episode_limit=600,
                    optimizer_updates=version_update,policy_update=version_update,
                    history_index=episode*2+chunk,reset_episode_feedback=method=='ppo_reset',
                    recurrent_reset_mask=method=='ppo_reset' and chunk==0)
                decision_arrays={'sampled_noise':noise,'issued_motor_joint_targets':np.zeros((32,14),np.float32)}
                if method!='sac':
                    token=policy_inputs(np.zeros(2062,np.float32),previous_noise,previous_reward,
                        method!='ppo_reset' and episode>0 and chunk==0,episode,31*chunk,
                        train_episodes=4,episode_limit=600)
                    decision_arrays['metadata/policy_input']=token
                    decision_arrays['metadata/pre_squash_actions']=np.arctanh(noise.reshape(-1)/1.5)
                    metadata['policy_input']={'numeric_array':'metadata/policy_input'}
                    metadata['pre_squash_actions']={'numeric_array':'metadata/pre_squash_actions'}
                decision=array_ref(root,f'episodes/{episode_id}/chunks/{chunk+1:06d}.npz',**decision_arrays)
                native=array_ref(root,f'episodes/{episode_id}/chunks/{chunk+1:06d}-native.npz',
                    valid_action_mask=np.asarray([[True]*duration+[False]*(32-duration)]))
                following=array_ref(root,f'frames/{episode_id}-{chunk+1}.npz',features=np.zeros(2062,np.float32))
                row=dict(schema=SCHEMA,status='complete',episode_id=episode_id,chunk_index=chunk+1,
                    context=context,observation=frame,next_observation=following,decision_arrays=decision,
                    decision_metadata=metadata,native_result=dict(reward=0. if done else -1.,
                        terminated=done,truncated=False,executed_primitive_steps=duration,arrays=native),
                    learning_result=dict(reward=0. if done else -1.,terminated=done,truncated=False,
                        budget_truncated=False,episode_primitive_steps=32 if done else 31))
                path=episode_dir/'chunks'/f'{chunk+1:06d}.json'
                write_json(path,row)
                refs.append(dict(path=f'chunks/{chunk+1:06d}.json',sha256=file_sha256(path)))
                previous_noise,previous_reward=noise.reshape(-1),row['learning_result']['reward']
                frame=following
            write_json(episode_dir/'manifest.json',dict(schema=SCHEMA,episode_id=episode_id,
                pending_decision=None,chunks=refs,completed_chunks=2,executed_primitive_steps=32))
    return root,dict(source_path=source_path,source_sha256=file_sha256(source_path),source_method=method,
        teacher_path=teacher_path,teacher_sha256=file_sha256(teacher_path),teacher_method='sac')


@pytest.mark.parametrize('method',['ppo','ppo_reset','sac'])
def test_native_paired_export_has_full_context_physical_scale_and_no_source_mutation(tmp_path,method):
    root,args=fixture(tmp_path,method)
    result=export_corpus(root,tmp_path/'paired',**args)
    with np.load(result['datasets']['ad']['path']) as ad,np.load(result['datasets']['dpt']['path']) as dpt:
        assert ad['observations'].shape==(16,2062)
        assert ad['history_inputs'].shape==(16,3090)
        assert ad['target_noise'].shape==(16,1024)
        for key in ad.files:
            if key!='target_noise': np.testing.assert_array_equal(ad[key],dpt[key])
        np.testing.assert_array_equal(ad['target_noise'],ad['executed_noise'])
        assert not np.array_equal(ad['target_noise'],dpt['target_noise'])
        assert np.abs(dpt['target_noise']).max()<=1.5
        assert ad['history_inputs'][1,-1]==np.float32(31/600)
        # At the next source-trial start, full student history retains actual
        # terminal feedback/action, unlike the independently reset teacher.
        np.testing.assert_array_equal(ad['history_inputs'][8,2062:3086],ad['executed_noise'][7])
        assert ad['history_inputs'][8,-3]==1.
    assert result['counts']['complete_K4_trials']==2
    assert result['counts']['selected_actions']==256
    assert all(file_sha256(path)==sha for path,sha in result['input_sha256'].items())
    repeated=export_corpus(root,tmp_path/'repeat',**args)
    assert result['datasets']['dpt']['sha256']==repeated['datasets']['dpt']['sha256']


def test_recurrent_teacher_replays_its_own_resets_not_behavior_cache(tmp_path):
    root,args=fixture(tmp_path,'ppo')
    snapshot=source_recovery(args['source_path'],args['source_sha256'],'ppo')
    _,trials,_=read_source(root,snapshot)
    rows=trials[0]['rows']
    teacher,reset=load_teacher(snapshot,device='cpu')
    before={key:value.clone() for key,value in teacher.state_dict().items()}
    full=teacher_labels(teacher,'ppo',rows,reset_each_episode=False)
    episode=teacher_labels(teacher,'ppo_reset',rows,reset_each_episode=True)
    np.testing.assert_array_equal(episode[0],episode[2])
    assert not np.array_equal(full[2:],episode[2:])
    changed=[dict(row) for row in rows]
    changed[1]['reward']=19.
    changed[1]['noise']=np.full(1024,.9,np.float32)
    altered=teacher_labels(teacher,'ppo',changed,reset_each_episode=False)
    np.testing.assert_array_equal(full[:2],altered[:2])
    assert not np.array_equal(full[2:],altered[2:])
    for key,value in teacher.state_dict().items(): torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_fixed_policy_history_supports_dpt_but_not_false_ad(tmp_path):
    root,args=fixture(tmp_path,fixed=True)
    with pytest.raises(ValueError,match='actual chronological behavior policy changes'):
        export_corpus(root,tmp_path/'invalid',**args)
    assert not (tmp_path/'invalid').exists()
    export_corpus(root,tmp_path/'dpt',objectives=('dpt',),**args)


def test_only_source_and_committed_cursor_are_selected(tmp_path):
    root,args=fixture(tmp_path)
    write_json(root/'episodes'/'target'/'reset.json',dict(context=dict(phase='target',method='ppo')))
    write_json(root/'episodes'/'future'/'reset.json',dict(context=dict(phase='source',method='ppo',trial_id='source:ppo:3')))
    result=export_corpus(root,tmp_path/'paired',**args)
    assert result['excluded']==dict(non_source_or_other_method_episodes=1,after_committed_cursor_episodes=1)
    assert result['counts']['decisions']==16


@pytest.mark.parametrize('kind',['missing_episode','unfinished','corrupt_frame','native_duration'])
def test_incomplete_or_corrupted_native_evidence_is_rejected(tmp_path,kind):
    root,args=fixture(tmp_path)
    directory=root/'episodes/trial1-ep0'
    if kind=='missing_episode':
        (directory/'reset.json').unlink()
    elif kind=='unfinished':
        path=directory/'manifest.json';value=json.loads(path.read_text());value['pending_decision']='pending';write_json(path,value)
    elif kind=='corrupt_frame':
        path=root/'frames/trial1-ep0-0.npz';path.write_bytes(b'corrupt')
    else:
        path=directory/'chunks/000001.json';value=json.loads(path.read_text());value['native_result']['executed_primitive_steps']=30;write_json(path,value)
        manifest=directory/'manifest.json';value=json.loads(manifest.read_text());value['chunks'][0]['sha256']=file_sha256(path);write_json(manifest,value)
    with pytest.raises((ValueError,FileNotFoundError)):
        export_corpus(root,tmp_path/'invalid',**args)


def test_target_adapted_teacher_cannot_label_source_data(tmp_path):
    root,args=fixture(tmp_path)
    teacher=torch.load(args['teacher_path'],weights_only=False)
    teacher['learner']['target_replay_mode']='fresh_target'
    torch.save(teacher,args['teacher_path'])
    args['teacher_sha256']=file_sha256(args['teacher_path'])
    with pytest.raises(ValueError,match='Target-updated SAC'):
        export_corpus(root,tmp_path/'invalid',**args)


def test_sac_warmup_has_no_actor_and_does_not_invent_an_update(tmp_path):
    root,args=fixture(tmp_path,'sac')
    path=root/'episodes/trial1-ep0/chunks/000001.json'
    value=json.loads(path.read_text())
    value['context'].update(policy_version='bounded_uniform_warmup',policy_checkpoint=None,
        policy_sha256=None,policy_checkpoint_sha256=None)
    value['decision_metadata']['source_warmup']=True
    write_json(path,value)
    manifest=path.parent.parent/'manifest.json'
    value=json.loads(manifest.read_text());value['chunks'][0]['sha256']=file_sha256(path);write_json(manifest,value)
    result=export_corpus(root,tmp_path/'paired',**args)
    with np.load(result['datasets']['ad']['path']) as data:
        assert data['behavior_versions'][0]=='bounded_uniform_warmup'
        assert data['behavior_updates'][0]==data['behavior_updates'][1]==0


def test_scientific_runtime_fields_are_not_normalized_away(tmp_path):
    root,args=fixture(tmp_path)
    path=root/'runtime-contract.json';value=json.loads(path.read_text());value['features']='different encoder'
    write_json(path,value)
    with pytest.raises(ValueError,match='source/runtime contract differs'):
        export_corpus(root,tmp_path/'invalid',**args)


def test_frozen_source_action_counters_must_reconcile(tmp_path):
    root,args=fixture(tmp_path)
    value=torch.load(args['source_path'],weights_only=False)
    value['source_steps']+=1
    torch.save(value,args['source_path'])
    args['source_sha256']=file_sha256(args['source_path'])
    with pytest.raises(ValueError,match='frozen source_steps/task_steps'):
        export_corpus(root,tmp_path/'invalid',**args)


def test_acting_archive_tensors_are_verified_not_just_declared_hash_strings(tmp_path):
    root,args=fixture(tmp_path)
    actor=root/'policies/0.pt'
    saved=torch.load(actor,weights_only=False)
    saved['state_dict']['weight']+=7.
    torch.save(saved,actor)
    digest=file_sha256(actor)
    for directory in (root/'episodes').glob('trial1-*'):
        manifest=json.loads((directory/'manifest.json').read_text())
        for ref in manifest['chunks']:
            path=directory/ref['path'];value=json.loads(path.read_text())
            value['context']['policy_checkpoint_sha256']=digest
            write_json(path,value);ref['sha256']=file_sha256(path)
        write_json(directory/'manifest.json',manifest)
    with pytest.raises(ValueError,match='tensor parameter digest'):
        export_corpus(root,tmp_path/'invalid',**args)
