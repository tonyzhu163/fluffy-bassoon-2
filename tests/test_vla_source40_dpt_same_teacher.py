"""Allocated CPU tests with NumPy actors; no SAC/VLA model construction."""
import copy
import time

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_source40_dpt_same_teacher import (
    METRIC,non_target_hashes,relabel_trial,sac_input,sha,verify_teacher_reference)


def history():
    obs=np.arange(18,dtype=np.float32).reshape(6,3)
    trial=dict(observations=obs,executed_noise=np.zeros((6,32),np.float32),target_noise=np.full((6,32),-.75,np.float32),
        rewards=np.asarray([-1,-1,0,-1,-1,0],np.float32),dones=np.asarray([0,0,1,0,0,1],bool),
        episode_starts=np.asarray([1,0,0,1,0,0],bool),teacher_steps=np.arange(10,16,dtype=np.int64))
    rows=[dict(public_features=obs[i].copy(),executed_noise=trial['executed_noise'][i].copy(),
        reward=float(trial['rewards'][i]),done=bool(trial['dones'][i]),source_updates_before=int(trial['teacher_steps'][i]),
        sac_observation=dict(pixels=np.zeros((64,64,3,1),np.uint8),state=np.full((8,1),i/10,np.float32))) for i in range(6)]
    return trial,rows


def test_only_targets_change_from_train_actor_and_behavior_clock_is_retained():
    trial,rows=history();before=non_target_hashes(trial);calls=[]
    def train_actor(obs):
        calls.append(len(obs['state']));return np.repeat(obs['state'][:,0,0,None],32,axis=1)
    derived=relabel_trial(trial,iter(rows),train_actor,deadline=time.time()+60,batch_size=2)
    assert calls==[2,2,2] and non_target_hashes(derived)==before==non_target_hashes(trial)
    np.testing.assert_array_equal(derived['target_noise'][:,0],np.arange(6,dtype=np.float32)/10)
    assert not np.array_equal(derived['target_noise'],trial['target_noise'])
    np.testing.assert_array_equal(derived['teacher_steps'],np.arange(10,16))


def test_reordered_behavior_or_wrong_input_shape_rejects_before_export():
    trial,rows=history();rows[0],rows[1]=rows[1],rows[0]
    with pytest.raises(ValueError,match='immutable VAL history'):
        relabel_trial(trial,rows,lambda obs:np.zeros((len(obs['state']),32)),deadline=time.time()+60)
    _,rows=history();rows[0]['sac_observation']['pixels']=np.zeros((64,64,3),np.uint8)
    with pytest.raises(ValueError,match='canonical'):
        sac_input(rows[0])


def test_missing_or_extra_decisions_are_not_silently_padded_or_filtered():
    trial,rows=history();evaluate=lambda obs:np.zeros((len(obs['state']),32),np.float32)
    with pytest.raises(ValueError,match='omitted'):
        relabel_trial(trial,rows[:-1],evaluate,deadline=time.time()+60)
    with pytest.raises(ValueError,match='extra'):
        relabel_trial(trial,rows+[rows[-1]],evaluate,deadline=time.time()+60)


def test_deadline_stops_without_inference_or_gpu_fallback():
    trial,rows=history()
    with pytest.raises(TimeoutError):
        relabel_trial(trial,rows,lambda obs:pytest.fail('actor must not run after deadline'),deadline=time.time()-1)


def test_val_actor_checkpoint_cannot_substitute_for_preselected_train_teacher():
    train=dict(checkpoint_path='/train/teacher',checkpoint_sha256='a'*64,dataset_sha256='b'*64,metadata=dict(source_updates=28620))
    teacher=dict(checkpoint_path='/train/teacher',checkpoint_sha256='a'*64,train_dpt_sha256='b'*64,source_updates=28620,pure_actor_sha256='c'*64)
    verify_teacher_reference(teacher,train)
    changed=dict(teacher,checkpoint_path='/validation/teacher',checkpoint_sha256='d'*64)
    with pytest.raises(ValueError,match='exact preselected TRAIN'):
        verify_teacher_reference(changed,train)


def test_export_preserves_original_metadata_and_all_nonlabel_bytes(tmp_path):
    from fluffy_bassoon.meta_rl.vla_history import load_dataset,save_dataset
    trial,rows=history();original=tmp_path/'original.npz';meta=dict(source_contract=dict(partition='validation'),
        source_updates=10500,source_actor_checkpoint_sha256='v'*64,history_policy='actual behavior')
    original_manifest=save_dataset(original,[trial],objective='dpt',metadata=meta);before=sha(original)
    derived=relabel_trial(trial,rows,lambda obs:np.ones((len(obs['state']),32),np.float32),deadline=time.time()+60)
    teacher=dict(source_updates=28620,checkpoint_sha256='t'*64,pure_actor_sha256='p'*64)
    metadata=copy.deepcopy(meta);metadata.update(label_teacher=teacher,metric_name=METRIC)
    out=tmp_path/'derived.npz';manifest=save_dataset(out,[derived],objective='dpt',metadata=metadata)
    loaded,saved=load_dataset(out,expected_sha256=manifest['sha256'])
    assert sha(original)==before==original_manifest['sha256']
    assert all(saved['metadata'][key]==value for key,value in meta.items())
    assert saved['metadata']['source_updates']==10500 and saved['metadata']['label_teacher']['source_updates']==28620
    assert non_target_hashes(loaded[0])==non_target_hashes(trial)
