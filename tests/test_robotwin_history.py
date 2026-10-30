"""Native supervised fits and frozen full/static deployment; allocated CPU only."""
import json

import numpy as np
import pytest
import torch

from test_robotwin_history_corpus import fixture
from fluffy_bassoon.meta_rl.robotwin_history_corpus import export_corpus
from fluffy_bassoon.meta_rl.robotwin_history import (NativeHistoryConfig,NativeHistoryPolicy,
    FrozenNativeHistoryController,load_dataset,load_policy,sample_batch,validate_partitions)
from fluffy_bassoon.meta_rl.robotwin_history_train import fit
from fluffy_bassoon.meta_rl.vla_history import file_sha256


def datasets(tmp_path,objective='ad'):
    train_root=tmp_path/'train'; train_root.mkdir()
    corpus,args=fixture(train_root,'sac')
    train=export_corpus(corpus,tmp_path/'train-data',**args)
    validation_root=tmp_path/'val'; validation_root.mkdir()
    corpus,validation_args=fixture(validation_root,partition='validation',feature_value=2.)
    # Both DPT sets use exactly the SAME source-trained teacher, independent of
    # the policy that collected held-out source-validation experience.
    validation_args.update(teacher_path=args['teacher_path'],teacher_sha256=args['teacher_sha256'])
    validation=export_corpus(corpus,tmp_path/'val-data',objectives=('dpt','behavior_cloning'),**validation_args)
    train_index=tmp_path/'train-data/index.json'; val_index=tmp_path/'val-data/index.json'
    a,am=load_dataset(train_index,file_sha256(train_index),objective)
    b,bm=load_dataset(val_index,file_sha256(val_index),'behavior_cloning' if objective=='ad' else 'dpt')
    return a,am,b,bm


def config(): return NativeHistoryConfig(width=8,layers=1,heads=2,context_length=8)


@pytest.mark.parametrize('objective',['ad','dpt'])
@pytest.mark.parametrize('mode',['full','static'])
def test_native_full_static_fit_has_source_only_norm_and_complete_portable_state(tmp_path,objective,mode):
    train,tm,validation,vm=datasets(tmp_path,objective)
    model,summary=fit(train,tm,validation,vm,tmp_path/'fit',source_commit='a'*40,mode=mode,
        config=config(),updates=2,batch_size=2,max_window=4,validation_batches=1,
        eval_every=1,milestones=(1,2),max_seconds=100.)
    assert summary['completed'] and summary['update']==2 and summary['fit_positions']>0
    assert model.core.input_dim==3090
    torch.testing.assert_close(model.observation_mean,torch.zeros(2062),rtol=0,atol=0)
    assert all(not value.requires_grad for value in model.parameters())
    recovery=torch.load(tmp_path/'fit/latest.pt',weights_only=False)
    assert recovery['optimizer']['state'] and set(recovery['rng'])>={'sampler','python','numpy','torch','cuda'}
    path=tmp_path/'fit/candidate-000002.pt'
    loaded,manifest=load_policy(path,file_sha256(path))
    assert manifest['mode']==mode and manifest['update']==2
    for key,value in model.state_dict().items(): torch.testing.assert_close(value,loaded.state_dict()[key],rtol=0,atol=0)


def test_static_batch_removes_previous_feedback_but_keeps_both_native_clocks(tmp_path):
    train,_,_,_=datasets(tmp_path)
    model=NativeHistoryPolicy(config())
    full=sample_batch(train,model,np.random.default_rng(3),batch_size=2,max_window=8,mode='full')
    static=sample_batch(train,model,np.random.default_rng(3),batch_size=2,max_window=8,mode='static')
    torch.testing.assert_close(full[0][...,:2062],static[0][...,:2062],rtol=0,atol=0)
    torch.testing.assert_close(full[0][...,-2:],static[0][...,-2:],rtol=0,atol=0)
    assert not static[0][...,2062:-2].any()
    assert torch.equal(full[1],static[1]) and torch.equal(full[2],static[2])


def test_partition_checks_reject_overlapping_cases_wrong_roles_and_teacher_changes(tmp_path):
    _,train,_,validation=datasets(tmp_path,'dpt')
    validate_partitions(train,validation)
    with pytest.raises(ValueError,match='disjoint'):
        validate_partitions(train,dict(validation,case_ids=train['case_ids']))
    with pytest.raises(ValueError,match='disjoint'):
        validate_partitions(train,dict(validation,partition='train'))
    with pytest.raises(ValueError,match='same pinned source teacher'):
        validate_partitions(train,dict(validation,teacher_recovery_sha256='0'*64))


def test_frozen_wrapper_uses_actual_physical_noise_and_explicit_episode_trial_lifecycle(tmp_path):
    _,metadata,_,_=datasets(tmp_path)
    model=NativeHistoryPolicy(config()).eval()
    with torch.no_grad(): model.actor.bias.fill_(10.) # Exercise the physical1.5 clip.
    manifest=dict(mode='full',interface=metadata['interface'])
    controller=FrozenNativeHistoryController(model,manifest,interface=metadata['interface'])
    before={key:value.clone() for key,value in model.state_dict().items()}
    with pytest.raises(ValueError,match='Begin an episode'): controller.act(np.zeros(2062))
    controller.begin_episode(episode_index=0,episode_limit=600)
    action=controller.act(np.zeros(2062),deterministic=True)
    assert action.shape==(32,32) and np.abs(action).max()==1.5
    with pytest.raises(RuntimeError,match='pending'): controller.reset_trial()
    with pytest.raises(ValueError,match='actual bounded noise'):
        controller.observe(action+.1,0.,True,executed_steps=1)
    controller.observe(action,-1.,False,executed_steps=32)
    with pytest.raises(ValueError,match='End the previous'): controller.begin_episode(episode_index=1,episode_limit=600)
    action=controller.act(np.zeros(2062),deterministic=True)
    controller.observe(action,0.,True,executed_steps=1)
    old_state=controller.state
    with pytest.raises(ValueError,match='native horizon'):
        controller.begin_episode(episode_index=1,episode_limit=500)
    controller.begin_episode(episode_index=1,episode_limit=600)
    assert controller.state is old_state and controller.previous_reward==0.
    action=controller.act(np.zeros(2062),deterministic=True)
    controller.observe(action,0.,True,executed_steps=1)
    controller.reset_trial(seed=99)
    assert controller.state is None and controller.episode_index==-1 and not controller.previous_noise.any()
    for key,value in model.state_dict().items(): torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_frozen_static_and_full_lesion_keep_clock_and_rng_contract(tmp_path):
    _,metadata,_,_=datasets(tmp_path)
    model=NativeHistoryPolicy(config())
    manifest=dict(mode='full',interface=metadata['interface'])
    controller=FrozenNativeHistoryController(model,manifest,interface=metadata['interface'],erase_each_episode=True)
    controller.reset_trial(seed=19);controller.begin_episode(episode_index=0,episode_limit=600)
    action=controller.act(np.zeros(2062));controller.observe(action,-1.,True,executed_steps=1)
    controller.begin_episode(episode_index=1,episode_limit=600)
    assert controller.state is None and not controller.previous_noise.any() and controller.previous_reward==0.
    assert controller.episode_index==1 # Reset lesion does not erase the current public clock.
    action=controller.act(np.zeros(2062));controller.observe(action,0.,True,executed_steps=1)
    controller.reset_trial(seed=19);controller.begin_episode(episode_index=0,episode_limit=600)
    first=controller.act(np.zeros(2062));controller.observe(first,0.,True,executed_steps=1)
    controller.reset_trial(seed=19);controller.begin_episode(episode_index=0,episode_limit=600)
    second=controller.act(np.zeros(2062))
    np.testing.assert_array_equal(first,second)


def test_model_loader_rejects_changed_weights_and_inference_interface(tmp_path):
    train,tm,validation,vm=datasets(tmp_path)
    _,_=fit(train,tm,validation,vm,tmp_path/'fit',source_commit='a'*40,config=config(),
        updates=1,batch_size=1,max_window=2,validation_batches=1,eval_every=1,milestones=(1,),max_seconds=100.)
    path=tmp_path/'fit/candidate-000001.pt'
    with pytest.raises(ValueError,match='checksum'): load_policy(path,'0'*64)
    model,manifest=load_policy(path,file_sha256(path))
    with pytest.raises(ValueError,match='interface differs'):
        FrozenNativeHistoryController(model,manifest,interface=dict(manifest['interface'],action_limit=1.))


def test_interrupted_native_fit_restores_optimizer_sampler_and_rng(tmp_path,monkeypatch):
    import fluffy_bassoon.meta_rl.robotwin_history_train as training
    train,tm,validation,vm=datasets(tmp_path)
    arguments=dict(source_commit='a'*40,config=config(),updates=3,batch_size=2,max_window=4,
        validation_batches=1,eval_every=1,milestones=(1,2,3),max_seconds=100.)
    expected,summary=fit(train,tm,validation,vm,tmp_path/'complete',**arguments)
    original=training.sample_batch
    calls=0
    def interrupted(trials,*args,**kwargs):
        nonlocal calls
        if trials is train:
            calls+=1
            if calls==2: raise RuntimeError('interrupted before next complete update')
        return original(trials,*args,**kwargs)
    monkeypatch.setattr(training,'sample_batch',interrupted)
    with pytest.raises(RuntimeError,match='interrupted'):
        fit(train,tm,validation,vm,tmp_path/'resumed',**arguments)
    monkeypatch.setattr(training,'sample_batch',original)
    actual,resumed=fit(train,tm,validation,vm,tmp_path/'resumed',
        resume_sha256=file_sha256(tmp_path/'resumed/latest.pt'),**arguments)
    assert resumed['fit_positions']==summary['fit_positions'] and resumed['update']==3
    for key,value in actual.state_dict().items(): torch.testing.assert_close(value,expected.state_dict()[key],rtol=0,atol=0)
