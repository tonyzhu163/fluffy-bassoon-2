"""Only new fixed-bank profile/state-fork/rescore checks; allocated Linux only.

Synthetic states below do not qualify the original four model files. The
separate inspect operation reads those actual SHA-bound originals without fits.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import vla_source40_fixed_bank_depth as depth
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy, save_dataset, sample_batch
from fluffy_bassoon.meta_rl.vla_history_train import _rng_state, supervised_update


@pytest.fixture(scope='module',autouse=True)
def bounded_synthetic_cpu_threads():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def state():
    contract = dict(source_commit=depth.ORIGINAL_SOURCE, config=deepcopy(depth.CONFIG), mode='full',
        objective='ad',train_sha256='a'*64,validation_sha256='b'*64,train_metadata={'source':'train'},
        validation_metadata={'source':'validation'},seed=0,batch_size=8,max_context=128,learning_rate=.0003,
        supervised_objective='Gaussian target-noise NLL; motor adapter applies its fixed noise map',
        status='candidate; imitation NLL does not establish robot performance')
    policy = dict(observation_mean=torch.zeros(4104),observation_std=torch.ones(4104),toy=torch.tensor([.2]))
    saved = dict(format='frozen_vla_history_resume_v1',contract=contract,update=100,policy=policy,
        optimizer=dict(state={0:dict(step=torch.tensor(100.),exp_avg=torch.zeros(1),exp_avg_sq=torch.ones(1))},
            param_groups=[dict(params=[0],lr=.0003,weight_decay=0.,betas=(.9,.999),eps=1e-8,amsgrad=False,maximize=False)]),
        rng=dict(sampler=np.random.default_rng(0).bit_generator.state,python=(1,2),numpy=np.arange(3),
            torch=torch.zeros(4,dtype=torch.uint8),cuda=[torch.ones(4,dtype=torch.uint8)]),
        best_validation_nll=1.5,elapsed_seconds=91.)
    endpoint = dict(format='frozen_vla_history_policy_v1',config=deepcopy(depth.CONFIG),policy=deepcopy(policy),
        manifest=dict(update=100,contract=deepcopy(contract)))
    return saved, endpoint, dict(old_contract={k:contract[k] for k in ('source_commit','config','mode','objective',
        'train_sha256','validation_sha256','seed','batch_size','max_context','learning_rate')})


def test_metadata_fork_preserves_original_model_adam_rng_and_normalizer_exactly():
    original,endpoint,spec = state()
    report = depth.validate_saved(original,endpoint,spec)
    assert report == json.loads(json.dumps(report))  # CPU receipt and GPU readback preserve tuple semantics.
    old = depth.tree_sha(original)
    migrated = depth.fork_state(original,'c'*40)
    assert depth.tree_sha(original) == old
    assert migrated['update'] == 100 and migrated['elapsed_seconds'] == 0
    assert migrated['best_validation_nll'] == original['best_validation_nll']
    for field in ('policy','optimizer','rng'):
        assert depth.tree_sha(migrated[field]) == depth.tree_sha(original[field])
    assert migrated['contract']['schedule'] == depth.SCHEDULE
    assert report['normalizer_sha256'] == depth.tree_sha({k:migrated['policy'][k]
        for k in ('observation_mean','observation_std')})


@pytest.mark.parametrize('corruption',['normalizer','optimizer','rng','contract'])
def test_original_state_mismatch_stops_before_migration(corruption):
    saved,endpoint,spec=state()
    if corruption=='normalizer': endpoint['policy']['observation_mean'][0]=1
    if corruption=='optimizer': saved['optimizer']['param_groups'][0]['lr']=.001
    if corruption=='rng': saved['rng'].pop('cuda')
    if corruption=='contract': saved['contract']['scheduler']='unreviewed_decay'
    with pytest.raises(ValueError): depth.validate_saved(saved,endpoint,spec)


def profile():
    path=Path(__file__).resolve().parents[1]/'outputs/libero40-fixed-bank500-fit-preparation-20261101-r1/PROFILE.json'
    return depth.load_profile(path,depth.sha(path))


def test_existing_trainer_command_uses_original_train_legacy_val_and_fixed_final500(tmp_path):
    p=profile()
    command=depth.trainer_command(p,'dpt_full',tmp_path,source_commit='c'*40,hostname='allocated-test')
    assert command[command.index('--updates')+1]=='500'
    assert command[command.index('--checkpoint-selection')+1]=='final'
    assert command[command.index('--validation-sha256')+1]==p['data']['dpt']['legacy_validation']['sha256']
    assert p['data']['dpt']['corrected_validation']['sha256'] not in command
    assert command[command.index('--resume')+1]==str(tmp_path/'latest.pt')
    assert '--execution-batch-size' not in command and '--max-episodes-per-window' not in command


def test_corrected_metric_changes_no_original_weights_normalizer_optimizer_or_rng(tmp_path):
    torch.manual_seed(17)
    model=HistoryNoisePolicy(HistoryConfig(2,1,width=8,layers=1,heads=2,context_length=4))
    trial=dict(observations=np.arange(12,dtype=np.float32).reshape(6,2),
        executed_noise=np.zeros((6,1),np.float32),target_noise=np.ones((6,1),np.float32),
        rewards=np.zeros(6,np.float32),dones=np.array([False,False,True,False,False,True]),
        episode_starts=np.array([True,False,False,True,False,False]),teacher_steps=np.arange(6,dtype=np.int64))
    artifact=tmp_path/'corrected.npz'
    saved_manifest=save_dataset(artifact,[trial],objective='dpt',metadata={'synthetic_only':True})
    saved=dict(update=100,contract=dict(config=dict(observation_dim=2,noise_dim=1,width=8,layers=1,
        heads=2,context_length=4,initial_std=.5)),policy=deepcopy(model.state_dict()),
        optimizer={'toy_state':torch.ones(1)},rng={'original_rng':torch.ones(4,dtype=torch.uint8)})
    sampler=np.random.default_rng(42)
    before=depth.tree_sha(_rng_state(sampler)); identity=depth.tree_sha(saved)
    result=depth.corrected_rescore(saved,dict(path=str(artifact),sha256=saved_manifest['sha256']),mode='full',device='cpu')
    assert depth.tree_sha(_rng_state(sampler))==before and depth.tree_sha(saved)==identity
    assert result['update']==100 and result['optimizer_updates']==0 and not result['checkpoint_selection_influence']


def test_clock_rejects_model_budget_borrowing_even_with_global_room(tmp_path):
    p=profile();p['root_authority_files']={}
    row=dict(path=str(tmp_path/'attempt'),status='started',lineage='ad_full',lineage_seconds=1800,
        slice_seconds=1740,reserved_seconds=1800,started_unix=1)
    clock=dict(contract=depth.clock_contract(p,'a'*64,'c'*40),worker_seconds=7200,attempts=[row])
    path=tmp_path/'worker-time.json';path.write_text(json.dumps(clock))
    assert depth.owned_attempt(tmp_path,tmp_path/'attempt',p,'a'*64,'c'*40,'ad_full')==row
    clock['attempts'].append(dict(row,path=str(tmp_path/'old-attempt'),status='finished',charged_seconds=1))
    path.write_text(json.dumps(clock))
    with pytest.raises(ValueError,match='subcap'):
        depth.owned_attempt(tmp_path,tmp_path/'attempt',p,'a'*64,'c'*40,'ad_full')


def test_current_fullbatch_path_matches_historical_single_loss_adam_step():
    """Only the active legacy->current update-path difference; tiny synthetic data."""
    torch.manual_seed(23)
    legacy=HistoryNoisePolicy(HistoryConfig(2,1,width=8,layers=1,heads=2,context_length=4))
    current=deepcopy(legacy)
    trial=dict(observations=np.arange(8,dtype=np.float32).reshape(4,2)/10,
        executed_noise=np.zeros((4,1),np.float32),target_noise=np.ones((4,1),np.float32),
        rewards=np.zeros(4,np.float32),dones=np.array([False,True,False,True]),
        episode_starts=np.array([True,False,True,False]),teacher_steps=np.arange(4,dtype=np.int64))
    batch=sample_batch([trial],legacy,np.random.default_rng(1),batch_size=2,max_context=4,mode='full')
    old_optimizer=torch.optim.AdamW(legacy.parameters(),lr=.0003,weight_decay=0.)
    new_optimizer=torch.optim.AdamW(current.parameters(),lr=.0003,weight_decay=0.)
    legacy.train();old_optimizer.zero_grad(set_to_none=True)
    loss=legacy.sequence_nll(*batch,mode='full');loss.backward()
    torch.nn.utils.clip_grad_norm_(legacy.parameters(),1.,error_if_nonfinite=True)
    old_optimizer.step()
    supervised_update(current,new_optimizer,batch,mode='full',execution_batch_size=None)
    assert depth.tree_sha(legacy.state_dict())==depth.tree_sha(current.state_dict())
    assert depth.tree_sha(old_optimizer.state_dict())==depth.tree_sha(new_optimizer.state_dict())
