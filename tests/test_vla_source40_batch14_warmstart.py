"""Allocated-Linux tests of lossless warm migration and restored RNG/counters."""
import copy

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.checkpoint import restore_rng, rng_state
from fluffy_bassoon.meta_rl.vla_rl2 import RL2Config
from fluffy_bassoon.meta_rl.vla_rl2_batch import training_contract
from fluffy_bassoon.meta_rl import vla_source40_batch14_warmstart as warm


def payload(mode='full'):
    policy=torch.nn.Linear(2,1)
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
    steps=476 if mode=='full' else 474
    # Populate the exact Adam schema without performing any restore-time SGD.
    for parameter in policy.parameters():
        optimizer.state[parameter]=dict(step=torch.tensor(float(steps)),
            exp_avg=torch.full_like(parameter,.123),exp_avg_sq=torch.full_like(parameter,.456))
    sampler=np.random.default_rng(17)
    config=RL2Config(observation_dim=3,noise_dim=2,width=8,layers=1,heads=2,context_length=4)
    history=dict(prior_actions=195964,mode=mode,marker='fixed hash-bound history')
    contract=training_contract(config,dict(source_only=True,source_data_from_targets=False,warm_start=history),
        updates=1,max_total_actions=20000,seed=2,mode=mode,learning_rate=1e-4,epochs=2)
    cycle=dict(format='source40_ppo_task_cursor_v1',cursor=steps//2,remaining=0,
        exposure={'task':dict(trials_started=steps//2,episodes=steps*2,primitive_steps=12345)})
    old=dict(version=1,boundary='between_trials',iteration=steps//2,
        contract=dict(mode=mode,config=contract['config'],optimizer=dict(name='Adam',learning_rate=1e-4)),
        policy=copy.deepcopy(policy.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
        environment_rng=copy.deepcopy(sampler.bit_generator.state),rng=rng_state(),
        runtime=dict(task_cycle=cycle,motor_rng=dict(format='jax_motor_key_v1',implementation='threefry2x32',
            key_data=np.asarray([872,931],dtype=np.uint32))),
        counters=dict(environment_steps=12345,settling_steps=steps*20,optimizer_steps=steps))
    newcycle=copy.deepcopy(cycle);newcycle.update(cursor=0,exposure={'task':dict(trials_started=0,episodes=0,primitive_steps=0)})
    return old,contract,history,newcycle,policy,optimizer,sampler


@pytest.mark.parametrize('mode',['full','episode'])
def test_phase_migration_preserves_every_inherited_leaf_and_separates_adam_clock(mode):
    old,contract,history,cycle,*_=payload(mode)
    before=warm.legacy_fingerprint(old)
    saved=warm.initial_payload(old,contract,history,cycle)
    inherited=warm.verify_inheritance(saved)
    assert warm.legacy_fingerprint(old)==before==warm.legacy_fingerprint(inherited['legacy_recovery'])
    assert warm.fingerprint(warm.operational(saved))==warm.fingerprint(warm.operational(old,legacy=True))
    assert saved['progress']['iteration']==saved['progress']['completed_trials']==0
    assert saved['progress']['counters']['optimizer_steps']==0
    assert saved['runtime']['task_cycle']['cursor']==0
    assert inherited['legacy_recovery']['runtime']['task_cycle']['cursor']==old['iteration']
    assert {float(row['step']) for row in saved['optimizer']['state'].values()}=={old['counters']['optimizer_steps']}
    saved['runtime']['task_cycle']['cursor']=14
    assert inherited['legacy_recovery']['runtime']['task_cycle']['cursor']==old['iteration']


@pytest.mark.parametrize('change',['mode','adam','typed_key','rng','partial_trial'])
def test_migration_rejects_changed_scientific_or_incomplete_rng_boundary(change):
    old,contract,history,cycle,*_=payload()
    if change=='mode':old['contract']['mode']='episode'
    if change=='adam':old['counters']['optimizer_steps']-=1
    if change=='typed_key':old['runtime']['motor_rng']['key_data']=np.asarray([872,931],dtype=np.int64)
    if change=='rng':del old['rng']['numpy']
    if change=='partial_trial':old['runtime']['task_cycle']['remaining']=1
    with pytest.raises(ValueError,match='Legacy recovery'):
        warm.initial_payload(old,contract,history,cycle)


@pytest.mark.parametrize('leaf',['policy','optimizer','environment_rng','motor_rng','exposure'])
def test_immutable_inherited_leaf_guard_rejects_mutation(leaf):
    old,contract,history,cycle,*_=payload()
    saved=warm.initial_payload(old,contract,history,cycle)
    inherited=saved['progress']['inheritance']['legacy_recovery']
    if leaf=='policy':inherited['policy']['bias']+=1
    if leaf=='optimizer':next(iter(inherited['optimizer']['state'].values()))['exp_avg']+=1
    if leaf=='environment_rng':inherited['environment_rng']['state']['state']+=1
    if leaf=='motor_rng':inherited['runtime']['motor_rng']['key_data'][0]+=1
    if leaf=='exposure':inherited['runtime']['task_cycle']['exposure']['task']['episodes']+=1
    with pytest.raises(ValueError,match='inherited state'):
        warm.verify_inheritance(saved)


def test_actual_restore_performs_zero_sgd_and_guards_sampler_global_motor_rng(tmp_path):
    old,contract,history,cycle,policy,optimizer,sampler=payload()
    saved=warm.initial_payload(old,contract,history,cycle)
    policy.load_state_dict(saved['policy']);optimizer.load_state_dict(saved['optimizer'])
    sampler.bit_generator.state=saved['sampler'];restore_rng(saved['rng'])
    runtime=copy.deepcopy(saved['runtime'])
    warm.verify_restoration(saved,policy,optimizer,sampler,runtime,tmp_path)
    assert len(list(tmp_path.glob('warm-restore-*.json')))==1
    assert {float(row['step']) for row in optimizer.state.values()}=={476.}
    runtime['motor_rng']['key_data'][1]+=1
    with pytest.raises(RuntimeError,match='motor RNG'):
        warm.verify_restoration(saved,policy,optimizer,sampler,runtime,tmp_path)
    runtime=copy.deepcopy(saved['runtime']);sampler.integers(2**31)
    with pytest.raises(RuntimeError,match='sampler'):
        warm.verify_restoration(saved,policy,optimizer,sampler,runtime,tmp_path)
    sampler.bit_generator.state=saved['sampler'];torch.rand(1)
    with pytest.raises(RuntimeError,match='global'):
        warm.verify_restoration(saved,policy,optimizer,sampler,runtime,tmp_path)
    restore_rng(saved['rng']);runtime['task_cycle']['cursor']+=1
    with pytest.raises(RuntimeError,match='task cursor'):
        warm.verify_restoration(saved,policy,optimizer,sampler,runtime,tmp_path)


def test_shared_source_qualifications_are_counted_once_but_conflicts_rejected():
    shared=dict(id='shared-qualification',actions=8313,sha256='abc')
    full=dict(charges=[dict(id='full',actions=171166),shared])
    episode=dict(charges=[dict(id='episode',actions=171707),copy.deepcopy(shared)])
    assert warm.deduplicated_actions([full,episode])==171166+171707+8313
    episode['charges'][1]['actions']+=1
    with pytest.raises(ValueError,match='conflicting'):
        warm.deduplicated_actions([full,episode])


def test_source_history_cap_counts_validation_qualification_and_rejects_duplicate_charge(monkeypatch):
    lineage=dict(terminal_index_path='/legacy',cumulative_physical_cost=dict(total_action_charge=100),
        recovery_sha256='old')
    documents={'/legacy':dict(physical_cost=dict(total_action_charge=100)),
        '/val':dict(format='source40_frozen_ppo_validation_v1',completed=True,weights_unchanged=True,
            contract=dict(slots=[dict(mode='full')],models=dict(full=dict(recovery_sha256='old'))),
            slot_costs=[dict(total_action_charge=20)]),
        '/qual':dict(source_contract=dict(source_only=True,source_data_from_targets=False),physical_cost=dict(total_action_charge=30))}
    charges=[dict(id='legacy',role='legacy_training',artifact='/legacy',sha256='sha',actions=100,action_key=['physical_cost','total_action_charge']),
        dict(id='val',role='source_validation',artifact='/val',sha256='sha',actions=20,slot_mode='full'),
        dict(id='qual',role='qualification',artifact='/qual',sha256='sha',actions=30,action_key=['physical_cost','total_action_charge'])]
    history=dict(format=warm.FORMAT,mode='full',legacy_root='/root',canonical_ppo_ledger='/ledger',
        maximum_cumulative_source_actions=1000000,legacy_lineage=lineage,cost_completeness_statement='All roles included',
        charges=charges,prior_actions=150)
    documents['/history']=history
    monkeypatch.setattr(warm,'file_sha256',lambda path:'sha')
    monkeypatch.setattr(warm,'read',lambda path:documents[str(path)])
    monkeypatch.setattr(warm,'terminal_source',lambda *args:lineage)
    monkeypatch.setattr(warm,'time_history',lambda *args:{'separately_verified':True})
    assert warm.load_history('/history','sha','full',maximum_new_actions=999850)['prior_actions']==150
    with pytest.raises(ValueError,match='1M cumulative cap'):
        warm.load_history('/history','sha','full',maximum_new_actions=999851)
    charges.append(copy.deepcopy(charges[-1]))
    with pytest.raises(ValueError,match='Repeated cost'):
        warm.load_history('/history','sha','full',maximum_new_actions=1)
