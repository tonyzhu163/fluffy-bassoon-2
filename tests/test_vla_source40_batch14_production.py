"""Allocated-Linux tests for the explicit phase transition, not a new PPO rule."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import vla_source40_batch14_production as prod
from fluffy_bassoon.meta_rl import vla_source40_batch14_warmstart as warm
from fluffy_bassoon.meta_rl.checkpoint import rng_state
from fluffy_bassoon.meta_rl.vla_rl2 import RL2Config
from fluffy_bassoon.meta_rl.vla_rl2_batch import training_contract
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import finish


def plan(mode='full'):
    prior,native=(195964,9272) if mode=='full' else (196776,0)
    return dict(format=prod.FORMAT,status='PROPOSAL_NOT_FUNDED',source_only=True,source_data_from_targets=False,
        mode=mode,legacy_history_path='/immutable/history-'+mode+'.json',legacy_history_sha256='h',
        legacy_prior_actions=prior,native_prior_actions=native,prior_source_resource_actions=prior+native,
        native_parent={'root':'/qualified','native_actions':9272} if mode=='full' else None,
        maximum_cumulative_source_actions=1000000,final_source_validation_actions=30000,
        maximum_native_stream_actions=1000000-prior-30000,maximum_new_actions=1000000-prior-native-30000,
        updates_target=100,complete_trials_per_update=14,attempts_per_trial=4,seed=2,
        architecture=dict(width=128,layers=2,heads=4,context_length=300),
        optimizer=dict(learning_rate=1e-4,epochs=2,valid_token_weighting=True),
        maximum_production_gpu_seconds=154800,final_source_validation_gpu_seconds=3600,
        maximum_total_gpu_seconds=158400,storage_allowance_bytes=64000000000,initialization='Explicit immutable phase')


@pytest.mark.parametrize('mode,prior,stream,new',[('full',205236,774036,764764),('episode',196776,773224,773224)])
def test_all_prior_costs_and_final_validation_hold_stay_within_per_learner1m(mode,prior,stream,new):
    value=prod.validate_plan(plan(mode))
    assert value['prior_source_resource_actions']==prior
    assert value['maximum_native_stream_actions']==stream and value['maximum_new_actions']==new
    assert prior+new+30000==1000000
    assert value['maximum_production_gpu_seconds']+value['final_source_validation_gpu_seconds']==158400
    wrong=copy.deepcopy(value);wrong['maximum_new_actions']+=1
    with pytest.raises(ValueError,match='cumulative costs'):prod.validate_plan(wrong)
    wrong=copy.deepcopy(value);wrong['final_source_validation_actions']=20000
    with pytest.raises(ValueError,match='reserve'):prod.validate_plan(wrong)


def test_source_invocation_cannot_reset_history_cap_seed_or_mode(monkeypatch):
    value=plan();monkeypatch.setattr(prod,'load_plan',lambda *args:value)
    source=dict(warm_start=dict(path=value['legacy_history_path'],sha256='h',prior_actions=195964))
    kwargs=dict(actions=774036,updates=100,mode='full',seed=2)
    merged=prod.source_contract(source,'/plan','sha',**kwargs)
    assert merged['warm_start']==source['warm_start']
    assert merged['production_continuation']['native_prior_actions']==9272
    for field,wrong in [('actions',794764),('updates',1),('mode','episode'),('seed',0)]:
        changed=dict(kwargs);changed[field]=wrong
        with pytest.raises(ValueError,match='exact phase'):prod.source_contract(source,'/plan','sha',**changed)


def qualified_native(tmp_path):
    policy=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
    for parameter in policy.parameters():
        optimizer.state[parameter]=dict(step=torch.tensor(476.),exp_avg=torch.full_like(parameter,.12),exp_avg_sq=torch.full_like(parameter,.34))
    config=RL2Config(observation_dim=3,noise_dim=2,width=8,layers=1,heads=2,context_length=4)
    history=dict(path='/immutable/history-full.json',sha256='h',prior_actions=195964)
    source=dict(source_only=True,source_data_from_targets=False,warm_start=history)
    old_contract=training_contract(config,source,updates=1,max_total_actions=20000,seed=2,mode='full',learning_rate=1e-4,epochs=2)
    old_cycle=dict(format='source40_ppo_task_cursor_v1',cursor=238,remaining=0,exposure={'source':dict(trials_started=238,episodes=952,primitive_steps=1234)})
    legacy=dict(version=1,boundary='between_trials',iteration=238,contract=dict(mode='full',config=old_contract['config'],optimizer=dict(name='Adam',learning_rate=1e-4)),
        policy=policy.state_dict(),optimizer=optimizer.state_dict(),environment_rng=np.random.default_rng(2).bit_generator.state,rng=rng_state(),
        runtime=dict(task_cycle=old_cycle,motor_rng=dict(format='jax_motor_key_v1',implementation='threefry2x32',key_data=np.asarray([2,3],dtype=np.uint32))),
        counters=dict(environment_steps=1234,settling_steps=9520,optimizer_steps=476))
    cycle=copy.deepcopy(old_cycle);cycle.update(cursor=0,exposure={'source':dict(trials_started=0,episodes=0,primitive_steps=0)})
    native=warm.initial_payload(legacy,old_contract,history,cycle)
    native['progress'].update(iteration=1,completed_trials=14)
    native['progress']['counters'].update(environment_steps=8702,settling_steps=560,optimizer_steps=2)
    native['runtime']['task_cycle'].update(cursor=14,exposure={'source':dict(trials_started=14,episodes=56,primitive_steps=8702)})
    for state in native['optimizer']['state'].values():state['step']=torch.tensor(478.)
    root=tmp_path/'qualified';learner=root/'experiment/learner';learner.mkdir(parents=True)
    torch.save(native,learner/'latest.pt')
    (learner/'warm-start-initial.pt').write_bytes(b'immutable original warm migration')
    (learner/'warm-start-migration.json').write_text('{}')
    (learner/'metrics.jsonl').write_text('{"update":1}\n')
    (learner/'traces').mkdir();(learner/'traces/behavior-000000.pt').write_bytes(b'original frozen behavior')
    for directory in ('execution-journal','task-resets'):
        folder=root/'experiment'/directory;folder.mkdir();(folder/'old.json').write_text('{"retained":true}')
    value=plan();value['native_parent'].update(root=str(root),recovery_path=str(learner/'latest.pt'),
        recovery_sha256=prod.sha(learner/'latest.pt'),native_updates=1)
    return native,value,config,source


def test_full_transition_preserves_all_native_leaves_adam_typed_rng_cursor_and_partial_resume(tmp_path,monkeypatch):
    old,value,config,source=qualified_native(tmp_path)
    monkeypatch.setattr(prod,'load_plan',lambda *args:value)
    output=tmp_path/'new/learner';output.parent.mkdir()
    source=prod.source_contract(source,'/plan','sha',actions=774036,updates=100,mode='full',seed=2)
    old_sha=value['native_parent']['recovery_sha256'];before=warm.legacy_fingerprint(old)
    kwargs=dict(updates=100,actions=774036,seed=2,mode='full')
    prod.prepare(output,'/plan','sha',source,config,{},**kwargs)
    saved=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
    assert warm.legacy_fingerprint(saved)==before
    assert saved['progress']['iteration']==1 and saved['runtime']['task_cycle']['cursor']==14
    assert {float(row['step']) for row in saved['optimizer']['state'].values()}=={478.}
    assert prod.sha(value['native_parent']['recovery_path'])==old_sha
    assert (output.parent/'execution-journal/old.json').read_text()=='{"retained":true}'
    proof=prod.read(output/'production-migration.json')
    assert proof['restore_optimizer_steps']==0 and proof['numerical_state_unchanged']
    assert proof['non_contract_sha256_before']==proof['non_contract_sha256_after']==before
    # A subsequent ordinary slice retains a committed partial batch exactly.
    saved['progress'].update(completed_trials=15,plan=[{'case':'fixed'}]*14,
        pending_rollouts=[{'actual':torch.ones(3)}],pending_receipts=[{'case':'actual'}])
    saved['runtime']['task_cycle']['cursor']=15
    torch.save(saved,output/'latest.pt');partial=warm.legacy_fingerprint(saved)
    prod.prepare(output,'/plan','sha',source,config,{},**kwargs)
    assert warm.legacy_fingerprint(torch.load(output/'latest.pt',weights_only=False))==partial


def test_episode_delegates_to_qualified_generic_r3_warmstart_without_new_optimizer_logic(tmp_path,monkeypatch):
    value=plan('episode');monkeypatch.setattr(prod,'load_plan',lambda *args:value)
    calls=[];monkeypatch.setattr(warm,'prepare',lambda *args,**kwargs:calls.append((args,kwargs)))
    history=dict(path=value['legacy_history_path'],sha256='h',prior_actions=196776)
    source=dict(warm_start=history)
    prod.prepare(tmp_path/'learner','/plan','sha',source,object(),{'cursor':0},updates=100,actions=773224,seed=2,mode='episode')
    assert len(calls)==1 and calls[0][0][1]==history
    assert calls[0][1]==dict(updates=100,maximum_new_actions=773224,seed=2,mode='episode')


def funded(tmp_path,value):
    record=dict(status='APPROVED',authorization='Explicit test funding only',run_id='fresh-run',source_commit='a'*40,
        plan_sha256='p',mode=value['mode'],production_gpu_seconds=154800,final_source_validation_gpu_seconds=3600,
        maximum_total_gpu_seconds=158400,maximum_native_stream_actions=value['maximum_native_stream_actions'])
    path=tmp_path/'funding.json';path.write_text(json.dumps(record));return path


def test_proposal_never_authorizes_production_or_borrows_final_validation_time(tmp_path,monkeypatch):
    value=plan();monkeypatch.setattr(prod,'load_plan',lambda *args:value)
    funding=funded(tmp_path,value);schedule=tmp_path/'schedule.json';schedule.write_text('{}')
    record=prod.read(funding);record['status']='PROPOSAL_NOT_FUNDED';funding.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='No exact approved44h'):
        prod.begin(tmp_path/'run','/plan','p',funding,'a'*40,'fresh-run',schedule,seconds=21540,resume=False,started=1000.)
    funding=funded(tmp_path,value)
    root=tmp_path/'run'
    attempt,stop,seconds=prod.begin(root,'/plan','p',funding,'a'*40,'fresh-run',schedule,seconds=21540,resume=False,started=1000.)
    assert stop==22540 and seconds==21540
    finish(root,attempt,1,now=1120.)
    clock=prod.read(root/'worker-time.json');assert clock['attempts'][0]['charged_seconds']==180
    # Near the end of the SAME funded phase, only remaining production time is available.
    clock['attempts'][0]['charged_seconds']=154000;prod.write(root/'worker-time.json',clock)
    _,_,seconds=prod.begin(root,'/plan','p',funding,'a'*40,'fresh-run',schedule,seconds=21540,resume=True,started=2000.)
    assert seconds==740
    with pytest.raises(RuntimeError,match='final-validation reserve remains separate'):
        prod.begin(root,'/plan','p',funding,'a'*40,'fresh-run',schedule,seconds=21540,resume=True,started=2001.)
