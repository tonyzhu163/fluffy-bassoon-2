"""New assessment checks with synthetic state only; no JAX/checkpoint/simulator."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import vla_source40_teacher_assessment as a
from fluffy_bassoon.meta_rl.vla_source40_evaluate import CostLedger,rollout_episode


def admission(tmp_path):
    source='a'*40;root=tmp_path/'new-assessment';output=root/'data'
    contract=a.worker_contract(source,'teacher-assessment-test')
    refs={k:dict(path=str(root/k),sha256=v) for k,v in dict(grant=a.GRANT_SHA,
        authority=a.AUTHORITY_SHA,protocol=a.PROTOCOL_SHA,asset_binding=a.ASSET_SHA,
        actor_table=a.TABLE_SHA,inventory=a.INVENTORY_SHA).items()}
    row=dict(path=str(root/'clock/attempts/one'),status='started',started_unix=100.,
        slice_seconds=10740,reserved_seconds=10800)
    ticket=dict(format=a.FORMAT,run_id='teacher-assessment-test',admitted=True,
        source_commit=source,hostname='allocated-host',job_id='actual-job',clock_root=str(root/'clock'),
        clock_contract=contract,resource_attempt=row['path'],output=str(output),refs=refs,
        driver_deadline_unix=10700.,gpu_uuid='allocated-uuid',
        allocation=dict(GPU=1,CPU=16,memory_GiB=96,shared_memory_GiB=8,automatic_retry=False,exclusive=True))
    clock=dict(contract=contract,worker_seconds=10800,attempts=[row])
    probe=dict(status='passed',hostname='allocated-host',devices=[dict(uuid='allocated-uuid')])
    return ticket,clock,probe,dict(source=source,output=output,hostname='allocated-host',now=200.)


def test_distinct_finite_clock_and_actual_allocation(tmp_path):
    ticket,clock,probe,kwargs=admission(tmp_path)
    value=a.validate_admission(ticket,clock,probe,**kwargs)
    assert value['planned_episodes']==320 and value['original_grant_actions']==196800
    assert value['optimizer_updates']==0 and value['source_pretraining_admitted'] is False


@pytest.mark.parametrize('mutation',[
    lambda t,c,p,k:t.update(admitted=False),
    lambda t,c,p,k:t.update(hostname='development-host'),
    lambda t,c,p,k:t['allocation'].update(GPU=0),
    lambda t,c,p,k:t['allocation'].update(automatic_retry=True),
    lambda t,c,p,k:t.update(driver_deadline_unix=float('inf')),
    lambda t,c,p,k:t.update(driver_deadline_unix=10800.),
    lambda t,c,p,k:c.update(worker_seconds=21600),
    lambda t,c,p,k:c['attempts'][0].update(reserved_seconds=10801),
    lambda t,c,p,k:c['attempts'].append(dict(path='old-failure',status='finished',reserved_seconds=60,charged_seconds=61)),
    lambda t,c,p,k:t['refs']['inventory'].update(sha256='b'*64),
    lambda t,c,p,k:p['devices'][0].update(uuid='different-uuid'),
    lambda t,c,p,k:p.update(status='unverified'),
])
def test_refuse_unbound_or_extra_resource_authority(tmp_path,mutation):
    ticket,clock,probe,kwargs=admission(tmp_path);mutation(ticket,clock,probe,kwargs)
    with pytest.raises(ValueError):a.validate_admission(ticket,clock,probe,**kwargs)


class Simulator:
    def __init__(self,*,mismatch=False,fail=False):
        self.metadata=dict(instruction='fixed instruction',horizon=400,settle_steps=10)
        self.resets=self.steps=0;self.mismatch=mismatch;self.fail=fail;self.closed=False
    def reset(self,state,seed):
        self.resets+=1
        changed=int(self.mismatch and self.resets>1)
        return dict(observation=dict(image=np.full((4,4,3),changed,np.uint8),
            wrist=np.zeros((4,4,3),np.uint8),state=np.zeros(8,np.float32)),
            initial_observation_sha256=f'public-{changed}',initial_state_vector_sha256='nominal',
            initial_simulator_state_sha256='actual-post-settle')
    def step(self,actions):
        self.steps+=1
        if self.fail:raise RuntimeError('interrupted physical RPC')
        return dict(observation=dict(image=np.ones((4,4,3),np.uint8),wrist=np.zeros((4,4,3),np.uint8),
            state=np.ones(8,np.float32)),primitive_steps=10,executed_actions=np.asarray(actions),
            rewards=[0.]*10,success=False,done=True)
    def close(self):self.closed=True


class Motor:
    horizon=10;metadata={}
    def __init__(self):
        self.policy=SimpleNamespace(_rng=0);self.noises=[]
    def feature_encoder(self,observation,instruction):return np.full(4104,observation['state'][0],np.float32)
    def infer(self,observation,instruction,noise):
        self.noises.append(np.asarray(noise).copy());return np.zeros((10,7),np.float32)


class Controller:
    public_observation_input=True
    def set_episode_seed(self,seed):self.seed=seed;self.assessment_rng_before=self.snapshot()
    def act(self,observation):self.pending=True;return np.full(32,3.,np.float32)
    def observe(self,*,executed_noise,reward,done):
        np.testing.assert_array_equal(executed_noise,np.ones(32,np.float32));self.pending=False
    def snapshot(self):return dict(seed=getattr(self,'seed',None),pending=getattr(self,'pending',False))


class Actors:
    def __init__(self,sim,*,mutate=False):self.sim=sim;self.mutate=mutate;self.activated=[]
    def activate(self,row):self.activated.append(row['task_key'])
    def controller(self,role):return Controller() if role in a.ROLES[2:] else None
    def fingerprint(self):return dict(full='unchanged' if not self.mutate else str(self.sim.steps),updates=12,replay='original')


@pytest.fixture
def runtime(monkeypatch):
    from fluffy_bassoon.meta_rl import vla_source40_corpus as corpus
    monkeypatch.setattr(corpus,'save_motor_rng',deepcopy)
    monkeypatch.setattr(corpus,'restore_motor_rng',deepcopy)
    table=dict(actors=[dict(source_ordinal=0,task_key='libero_spatial:00',suite='libero_spatial',task_id=0,
        BDDL=dict(actual_sha256='bound-bddl'),checkpoint=dict(published_sha256='bound-checkpoint'),recorded_updates=12)])
    original=a.panel
    # Run only a prefix in a fake world, retaining the full planned/omitted reporting grid.
    monkeypatch.setattr(a,'panel',lambda t,include_omitted=False:original(t,include_omitted=include_omitted) if include_omitted else original(t)[:4])
    return table


def run_fake(tmp_path,table,*,sim=None,mutate=False):
    sim=sim or Simulator();motor=Motor();actors=Actors(sim,mutate=mutate);qualified=[]
    forks=SimpleNamespace(new_case=lambda seed:{'global':dict(seed=seed),'motor':seed},
        capture_global=lambda:dict(saved=True),restore_global=lambda value:None,
        seed_motor=lambda seed:setattr(motor.policy,'_rng',seed))
    def execute():
        return a.run_panel(table,motor,actors,tmp_path/'data',server_factory=lambda task,n:sim,
            deadline=float('inf'),qualify=lambda *args:qualified.append('same-first-episode'),forks=forks,
            provenance=dict(source_commit='a'*40,immutable_assets='test-only'))
    return execute,sim,motor,actors,qualified


def test_fixed_role_draws_and_embedded_qualification_without_extra_reset(tmp_path,runtime):
    execute,sim,motor,actors,qualified=run_fake(tmp_path,runtime);result=execute()
    assert result['cursor']==4 and result['completed'] is False
    assert sim.resets==sim.steps==4 and qualified==['same-first-episode'] and sim.closed
    assert result['costs']==dict(all_in_actions=80,physical_episodes=4,uncertain_actions=0)
    assert actors.activated==['libero_spatial:00']
    assert not np.array_equal(motor.noises[0][0],motor.noises[0][1])
    np.testing.assert_array_equal(motor.noises[1][0],motor.noises[1][1])
    assert np.abs(motor.noises[1]).max()>1  # historical warmup is unbounded
    np.testing.assert_array_equal(motor.noises[2],np.ones((10,32),np.float32))
    assert json.loads((tmp_path/'data/results-summary.json').read_text())['omitted_initial_slots']==160
    with gzip.open(tmp_path/'data/episode-003/rng-context.pkl.gz','rb') as stream:context=pickle.load(stream)
    assert context['actor_before']['seed']==20261141 and context['actor_after']['pending'] is False


def test_exact_public_pair_refusal_happens_before_next_steering_action(tmp_path,runtime):
    execute,sim,*_=run_fake(tmp_path,runtime,sim=Simulator(mismatch=True))
    with pytest.raises(ValueError,match='BEFORE steering'):execute()
    assert sim.resets==2 and sim.steps==1 and sim.closed
    assert a.action_charge(tmp_path/'data')['all_in_actions']==30


def test_frozen_full_state_or_replay_mutation_refuses_publication(tmp_path,runtime):
    execute,sim,*_=run_fake(tmp_path,runtime,mutate=True)
    with pytest.raises(RuntimeError,match='learning state/replay'):execute()
    assert sim.resets==1 and not (tmp_path/'data/episode-000/receipt.json').exists()


def test_unresolved_intent_is_charged_and_interrupted_episode_never_replayed(tmp_path,runtime):
    execute,sim,*_=run_fake(tmp_path,runtime,sim=Simulator(fail=True))
    with pytest.raises(RuntimeError,match='interrupted'):execute()
    assert a.action_charge(tmp_path/'data')==dict(all_in_actions=20,physical_episodes=1,uncertain_actions=10)
    with pytest.raises(ValueError,match='interrupted physical trial'):execute()
    assert sim.resets==1


def test_closed_resume_does_not_replay_controls_or_qualification(tmp_path,runtime):
    execute,sim,motor,actors,qualified=run_fake(tmp_path,runtime);first=execute();second=execute()
    assert first['receipts']==second['receipts'] and sim.resets==4 and len(qualified)==1


@pytest.mark.parametrize('seed',[-1,2**32,True,1.5])
def test_invalid_explicit_seed_refuses_before_physical_reset(tmp_path,seed):
    sim=Simulator();ledger=CostLedger(tmp_path,max_primitive_steps=400,max_settling_steps=10)
    with pytest.raises(ValueError,match='uint32'):
        rollout_episode(sim,Motor(),None,case=SimpleNamespace(initial_state=40,scene_seed=123),
            method='native',role='query',ordinal=0,episode_id='fake',ledger=ledger,
            reset_motor_rng=lambda seed:None,episode_seed=seed)
    assert sim.resets==0


def test_explicit_noise_seed_is_exact_and_default_old_seed_path_stays_available(tmp_path):
    motor=Motor();sim=Simulator();seeds=[]
    ledger=CostLedger(tmp_path,max_primitive_steps=400,max_settling_steps=10)
    _,receipt,_=rollout_episode(sim,motor,None,case=SimpleNamespace(initial_state=40,scene_seed=123),
        method='native',role='query',ordinal=0,episode_id='fake',ledger=ledger,
        reset_motor_rng=seeds.append,episode_seed=20261141)
    expected=np.random.default_rng(20261141).standard_normal((10,32)).astype(np.float32)
    np.testing.assert_array_equal(motor.noises[0],expected)
    assert seeds==[20261141] and receipt['episode_random_seed']==20261141


def test_actual_feature_value_drift_fails_even_when_all_hash_labels_match():
    initial=Simulator().reset(40,123);features=np.zeros(4104,np.float32)
    reference=dict(initial=deepcopy(initial),features=features,instruction='fixed')
    a.check_pair(reference,initial,features,'fixed')
    with pytest.raises(ValueError,match='BEFORE steering'):
        a.check_pair(reference,initial,features+1,'fixed')
