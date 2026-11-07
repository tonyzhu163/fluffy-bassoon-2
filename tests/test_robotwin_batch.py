"""Core frozen batched rollout and owned simulator-process checks on Linux CPU."""
from dataclasses import asdict
from copy import deepcopy
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.rl.robotwin_batch import RoboTwinBatchRuntime
from fluffy_bassoon.rl.robotwin_execution import ChunkExecution
from fluffy_bassoon.rl.robotwin_production_process import ProductionSimulatorPool,ProductionRPCError


def observation(seed):
    return dict(main_images=np.full((1,4,4,3),seed%255,np.uint8),
        wrist_images=np.zeros((1,2,4,4,3),np.uint8),states=np.full((1,14),seed,np.float32),
        task_descriptions=['place the phone on its stand'])


class Motor:
    noise_shape = (4,32)
    model = SimpleNamespace(device=torch.device('cpu'))
    def __init__(self): self.calls = [];self.observations = []
    def check_frozen(self): pass
    def infer_batch(self,raw,*,noise):
        self.calls.append(noise.clone())
        self.observations.append({key:(value.clone() if torch.is_tensor(value) else list(value))
                                  for key,value in raw.items()})
        return noise[...,:14]+raw['states'][:,None],raw['states'].clone()


class Pool:
    num_workers = 2
    def __init__(self): self.steps = [0,0];self.closed = False;self.bad = False
    def reset(self,tasks,seeds,*,indices):
        for i in indices: self.steps[i] = 0
        return [observation(s) for s in seeds],[dict(actual_seed=s,episode_limit=8) for s in seeds]
    def step(self,commands,*,indices):
        result = []
        for lane,command in zip(indices,commands,strict=True):
            self.steps[lane] += 1
            execution = ChunkExecution(4,1,2,1,7,.004) if lane == 0 else ChunkExecution(4,4,4,0,12,.004)
            info = {} if self.bad else dict(execution=asdict(execution))
            result.append((observation(lane+1),float(lane==0),lane==0,self.steps[lane]==2,info))
        return result
    def close(self): self.closed = True


def runtime():
    motor,pool = Motor(),Pool()
    result = RoboTwinBatchRuntime(motor,pool,seed=7)
    result.reset('place_phone_stand',[11,13],case_ids=['trial0/episode0','trial1/episode0'])
    return result,motor,pool


def test_one_batched_motor_call_owned_proposals_and_measured_feedback():
    rt,motor,pool = runtime()
    base,features,meta = rt.infer()
    before = base.clone();features.fill_(-1);base.add_(100);meta['lanes'].reverse()
    rows = rt.step(before+.01)
    assert len(motor.calls) == 1 and motor.calls[0].shape == (2,4,32)
    assert np.array_equal(rows[0]['base_actions'],before[0].numpy())
    assert np.all(rows[0]['public_features'] == 11)
    assert rows[0]['native_batch']['lanes'] == [0,1]
    assert rows[0]['measured_valid_target_mask'].tolist() == [True,True,False,False]
    assert rows[0]['physics_duration_seconds'] == .028
    assert rows[0]['requested_actions'].shape == (4,14)
    assert rows[0]['terminated'] and rows[0]['reward'] == 1.
    # Learners may normalize or augment returned transitions in place.
    for key in ('main_images','wrist_images','states'):
        rows[1]['next_observation'][key].fill(99)
    rows[1]['next_observation']['task_descriptions'][0] = 'caller annotation'
    base,_,meta = rt.infer()
    assert torch.all(motor.observations[-1]['states'] == 2)
    assert torch.all(motor.observations[-1]['main_images'] == 2)
    assert torch.all(motor.observations[-1]['wrist_images'] == 0)
    assert motor.observations[-1]['task_descriptions'] == ['place the phone on its stand']
    assert meta['lanes'] == [1] and meta['decision_ids'] == ['trial1/episode0/chunk-1']
    rows = rt.step(base)
    assert rows[0]['truncated']
    with pytest.raises(RuntimeError): rt.infer()
    rt.close();assert pool.closed


def test_prefix_and_complete_episode_lifecycle_rejects_reuse():
    rt,_,_ = runtime()
    with pytest.raises(RuntimeError): rt.reset('place_phone_stand',[1],case_ids=['new'],indices=[0])
    with pytest.raises(RuntimeError): rt.step(np.zeros((2,4,14)))
    base,_,_ = rt.infer()
    with pytest.raises(RuntimeError): rt.infer()
    with pytest.raises(RuntimeError): rt.reset('place_phone_stand',[1],case_ids=['new'],indices=[0])
    with pytest.raises(ValueError): rt.step(np.zeros((2,3,14)))
    rt.step(base)
    with pytest.raises(ValueError):
        rt.reset('place_phone_stand',[11],case_ids=['trial0/episode0'],indices=[0])
    rt.reset('place_phone_stand',[17],case_ids=['trial0/episode1'],indices=[0])
    rt.close()


def test_case_noise_follows_episode_addresses_not_selected_lane_order():
    first,first_motor,_ = runtime();second,second_motor,_ = runtime()
    a,_,_ = first.infer(indices=[0,1]);b,_,_ = second.infer(indices=[1,0])
    assert torch.equal(first_motor.calls[0],second_motor.calls[0].flip(0))
    assert torch.equal(a,b.flip(0))
    first.close();second.close()


def test_missing_execution_never_falls_back_to_requested_counts():
    rt,_,pool = runtime();pool.bad = True
    base,_,_ = rt.infer()
    with pytest.raises(KeyError): rt.step(base)
    assert rt.failed and rt._pending is not None
    with pytest.raises(RuntimeError): rt.infer()
    rt.close()


def test_fixed_motor_slots_execute_only_active_physics_and_record_original_rows():
    rt,motor,pool = runtime()
    base,_,meta = rt.infer(fixed_native_batch=True)
    rt.step(base)
    base,_,meta = rt.infer(fixed_native_batch=True)
    assert meta['batch_size']==2 and meta['lanes']==[0,1] and meta['active_lanes']==[1]
    assert len(motor.calls)==2 and all(len(value)==2 for value in motor.calls)
    rows = rt.step(base)
    assert len(rows)==1 and rows[0]['native_batch']['row']==1
    assert rows[0]['native_batch']['decision_ids'][1]=='trial1/episode0/chunk-1'
    np.testing.assert_array_equal(rows[0]['requested_actions'],base[1].numpy())
    assert pool.steps==[1,2] and not any(rt._active)
    rt.close()


def test_fixed_motor_batch_requires_initialized_slots_and_prevents_reordering():
    rt = RoboTwinBatchRuntime(Motor(),Pool())
    rt.reset('place_phone_stand',[11],case_ids=['only-first'],indices=[0])
    with pytest.raises(RuntimeError,match='Initialize every'):
        rt.infer(fixed_native_batch=True)
    with pytest.raises(ValueError,match='original order'):
        rt.infer(indices=[0],fixed_native_batch=True)
    assert rt._pending is None
    rt.close()


def chunk_session(mode='full'):
    from fluffy_bassoon.meta_rl.chunk_controller import FrozenChunkController
    from fluffy_bassoon.meta_rl.chunk_history import ChunkHistoryLayout
    from fluffy_bassoon.meta_rl.chunk_policy import ChunkModelConfig,ChunkResidualPolicy
    from fluffy_bassoon.meta_rl.chunk_robotwin import RoboTwinChunkSession
    torch.manual_seed(317)
    model = ChunkModelConfig(ChunkHistoryLayout(14,4,14),width=8,layers=1,heads=2,context_length=8)
    controller = FrozenChunkController(ChunkResidualPolicy(model,torch.full((4,14),.1)),
                                      native_batch=2,mode=mode)
    motor,pool = Motor(),Pool()
    return RoboTwinChunkSession(RoboTwinBatchRuntime(motor,pool,seed=7),controller),motor,pool


@pytest.mark.parametrize('mode',['full','episode','static'])
def test_frozen_student_session_joins_physics_preserves_memory_and_resets_trials(mode):
    session,motor,pool = chunk_session(mode)
    control = session.controller
    session.reset('place_phone_stand',[11,13],case_ids=['t0/support','t1/support'],new_trials=True)
    commands,_,_ = session.propose()
    session.step(commands)
    state = deepcopy(control.state)
    feedback = control.encoders[0].previous.copy()
    commands,_,meta = session.propose()
    assert meta['active_lanes']==[1] and meta['batch_size']==2
    for key in state:
        torch.testing.assert_close(control.state[key][0],state[key][0],rtol=0,atol=0)
    np.testing.assert_array_equal(control.encoders[0].previous,feedback)
    rows = session.step(commands)
    assert rows[0]['native_batch']['row']==1 and pool.steps==[1,2]
    assert control.state['length'].tolist()==([1,1] if mode=='static' else [1,2])
    # Support feedback survives a new physical episode within the same trial.
    session.reset('place_phone_stand',[11,13],case_ids=['t0/query','t1/query'])
    commands,_,_ = session.propose()
    assert control.state['length'].tolist()==([2,3] if mode=='full' else [1,1])
    session.step(commands)
    commands,_,_ = session.propose();session.step(commands)
    old = deepcopy(control.state)
    session.reset('place_phone_stand',[17],case_ids=['new-t0/support'],indices=[0],new_trials=True)
    commands,_,_ = session.propose()
    assert control.state['length'][0]==1
    for key in old:
        torch.testing.assert_close(control.state[key][1],old[key][1],rtol=0,atol=0)
    session.step(commands)
    assert all(len(value)==2 for value in motor.calls)
    assert not any(p.requires_grad for p in control.policy.parameters())
    session.close();assert pool.closed


def test_student_command_tampering_is_rejected_before_any_physics_dispatch():
    session,_,pool = chunk_session()
    session.reset('place_phone_stand',[11,13],case_ids=['first','second'])
    commands,_,_ = session.propose()
    commands[0,-1,-1] += .01
    with pytest.raises(ValueError,match='before physical dispatch'):
        session.step(commands)
    assert pool.steps==[0,0] and session.failed
    with pytest.raises(RuntimeError,match='without automatic replay'):
        session.step(commands)
    session.close()


class ProcessSimulator:
    """Spawned CPU-only native boundary fake, never loads a VLA or renderer."""
    def __init__(self,config): self.config = config;self.steps = 0
    def reset(self,task,seed):
        self.steps = 0
        return observation(seed),dict(actual_seed=seed,episode_limit=8,lane=self.config['lane'],pid=os.getpid(),
                                     marker=self.config['worker_options'].get('marker'))
    def step(self,actions):
        if self.config['worker_options'].get('fail_lane') == self.config['lane']:
            raise RuntimeError('native worker failed')
        self.steps += 1
        return observation(self.steps),0.,False,self.steps==2,dict(worker_pid=os.getpid())
    def close(self): pass


def test_process_pool_preserves_selected_order_and_owns_lifecycle(tmp_path):
    pool = ProductionSimulatorPool(2,rlinf_root=tmp_path,robotwin_root=tmp_path,
        output_dir=tmp_path/'workers',task_episode_limits={'place_phone_stand':8},
        _simulator_factory=ProcessSimulator,worker_options={'marker':'qualified'})
    try:
        raw,info = pool.reset('place_phone_stand',[13,17],indices=[1,0])
        assert [x['lane'] for x in info] == [1,0]
        assert len({x['pid'] for x in info}) == 2 and all(x['pid'] != os.getpid() for x in info)
        assert all(x['marker'] == 'qualified' for x in info)
        pool.step(np.zeros((1,4,14),np.float32),indices=[1])
        pool.step(np.zeros((1,4,14),np.float32),indices=[1])
        with pytest.raises(ValueError): pool.step(np.zeros((1,4,14),np.float32),indices=[1])
    finally: pool.close()
    assert pool.closed and not any(p.is_alive() for p in pool.processes)


def test_worker_failure_retains_completed_replies_and_never_restarts(tmp_path):
    pool = ProductionSimulatorPool(2,rlinf_root=tmp_path,robotwin_root=tmp_path,
        output_dir=tmp_path/'failure',task_episode_limits={'place_phone_stand':8},
        _simulator_factory=ProcessSimulator,worker_options={'fail_lane':1})
    pool.reset('place_phone_stand',[13,17])
    with pytest.raises(ProductionRPCError,match='native worker failed') as caught:
        pool.step(np.zeros((2,4,14),np.float32))
    assert set(caught.value.completed) == {0}
    assert pool.failed and pool.closed and not any(p.is_alive() for p in pool.processes)
