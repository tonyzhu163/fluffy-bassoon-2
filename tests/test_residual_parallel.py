"""Numerical equivalence of serial and scheduled parallel residual rollouts."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig, stack_records, update
from fluffy_bassoon.meta_rl.robocasa_residual import collect_pending, install_terminal_labels
from fluffy_bassoon.meta_rl.robocasa_residual_parallel import (
    ParallelResidualEnvironment, draw_schedule, episode_partitions, native_chunk_offsets,
)
from fluffy_bassoon.rl.action_residual import ResidualBound


class Motor:
    def __init__(self):
        self.seed_rng = np.random.default_rng(710100)

    def infer(self, observation, *, noise_seed=None):
        seed = int(self.seed_rng.integers(0, 2**63 - 1)) if noise_seed is None else noise_seed
        return np.full((50, 12), .1, np.float32), np.array([seed / 2**63, 1.], np.float32), seed


class Environment:
    def __init__(self, first=0, chunks=2, lengths=None):
        self.cursor = first
        self.chunks = chunks
        self.lengths = lengths

    def reset(self):
        self.steps = 0
        self.cursor += 1
        if self.lengths is not None:self.chunks=self.lengths[self.cursor-1]
        return None

    def step_chunk(self, chunk):
        self.steps += 1
        done = self.steps == self.chunks
        return None, done, dict(id=str(self.cursor)) if done else {}


def learner():
    torch.manual_seed(42)
    policy = ResidualActorCritic(ResidualPPOConfig(2, 6, width=8, epochs=2, minibatch_size=8))
    return policy, torch.optim.Adam(policy.parameters(), lr=policy.config.learning_rate)


def test_eight_partitions_preserve_draws_complete_gae_and_next_adam_update():
    policy, optimizer = learner()
    bound = ResidualBound(12, tuple(range(6)), (.25,) * 6)
    serial_motor, serial_generator = Motor(), torch.Generator().manual_seed(42)
    expected, episodes = collect_pending(policy, bound, serial_motor, Environment(), episodes=16,
                                         generator=serial_generator)
    scheduled_motor, scheduled_generator = Motor(), torch.Generator().manual_seed(42)
    seeds, noise = draw_schedule(scheduled_motor, policy, scheduled_generator, 32, deterministic=False)
    actual, actual_episodes = [], []
    for rank in range(8):
        rows, finished = collect_pending(policy, bound, Motor(), Environment(rank * 2), episodes=2,
            generator=None, planned_motor_seeds=seeds[rank * 4:(rank + 1) * 4],
            planned_action_noise=noise[rank * 4:(rank + 1) * 4])
        actual.extend(rows)
        actual_episodes.extend(finished)
    assert actual_episodes == episodes
    for left, right in zip(expected, actual):
        for key in left:
            if isinstance(left[key], np.ndarray):
                np.testing.assert_array_equal(left[key], right[key])
            else:
                assert left[key] == right[key]
    assert serial_motor.seed_rng.integers(2**63 - 1) == scheduled_motor.seed_rng.integers(2**63 - 1)
    torch.testing.assert_close(torch.randn((1, 6), generator=serial_generator),
                               torch.randn((1, 6), generator=scheduled_generator), rtol=0, atol=0)
    second, second_optimizer = learner()
    labels = {row["id"]: index % 2 for index, row in enumerate(episodes)}
    update(policy, optimizer, stack_records(install_terminal_labels(expected, episodes, labels, arm="S")),
           generator=torch.Generator().manual_seed(43))
    update(second, second_optimizer, stack_records(install_terminal_labels(actual, actual_episodes, labels, arm="S")),
           generator=torch.Generator().manual_seed(43))
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    for left, right in zip(optimizer.state_dict()["state"].values(), second_optimizer.state_dict()["state"].values()):
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_deterministic_parallel_plan_keeps_training_actor_rng_untouched():
    policy, _ = learner()
    generator = torch.Generator().manual_seed(42)
    before = generator.get_state().clone()
    seeds, noise = draw_schedule(Motor(), policy, generator, 16, deterministic=True)
    assert len(seeds) == len(set(seeds)) == 16 and noise is None
    torch.testing.assert_close(before, generator.get_state(), rtol=0, atol=0)
    rows, episodes = collect_pending(policy, ResidualBound(12, tuple(range(6)), (.25,) * 6),
        Motor(), Environment(), episodes=8, generator=generator, deterministic=True,
        planned_motor_seeds=seeds)
    assert len(episodes) == 8 and all(np.array_equal(row["control"], np.zeros(6)) for row in rows)
    torch.testing.assert_close(before, generator.get_state(), rtol=0, atol=0)


@pytest.mark.parametrize('world',[4,8])
def test_fifty_complete_episodes_without_padding_keep_draws_gae_and_next_adam(world):
    policy, optimizer = learner()
    bound = ResidualBound(12, tuple(range(6)), (.25,) * 6)
    motor, generator = Motor(), torch.Generator().manual_seed(42)
    expected, episodes = collect_pending(policy, bound, motor, Environment(), episodes=50,
                                         generator=generator)
    second_motor, second_generator = Motor(), torch.Generator().manual_seed(42)
    seeds, noise = draw_schedule(second_motor, policy, second_generator, 100, deterministic=False)
    actual, finished = [], []
    for first, stop in episode_partitions(50,world):
        rows, completed = collect_pending(policy, bound, Motor(), Environment(first), episodes=stop-first,
            generator=None, planned_motor_seeds=seeds[first*2:stop*2], planned_action_noise=noise[first*2:stop*2])
        actual.extend(rows);finished.extend(completed)
    assert [stop-first for first,stop in episode_partitions(50,world)] == ([13,13,12,12] if world==4 else [7,7,6,6,6,6,6,6])
    assert finished == episodes and len(actual) == len(expected) == 100
    for left, right in zip(expected, actual):
        for key in left:
            if isinstance(left[key], np.ndarray):np.testing.assert_array_equal(left[key],right[key])
            else:assert left[key] == right[key]
    assert motor.seed_rng.integers(2**63-1) == second_motor.seed_rng.integers(2**63-1)
    torch.testing.assert_close(generator.get_state(), second_generator.get_state(), rtol=0, atol=0)
    second, second_optimizer = learner()
    labels={row['id']:index%2 for index,row in enumerate(episodes)}
    for model, opt, records in ((policy,optimizer,expected),(second,second_optimizer,actual)):
        update(model,opt,stack_records(install_terminal_labels(records,episodes,labels,arm='S')),
               generator=torch.Generator().manual_seed(43))
    for key,value in policy.state_dict().items():
        torch.testing.assert_close(value,second.state_dict()[key],rtol=0,atol=0)
    for left,right in zip(optimizer.state_dict()['state'].values(),second_optimizer.state_dict()['state'].values()):
        for key in left:torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)


def test_retained_eight_then_forty_two_keep_exact_fifty_motor_draw_stream():
    policy,_=learner()
    serial=Motor();all_seeds,_=draw_schedule(serial,policy,None,100,deterministic=True)
    continued=Motor()
    for _ in range(8*2):continued.seed_rng.integers(0,2**63-1)
    suffix,_=draw_schedule(continued,policy,None,42*2,deterministic=True)
    assert suffix == all_seeds[16:]
    assert [stop-first for first,stop in episode_partitions(42)] == [6,6,5,5,5,5,5,5]
    assert serial.seed_rng.integers(2**63-1)==continued.seed_rng.integers(2**63-1)


@pytest.mark.parametrize('distributed',[False,True])
@pytest.mark.parametrize('world',[4,8])
def test_uneven_worker_results_merge_all_episodes_and_preserve_recovery(tmp_path,monkeypatch,distributed,world):
    import time
    from fluffy_bassoon.meta_rl.warmstart import atomic_save
    from fluffy_bassoon.meta_rl import robocasa_residual_parallel as module
    policy,_=learner();bound=ResidualBound(12,tuple(range(6)),(.25,)*6)
    def child(command, **kwargs):
        request=torch.load(command[-1],weights_only=False)
        first=request['cases'][0]['slot']; count=len(request['cases'])
        rows,completed=collect_pending(policy,bound,Motor(),Environment(first,chunks=12),episodes=count,
            generator=None,planned_motor_seeds=request['seeds'],planned_action_noise=request['noise'])
        assert request['episode_id_prefix'].startswith('PPO-task-')
        atomic_save(module.Path(request['output'])/'result.pt',dict(cases=request['cases'],records=rows,
            episodes=completed,environment=dict(cursor=count),global_rng=dict(rank=request['rank']),
            motor_rng=dict(rank=request['rank'])))
        return SimpleNamespace(returncode=0,poll=lambda:0,wait=lambda **kw:0)
    monkeypatch.setattr(module.subprocess,'Popen',child)
    cases=[dict(task='OpenDrawer',slot=i) for i in range(50)]
    profile=dict(rollout_devices=[dict(visibility_token=str(i),uuid=str(i),allocation_ordinal=0) for i in range(world)],
        attempt='core',source_commit='a'*40,episode_id_namespace='PPO-task-',
        deadline_unix=time.time()+60,closure_seconds=1)
    if distributed:
        from fluffy_bassoon.meta_rl import rlinf_flora_ppo
        original_write=rlinf_flora_ppo.write_new
        def queued(path,value):
            assert path.name.startswith('core-')
            original_write(path,value)
            child([value['request']])
        monkeypatch.setattr(rlinf_flora_ppo,'write_new',queued)
        profile['distributed_rollout_root']=str(tmp_path/'dispatch')
    env=ParallelResidualEnvironment(cases,horizon=600,output=tmp_path/'bank',monitoring_root=tmp_path/'monitor',
        profile=profile,code=tmp_path,report=lambda *a,**k:None)
    rows,episodes=env.collect(policy,bound,Motor(),episodes=50,generator=None,deterministic=True)
    assert len(rows)==600 and [e['id'] for e in episodes]==[str(i) for i in range(1,51)]
    state=env.state_dict()
    assert state['cursor']==50 and len(state['worker_recovery'])==world
    assert [r['environment']['cursor'] for r in state['worker_recovery']]==([13,13,12,12] if world==4 else [7,7,6,6,6,6,6,6])
    restored=ParallelResidualEnvironment(cases,horizon=600,output=tmp_path/'recovered',monitoring_root=tmp_path/'monitor',
        profile=profile,code=tmp_path,report=lambda *a,**k:None)
    restored.load_state_dict(state)
    assert restored.state_dict()==state


    other=deepcopy(profile)
    other['rollout_devices']=[dict(visibility_token=str(i),uuid=str(i),allocation_ordinal=0)
                              for i in range(8 if world==4 else 4)]
    incompatible=ParallelResidualEnvironment(cases,horizon=600,output=tmp_path/'other',monitoring_root=tmp_path/'monitor',
        profile=other,code=tmp_path,report=lambda *a,**k:None)
    with pytest.raises(ValueError,match='case bank'):incompatible.load_state_dict(state)


def test_joint_eighty_full_native_episodes_preserve_serial_gae_and_next_adam():
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import PPO_TASKS
    cases=[dict(task=task) for _ in range(16) for task in PPO_TASKS]
    offsets=native_chunk_offsets(cases)
    lengths=[b-a for a,b in zip(offsets,offsets[1:])]
    assert offsets[-1]==960
    assert [offsets[b]-offsets[a] for a,b in episode_partitions(80)]==[120]*8
    policy,optimizer=learner();bound=ResidualBound(12,tuple(range(6)),(.25,)*6)
    motor,generator=Motor(),torch.Generator().manual_seed(42)
    serial,episodes=collect_pending(policy,bound,motor,Environment(lengths=lengths),episodes=80,generator=generator)
    scheduled,scheduled_generator=Motor(),torch.Generator().manual_seed(42)
    seeds,noise=draw_schedule(scheduled,policy,scheduled_generator,offsets[-1],deterministic=False)
    actual,completed=[],[]
    for first,stop in episode_partitions(80):
        rows,finished=collect_pending(policy,bound,Motor(),Environment(first,lengths=lengths),episodes=stop-first,
            generator=None,planned_motor_seeds=seeds[offsets[first]:offsets[stop]],
            planned_action_noise=noise[offsets[first]:offsets[stop]])
        actual.extend(rows);completed.extend(finished)
    assert completed==episodes and len(serial)==len(actual)==960
    for left,right in zip(serial,actual):
        for key in left:
            if isinstance(left[key],np.ndarray):np.testing.assert_array_equal(left[key],right[key])
            else:assert left[key]==right[key]
    assert motor.seed_rng.integers(2**63-1)==scheduled.seed_rng.integers(2**63-1)
    torch.testing.assert_close(generator.get_state(),scheduled_generator.get_state(),rtol=0,atol=0)
    second,opt2=learner();labels={e['id']:i%2 for i,e in enumerate(episodes)}
    for model,opt,rows in ((policy,optimizer,serial),(second,opt2,actual)):
        update(model,opt,stack_records(install_terminal_labels(rows,episodes,labels,arm='S')),
            generator=torch.Generator().manual_seed(43))
    for key,value in policy.state_dict().items():torch.testing.assert_close(value,second.state_dict()[key],rtol=0,atol=0)
    for left,right in zip(optimizer.state_dict()['state'].values(),opt2.state_dict()['state'].values()):
        for key in left:torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)


def test_single_and_dual_gpu_node_routing_keeps_distinct_worker_ownership():
    from fluffy_bassoon.meta_rl.robocasa_residual_parallel import native_worker_rank
    assert [native_worker_rank(node,0,1) for node in range(4)]==list(range(4))
    assert [native_worker_rank(node,local,2) for node in range(4) for local in range(2)]==list(range(8))
    for args in ((0,1,1),(0,0,4),(-1,0,1),(8,0,1),(0,True,1)):
        with pytest.raises(ValueError):native_worker_rank(*args)
