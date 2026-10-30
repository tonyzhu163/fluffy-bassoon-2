"""Allocated Linux CPU tests; fakes execute no model or real simulator."""
from copy import deepcopy
from functools import partial
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest
import torch

from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter
from fluffy_bassoon.rl.robotwin_production_process import ProductionRPCError, ProductionSimulatorPool
from fluffy_bassoon.rl.robotwin_production_vector import (
    ProductionAdmissionError, ProductionRobotwinVectorEnv, ProductionVectorError, RESUME_BLOCKER,
)


TASKS = ["handover_mic", "place_phone_stand", "scan_object"]
LIMITS = dict(zip(TASKS, (600, 500, 400)))


def raw(lane, elapsed):
    return dict(main_images=np.full((1, 4, 5, 3), lane+elapsed%200, np.uint8),
        wrist_images=np.full((1, 2, 4, 5, 3), lane+1, np.uint8),
        states=np.array([[lane+1, elapsed]+[0.]*12], np.float32),
        task_descriptions=[f"native instruction lane {lane}"])


class Backend:
    def __init__(self, path):
        self.output_dir = path / "runtime"
        self.rlinf_root = self.robotwin_root = path
        self.final_observation_only = True
        self.task_config_name = "demo_clean"
        self.contract = dict(motor_sha256="b"*64, rlinf_commit="c"*40, robotwin_commit="d"*40)
        self.torch = torch
        self._prepared = "caller-prepared"; self._prefix_mask = "caller-mask"
        self._kv_cache = "caller-cache"; self._raw_obs = "caller-raw"
        self.fail_encode = False

    def _encode(self, observation):
        state = observation["states"].numpy()[0]
        if self.fail_encode and state[1] > 0:
            raise RuntimeError("deliberate post-RPC encoder failure")
        self._prepared = state.copy(); self._prefix_mask = int(state[0])
        self._kv_cache = state.copy(); self._raw_obs = observation
        return np.array([state[0],state[1],1.],np.float32)

    def decode(self, noise):
        return np.full((32,14), self._prepared[0]+noise.reshape(-1)[0], np.float32)


class Scalar:
    observation_dim = 3
    noise_shape = (32,32)
    native_horizon = 32
    task_episode_limits = LIMITS

    def __init__(self, root): self.backend = Backend(root)
    def episode_limit_for_task(self, task): return LIMITS[task]


def records_for_lane(writer,lane):
    return [(path,json.loads(path.read_text())) for path in sorted((writer.root/"episodes").glob("*/chunks/*.json"))
            if json.loads(path.read_text())["context"]["collector_lane"] == lane]


def acknowledge(vector,budget,path):
    torch.save(dict(parallel_collection_group=vector.group_recovery_metadata(),
                    campaign_journal_fence=vector.campaign_journal_fence()),path)
    return vector.acknowledge_checkpoint(path)


class Pool:
    def __init__(self, count, *, task_episode_limits, on_step=None, partial_failure=False,
                 both_flags=False, **kwargs):
        self.count, self.limits, self.on_step = count, task_episode_limits, on_step
        self.partial_failure, self.both_flags = partial_failure, both_flags
        self.tasks = [None]*count; self.elapsed = [0]*count
        self.reset_calls = self.step_calls = 0
        self.closed = False

    def reset(self, tasks, seeds, *, indices):
        self.reset_calls += 1
        rows, receipts = [], []
        for lane, task, seed in zip(indices,tasks,seeds):
            self.tasks[lane], self.elapsed[lane] = task, 0
            rows.append(raw(lane,0))
            receipts.append(dict(task=task,requested_seed=seed,actual_seed=seed,
                reset_policy_actions=0,episode_limit=self.limits[task],worker_pid=1000+lane))
        return rows, receipts

    def step(self, actions, *, indices):
        self.step_calls += 1
        if self.on_step: self.on_step(indices,actions)
        rows = []
        for lane, commands in zip(indices,actions):
            # Decode must read the saved B1 cache for this lane, not its neighbor.
            np.testing.assert_array_equal(commands, np.full_like(commands,lane+1))
            actual = min(len(commands),3) if self.both_flags and lane == 0 else len(commands)
            self.elapsed[lane] += actual
            term = bool(self.both_flags and lane == 0)
            trunc = bool(term or self.elapsed[lane] >= self.limits[self.tasks[lane]])
            mask = (np.arange(len(commands)) < actual)[None]
            rows.append((raw(lane,self.elapsed[lane]),float(term),term,trunc,dict(
                primitive_steps=actual,issued_primitive_steps=len(commands),
                valid_action_mask=mask,success=term,duration_seconds=.1,
                raw_rewards=np.array([float(term)],np.float32),
                raw_terminations=np.array([term]),raw_truncations=np.array([trunc]),
                adapter_rewards=np.array([[float(term)]]),adapter_terminations=np.array([[term]]),
                adapter_truncations=np.array([[trunc]]),
                native_info=dict(executed_action_count=actual,native_semantics="unchanged"))))
        if self.partial_failure:
            raise ProductionRPCError("deliberate one-lane failure", completed={indices[-1]:rows[-1]})
        return rows

    def close(self): self.closed = True


def setup(tmp_path, *, lanes=3, cap=10000, pool_factory=Pool):
    scalar = Scalar(tmp_path)
    budget = CampaignBudget(tmp_path/"budget",campaign_id="production-test",total_actions=cap,
        source_caps={"ppo":cap},contract={"native":True},run_id="test-run",source_commit="a"*40)
    writers = RobotwinCorpusWriter(tmp_path/"corpus",
        dict(source_commit="a"*40,motor_sha256="b"*64))
    vector = ProductionRobotwinVectorEnv(scalar,budget,"test-run",lanes,writers,
        output_dir=tmp_path,_pool_factory=pool_factory)
    tasks = [TASKS[lane%len(TASKS)] for lane in range(lanes)]
    for lane, task in enumerate(tasks):
        vector.set_lane_context(lane,dict(phase="source",method="ppo",trial_id=f"trial-{lane}",
            policy_version="v0",checkpoint_sha256="e"*64,task=task))
    return scalar,budget,writers,vector,tasks


def decision(vector, active):
    for lane in np.flatnonzero(active).tolist():
        vector.record_decision_metadata(lane,dict(episode_index=int(vector.episode_positions[lane]),
            chunk_index=int(vector.chunks[lane]),history_index=int(vector.chunks[lane]),
            pre_squash_actions=np.zeros(1024,np.float32),old_log_prob=-1.,old_value=0.,
            policy_input=np.zeros(1031,np.float32),recurrent_reset_mask=False))


def test_native_limits_parent_cache_and_all_intents_before_every_parallel_rpc(tmp_path):
    scalar,budget,writers,vector,tasks = setup(tmp_path)
    vector.admit_group(tasks,[[11],[12],[13]])
    assert vector.group_recovery_metadata()["worst_case_actions"] == 1500
    assert vector.group_recovery_metadata()["dispatch_count"] == 0
    vector.reset_batch(tasks,[11,12,13])
    def before_rpc(indices,actions):
        snapshot = budget.snapshot(include_decisions=True)
        current = [row for row in snapshot["decisions"].values() if row["status"] == "pending"]
        assert len(current) == len(indices)
        assert sum(row["granted"] for row in current) == sum(len(value) for value in actions)
        for lane in indices:
            assert writers.for_lane(lane).pending["record"]["status"] == "intent"
    vector._pool.on_step = before_rpc
    while vector.active.any():
        active = vector.active.copy(); decision(vector,active)
        vector.step_batch(np.zeros((3,32,32),np.float32),active_mask=active)
    receipt = vector.finish_group()
    assert receipt["per_lane_actions"] == [600,500,400]
    assert receipt["physical_actions"] == budget.snapshot()["actual_actions"] == 1500
    assert receipt["completed_episodes"] == [1,1,1]
    assert receipt["wall_seconds"] > 0 and receipt["phase_seconds"]["step_rpc"] > 0
    assert receipt["status"] == "physical_complete_uncheckpointed"
    assert (tmp_path/RESUME_BLOCKER).is_file()
    assert not budget.snapshot()["unresolved_decisions"]
    for lane, expected_last in enumerate((24,20,16)):
        final = records_for_lane(writers,lane)[-1][1]
        assert final["issued_primitive_steps"] == expected_last
        assert final["status"] == "complete"
    assert scalar.backend._prepared == "caller-prepared"
    assert scalar.backend._prefix_mask == "caller-mask"
    assert scalar.backend._kv_cache == "caller-cache"
    assert scalar.backend._raw_obs == "caller-raw"
    assert writers.statistics()["executed_primitive_steps"] == 1500
    acknowledged = acknowledge(vector,budget,tmp_path/"coordinator.pt")
    assert acknowledged["status"] == "checkpointed" and not (tmp_path/RESUME_BLOCKER).exists()
    vector.close(); budget.close()


def test_partial_terminal_count_and_native_vs_learning_flags_are_faithful(tmp_path):
    scalar,budget,writers,vector,tasks = setup(tmp_path,lanes=2,
        pool_factory=partial(Pool,both_flags=True))
    vector.admit_group(tasks,[[11],[12]])
    vector.reset_batch(tasks,[11,12]); decision(vector,np.array([True,True]))
    _,rewards,terms,truncs,info = vector.step_batch(np.zeros((2,32,32)),active_mask=np.array([True,True]))
    assert rewards[0] == 1 and terms[0] and not truncs[0]
    assert info["primitive_steps"].tolist() == [3,32]
    assert budget.snapshot()["actual_actions"] == 35
    record = records_for_lane(writers,0)[0][1]
    assert record["native_result"]["truncated"] and record["native_result"]["terminated"]
    assert not record["learning_result"]["truncated"]
    with np.load(writers.for_lane(0).root/record["native_result"]["arrays"]["path"]) as arrays:
        assert arrays["valid_action_mask"].sum() == 3
        assert arrays["raw_rewards"].tolist() == [1.]
        assert arrays["raw_terminations"].tolist() == [True]
    vector.close(); budget.close()


def test_every_returned_count_resolves_before_encoder_failure_and_blocker(tmp_path):
    scalar,budget,writers,vector,tasks = setup(tmp_path,lanes=2)
    vector.admit_group(tasks,[[11],[12]]); vector.reset_batch(tasks,[11,12])
    scalar.backend.fail_encode = True
    decision(vector,np.array([True,True]))
    with pytest.raises(ProductionVectorError,match="encoder failure"):
        vector.step_batch(np.zeros((2,32,32)),active_mask=np.array([True,True]))
    assert budget.snapshot()["actual_actions"] == 64
    assert budget.snapshot()["pending_actions"] == 0
    assert (tmp_path/RESUME_BLOCKER).is_file()
    assert vector.resume_blocker_metadata["resume_blocked"]
    assert vector.group_recovery_metadata()["physical_actions"] == 64
    assert all(writers.for_lane(lane).pending["record"]["status"] == "intent" for lane in range(2))
    assert (vector.root/"failure-returned-replies.pkl").is_file()
    vector.close(); budget.close()


def test_partial_rpc_failure_keeps_unresolved_reservation_charged(tmp_path):
    scalar,budget,writers,vector,tasks = setup(tmp_path,lanes=2,
        pool_factory=partial(Pool,partial_failure=True))
    vector.admit_group(tasks,[[11],[12]]); vector.reset_batch(tasks,[11,12])
    decision(vector,np.array([True,True]))
    with pytest.raises(ProductionVectorError,match="one-lane failure"):
        vector.step_batch(np.zeros((2,32,32)),active_mask=np.array([True,True]))
    snapshot = budget.snapshot()
    assert snapshot["actual_actions"] == 32 and snapshot["pending_actions"] == 32
    assert snapshot["charged_actions"] == 64
    assert all(writers.for_lane(lane).pending["record"]["status"] == "intent" for lane in range(2))
    vector.close(); budget.close()


def test_full_group_admission_cancel_and_no_interleaved_budget_consumers(tmp_path):
    scalar,budget,writers,vector,tasks = setup(tmp_path,lanes=2,cap=1000)
    with pytest.raises(ProductionAdmissionError,match="does not fit"):
        vector.admit_group(tasks,[[11],[12]])
    assert vector._pool is None and budget.snapshot()["charged_actions"] == 0
    vector.close(); budget.close()

    root = tmp_path/"second"
    scalar,budget,writers,vector,tasks = setup(root,lanes=2)
    vector.admit_group(tasks,[[11],[12]])
    assert (root/RESUME_BLOCKER).exists()
    vector.cancel_group("allocation boundary before dispatch")
    assert vector._pool is None and not (root/RESUME_BLOCKER).exists()
    vector.admit_group(tasks,[[11],[12]])
    vector.reset_batch(tasks,[11,12])
    budget.reserve("foreign",1,phase="source",method="ppo",task=tasks[0],
                   episode="foreign",chunk=0,policy_version="v0")
    decision(vector,np.array([True,True]))
    with pytest.raises(ProductionVectorError,match="interleaved"):
        vector.step_batch(np.zeros((2,32,32)),active_mask=np.array([True,True]))
    assert vector._pool.step_calls == 0
    vector.close(); budget.close()


@pytest.mark.parametrize("key",["policy_checkpoint_sha256","policy_sha256","policy_checkpoint"])
def test_checkpoint_lineage_cannot_change_inside_group(tmp_path,key):
    scalar,budget,writers,vector,tasks = setup(tmp_path,lanes=2)
    for lane in range(2): vector.set_lane_context(lane,{key:"f"*64})
    vector.admit_group(tasks,[[11],[12]])
    with pytest.raises(RuntimeError,match="behavior identity"):
        vector.set_lane_context(0,{key:"0"*64})
    vector.cancel_group("no dispatch")
    vector.close(); budget.close()


def test_physical_completion_is_fenced_until_exact_checkpoint_acknowledgement(tmp_path):
    scalar,budget,writers,vector,tasks = setup(tmp_path,lanes=1,
        pool_factory=partial(Pool,both_flags=True))
    vector.admit_group(tasks,[[11]]); vector.reset_batch(tasks,[11])
    decision(vector,np.array([True]))
    vector.step_batch(np.zeros((1,32,32)),active_mask=np.array([True]))
    vector.finish_group()
    with pytest.raises(ProductionAdmissionError): vector.admit_group(tasks,[[12]])
    stale = tmp_path/"stale-coordinator.pt"
    torch.save(dict(parallel_collection_group={"group_id":"older"},
                    campaign_journal_fence=vector.campaign_journal_fence()),stale)
    with pytest.raises(ProductionVectorError,match="does not commit"):
        vector.acknowledge_checkpoint(stale)
    assert (tmp_path/RESUME_BLOCKER).exists() and vector.resume_blocker_metadata["resume_blocked"]
    vector.close(); budget.close()


class ConcurrentSimulator:
    def __init__(self,config,barrier):
        self.lane, self.barrier = config["lane"], barrier
    def reset(self,task,seed):
        np.random.seed(seed)
        return raw(self.lane,0),dict(reset_policy_actions=0,worker_pid=os.getpid())
    def step(self,actions):
        self.barrier.wait(timeout=15)
        return raw(self.lane,len(actions)),0.,False,False,dict(
            primitive_steps=len(actions),worker_pid=os.getpid(),random_value=float(np.random.random()))
    def close(self): pass


def test_six_isolated_processes_execute_concurrently_with_independent_rng(tmp_path):
    barrier = mp.get_context("spawn").Barrier(6)
    with ProductionSimulatorPool(6,rlinf_root=tmp_path,robotwin_root=tmp_path,
        output_dir=tmp_path/"six-processes",task_episode_limits=LIMITS,timeout_seconds=30,
        _simulator_factory=partial(ConcurrentSimulator,barrier=barrier)) as pool:
        _,receipts = pool.reset([TASKS[i%3] for i in range(6)],list(range(11,17)))
        assert len({row["worker_pid"] for row in receipts}) == 6
        rows = pool.step(np.zeros((6,1,14),np.float32))
        assert [row[4]["random_value"] for row in rows] == [
            np.random.RandomState(seed).random_sample() for seed in range(11,17)]
