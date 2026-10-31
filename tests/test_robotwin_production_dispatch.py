"""CPU-only planning/recovery tests; run on allocated Linux."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch


path=Path(__file__).parents[1]/"scripts/diagnostics/robotwin_noise_compare.py"
spec=importlib.util.spec_from_file_location("production_dispatch_tests",path)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def state():
    return dict(learner=SimpleNamespace(sampler=np.random.default_rng(19)),
        schedule=driver.BalancedSchedule(driver.SOURCE_TASKS,17),pending=[])


def args(tmp_path):
    return ["--source-commit","a"*40,"--model-path","/model","--norm-stats","/norm",
        "--robotwin-root","/robotwin","--rlinf-root","/rlinf","--output",str(tmp_path/"run"),
        "--campaign-root",str(tmp_path/"campaign"),"--record-corpus",
        "--source-iterations","1","--ppo-trials-per-update","6",
        "--benchmark-episode-limits","--ppo-max-update-kl",".1",
        "--collector-group-seconds","120"]


@pytest.mark.parametrize("lanes",[2,3,6])
def test_supported_lane_counts_preserve_the_same_global_six_trial_round(tmp_path,lanes):
    parsed=driver.parse_args(args(tmp_path)+["--collector-lanes",str(lanes)])
    assert parsed.ppo_trials_per_update==6 and parsed.train_episodes==4 and parsed.native_horizon==32
    source=state(); seen=[]
    for _ in range(6//lanes):
        plan=driver.plan_parallel_source_group(source,lanes=lanes,episodes=4,
            episode_limit_for_task=lambda task:3 if task=="scan_object" else 5,
            remaining_actions=1000,admit_time=lambda:None)
        seen.extend(item["task"] for item in plan["plans"])
        source["pending"].extend([None]*lanes)
    assert len(seen)==6 and set(seen)==set(driver.SOURCE_TASKS)
    with pytest.raises(ValueError,match="cross"):
        driver.plan_parallel_source_group(source,lanes=lanes,episodes=4,
            episode_limit_for_task=lambda task:5,remaining_actions=1000,admit_time=lambda:None)


@pytest.mark.parametrize("reason",["budget","time","bad_horizon"])
def test_prelaunch_rejection_restores_schedule_private_sampler_and_all_process_rng(reason):
    source=state()
    schedule=deepcopy(source["schedule"].state_dict())
    sampler=deepcopy(source["learner"].sampler.bit_generator.state)
    random.seed(3);np.random.seed(5);torch.manual_seed(7)
    before=driver.capture_rng()
    expected=(random.random(),np.random.random(),torch.rand(1))
    driver.restore_rng(before)
    def horizon(task):
        random.random();np.random.random();torch.rand(1);source["learner"].sampler.random()
        return 0 if reason=="bad_horizon" else 5
    def time_gate():
        if reason=="time":raise driver.WallBudgetReached("test")
    with pytest.raises((ValueError,driver.CampaignBudgetReached,driver.WallBudgetReached)):
        driver.plan_parallel_source_group(source,lanes=2,episodes=4,episode_limit_for_task=horizon,
            remaining_actions=1 if reason=="budget" else 100,admit_time=time_gate)
    assert source["schedule"].state_dict()==schedule
    assert source["learner"].sampler.bit_generator.state==sampler
    assert random.random()==expected[0] and np.random.random()==expected[1]
    torch.testing.assert_close(torch.rand(1),expected[2],rtol=0,atol=0)


def test_time_gate_carries_measured_max_and_checkpoint_reserve_across_resume():
    gate=driver.CollectorTimeAdmission(100,30)
    with pytest.raises(driver.WallBudgetReached):gate.require(130)
    gate.require(131)
    gate.observe(120)
    with pytest.raises(driver.WallBudgetReached):gate.require(180)
    restored=driver.CollectorTimeAdmission(100,30)
    restored.load_state_dict(gate.state_dict())
    assert restored.maximum_observed==120
    restored.require(181)
    with pytest.raises(ValueError):driver.CollectorTimeAdmission(99,30).load_state_dict(gate.state_dict())


@pytest.mark.parametrize("change",[
    ["--ppo-trials-per-update","8"],["--train-episodes","2"],["--native-horizon","50"],
    ["--collector-lanes","4"],["--collector-group-seconds","nan"],
])
def test_incompatible_parallel_schedules_rejected_before_runtime(tmp_path,change):
    with pytest.raises(SystemExit):driver.parse_args(args(tmp_path)+["--collector-lanes","2"]+change)
