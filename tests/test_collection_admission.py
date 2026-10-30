"""Allocated-Linux CPU tests of whole-group accounting and replay admission."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.collection_admission import (
    RESUME_BLOCKER, admit_collected_group, assert_resume_allowed,
)


class Schedule:
    def state_dict(self): return {"pending": ["b", "a"], "trials": 3}


class Learner:
    def __init__(self, failures=None):
        self.failures = failures or {}
        self.sampler = np.random.default_rng(19)
        self.primitive_steps = 10
        self.seen = []
        self.state = None

    def replay_error(self, rollout):
        self.seen.append(rollout["lane"])
        # Every physical cost is charged, and no prefix has entered pending.
        assert self.state["steps"] == 22
        assert self.state["task_steps"] == {"a": 13, "b": 9}
        assert self.state["pending"] == [("old", {"primitive_steps": 10})]
        failure = self.failures.get(rollout["lane"])
        if isinstance(failure, Exception):
            raise failure
        return {"log_prob_max_error": 0. if failure is None else failure,
                "value_max_error": 0.}

    def state_dict(self):
        return dict(weights=np.array([3., 5.]), optimizer={"step": 7},
                    sampler=deepcopy(self.sampler.bit_generator.state))


def fixture(failures=None):
    learner = Learner(failures)
    state = dict(learner=learner, schedule=Schedule(), steps=10,
                 task_steps={"a": 7, "b": 3}, pending=[("old", {"primitive_steps": 10})])
    learner.state = state
    group = [(dict(lane=index, actions=np.array([index])), dict(task=task, primitive_steps=count))
             for index, (task, count) in enumerate((("a", 4), ("b", 6), ("a", 2)))]
    return learner, state, group


def save_pickle(payload, path):
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def admit(state, group, path, save_payload=save_pickle):
    return admit_collected_group(state, group, output_dir=path, method="ppo",
                                 capture_rng=lambda: {"saved_rng": 31}, save_payload=save_payload)


def test_valid_group_is_admitted_only_after_every_lane_passes(tmp_path):
    learner, state, group = fixture()
    errors = admit(state, group, tmp_path)
    assert learner.seen == [0, 1, 2]
    assert all(row["passed"] for row in errors)
    assert state["pending"][1:] == group
    assert learner.primitive_steps == 22
    assert not (tmp_path / RESUME_BLOCKER).exists()


@pytest.mark.parametrize("failure", [.01, float("nan"), RuntimeError("bad replay")])
def test_rejected_lane_never_admits_valid_prefix_and_preserves_whole_group(tmp_path, failure):
    learner, state, group = fixture({1: failure})
    with pytest.raises(RuntimeError, match="entire executed group quarantined"):
        admit(state, group, tmp_path)
    assert learner.seen == [0, 1, 2]
    assert state["steps"] == learner.primitive_steps == 22
    assert state["task_steps"] == {"a": 13, "b": 9}
    assert state["pending"] == [("old", {"primitive_steps": 10})]
    with (tmp_path / "quarantine.pt").open("rb") as stream:
        evidence = pickle.load(stream)
    assert [row[0]["lane"] for row in evidence["collected_group"]] == [0, 1, 2]
    assert [row["passed"] for row in evidence["replay_errors"]] == [True, False, True]
    assert evidence["learner"]["optimizer"] == {"step": 7}
    assert evidence["sampler"] == learner.sampler.bit_generator.state
    assert evidence["rng"] == {"saved_rng": 31}
    assert evidence["schedule"] == {"pending": ["b", "a"], "trials": 3}
    assert state["rejected_group"]["status"] == "quarantined"
    with pytest.raises(RuntimeError, match="Ordinary resume blocked"):
        assert_resume_allowed(tmp_path)
    # Relocation cannot bypass the rejected-state check embedded in recovery.
    with pytest.raises(RuntimeError, match="checkpoint metadata"):
        assert_resume_allowed(tmp_path / "elsewhere/recovery.pt", checkpoint={
            "states": {"ppo": {"rejected_group": state["rejected_group"]}}})


def test_payload_write_failure_keeps_durable_resume_blocker_and_no_admission(tmp_path):
    learner, state, group = fixture({1: .01})
    def broken_writer(payload, path): raise OSError("deliberate disk failure")
    with pytest.raises(OSError, match="disk failure"):
        admit(state, group, tmp_path, save_payload=broken_writer)
    assert (tmp_path / RESUME_BLOCKER).exists()
    assert not (tmp_path / "quarantine.pt").exists()
    assert state["pending"] == [("old", {"primitive_steps": 10})]
    with pytest.raises(RuntimeError, match="Ordinary resume blocked"):
        assert_resume_allowed(tmp_path)


def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnostics/robotwin_noise_compare.py"
    spec = importlib.util.spec_from_file_location("robotwin_dispatch_admission_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path):
    return ["--source-commit", "0"*40, "--model-path", str(tmp_path / "motor"),
            "--norm-stats", str(tmp_path / "norm.json"), "--robotwin-root", str(tmp_path / "robotwin"),
            "--rlinf-root", str(tmp_path / "rlinf"), "--output", str(tmp_path / "new-output")]


def test_parallel_dispatch_requires_the_production_fixed_round_contract(tmp_path):
    module = driver()
    args = arguments(tmp_path) + ["--collector-lanes", "2"]
    with pytest.raises(SystemExit): module.parse_args(args)
    with pytest.raises(SystemExit): module.parse_args(args + ["--qualification-smoke"])
    contract=["--campaign-root",str(tmp_path/"campaign"),"--record-corpus",
        "--source-iterations","1","--ppo-trials-per-update","6",
        "--benchmark-episode-limits","--ppo-max-update-kl",".1",
        "--collector-group-seconds","120"]
    assert module.parse_args(args+contract).collector_lanes==2


def test_marker_and_relocated_recovery_fail_before_any_cuda_runtime(tmp_path, monkeypatch):
    module = driver()
    def forbidden_cuda(): raise AssertionError("CUDA was touched before resume rejection")
    monkeypatch.setattr(module.torch.cuda, "is_available", forbidden_cuda)
    run = tmp_path / "rejected-run"
    run.mkdir()
    (run / RESUME_BLOCKER).write_text("{}")
    with pytest.raises(SystemExit):
        module.main(arguments(tmp_path) + ["--resume", str(run)])
    relocated = tmp_path / "relocated-recovery.pt"
    torch.save({"states": {"ppo": {"rejected_group": {"resume_blocked": True}}}}, relocated)
    with pytest.raises(RuntimeError, match="checkpoint metadata"):
        module.main(arguments(tmp_path) + ["--resume", str(relocated)])
