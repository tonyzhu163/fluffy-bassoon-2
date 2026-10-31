"""Allocated-Linux CPU tests of whole-group accounting and replay admission."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pickle
import json
import hashlib

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.collection_admission import (
    RESUME_BLOCKER, admit_collected_group, assert_resume_allowed,
    campaign_checkpoint_fence, assert_campaign_checkpoint_fence, quarantine_collection_failure,
)
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget


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


def test_incomplete_native_group_preserves_forensics_and_never_enters_pending(tmp_path):
    learner,state,group=fixture()
    quarantine_collection_failure(state,output_dir=tmp_path,method="ppo",error=OSError("RPC failure"),
        capture_rng=lambda:{"saved":19},save_payload=save_pickle,
        details={"charged_actions":33,"pending_physical_intent":True})
    assert state["pending"]==[("old",{"primitive_steps":10})]
    assert state["steps"]==10
    with (tmp_path/"collection-failure.pt").open("rb") as stream: saved=pickle.load(stream)
    assert saved["details"]["charged_actions"]==33
    with pytest.raises(RuntimeError,match="blocked"):assert_resume_allowed(tmp_path)


def campaign(root,run):
    return CampaignBudget(root,campaign_id="fence-test",total_actions=30,source_caps={"ppo":30},
        contract={"fixed":True},run_id=run,source_commit="a"*40)


def test_canonical_campaign_fence_allows_run_registration_but_rejects_uncheckpointed_physics(tmp_path):
    root=tmp_path/"ledger"
    with campaign(root,"first") as budget:
        budget.reserve("one",4,phase="source",method="ppo",task="scan_object",episode="e",chunk=0,policy_version="p0")
        budget.resolve("one",2)
        checkpoint={"campaign_journal_fence":campaign_checkpoint_fence(budget)}
        assert_campaign_checkpoint_fence(root,checkpoint,live_budget=budget)
    assert_campaign_checkpoint_fence(root,checkpoint)
    with campaign(root,"second") as budget:
        assert_campaign_checkpoint_fence(root,checkpoint,live_budget=budget)
        budget.reserve("two",4,phase="source",method="ppo",task="scan_object",episode="e2",chunk=0,policy_version="p0")
        with pytest.raises(RuntimeError,match="Uncheckpointed physical"):
            assert_campaign_checkpoint_fence(root,checkpoint,live_budget=budget)
    with pytest.raises(RuntimeError,match="Uncheckpointed physical"):
        assert_campaign_checkpoint_fence(root,checkpoint)


def test_copied_unacknowledged_group_cannot_drop_the_original_blocker(tmp_path):
    canonical=tmp_path/"canonical-vector";canonical.mkdir()
    journal=canonical/"group-00000001.json"
    group=dict(group_id="g",status="physical_complete_uncheckpointed",journal_path=str(journal))
    journal.write_text(json.dumps(group))
    payload=dict(parallel_collection_group=group,parallel_vector_root=str(canonical))
    copy=tmp_path/"copied-recovery.pt";torch.save(payload,copy)
    with pytest.raises(RuntimeError,match="Uncheckpointed native"):
        assert_resume_allowed(copy,checkpoint=payload)
    journal.write_text(json.dumps(dict(group,status="checkpointed",
        checkpoint_sha256=hashlib.sha256(copy.read_bytes()).hexdigest())))
    assert_resume_allowed(copy,checkpoint=payload)
    later=canonical/"group-00000002.json"
    later.write_text(json.dumps(dict(group_id="later",status="admitted",dispatch_count=1)))
    with pytest.raises(RuntimeError,match="Uncheckpointed native"):
        assert_resume_allowed(copy,checkpoint=payload)


def test_default_scalar_cli_cannot_bypass_parallel_recovery_fence_before_cuda(tmp_path,monkeypatch):
    module=driver()
    def forbidden_cuda():raise AssertionError("CUDA touched before explicit parallel recovery configuration")
    monkeypatch.setattr(module.torch.cuda,"is_available",forbidden_cuda)
    copy=tmp_path/"stale-parallel.pt"
    torch.save(dict(contract={"parallel_source_collection":{"lanes":6,"group_seconds_bound":120}},
                    campaign_journal_fence={"canonical_root":"/original"}),copy)
    with pytest.raises(RuntimeError,match="same explicit collector"):
        module.main(arguments(tmp_path)+["--resume",str(copy)])


def test_healthy_ack_then_optimizer_checkpoint_uses_historical_ack_digest(tmp_path):
    root=tmp_path/"ledger";vector=tmp_path/"vector";vector.mkdir()
    journal=vector/"group-00000001.json"
    group=dict(group_id="g",status="physical_complete_uncheckpointed",journal_path=str(journal),physical_actions=2)
    with campaign(root,"first") as budget:
        budget.reserve("one",4,phase="source",method="ppo",task="scan_object",episode="e",chunk=0,policy_version="p0")
        budget.resolve("one",2)
        fence=campaign_checkpoint_fence(budget)
    preupdate=dict(parallel_collection_group=group,parallel_vector_root=str(vector),
        campaign_journal_fence=fence,states={"ppo":{"updates":0,"pending":[1,2,3,4,5,6]}})
    before=tmp_path/"preupdate.pt";torch.save(preupdate,before)
    acknowledged=dict(group,status="checkpointed",checkpoint_sha256=hashlib.sha256(before.read_bytes()).hexdigest())
    journal.write_text(json.dumps(acknowledged))
    assert_resume_allowed(before,checkpoint=preupdate)
    postupdate=dict(preupdate,parallel_collection_group=acknowledged,
        states={"ppo":{"updates":1,"optimizer_steps":5,"pending":[]}})
    after=tmp_path/"postupdate.pt";torch.save(postupdate,after)
    assert hashlib.sha256(after.read_bytes()).hexdigest()!=acknowledged["checkpoint_sha256"]
    assert_resume_allowed(after,checkpoint=postupdate)
    assert_campaign_checkpoint_fence(root,postupdate)
    # Relocation preserves eligibility only while the canonical physical fence
    # remains current; later physical work must block this otherwise valid copy.
    copy=tmp_path/"relocated.pt";copy.write_bytes(after.read_bytes())
    assert_resume_allowed(copy,checkpoint=postupdate)
    with campaign(root,"later") as budget:
        budget.reserve("later",1,phase="source",method="ppo",task="scan_object",episode="e2",chunk=0,policy_version="p1")
        budget.resolve("later",1)
    with pytest.raises(RuntimeError,match="Uncheckpointed physical"):
        assert_campaign_checkpoint_fence(root,postupdate)
