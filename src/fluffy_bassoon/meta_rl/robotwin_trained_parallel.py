"""Admission and identity checks for trained, complete-update N6 migrations."""
from __future__ import annotations

import json
from pathlib import Path

from .collection_admission import assert_campaign_checkpoint_fence
from .robotwin_trial_resume import artifact_digest, file_sha256


def _require(condition, message):
    if not condition:
        raise ValueError("Trained parallel migration: " + message)


def continuation_campaign_fence(manifest_path, campaign_root):
    """Use the hash-bound predecessor ledger without opening a live ledger."""
    manifest=json.loads(Path(manifest_path).read_text())
    ref=manifest["ledger_snapshot"];path=Path(ref["path"])
    _require(file_sha256(path)==ref["sha256"],"ledger artifact changed")
    snapshot=json.loads(path.read_text())
    return dict(canonical_root=str(Path(campaign_root).resolve()),
        **{key:snapshot[key] for key in
           ("journal_events","journal_sha256","charged_actions","decision_count")})


def validate_trained_parallel_boundary(continuation, checkpoint, *, manifest_path, campaign_root):
    """Called only after load_trial_continuation verified every artifact hash."""
    method=continuation.method
    _require(method in ("ppo","ppo_reset") and checkpoint["contract"]["methods"]==[method],
             "one independently supervised PPO learner is required")
    _require(continuation.trial is None and not continuation.episodes
             and continuation.prefix_rollout is None and continuation.prefix_steps==0,
             "selected/partial trials must finish with the scalar collector")
    saved=checkpoint["states"][method];learner=saved["learner"]
    trials=saved["completed_source_trials"];iterations=saved["completed_source_iterations"]
    _require(not saved.get("selected_trial_recovery") and not saved.get("pending_rollouts"),
             "selected trials and pending PPO data are not eligible")
    _require(iterations>0 and trials%6==0 and trials==6*iterations
             and saved["schedule"]["trials"]==trials and learner["updates"]==iterations,
             "a trained completed PPO update boundary is required")
    config=learner["config"]
    _require(config["train_episodes"]==4 and config["epochs"]==5
             and config["minibatch_trials"]>=6 and learner["optimizer_steps"]==5*iterations,
             "the original complete-six-trial five-step optimizer schedule is required")
    _require(learner["primitive_steps"]==saved["source_steps"]
             and sum(saved["task_steps"].values())==saved["source_steps"],
             "source/task/learner action counters differ")
    _require(config["reset_each_episode"]==(method=="ppo_reset"),"trained memory convention differs")
    optimizer=learner["optimizer"]
    _require(bool(optimizer["state"]) and all(
        int(value["step"])==learner["optimizer_steps"] for value in optimizer["state"].values()),
        "trained Adam moments/steps are missing or inconsistent")
    _require(all(group["lr"]==config["learning_rate"] for group in optimizer["param_groups"]),
             "saved optimizer learning rate differs")
    _require(all(key in learner for key in ("policy","sampler","rng"))
             and all(key in checkpoint["rng"] for key in ("python","numpy","torch","cuda")),
             "complete policy/sampler/global RNG recovery is required")
    fence=continuation_campaign_fence(manifest_path,campaign_root)
    assert_campaign_checkpoint_fence(campaign_root,{"campaign_journal_fence":fence})
    return dict(schema="robotwin_trained_parallel_boundary_v1",method=method,
        predecessor_manifest_sha256=continuation.manifest_sha256,
        source_steps=saved["source_steps"],completed_source_trials=trials,
        completed_source_iterations=iterations,optimizer_steps=learner["optimizer_steps"],
        campaign_journal_fence=fence,
        identity_digests={key:artifact_digest(value) for key,value in dict(
            learner=learner,policy=learner["policy"],optimizer=optimizer,
            sampler=learner["sampler"],learner_rng=learner["rng"],global_rng=checkpoint["rng"],
            schedule=saved["schedule"],task_steps=saved["task_steps"]).items()})


def assert_loaded_trained_identity(learner, schedule, saved):
    """Constructors may allocate storage; loading must preserve every acting stream."""
    loaded=learner.state_dict()
    for key in ("policy","optimizer","sampler","config","updates","optimizer_steps",
                "primitive_steps","chunks"):
        _require(artifact_digest(loaded[key])==artifact_digest(saved["learner"][key]),
                 f"loaded {key} identity differs")
    _require(artifact_digest(schedule.state_dict())==artifact_digest(saved["schedule"]),
             "loaded task sampler identity differs")
