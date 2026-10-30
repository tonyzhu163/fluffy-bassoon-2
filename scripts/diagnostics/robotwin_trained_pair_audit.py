#!/usr/bin/env python3
"""CPU-only final audit; failure never retries or refunds physical work."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path

from robotwin_trained_pair import (HEAD_KEYS, LINEAGE, METHODS, RUNTIME, atom,
    ledger_root, read_ref, require, sha, validate_plan)


def bound(ref):
    return {"path": str(Path(ref).resolve()), "sha256": sha(ref)}


def audit_corpus(root):
    """Reconcile every retained native action, including original due evals."""
    import numpy as np
    root = Path(root).resolve()
    checked = {}
    def check_arrays(ref):
        path = (root / ref["path"]).resolve()
        require(path.is_relative_to(root) and path.is_file(), "raw artifact escaped or disappeared")
        if str(path) not in checked:
            checked[str(path)] = sha(path)
        require(checked[str(path)] == ref["sha256"], "raw numeric/RGB artifact changed")
        return path
    total = chunks = source = 0
    source_episodes = []
    for reset_path in sorted((root / "episodes").glob("*/reset.json")):
        reset = json.loads(reset_path.read_text())
        episode_path = reset_path.parent
        manifest = json.loads((episode_path / "manifest.json").read_text())
        require(not manifest.get("pending_decision"), "unresolved corpus intent")
        require(len(manifest["chunks"]) == manifest["completed_chunks"], "corpus chunk count differs")
        require(len(list((episode_path / "chunks").glob("*.json"))) == manifest["completed_chunks"],
                "unlisted physical decision")
        count = 0
        previous_frame = reset["initial_observation"]["sha256"]
        if "initial_observation" in reset:
            check_arrays(reset["initial_observation"])
        for index, ref in enumerate(manifest["chunks"]):
            path = (episode_path / ref["path"]).resolve()
            require(path.is_relative_to(episode_path) and sha(path) == ref["sha256"], "chunk hash differs")
            record = json.loads(path.read_text())
            require(record["status"] == "complete" and record["chunk_index"] == index + 1,
                    "incomplete/reordered native decision")
            require(record["observation"]["sha256"] == previous_frame,
                    "raw observation chronology changed")
            previous_frame = record["next_observation"]["sha256"]
            require(all(record["context"].get(key) == reset["context"].get(key)
                        for key in ("phase", "method", "trial_id")), "raw episode context changed")
            for key in ("observation", "next_observation", "decision_arrays"):
                check_arrays(record[key])
            native = record["native_result"]
            with np.load(check_arrays(native["arrays"]), allow_pickle=False) as arrays:
                mask = arrays["valid_action_mask"]
                require(np.isin(mask, (0, 1)).all()
                        and int(mask.sum()) == native["executed_primitive_steps"], "native mask count differs")
            executed = native["executed_primitive_steps"]
            require(type(executed) is int and 0 <= executed <= record["issued_primitive_steps"],
                    "invalid actual native action count")
            learning = record["learning_result"]
            require(learning["terminated"] == native["terminated"]
                    and not (learning["terminated"] and learning["truncated"])
                    and (not native["truncated"] or native["terminated"] or learning["truncated"]),
                    "native terminal feedback changed at the learning boundary")
            require(executed > 0 or bool(native["terminated"] or native["truncated"]),
                    "zero-action decision lacks a native terminal boundary")
            count += executed
            require(not learning["budget_truncated"] and learning["episode_primitive_steps"] == count,
                    "episode was externally cut or clock differs")
            require(bool(learning["terminated"] or learning["truncated"])
                    == (index == len(manifest["chunks"]) - 1), "episode is not a complete native unit")
            chunks += 1
        require(count == manifest["executed_primitive_steps"], "episode action sum differs")
        total += count
        context = reset["context"]
        if context.get("phase") == "source":
            require(manifest["completed_chunks"] > 0, "source reset has no completed episode")
            source += count
            source_episodes.append(dict(trial_id=context["trial_id"],
                task=reset["reset_info"]["task"], seed=reset["reset_info"]["requested_seed"], actions=count))
    statistics = json.loads((root / "statistics.json").read_text())
    require(not statistics.get("pending_decision") and not statistics.get("unresolved_decisions")
            and statistics["native_completed_chunks"] == 0
            and statistics["completed_chunks"] == chunks
            and statistics["executed_primitive_steps"] == total, "raw corpus totals differ")
    return dict(actual_actions=total, source_actions=source, completed_chunks=chunks,
                hashed_raw_artifacts=len(checked), source_episodes=source_episodes)


def audit_arm(plan, method):
    import torch
    from fluffy_bassoon.meta_rl.collection_admission import assert_resume_allowed
    from fluffy_bassoon.meta_rl.robotwin_trial_resume import load_trial_continuation
    from fluffy_bassoon.meta_rl.robotwin_trained_parallel import validate_trained_parallel_boundary
    arm = plan["methods"][method]
    before_manifest = read_ref(arm["manifest"])
    require(before_manifest["checkpoint"] == arm["checkpoint"], "input checkpoint binding differs")
    before_path = Path(arm["checkpoint"]["path"])
    require(sha(before_path) == arm["checkpoint"]["sha256"], "input trained checkpoint changed")
    before = torch.load(before_path, map_location="cpu", weights_only=False)
    old = before["states"][method]
    output = Path(arm["output"])
    summary = json.loads((output / "summary.json").read_text())
    require(summary["status"] == "wall_budget_stop" and summary["process_source_trials"] == 6
            and summary["process_trial_lifetime_limit"] == 6 and summary["recycle_at_round_boundary"] is True,
            "expected exactly one complete lifetime-bounded source round")
    next_ref = summary["continuation_manifest"]
    manifest = read_ref(next_ref)
    require(Path(next_ref["path"]).resolve() == output / "continuation-manifest.json"
            and Path(manifest["checkpoint"]["path"]).resolve() == output / "coordinator-recovery.pt",
            "final recovery escaped its output")
    checkpoint_path = Path(manifest["checkpoint"]["path"])
    require(sha(checkpoint_path) == manifest["checkpoint"]["sha256"], "final checkpoint hash differs")
    final = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(final["contract"] == before["contract"]
            and json.loads(json.dumps(final["contract"])) == summary["contract"],
            "original scientific contract changed")
    assert_resume_allowed(checkpoint_path, checkpoint=final)
    continuation = load_trial_continuation(next_ref["path"], next_ref["sha256"],
        checkpoint_path=checkpoint_path, checkpoint=final, execution_commit=RUNTIME, lineage_commit=LINEAGE)
    identity = validate_trained_parallel_boundary(continuation, final,
        manifest_path=next_ref["path"], campaign_root=ledger_root(method))
    saved = final["states"][method]
    require(saved["completed_source_iterations"] == old["completed_source_iterations"] + 1
            and saved["completed_source_trials"] == old["completed_source_trials"] + 6
            and saved["learner"]["optimizer_steps"] == old["learner"]["optimizer_steps"] + 5,
            "genuine trained round/optimizer increment differs")
    source_delta = saved["source_steps"] - old["source_steps"]
    require(0 < source_delta <= 10800, "one-round source action bound differs")
    old_units, new_units = set(before["completed_units"]), set(final["completed_units"])
    require(old_units <= new_units and final["history"][:len(before["history"])] == before["history"],
            "completed evaluation evidence/cursors regressed")
    initial_iteration = old["completed_source_iterations"]
    due = []
    for iteration in (10, 25, 50, 100):
        unit = f"stage:iteration:{iteration}:complete"
        if iteration <= initial_iteration:
            require(unit in new_units, "an input-due original evaluation was skipped")
            if unit not in old_units:
                due.append(iteration)
        elif iteration == initial_iteration + 1:
            require(unit not in new_units, "newly reached evaluation should remain due at the round boundary")
    for key, stage in before["stages"].items():
        if stage.get("status") == "complete":
            require(final["stages"][key] == stage, "an earlier completed evaluation stage changed")
    motor = read_ref(arm["frozen_motor"])
    frozen = json.loads((output / "frozen-motor.json").read_text())
    require(frozen["unchanged"] is True and frozen["before"] == frozen["after"] == motor["before"] == motor["after"],
            "frozen motor identity changed")
    migration = json.loads((output / "execution-migration.json").read_text())
    require(migration["execution_source_commit"] == RUNTIME and migration["lineage_source_commit"] == LINEAGE
            and migration["manifest_sha256"] == arm["manifest"]["sha256"], "migration ancestry differs")
    input_ledger = read_ref(before_manifest["ledger_snapshot"])
    ledger = read_ref(manifest["ledger_snapshot"])
    require(ledger["header"] == input_ledger["header"], "canonical header/caps changed")
    require(ledger["pending_actions"] == ledger["uncertain_actions"] == 0
            and not ledger["unresolved_decisions"], "unresolved physical decisions")
    actual_delta = ledger["actual_actions"] - input_ledger["actual_actions"]
    source_charge = ledger["by_phase"].get("source", 0) - input_ledger["by_phase"].get("source", 0)
    require(source_charge == source_delta, "source checkpoint/physical accounting differs")
    maximum_eval = (31600 if method == "ppo" else 17200) if due else 0
    require(source_delta <= actual_delta <= 10800 + maximum_eval, "new original-phase allowance exceeded")
    corpus = audit_corpus(output / "corpus")
    require(corpus["actual_actions"] == actual_delta and corpus["source_actions"] == source_delta,
            "raw retained counts differ from canonical physical actions")
    episodes = corpus["source_episodes"]
    require(len(episodes) == 24, "six complete K4 source trials required")
    trial_ids = {f"source:{method}:{number}" for number in
                 range(old["completed_source_trials"] + 1, saved["completed_source_trials"] + 1)}
    require({row["trial_id"] for row in episodes} == trial_ids, "source trials duplicated or omitted")
    tasks = []
    for trial_id in trial_ids:
        rows = [row for row in episodes if row["trial_id"] == trial_id]
        require(len(rows) == 4 and len({row["task"] for row in rows}) == 1,
                "source K4 task/episode counts differ")
        tasks.append(rows[0]["task"])
    require(sorted(tasks) == sorted(before["contract"]["source_tasks"]), "balanced source task block changed")
    groups = [json.loads(path.read_text()) for path in (output / "production-vector").glob("group-*.json")]
    require(len(groups) == 1 and groups[0]["status"] == "checkpointed"
            and groups[0]["physical_actions"] == source_delta
            and groups[0]["completed_episodes"] == [4] * 6
            and groups[0].get("checkpoint_sha256"), "durable group ACK is missing")
    # Exercise only the qualified CPU task sampler: evaluation must not consume
    # its stream, and parallel execution must select the exact next six trials.
    from robotwin_trained_pair_preflight import load_driver
    from fluffy_bassoon.meta_rl.robotwin_trial_resume import artifact_digest
    driver = load_driver(Path(plan["worktree"]))
    schedule = driver.BalancedSchedule(before["contract"]["source_tasks"], 0)
    schedule.load_state_dict(deepcopy(old["schedule"]))
    expected = [schedule.sample(4) for _ in range(6)]
    require(artifact_digest(schedule.state_dict()) == artifact_digest(saved["schedule"])
            and groups[0]["tasks"] == [task for task, _ in expected]
            and groups[0]["episode_seeds"] == [seeds for _, seeds in expected],
            "task sampler/seed stream changed across parallel migration")
    for number, (task, seeds) in enumerate(expected, old["completed_source_trials"] + 1):
        rows = [row for row in episodes if row["trial_id"] == f"source:{method}:{number}"]
        require(sorted(row["seed"] for row in rows) == sorted(seeds)
                and {row["task"] for row in rows} == {task}, "raw source case provenance differs")
    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines() if line]
    replay = [row for row in progress if row["stage"] == "source_execution_group_complete"]
    updates = [row for row in progress if row["stage"] == "ppo_update"]
    require(len(replay) == len(updates) == 1 and replay[0]["lanes"] == 6, "one replay-checked update required")
    require(len(replay[0]["replay_errors"]) == 6, "a replay lane is missing")
    for row in replay[0]["replay_errors"]:
        metrics = row["metrics"]
        require(row["passed"] is True and 0 <= metrics["log_prob_max_error"] <= 2e-4
                and 0 <= metrics["value_max_error"] <= 1e-5, "old-policy replay tolerance failed")
    update = updates[0]
    require(math.isfinite(update["approximate_kl"]) and update["approximate_kl"] <= .1
            and all(math.isfinite(value) for value in update.values() if isinstance(value, (int, float))),
            "nonfinite/rejected PPO update")
    corpus.pop("source_episodes")
    return dict(method=method, input_manifest=arm["manifest"], next_manifest=next_ref,
        checkpoint=manifest["checkpoint"], completed_iterations=saved["completed_source_iterations"],
        source_actions=source_delta, actual_actions=actual_delta, input_due_stages_completed=due,
        optimizer_step_increment=5, trained_boundary=identity, corpus=corpus,
        final_canonical_head={key: ledger[key] for key in HEAD_KEYS}, frozen_motor_unchanged=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "audit must hide CUDA")
    require(sha(args.plan) == args.plan_sha256, "plan changed")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    arms = {method: audit_arm(plan, method) for method in METHODS}
    result = dict(status="passed_one_trained_round_each", plan_sha256=args.plan_sha256,
        runtime_commit=RUNTIME, arms=arms,
        total_new_source_actions=sum(row["source_actions"] for row in arms.values()),
        total_new_physical_actions=sum(row["actual_actions"] for row in arms.values()),
        no_new_ledger_or_cap=True, no_fresh_initialization=True, no_automatic_retry=True)
    require(result["total_new_source_actions"] <= 21600 and result["total_new_physical_actions"] <= 70400,
            "combined original phase bound exceeded")
    atom(args.receipt, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
