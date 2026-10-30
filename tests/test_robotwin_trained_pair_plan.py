"""Prepared for allocated Linux CPU execution; no model, CUDA or live ledger."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "scripts/diagnostics"
    monkeypatch.syspath_prepend(str(root))
    loaded = []
    for name in ("robotwin_trained_pair", "robotwin_trained_pair_audit"):
        spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return tuple(loaded)


def ref(path, pair, value=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({} if value is None else value))
    return {"path": str(path.resolve()), "sha256": pair.sha(path)}


def plan(pair, tmp_path):
    identity = "robotwin-trained-pair-cpu-plan"
    output = pair.ROOT / "outputs" / identity
    result = dict(schema="robotwin_trained_pair_plan_v1", runtime_commit=pair.RUNTIME,
        lineage_commit=pair.LINEAGE, run_id=identity, output=str(output), worktree="/pinned/runtime",
        parent_manifest={"path": str(tmp_path / "parent.json"), "sha256": pair.PARENT_SHA},
        resources=deepcopy(pair.RESOURCES), maximum_seconds=2520, child_wall_seconds=2160,
        maximum_new_source_actions=21600, maximum_input_due_evaluation_actions=48800,
        owner_release=ref(tmp_path / "release.json", pair),
        compute_binding=ref(tmp_path / "compute.json", pair), methods={})
    for method in pair.METHODS:
        result["methods"][method] = dict(output=str(output / method),
            **{key: ref(tmp_path / method / (key + ".json"), pair)
               for key in ("manifest", "checkpoint", "frozen_motor")})
    result["child_commands"] = pair.build_commands(result)
    return result


def option(command, name):
    return command[command.index(name) + 1]


def test_commands_keep_original_caps_goals_and_complete_round_execution(modules, tmp_path):
    pair, _ = modules
    value = plan(pair, tmp_path)
    pair.validate_plan(value, read_evidence=False)
    for method, command in value["child_commands"].items():
        for name, expected in {"--source-commit": pair.RUNTIME,
            "--lineage-source-commit": pair.LINEAGE, "--source-steps": "1100000",
            "--source-iterations": "100", "--ppo-trials-per-update": "6",
            "--train-episodes": "4", "--eval-episodes": "8", "--eval-trials": "1",
            "--source-validation-episodes": "4", "--ppo-epochs": "5",
            "--ppo-minibatch-trials": "16", "--learning-rate": "1e-6",
            "--action-limit": "1.5", "--collector-lanes": "6",
            "--max-source-trials-per-process": "6", "--methods": method,
            "--campaign-root": str(pair.ledger_root(method)),
            "--campaign-max-actions": str(pair.CAPS[method]),
            "--native-episodes": "1" if method == "ppo" else "0"}.items():
            assert option(command, name) == expected
        start = command.index("--source-eval-iterations") + 1
        assert command[start:start + 4] == ["10", "25", "50", "100"]
        start = command.index("--source-tasks") + 1
        assert command[start:start + 6] == ["handover_mic", "pick_diverse_bottles", "place_a2b_left",
            "place_object_stand", "place_phone_stand", "scan_object"]
        assert "--recycle-at-round-boundary" in command and "--record-corpus" in command
        assert "--stage-source-validation-only" not in command
        assert option(command, "--resume") == value["methods"][method]["checkpoint"]["path"]


@pytest.mark.parametrize("change", ["pin", "parent", "output", "resources", "wall", "source_cap", "eval_cap", "command"])
def test_plan_rejects_recipe_budget_or_command_expansion(modules, tmp_path, change):
    pair, _ = modules
    value = plan(pair, tmp_path)
    if change == "pin": value["runtime_commit"] = "a" * 40
    elif change == "parent": value["parent_manifest"]["sha256"] = "b" * 64
    elif change == "output": value["methods"]["ppo"]["output"] += "-other"
    elif change == "resources": value["resources"]["gpus"] = 2
    elif change == "wall": value["child_wall_seconds"] = 2700
    elif change == "source_cap": value["maximum_new_source_actions"] += 1
    elif change == "eval_cap": value["maximum_input_due_evaluation_actions"] += 1
    else:
        command = value["child_commands"]["ppo"]
        command[command.index("--source-iterations") + 1] = "1"
    with pytest.raises(ValueError): pair.validate_plan(value, read_evidence=False)


def evidence_plan(pair, tmp_path, monkeypatch):
    value = plan(pair, tmp_path)
    parent = dict(source_commit=pair.LINEAGE, campaign_id=pair.CAMPAIGN, workers={
        method: dict(ledger_root=str(pair.ledger_root(method)), maximum_actions=pair.CAPS[method],
                     source_safety_actions=1100000) for method in pair.METHODS})
    parent_ref = ref(tmp_path / "parent.json", pair, parent)
    # Use a genuine hash for the test fixture, without requiring production files.
    monkeypatch.setattr(pair, "PARENT_SHA", parent_ref["sha256"])
    value["parent_manifest"] = parent_ref
    release = dict(schema="robotwin_trained_pair_owner_release_v1", runtime_commit=pair.RUNTIME,
        parent_manifest_sha256=pair.PARENT_SHA, methods={})
    for method in pair.METHODS:
        release["methods"][method] = dict(released=True, allocation_id="job-cpu123",
            terminal_state="Succeeded", manifest=value["methods"][method]["manifest"],
            checkpoint=value["methods"][method]["checkpoint"],
            canonical_head=dict(journal_events=2, journal_sha256="c" * 64, charged_actions=0, decision_count=0),
            terminal_receipt=ref(tmp_path / method / "terminal.json", pair, {"owner_attested": True}))
    value["owner_release"] = ref(tmp_path / "release.json", pair, release)
    compute = dict(schema="robotwin_trained_pair_compute_v1", campaign_id=pair.CAMPAIGN,
        original_total_compute_ceiling_seconds=None, source_goal_rounds=100,
        qualification_capacity_reused=False, maximum_scheduler_gpu_seconds=2700,
        admitted_scheduler_gpu_seconds=2700,
        resources=deepcopy(pair.RESOURCES), owner_release=value["owner_release"],
        authority=ref(tmp_path / "authority.json", pair, {"continue_to_source_goal": 100}),
        prior_charge_and_physical_snapshot=ref(tmp_path / "prior-charge.json", pair, {"actual_charges": True}))
    value["compute_binding"] = ref(tmp_path / "compute.json", pair, compute)
    value["child_commands"] = pair.build_commands(value)
    return value, release, compute


def test_hash_bound_owner_release_and_goal_authority_are_required(modules, tmp_path, monkeypatch):
    pair, _ = modules
    value, _, compute = evidence_plan(pair, tmp_path, monkeypatch)
    pair.validate_plan(value)
    assert compute["original_total_compute_ceiling_seconds"] is None
    assert "residual_scheduler_gpu_seconds" not in compute and "original_compute_budget_id" not in compute
    compute["qualification_capacity_reused"] = True
    value["compute_binding"] = ref(tmp_path / "compute.json", pair, compute)
    with pytest.raises(ValueError, match="operational ceiling differs"): pair.validate_plan(value)


def test_short_slice_admission_and_nonterminal_source_release_fail_closed(modules, tmp_path, monkeypatch):
    pair, _ = modules
    value, release, compute = evidence_plan(pair, tmp_path, monkeypatch)
    compute["admitted_scheduler_gpu_seconds"] = 2699
    value["compute_binding"] = ref(tmp_path / "compute.json", pair, compute)
    with pytest.raises(ValueError, match="operational ceiling differs"): pair.validate_plan(value)
    release["methods"]["ppo"]["released"] = False
    value["owner_release"] = ref(tmp_path / "release.json", pair, release)
    with pytest.raises(ValueError, match="not attested terminal"): pair.validate_plan(value)


def test_invented_global_compute_ceiling_or_cloned_worker_ledger_is_rejected(modules, tmp_path, monkeypatch):
    pair, _ = modules
    value, _, compute = evidence_plan(pair, tmp_path, monkeypatch)
    compute["original_total_compute_ceiling_seconds"] = 7200
    value["compute_binding"] = ref(tmp_path / "compute.json", pair, compute)
    with pytest.raises(ValueError, match="goal-completion authority"): pair.validate_plan(value)
    parent = json.loads(Path(value["parent_manifest"]["path"]).read_text())
    parent["workers"]["ppo"]["ledger_root"] = "/cloned/worker-ledger"
    value["parent_manifest"] = ref(tmp_path / "parent.json", pair, parent)
    monkeypatch.setattr(pair, "PARENT_SHA", value["parent_manifest"]["sha256"])
    value["child_commands"] = pair.build_commands(value)
    with pytest.raises(ValueError, match="canonical worker changed"): pair.validate_plan(value)


def corpus(tmp_path, *, count=2, terminated=True, truncated=False, phase="source",
           learning_terminated=None, learning_truncated=None):
    writer = RobotwinCorpusWriter(tmp_path / "corpus", {"source_commit": "a" * 40})
    raw = dict(main_images=np.zeros((1, 2, 3, 3), np.uint8),
        wrist_images=np.zeros((1, 2, 2, 3, 3), np.uint8), states=np.zeros((1, 14), np.float32),
        task_descriptions=["native task instruction"])
    writer.set_context(dict(phase=phase, method="ppo" if phase == "source" else "native",
        trial_id="source:ppo:1", policy_version="frozen-v1"))
    writer.begin_episode(raw, np.zeros(2, np.float32), {"task": "scan_object", "requested_seed": 11,
        "actual_seed": 11, "observation_sha256": "b" * 64, "features_sha256": "c" * 64})
    writer.begin_decision(np.zeros((32, 32), np.float32), np.zeros((2, 14), np.float32), {})
    writer.complete_native(next_raw=raw, next_features=np.zeros(2, np.float32),
        raw_rewards=np.array([0.]), raw_terminations=np.array([terminated]),
        raw_truncations=np.array([truncated]), valid_action_mask=(np.arange(2) < count)[None],
        reward=0., terminated=terminated, truncated=truncated, executed_steps=count,
        success=terminated, duration_seconds=.01)
    writer.finalize_learning_flags(reward=0.,
        terminated=terminated if learning_terminated is None else learning_terminated,
        truncated=(truncated and not terminated) if learning_truncated is None else learning_truncated,
        budget_truncated=False, episode_primitive_steps=count)
    return writer


def test_audit_counts_real_writer_arrays_and_rejects_tampering(modules, tmp_path):
    _, audit = modules
    writer = corpus(tmp_path)
    result = audit.audit_corpus(writer.root)
    assert result["actual_actions"] == result["source_actions"] == 2
    assert result["completed_chunks"] == 1 and len(result["source_episodes"]) == 1
    path = next(writer.root.glob("episodes/*/chunks/*-native.npz"))
    with path.open("ab") as stream: stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"): audit.audit_corpus(writer.root)


def test_audit_accepts_zero_native_terminal_but_rejects_unfinished_episode(modules, tmp_path):
    _, audit = modules
    writer = corpus(tmp_path / "terminal", count=0, phase="target")
    result = audit.audit_corpus(writer.root)
    assert result["actual_actions"] == result["source_actions"] == 0
    assert result["completed_chunks"] == 1 and result["source_episodes"] == []
    unfinished = corpus(tmp_path / "unfinished", count=1, terminated=False)
    with pytest.raises(ValueError, match="not a complete native unit"): audit.audit_corpus(unfinished.root)


@pytest.mark.parametrize("native_term,native_trunc,learning_term,learning_trunc", [
    (True, False, False, True), (True, True, True, True), (False, True, False, False),
])
def test_audit_preserves_native_terminal_feedback_at_learning_boundary(
        modules, tmp_path, native_term, native_trunc, learning_term, learning_trunc):
    _, audit = modules
    writer = corpus(tmp_path, terminated=native_term, truncated=native_trunc,
        learning_terminated=learning_term, learning_truncated=learning_trunc)
    with pytest.raises(ValueError, match="native terminal feedback changed"):
        audit.audit_corpus(writer.root)
