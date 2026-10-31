"""Allocated Linux CPU regressions. These do not qualify the real VLA encoder."""
from contextlib import nullcontext
from copy import deepcopy
from types import ModuleType, SimpleNamespace
import random
import sys
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from fluffy_bassoon.rl.robotwin_noise import (
    _RoboTwinBackend, RobotwinNoiseEnv, _observation_hash, bind_paired_instruction,
    validate_paired_instruction,
)
from fluffy_bassoon.meta_rl.robotwin_steering_screen import (
    paired_reset_fields, run_complete_episode, build_plan, next_episode, reset_identity,
    validate_instruction_migration, validate_migration_ledger, FORMAT, sha, TASKS,
    repeat_paired_prefix,
)
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter


def observation(text="original accepted instruction"):
    return dict(main_images=np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3),
                wrist_images=np.arange(24, dtype=np.uint8).reshape(1, 2, 2, 2, 3),
                states=np.arange(14, dtype=np.float64).reshape(1, 14), task_descriptions=[text])


@pytest.mark.parametrize("value", ["text", [], [""], [" "], [None], ["a", "b"], None])
def test_binding_rejects_ambiguous_or_empty_scalar_instructions(value):
    with pytest.raises(ValueError, match="one nonempty"):
        validate_paired_instruction(value)


def test_binding_changes_only_language_and_retains_sampled_physical_hash():
    raw = observation("sampled replacement")
    original = deepcopy(raw)
    effective, audit = bind_paired_instruction(raw, ["original accepted instruction"])
    assert effective["task_descriptions"] == ["original accepted instruction"]
    assert raw["task_descriptions"] == original["task_descriptions"]
    for key in ("main_images", "wrist_images", "states"):
        assert effective[key] is raw[key]
        np.testing.assert_array_equal(raw[key], original[key])
    assert audit["sampled_instruction"] == raw["task_descriptions"]
    assert audit["bound_instruction"] == effective["task_descriptions"]
    assert audit["sampled_observation_sha256"] == _observation_hash(raw)
    assert audit["effective_observation_sha256"] == _observation_hash(effective)
    assert audit["sampled_observation_sha256"] != audit["effective_observation_sha256"]


def encoder(monkeypatch):
    module = ModuleType("rlinf.models.embodiment.openpi.modules.model")
    module.preprocess_observation = lambda value, train: value
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    seen = []

    def prepare(raw):
        seen.append(raw["task_descriptions"].copy())
        return SimpleNamespace(state=torch.as_tensor(raw["states"]), raw=raw)

    def prefix(prepared):
        scalar = sum(map(ord, prepared.raw["task_descriptions"][0]))
        values = torch.full((1, 2, 2048), float(scalar))
        return values, torch.ones(1, 2, dtype=torch.bool), ((values[:, :, :4, None].clone(),),)

    backend = object.__new__(_RoboTwinBackend)
    backend.torch, backend.observation_dim = torch, 2062
    backend.model = SimpleNamespace(env_obs_to_observation=prepare, build_prefix_cache=prefix)
    backend.prefix_calls, backend.prefix_seconds = 0, 0.
    backend.motor_queries, backend.motor_seconds = 0, 0.
    backend._paired_instruction = ("original accepted instruction",)
    return backend, seen


def test_binding_precedes_prefix_for_reset_and_later_observations(monkeypatch):
    backend, seen = encoder(monkeypatch)
    first = backend._encode(observation())
    initial_cache = deepcopy(backend._kv_cache)
    following = backend._encode(observation("later simulator-sampled instruction"))
    np.testing.assert_array_equal(first, following)
    assert all(text == ["original accepted instruction"] for text in seen)
    assert torch.equal(initial_cache[0][0], backend._kv_cache[0][0])
    assert backend._instruction_binding["sampled_instruction"] == ["later simulator-sampled instruction"]
    assert backend._raw_obs["task_descriptions"] == ["original accepted instruction"]


def test_saved_original_frame_repeat_is_exact_and_keeps_actual_sampled_audit(monkeypatch, tmp_path):
    backend, _ = encoder(monkeypatch)
    raw = observation()
    features = backend._encode(raw)
    writer = RobotwinCorpusWriter(tmp_path/"corpus", {})
    writer.set_context(dict(method="native", phase="a1", trial_id="baseline", policy_version="baseline"))
    identity = dict(actual_seed=7, observation_sha256=_observation_hash(raw),
                    features_sha256=hashlib.sha256(features.tobytes()).hexdigest())
    episode = writer.begin_episode(raw, features, dict(identity, instruction=raw["task_descriptions"]))
    path = writer.root/"episodes"/episode/"reset.json"
    backend._encode(observation("actual new simulator sample"))
    audit = deepcopy(backend._instruction_binding)
    result = repeat_paired_prefix(backend, dict(expected_reset=identity, paired_instruction=raw["task_descriptions"]),
                                 dict(path=str(path), sha256=sha(path)))
    assert result["saved_original_frame"] is not None
    assert result["public_features_exact"] and result["kv_cache_exact"] and result["prefix_mask_exact"]
    assert backend._instruction_binding == audit


@pytest.mark.parametrize("fault", ["features", "mask", "cache"])
def test_prefix_repeat_rejects_actual_numerical_or_layout_drift(monkeypatch, fault):
    backend, _ = encoder(monkeypatch)
    backend._encode(observation())
    original = backend._encode

    def drifting(raw):
        result = original(raw)
        if fault == "features": backend._public_features[0, 0] += 1
        if fault == "mask": backend._prefix_mask[0, 0] = False
        if fault == "cache": backend._kv_cache[0][0][0, 0, 0, 0] += 1
        return result

    backend._encode = drifting
    with pytest.raises(RuntimeError, match="not exactly equal"):
        repeat_paired_prefix(backend, dict(paired_instruction=["original accepted instruction"]))


def test_later_corpus_frame_and_decision_use_same_bound_text(monkeypatch):
    backend, _ = encoder(monkeypatch)
    backend._encode(observation("initial sampled instruction"))
    backend.decode = lambda noise: np.zeros((3, 14), np.float32)
    backend._decision_metadata, backend._last_residual_metadata = {}, None
    following = observation("new text after chunk")
    backend.env = SimpleNamespace(venv=SimpleNamespace(envs=[SimpleNamespace(task=object())]),
        chunk_step=lambda actions: ([following], torch.ones(1), torch.zeros(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool), [dict(valid_action_mask=[1, 1, 1], episode=dict(success_once=False))]))
    captured = {}
    backend.corpus_writer = SimpleNamespace(
        begin_decision=lambda noise, actions, metadata: captured.update(metadata=metadata),
        complete_native=lambda **values: captured.update(values))
    monkeypatch.setattr("fluffy_bassoon.rl.robotwin_noise._capture_sparse_feedback", lambda task: nullcontext({}))
    backend.step(np.zeros((3, 32), np.float32), 3)
    assert captured["metadata"]["paired_instruction"]["sampled_instruction"] == ["initial sampled instruction"]
    assert captured["next_raw"]["task_descriptions"] == ["original accepted instruction"]
    assert captured["native_info"]["paired_instruction"]["sampled_instruction"] == ["new text after chunk"]
    assert following["task_descriptions"] == ["new text after chunk"]


def test_paired_step_without_corpus_writer_keeps_bound_input_after_real_step_boundary(monkeypatch):
    backend, _ = encoder(monkeypatch)
    backend._encode(observation("initial sampled instruction"))
    backend.decode = lambda noise: np.zeros((3,14), np.float32)
    backend._decision_metadata, backend._last_residual_metadata = {}, None
    backend.corpus_writer = None
    commands = []
    def step(actions):
        commands.append(actions.clone())
        return [observation("later sampled text")], torch.ones(1), torch.ones(1,dtype=torch.bool), torch.zeros(1,dtype=torch.bool), [
            dict(valid_action_mask=[1,1,1],episode=dict(success_once=True))]
    backend.env = SimpleNamespace(chunk_step=step)
    _,_,end,_,info = backend.step(np.zeros((3,32),np.float32),3)
    assert end and info["primitive_steps"] == 3 and len(commands) == 1
    assert backend._raw_obs["task_descriptions"] == ["original accepted instruction"]
    assert backend._instruction_binding["sampled_instruction"] == ["later sampled text"]


def test_new_unpaired_reset_clears_binding_and_restores_global_rng(monkeypatch):
    backend = object.__new__(_RoboTwinBackend)
    backend.torch = torch
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda value: None)
    backend._reset_scene = lambda task, seed: (backend._paired_instruction, random.random(), np.random.rand())
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    bound = backend.reset("task", 7, paired_instruction=["keep"])
    assert bound[0] == ("keep",)
    assert backend.reset("task", 8)[0] is None
    assert random.getstate() == before[0]
    assert np.random.get_state()[0] == before[1][0]
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_common_baseline_fields_bind_actual_text_without_aliasing_record():
    identity = dict(actual_seed=7, observation_sha256="a"*64, features_sha256="b"*64)
    record = dict(reset_identity=identity, reset_info=dict(identity, instruction=["accepted"]))
    fields = paired_reset_fields(record)
    assert fields == dict(expected_reset=identity, paired_instruction=["accepted"])
    fields["paired_instruction"][0] = "changed"
    assert record["reset_info"]["instruction"] == ["accepted"]
    record["reset_info"]["actual_seed"] = 8
    with pytest.raises(ValueError, match="identity"):
        paired_reset_fields(record)


def test_pair_without_original_instruction_fails_before_budget_or_reset():
    spec = dict(arm="noise", expected_reset={})
    with pytest.raises(ValueError, match="original actual"):
        run_complete_episode(object(), spec, object(), run_id="fixture")


def test_public_env_forwards_explicit_binding_and_default_reset_unchanged():
    seen = []
    backend = SimpleNamespace(observation_dim=3, noise_shape=(4, 2),
        reset=lambda task, seed, **options: (seen.append(options) or np.zeros(3), {}))
    env = RobotwinNoiseEnv(backend=backend)
    env.reset("place_phone_stand", 7, paired_instruction=["accepted"])
    env.reset("place_phone_stand", 8)
    assert seen == [dict(paired_instruction=("accepted",)), {}]


@pytest.mark.parametrize("fault", ["seed", "raw", "features", "instruction"])
def test_strict_pair_identity_still_stops_before_first_reservation(fault):
    expected = dict(actual_seed=7, observation_sha256="a"*64, features_sha256="b"*64)
    info = dict(expected, instruction=["accepted"], reset_policy_actions=0)
    if fault == "seed": info["actual_seed"] += 1
    if fault == "raw": info["observation_sha256"] = "f"*64
    if fault == "features": info["features_sha256"] = "f"*64
    if fault == "instruction": info["instruction"] = ["unbound"]
    calls = []
    control = SimpleNamespace(episode_limit_for_task=lambda task: 1,
        set_recording_context=lambda value: None, set_coefficients=lambda arm, values: None,
        reset=lambda *args, **kwargs: calls.append(kwargs) or np.zeros(2062), last_reset_info=info)
    budget = SimpleNamespace(snapshot=lambda: dict(unresolved_decisions={}), remaining=lambda **kwargs: 1)
    spec = dict(id="paired", task=TASKS[0], reset_seed=7, motor_seed=8, horizon=1, arm="noise",
                coefficients=[0.]*4, expected_reset=expected, paired_instruction=["accepted"])
    with pytest.raises(RuntimeError):
        run_complete_episode(control, spec, budget, run_id="fixture", before_step=lambda *args: pytest.fail("issued"))
    assert calls == [dict(motor_seed=8, paired_instruction=("accepted",))]


def json_ref(path, value):
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=sha(path))


@pytest.fixture
def migration(tmp_path):
    """Closed real corpus/ledger APIs, synthetic observations; no simulator."""
    plan = build_plan(dict.fromkeys(TASKS, 1))
    writer = RobotwinCorpusWriter(tmp_path/"corpus", {})
    features = np.arange(2062, dtype=np.float32)
    records = {}

    def begin(spec, text, values):
        raw = observation(text)
        writer.set_context(dict(method=spec["arm"], phase="a1", trial_id=spec["id"], policy_version=spec["id"]))
        info = dict(actual_seed=spec["reset_seed"], requested_seed=spec["reset_seed"], task=spec["task"],
                    observation_sha256=_observation_hash(raw), features_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                    instruction=[text], reset_policy_actions=0)
        episode = writer.begin_episode(raw, values, info)
        info["corpus_episode_id"] = episode
        return raw, info

    old_source, new_source = "a"*40, "b"*40
    ledger = dict(root=tmp_path/"ledger", campaign_id="test", total_actions=300000, source_caps={},
                  phase_caps={"a1":30000}, contract={})
    with CampaignBudget(**ledger, run_id="old", source_commit=old_source) as budget:
        for spec in plan["development"]:
            raw, info = begin(spec, "original accepted instruction", features)
            writer.begin_decision(np.zeros((1, 32)), np.zeros((1, 14)), {})
            writer.complete_native(next_raw=raw, next_features=features, raw_rewards=[0], raw_terminations=[0],
                raw_truncations=[1], valid_action_mask=[1], reward=0., terminated=False, truncated=True,
                executed_steps=1, success=False, duration_seconds=.01)
            writer.finalize_learning_flags(reward=0., terminated=False, truncated=True, budget_truncated=False,
                                           episode_primitive_steps=1)
            path = writer.root/"episodes"/info["corpus_episode_id"]/"manifest.json"
            record = dict(spec=spec, complete=True, success=False, primitive_steps=1, chunks=1,
                reset_info=info, reset_identity=reset_identity(info),
                episode_manifest_ref=dict(root=str(writer.root), path=str(path.relative_to(writer.root)), sha256=sha(path)))
            records[spec["id"]] = record
            budget.reserve("old:"+spec["id"], 1, phase="a1", method="native", task=spec["task"],
                           episode=spec["id"], chunk=0, policy_version=spec["id"])
            budget.resolve("old:"+spec["id"], 1)
        snapshot = budget.snapshot()
    pending = next_episode(plan, records)
    pending.pop("paired_instruction")
    _, failed_info = begin(pending, "changed simulator instruction", features+1)
    failed_path = writer.root/"episodes"/failed_info["corpus_episode_id"]/"reset.json"
    baseline = records[f"dev-{pending['task']}-{pending['case']:02d}"]
    original_path = writer.root/Path(baseline["episode_manifest_ref"]["path"]).with_name("reset.json")
    prior = dict(status="blocked_failure", plan=plan, records=records, pending=pending, scope_sha256="c"*64,
        interface_sha256="d"*64, source_commit=old_source, actions=snapshot,
        action_ledger=dict(root=str(ledger["root"])),
        failure=dict(type="RuntimeError", message="Actual seed/raw observation/public features differ from the paired development reset"))
    recovery = tmp_path/"recovery.pt"
    recovery.write_bytes(b"stand-in whose content is checked by the CLI before torch.load")
    proof = dict(schema="robotwin_paired_instruction_migration_v1", source_commit_before=old_source,
        source_commit_after=new_source, recovery=dict(path=str(recovery), sha256=sha(recovery)),
        result=json_ref(tmp_path/"result.json", prior),
        terminal=json_ref(tmp_path/"terminal.json", dict(format="frozen_pi05_a1_terminal_failure_receipt_v1",
            status="blocked_failure", scheduler=dict(status="Failed"), exit_code=1,
            charged_gpu_seconds=632, source_commit=old_source)),
        allocation_closeout=json_ref(tmp_path/"closeout.json", dict(source_commit=old_source, exit_code=1,
            error=None, attempt=dict(status="finished"), charged_gpu_seconds=632)),
        original_reset=dict(path=str(original_path), sha256=sha(original_path)),
        failed_reset=dict(path=str(failed_path), sha256=sha(failed_path)),
        failed_manifest=dict(path=str(failed_path.with_name("manifest.json")), sha256=sha(failed_path.with_name("manifest.json"))),
        native_language_audit=json_ref(tmp_path/"language.json", dict(format="frozen_pi05_native_language_constancy_v1",
            all_sixteen_constant=True, source_commit=old_source,
            episodes=[dict(episode=key, all_languages_constant=True, chunks=row["chunks"],
                           language=row["reset_info"]["instruction"]) for key,row in records.items()])),
        journal=dict(root=str(ledger["root"]), event_count=snapshot["journal_events"],
                     sha256=snapshot["journal_sha256"], actual_actions=snapshot["actual_actions"]))
    allocation = dict(prior_a1_gpu_seconds=632, prior_phase_gpu_seconds=632, action_ledger=dict(root=str(ledger["root"])))
    saved = dict(format=FORMAT, state=prior)
    kwargs = dict(resume=recovery, resume_sha=sha(recovery), plan=plan, source=new_source,
                  scope_sha="c"*64, interface_sha="d"*64, allocation=allocation)
    return saved, proof, kwargs, ledger


def test_exact_failed_instruction_migration_preserves_records_cursor_and_costs(migration):
    saved, proof, kwargs, ledger = migration
    before = deepcopy(saved)
    successor = validate_instruction_migration(saved, proof, **kwargs)
    assert saved == before
    assert {k:v for k,v in successor.items() if k != "paired_instruction"} == saved["state"]["pending"]
    assert successor["paired_instruction"] == ["original accepted instruction"]
    with CampaignBudget(**ledger, run_id="new", source_commit=kwargs["source"]) as budget:
        validate_migration_ledger(budget, proof)
        assert budget.snapshot()["actual_actions"] == 16


@pytest.mark.parametrize("fault", ["failure", "cursor", "cost", "incomplete", "physical", "zero_chunk", "language",
                                  "foreign_same_total_root", "predecessor_tip", "predecessor_count"])
def test_instruction_migration_rejects_unproven_or_changed_predecessor(migration, fault):
    saved, proof, kwargs, _ = migration
    if fault == "failure": saved["state"]["failure"]["message"] = "unrelated crash"
    if fault == "cursor": saved["state"]["pending"]["motor_seed"] += 1
    if fault == "cost": kwargs["allocation"]["prior_a1_gpu_seconds"] = 0
    if fault == "incomplete": next(iter(saved["state"]["records"].values()))["complete"] = False
    if fault == "foreign_same_total_root":
        old = Path(proof["journal"]["root"])
        foreign = old.with_name("copied-equal-total-journal")
        shutil.copytree(old,foreign)
        proof["journal"]["root"] = str(foreign)
        kwargs["allocation"]["action_ledger"]["root"] = str(foreign)
    if fault == "predecessor_tip": proof["journal"]["sha256"] = "f"*64
    if fault == "predecessor_count": proof["journal"]["event_count"] += 1
    if fault in ("physical", "zero_chunk", "language"):
        key = {"physical":"failed_reset", "zero_chunk":"failed_manifest", "language":"native_language_audit"}[fault]
        path = Path(proof[key]["path"])
        data = json.loads(path.read_text())
        if fault == "physical": data["initial_observation"]["arrays"]["states"]["sha256"] = "f"*64
        if fault == "zero_chunk": data["completed_chunks"] = 1
        if fault == "language": data["all_sixteen_constant"] = False
        proof[key] = json_ref(path, data)
    # Keep result/recovery agreement for faults deeper than that independent gate.
    if fault in ("cursor", "incomplete"):
        proof["result"] = json_ref(Path(proof["result"]["path"]), saved["state"])
    with pytest.raises(ValueError):
        validate_instruction_migration(saved, proof, **kwargs)


@pytest.mark.parametrize("fault", ["reserved_failed", "other_event", "changed_tip"])
def test_migration_ledger_refuses_prior_commands_or_intervening_events(migration, fault):
    saved, proof, kwargs, ledger = migration
    with CampaignBudget(**ledger, run_id="new", source_commit=kwargs["source"]) as budget:
        if fault == "reserved_failed":
            spec = saved["state"]["pending"]
            budget.reserve("new:failed:0", 1, phase="a1", method="noise", task=spec["task"],
                           episode=spec["id"], chunk=0, policy_version=spec["id"])
        if fault == "other_event": proof["journal"]["event_count"] -= 1
        if fault == "changed_tip": proof["journal"]["sha256"] = "f"*64
        with pytest.raises(ValueError): validate_migration_ledger(budget, proof)
