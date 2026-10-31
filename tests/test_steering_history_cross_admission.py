"""Allocated Linux: real checkpoint tensors and canonical raw24 A2 evidence.

Tiny synthetic worlds verify admission semantics, not physical robot success.
"""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import steering_history as sh
from fluffy_bassoon.meta_rl.steering_history_cross_admission import verify_condition_cross
from fluffy_bassoon.meta_rl.steering_condition_collection import _state_digest
from fluffy_bassoon.meta_rl.robotwin_history_corpus import parameter_digest
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter
from fluffy_bassoon.rl.steering_actuator_condition import CalibrationDistribution, CalibrationSpec
from test_steering_history import source, profile, ref, json_ref, raw_observation


@pytest.fixture
def raw_cross(source, tmp_path):
    p, prefix = deepcopy(source)
    p["prefix_episodes"] = 24
    prefix["source_publications"] = {}
    names=dict(left="calibration_minus",right="calibration_plus")
    p["conditions"]=[names[c] for c in p["conditions"]]
    p["teacher_lineage_ids"]={names[c]:key for c,key in p["teacher_lineage_ids"].items()}
    for slot in prefix["lineages"]:
        slot["condition_id"]=names[slot["condition_id"]]
    prefix["teacher_qualifications"]={names[c]:r for c,r in prefix["teacher_qualifications"].items()}
    calibration=CalibrationDistribution([CalibrationSpec(sign=sign,bias_radians=(.01,)+(0.,)*13,
        command_lower=(-3.,)*14,command_upper=(3.,)*14,range_receipt_sha256="d"*64,
        mapping_sha256=p["interface"]["mapping_sha256"],motor_sha256=p["interface"]["motor_sha256"],
        reference_basis_sha256="e"*64,reference_arm_bound=.02) for sign in (-1,1)]).contract()
    for slot in prefix["lineages"]:
        state=torch.load(slot["final_recovery"]["path"],weights_only=False)
        full=dict(format="source_coefficient_full_recovery_v1",python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=None,event="published_prefix_full_state",
            control=dict(mapping=dict(condition_audit_contract=calibration)),
            collector=dict(format="source_coefficient_teacher_v1",learner=state,active_episode=None,pending=None,
                pending_update_cycles=0,updating=False,
                lineage=dict(method=slot["lineage_id"],partition=slot["partition"],learner_seed=slot["seed"]),
                profile=dict(task=p["source_task"],motor_sha256=p["interface"]["motor_sha256"],
                    mapping_sha256=p["interface"]["mapping_sha256"],interface_sha256=p["interface"]["control_interface_sha256"],
                    scope_sha256=p["scope_sha256"])))
        path=tmp_path/(slot["lineage_id"]+"-full.pt")
        torch.save(full,path)
        slot["final_recovery"]=dict(ref(path),retention="immutable",size_bytes=path.stat().st_size)
    methods = [p["teacher_lineage_ids"][c] for c in p["conditions"]]
    controller_refs, frozen = {}, {}
    writer = RobotwinCorpusWriter(tmp_path / "cross-corpus", {"source_commit": "f" * 40})

    def episode(spec, metadata, success):
        writer.set_context(dict(phase="condition_cross", method=spec["method"],
            trial_id=spec["id"], policy_version=metadata["policy_version"]))
        features = np.full(3, spec["case"], np.float32)
        info = dict(task=p["source_task"], requested_seed=spec["reset_seed"], actual_seed=spec["reset_seed"],
            observation_sha256=sh.content_hash(["observation", spec["reset_seed"]]),
            features_sha256=sh.content_hash(["features", spec["reset_seed"]]),
            instruction=["Put the phone on its stand."], reset_policy_actions=0)
        eid = writer.begin_episode(raw_observation(spec["case"]), features, info)
        writer.begin_decision(np.zeros((32, 32), np.float32), np.zeros((32, 14), np.float32), metadata)
        mask = np.zeros((1, 32), bool)
        mask[0, 0] = True
        writer.complete_native(next_raw=raw_observation(spec["case"]), next_features=features,
            raw_rewards=np.array([float(success)]), raw_terminations=np.array([True]),
            raw_truncations=np.array([False]), valid_action_mask=mask, reward=float(success),
            terminated=True, truncated=False, executed_steps=1, success=success, duration_seconds=.001)
        writer.finalize_learning_flags(reward=float(success), terminated=True, truncated=False,
            budget_truncated=False, episode_primitive_steps=1)
        manifest = writer.root / "episodes" / eid / "manifest.json"
        return dict(spec=spec, success=success, primitive_steps=1, chunks=1, complete=True, reset_info=info,
            episode_manifest_ref=dict(ref(manifest), root=str(writer.root)))

    validation_refs = []
    for method in methods:
        slot = next(x for x in prefix["lineages"] if x["lineage_id"] == method)
        state = sh._source_state(slot["final_recovery"],lineage=slot,profile=p)
        sha = parameter_digest(state["actor"])
        controller_refs[method] = dict(slot["final_recovery"], policy_sha256=sha,
            optimizer_updates=state["updates"], source_method=method, sampling_mode="sampled_gaussian")
        frozen[method] = dict(full_state_sha256=_state_digest(state), policy_sha256=sha,
                              optimizer_updates=state["updates"])
        actor_path = tmp_path / (method + "-actor.pt")
        torch.save(dict(kind="sac_actor", state_dict=state["actor"], parameter_sha256=sha), actor_path)
        records = []
        for case in range(2):
            spec = dict(id=f"val-{method}-{case}", method=method, case=case,
                        reset_seed=4000000 + 1000 * methods.index(method) + case)
            records.append(episode(spec, dict(policy_version=f"sac:update:{state['updates']}:{sha}",
                optimizer_updates=state["updates"], source_warmup=False,
                policy_checkpoint=str(actor_path), policy_checkpoint_sha256=sh.digest(actor_path)), True))
        val_ref = json_ref(tmp_path / (method + "-validation.json"), dict(complete=True,
            phase="source_validation", source_checkpoint=slot["final_recovery"], results=records))
        validation_refs.append(val_ref)
        prefix["source_publications"][method] = dict(actual_resets=[dict(actual_seed=42)],
            source_validation_receipts={"24": val_ref})
    blind = "source_validation_selected_blind"
    controller_refs[blind] = deepcopy(controller_refs[methods[0]])
    frozen[blind] = deepcopy(frozen[methods[0]])
    selection_ref = json_ref(tmp_path / "blind.json", dict(selected_method=blind,
        selected_source_method=methods[0], selected_from="source_validation", cross_cases_opened=False,
        controller_ref=controller_refs[blind], final_validation_successes={m: 2 for m in methods},
        source_validation_refs=validation_refs))
    control = dict(task=p["source_task"], mapping_sha256=p["interface"]["mapping_sha256"],
        interface_sha256=p["interface"]["control_interface_sha256"], motor_sha256=p["interface"]["motor_sha256"],
        scope_sha256=p["scope_sha256"], public_dim=3, native_horizon=32, episode_limit=600,
        arm="kv", teacher_label_selection=p["teacher_lineage_ids"])

    def build(*, fail=False, forged_flags=False):
        specs, records = [], []
        for condition, matched in zip(p["conditions"], methods):
            for case in range(4):
                for method in (*methods, blind):
                    spec = dict(id=f"cross-{condition}-{case}-{method}", condition_id=condition,
                        task=p["source_task"], arm="kv", case=case, method=method, horizon=600,
                        reset_seed=2300000000 + 997 * case, motor_seed=1730000000 + 997 * case)
                    specs.append(deepcopy(spec))
                    success = (method == matched or (method == blind and matched == methods[0]))
                    if fail and condition == p["conditions"][1]:
                        success = False
                    records.append(episode(spec, dict(policy_version=f"frozen-cross:{method}:{controller_refs[method]['policy_sha256']}",
                        frozen_controller_ref=controller_refs[method], cross_actor_sampling_mode="sampled_gaussian",
                        cross_actor_innovation_seed=1930000000 + 997 * case,
                        steering_condition_id=condition,steering_mapping_sha256=p["interface"]["mapping_sha256"],
                        steering_interface_sha256=p["interface"]["control_interface_sha256"],
                        coefficient_steering=dict(audit_only=dict(mapping=dict(condition_audit_contract=calibration))),
                        optimizer_updates=controller_refs[method]["optimizer_updates"]), success))
        plan = dict(format="source_only_frozen_condition_cross_v1", profile=control, specs=specs,
            controller_refs=controller_refs, condition_blind_method=blind, frozen=frozen,
            blind_selection_ref=selection_ref, actor_seeds_by_scene=[1930000000 + 997 * i for i in range(4)],
            paired_gap=.25, aggregate_blind_gap=.125, source_sgd=0)
        plan["condition_contract"]=deepcopy(calibration)
        plan_ref = json_ref(tmp_path / "plan.json", plan)
        result = dict(format=plan["format"], status="complete", plan_sha256=sh.content_hash(plan),
            records=records, pending=None, active_actor_rng=None,
            summary=dict(criterion_pass=forged_flags, both_directions_pass=forged_flags))
        result_ref = json_ref(tmp_path / "result.json", result)
        prefix["condition_cross"] = dict(result=dict(result_ref, complete=True, phase="condition_cross",
            criterion_pass=forged_flags, both_directions_pass=forged_flags), plan=plan_ref)
        return p, prefix, plan, result

    return build


@pytest.mark.parametrize("objective", ["ad", "dpt"])
def test_original_raw24_not_pass_flags_admits_both_objectives(raw_cross, objective):
    p, prefix, _, _ = raw_cross(forged_flags=False)
    verified = verify_condition_cross(p, prefix, objective=objective)
    assert verified["matched_mismatch_gaps"] == {c:1. for c in p["conditions"]}
    assert verified["pooled_blind_gap"] == .5 and verified["actual_actions"] == 24
    # Actual CalibrationDistribution -> torch full recovery retains tuples;
    # the original A2 JSON plan has lists. The physical contract is identical.
    original=torch.load(prefix["lineages"][0]["final_recovery"]["path"],weights_only=False)
    source_contract=original["control"]["mapping"]["condition_audit_contract"]
    plan_contract=sh.read_json(prefix["condition_cross"]["plan"])["condition_contract"]
    assert isinstance(source_contract["conditions"][0]["bias_radians"],tuple)
    assert isinstance(plan_contract["conditions"][0]["bias_radians"],list)
    assert source_contract!=plan_contract and sh.content_hash(source_contract)==sh.content_hash(plan_contract)


def test_failed_raw24_cannot_be_admitted_by_success_flags(raw_cross):
    p, prefix, _, _ = raw_cross(fail=True, forged_flags=True)
    with pytest.raises(ValueError, match="condition dependence failed"):
        verify_condition_cross(p, prefix, objective="ad")


@pytest.mark.parametrize("mutation,expected", [
    ("sampling", "actor/update/RNG"), ("actor", "actor/update/RNG"),
    ("duplicate", "duplicated"), ("teacher", "canonical collector full recovery"),
    ("calibration", "original source full-recovery"),
])
def test_cross_binds_actual_actor_sampling_cells_and_dpt_teacher(raw_cross, tmp_path, mutation, expected):
    p, prefix, plan, result = raw_cross()
    if mutation == "sampling":
        # A raw sampled trace cannot be re-described as deterministic mean.
        method = p["teacher_lineage_ids"][p["conditions"][0]]
        plan["controller_refs"][method]["sampling_mode"] = "deterministic_mean"
    elif mutation == "actor":
        method = p["teacher_lineage_ids"][p["conditions"][0]]
        plan["controller_refs"][method]["optimizer_updates"] += 1
    elif mutation == "duplicate":
        result["records"][-1] = deepcopy(result["records"][0])
    elif mutation == "calibration":
        plan["condition_contract"]["conditions"][0]["bias_radians"]=[999.]*14
    else:
        condition=p["conditions"][0]
        q = sh.read_json(prefix["teacher_qualifications"][condition])
        other = next(x for x in prefix["lineages"] if x["lineage_id"] == p["teacher_lineage_ids"][p["conditions"][1]])
        q["teacher_recovery"] = other["final_recovery"]
        prefix["teacher_qualifications"][condition] = json_ref(tmp_path / "wrong-teacher.json", q)
    prefix["condition_cross"]["plan"] = json_ref(tmp_path / "plan.json", plan)
    result["plan_sha256"] = sh.content_hash(plan)
    prefix["condition_cross"]["result"] = json_ref(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match=expected):
        verify_condition_cross(p, prefix, objective="dpt")


def test_missing_a2_remains_pending():
    with pytest.raises(ValueError, match="pending"):
        verify_condition_cross(profile(), {}, objective="dpt")


def test_source_admission_calls_production_cross_gate(source, monkeypatch):
    p, prefix = source
    monkeypatch.setattr(sh, "verify_condition_cross", verify_condition_cross)
    for objective in ("ad", "dpt"):
        with pytest.raises(ValueError, match="pending"):
            sh.admit_prefix(p, prefix, objective=objective)


def test_original_cross_does_not_qualify_smaller_pending_prefix(raw_cross):
    p, prefix, _, _ = raw_cross()
    p["prefix_episodes"] = 12
    with pytest.raises(ValueError, match="completed24 prefix"):
        verify_condition_cross(p, prefix, objective="ad")
