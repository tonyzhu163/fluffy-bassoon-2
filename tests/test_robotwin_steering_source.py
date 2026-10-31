"""Source-entry wiring qualification; only execute on allocated Linux CPU."""
from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import robotwin_steering_source as source
from fluffy_bassoon.meta_rl.robotwin_steering_screen import build_plan, next_episode, restricted_summary
from fluffy_bassoon.meta_rl.steering_condition_collection import BoundedTeacherRecovery, _state_digest, lineage_specs
from fluffy_bassoon.meta_rl.vla_noise_sac import NoiseSAC
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.steering_actuator_condition import CalibrationDistribution
from test_steering_actuator_condition import make_pair
from test_steering_condition_collection import profile, teacher


def tiny_factory(monkeypatch):
    existing, _ = teacher()
    def make(p, lineage, *, device="cpu"):
        return NoiseSAC(2062, 4, action_limit=1., device=device, seed=lineage.learner_seed,
                        config=existing.learner.config)
    monkeypatch.setattr(source, "make_learner", make)
    return make


def a1_fixture():
    p = replace(profile(), arm="noise")
    plan, records = build_plan(dict.fromkeys(source.TASKS, 400)), {}
    while (spec := next_episode(plan, records)) is not None:
        if spec["stage"] == "development":
            success = not (spec["task"] == p.task and spec["case"] == 0)
        else:
            success = spec["arm"] in ("noise", "kv")
        records[spec["id"]] = dict(spec=spec, complete=True, success=success,
            primitive_steps=50 if success else 400,
            reset_identity=dict(actual_seed=spec["reset_seed"], observation_sha256="a"*64,
                                features_sha256="b"*64))
    a1 = dict(status="complete_restricted_development_search", plan=plan, records=records,
        interface_sha256=p.interface_sha256, frozen_motor_sha256="1"*64,
        restricted_summary=restricted_summary(plan, records))
    q = dict(status="passed", zero_kv_exact=True, zero_fixed_kv_exact=True,
        zero_action_exact=True, frozen_before="1"*64, frozen_after="1"*64,
        actual_actions=400, motor_gradients=0, native_parity=dict(passed=True), kv_gradient_norm=1.,
        qualification_episode=plan["development"][0]["id"], complete_real_episode=plan["development"][0]["id"])
    selection = dict(schema="frozen_motus_source_selection_v1", status="selected_from_completed_A1",
        selection_basis="natural_A1_predeclared_criteria", condition_outcomes_used=False,
        task=p.task, arm=p.arm, mapping_sha256=p.mapping_sha256, interface_sha256=p.interface_sha256,
        motor_sha256=p.motor_sha256, episode_limit=p.episode_limit,
        task_horizon_sha256=p.task_horizon_sha256,
        selection_rule=dict(path="actual-rule.json", sha256=source.SELECTION_RULE_SHA256),
        evidence_episode_ids=[key for key, r in records.items()
            if r["spec"]["stage"] == "confirmation" and r["spec"]["arm"] in ("native", "noise")])
    rule = dict(format="frozen_motus_A1_operational_selection_v1", tie_order=["noise", "kv", "action"])
    return p, selection, q, a1, rule


def test_selection_recomputes_confirmed_rescue_and_fixed_tie_rule():
    p, selection, q, a1, rule = a1_fixture()
    source.validate_selection(selection, q, a1, p, dict(motor_sha256=p.motor_sha256),
                              interface_sha=p.interface_sha256, selection_rule=rule)
    assert a1["restricted_summary"]["selected_source_task"] == p.task
    with pytest.raises(ValueError, match="greatest strict"):
        source.validate_selection(dict(selection, arm="kv"), q, a1, replace(p, arm="kv"),
            dict(motor_sha256=p.motor_sha256), interface_sha=p.interface_sha256, selection_rule=rule)
    with pytest.raises(ValueError, match="Cite all actual"):
        source.validate_selection(dict(selection, evidence_episode_ids=selection["evidence_episode_ids"][:1]),
            q, a1, p, dict(motor_sha256=p.motor_sha256), interface_sha=p.interface_sha256, selection_rule=rule)


def test_selection_without_true_repeated_recovery_is_inconclusive_not_eligible():
    p, selection, q, a1, rule = a1_fixture()
    for record in a1["records"].values():
        if record["spec"]["stage"] == "confirmation":
            record["success"] = True  # Native succeeds too: not a strict rescue.
    a1["restricted_summary"] = restricted_summary(a1["plan"], a1["records"])
    with pytest.raises(ValueError, match="No eligible"):
        source.validate_selection(selection, q, a1, p, dict(motor_sha256=p.motor_sha256),
                                  interface_sha=p.interface_sha256, selection_rule=rule)


def test_validation_crossed_warmup_counter_uses_uniform_until_completed_update(monkeypatch):
    make = tiny_factory(monkeypatch)
    p, lineage = profile(), lineage_specs()[0]
    learner = make(p, lineage)
    for _ in range(33):
        learner.observe(np.zeros(2062), np.zeros(4), 0., np.zeros(2062), False, False, 1)
    assert not learner.warming_up and learner.updates == 0
    saved = dict(learner=learner.state_dict())
    before = _state_digest(learner.state_dict())
    spec = dict(reset_seed=2100000000)
    def forbidden(_):
        raise AssertionError("The untrained NN actor must not label initial source validation")
    one = source.FrozenSourceBehavior(saved, p, lineage, spec, archive_actor=forbidden,
                                      guard=lambda: None, device="cpu")
    two = source.FrozenSourceBehavior(saved, p, lineage, spec, archive_actor=forbidden,
                                      guard=lambda: None, device="cpu")
    one.learner.act = lambda *args, **kwargs: forbidden(None)
    action = one.act(np.zeros(2062))
    np.testing.assert_array_equal(action, two.act(np.zeros(2062)))
    assert one.decision_metadata()["source_warmup"] and np.max(np.abs(action)) <= 1
    one.observe(None)
    assert _state_digest(learner.state_dict()) == before


def test_frozen_learned_validation_uses_existing_actor_archive_and_no_observe_update(monkeypatch, tmp_path):
    make = tiny_factory(monkeypatch)
    p, lineage = profile(), lineage_specs()[0]
    learner = make(p, lineage)
    for _ in range(33):
        learner.observe(np.zeros(2062), np.zeros(4), 0., np.zeros(2062), False, False, 1)
    learner.update(gradient_steps=1)
    archive = source.existing_actor_archive(tmp_path/"actors")
    frozen = source.FrozenSourceBehavior(dict(learner=learner.state_dict()), p, lineage,
        dict(reset_seed=2100000000), archive_actor=lambda l: archive.identify(l, "sac"),
        guard=lambda: None, device="cpu")
    frozen.learner.sample_warmup = lambda *args: pytest.fail("Learned validation must use its actual actor")
    before = _state_digest(learner.state_dict())
    action = frozen.act(np.zeros(2062))
    frozen.observe(np.zeros(2062), action, 1., np.zeros(2062), True, False, 1, {})
    metadata = frozen.decision_metadata()
    assert not metadata["source_warmup"] and metadata["optimizer_updates"] == 1
    assert metadata["policy_version"].startswith("sac:update:1:")
    assert _state_digest(learner.state_dict()) == before
    assert frozen.learner.updates == 1 and len(frozen.learner.replay) == 33


def test_global_resume_restores_real_python_numpy_torch_streams():
    saved = dict(python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                 torch_rng=torch.get_rng_state(), cuda_rng=None)
    expected = random.random(), float(np.random.random()), torch.rand(3)
    random.random(); np.random.random(); torch.rand(3)
    source.restore_global_rng(saved)
    actual = random.random(), float(np.random.random()), torch.rand(3)
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_source_admission_reuses_existing_ledger_and_cumulative_resource_envelope(tmp_path):
    p, phase_id = profile(), "frozen-pi05-endtoend-20261031-r1"
    contract = dict(scope_sha256=p.scope_sha256, phase_id=phase_id, motor_sha256=p.motor_sha256)
    with CampaignBudget(tmp_path/"ledger", campaign_id=phase_id, total_actions=300000,
            source_caps=p.source_caps(), phase_caps=source.PHASE_CAPS, contract=contract,
            run_id="cpu-fixture", source_commit="a"*40) as budget:
        ledger = dict(budget.header, root=str(tmp_path/"ledger"))
    scope = dict(schema="frozen_pi05_endtoend_scope_v1", phase_id=phase_id,
        phase_limits=dict(cumulative_gpu_seconds=28800, all_in_simulator_actions=300000,
                          outputs_gib=64, fresh_cpu_qualification_worker_seconds=1800))
    allocation = dict(schema="frozen_motus_source_allocation_v1", scope_sha256=p.scope_sha256,
        profile_sha256="d"*64, source_commit="a"*40, hostname="allocated-cpu-fixture",
        gpu_count=1, job_id="synthetic-ticket", output=str(tmp_path/"output"),
        phase_output_root=str(tmp_path), reserved_gpu_seconds=1000, prior_phase_gpu_seconds=632,
        other_reserved_phase_gpu_seconds=5000, driver_deadline_unix=2000,
        allocation_started_unix=1000, cleanup_reserve_seconds=60, maximum_episode_seconds=180,
        maximum_episode_update_seconds=30, action_ledger=ledger, max_full_recovery_bytes=1000000,
        recovery_output_reservation_bytes=60000000, other_future_output_reservation_bytes=0)
    args = dict(source="a"*40, hostname="allocated-cpu-fixture", output=tmp_path/"output", profile_sha="d"*64)
    source.validate_source_admission(scope, allocation, p, **args)
    with pytest.raises(ValueError, match="No fresh child"):
        source.validate_source_admission(scope, dict(allocation, prior_phase_gpu_seconds=28000), p, **args)
    with pytest.raises(ValueError, match="recovery-size"):
        source.validate_source_admission(scope, dict(allocation, recovery_output_reservation_bytes=59999999), p, **args)


def test_actual_orchestration_publishes_balanced_train_and_val_with_immutable_sink(monkeypatch, tmp_path):
    tiny_factory(monkeypatch)
    p, output = profile(), tmp_path/"source"
    output.mkdir()
    _, pair = make_pair()
    distribution = CalibrationDistribution(pair)
    control = SimpleNamespace(active=False, post_command_transform=distribution,
        condition_contract=distribution.contract(), interface_sha256=p.interface_sha256,
        episode_limit_for_task=lambda task: 400, set_coefficients=lambda *args: None,
        mapping_sha256=lambda: p.mapping_sha256, snapshot=lambda: dict(actual_control_state=True))
    class Guard:
        maximum_episode_seconds = .1
        def __call__(self):
            pass
        def admit_episode(self):
            pass
    guard, received, validation_modes = Guard(), [], []
    def runner(current, spec, budget, *, controller, run_id, phase, **kwargs):
        assert distribution.selected_sign == controller.lineage.sign
        obs = np.zeros(2062, np.float32)
        action = controller.act(obs)
        if phase == "source_validation":
            validation_modes.append(controller.decision_metadata()["source_warmup"])
        key = run_id+":"+spec["id"]
        reserve = budget.reserve(key, 1, phase=phase, method=spec["method"],
            task=spec["task"], episode=spec["id"], chunk=0, policy_version="test-only")
        assert reserve.execute
        budget.resolve(key, 1)
        ref = dict(path=str(output/(key.replace(":", "_")+".json")), sha256="a"*64)
        controller.observe(obs, action, 1., obs, True, False, 1, ref)
        return dict(spec=spec, complete=True, primitive_steps=1, chunks=1, wall_seconds=.001,
            success=True, reset_info=dict(actual_seed=spec["reset_seed"]), corpus_refs=[ref],
            episode_manifest_ref=ref)
    monkeypatch.setattr(source, "run_complete_episode", runner)
    def builder(student_profile, publications, *, output, **kwargs):
        count = student_profile["prefix_episodes"]
        assert len(publications) == 6
        assert all(len(item["complete_episode_manifests"]) == count for item in publications.values())
        assert sorted(item["lineage"]["partition"] for item in publications.values()) == ["train"]*4+["validation"]*2
        for item in publications.values():
            assert item["final_full_state"]["retention"] == "immutable"
            current = BoundedTeacherRecovery.load(item["final_full_state"])["collector"]
            validation_ref = item["source_validation_receipts"][str(count)]["input_checkpoint"]
            validated = BoundedTeacherRecovery.load(validation_ref)["collector"]
            assert item["final_full_state"] != validation_ref
            assert _state_digest(current["learner"]) == _state_digest(validated["learner"])
        received.append(count)
        source._atomic_json(output, dict(completed=True, actual_prefix=count, gates="pending"))
    with CampaignBudget(tmp_path/"ledger", campaign_id="test-only", total_actions=300000,
            source_caps=p.source_caps(), phase_caps=source.PHASE_CAPS,
            contract=dict(scope_sha256=p.scope_sha256), run_id="test", source_commit="a"*40) as budget:
        status = source.run_source_collection(control=control, budget=budget, profile=p,
            distribution=distribution, output=output, guard=guard,
            archive=source.existing_actor_archive(output/"actors"), run_id="test", device="cpu",
            source_commit="a"*40,
            student_profiles={12: dict(prefix_episodes=12), 24: dict(prefix_episodes=24)},
            control_qualification=dict(path="pending-owner-proof", sha256="a"*64), prefix_builder=builder)
        assert status == "initial_prefix_complete"
        assert budget.snapshot()["unresolved_decisions"] == 0
    assert received == [12, 24] and len(validation_modes) == 36 and all(validation_modes)
    final = source.json.loads((output/"source-state.json").read_text())
    assert final["closed_physical_boundary"] and set(final["complete_episodes"].values()) == {24}
    assert set(final["actual_optimizer_cycles"].values()) == {0}  # Early successes cannot invent learning.
