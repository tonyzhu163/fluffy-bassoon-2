"""Scientific collector/recovery regressions; allocated Linux CPU execution only."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.steering_condition_collection import (
    PHASE_CAPS, SOURCE_METHOD_KEYS, BoundedTeacherRecovery, CoefficientTeacher, QualifiedCoefficientTeacher,
    SourceProfile, _state_digest, collect_lineage, collect_round_robin,
    cross_evaluation_specs, episode_spec, lineage_specs, make_learner,
)
from fluffy_bassoon.meta_rl.robotwin_history_corpus import parameter_digest
from fluffy_bassoon.meta_rl.vla_noise_sac import NoiseSAC, SACConfig
from fluffy_bassoon.rl.steering_actuator_condition import CalibrationDistribution
from test_steering_actuator_condition import make_pair


def profile():
    return SourceProfile(task="place_phone_stand", episode_limit=400, mapping_sha256="c"*64,
        interface_sha256="d"*64, motor_sha256="b"*64, scope_sha256="e"*64,
        task_horizon_sha256="f"*64, source_selection_sha256="a"*64,
        arm="decoded_ALOHA_joint_coefficients")


class RecoverySink:
    """Test-only bounded retention; production sink is owned by the launcher."""
    def __init__(self):
        self.blobs, self.events, self.serial = {}, [], 0

    def __call__(self, event, payload, *, immutable):
        self.serial += 1
        path = f"immutable/{self.serial}.pt" if immutable else "latest.pt"
        reference = dict(path=path, sha256=_state_digest(payload),
                         retention="immutable" if immutable else "rolling")
        self.blobs[path] = deepcopy(payload)
        self.events.append((event, reference))
        return reference


def archive(learner):
    digest = parameter_digest(learner.actor.state_dict())
    return dict(policy_sha256=digest, policy_version=f"sac:update:{learner.updates}:{digest}",
                policy_checkpoint=f"actors/{digest}.pt", policy_checkpoint_sha256="a"*64)


def teacher(index=0):
    p, sink = profile(), RecoverySink()
    lineage = lineage_specs()[index]
    # Same scientific profile with a small CPU network for recovery/update tests.
    learner = NoiseSAC(2062, 4, action_limit=1., device="cpu", seed=lineage.learner_seed,
        config=SACConfig(hidden_dims=(8,), layer_norm=False, replay_capacity=64,
                         batch_size=32, warmup_chunks=32, utd=4))
    return CoefficientTeacher(learner, p, lineage, checkpoint=sink,
                              archive_actor=archive, guard=lambda: None), sink


def transition(t, index, *, terminal=False):
    obs = np.full(2062, index/100, np.float32)
    action = t.act(obs)
    metadata = t.decision_metadata()
    t.observe(obs, action, float(terminal), obs+.001, terminal, False, 1,
              dict(path=f"chunks/{index}.json", sha256="b"*64))
    return metadata


def receipt(t, count):
    return dict(spec=deepcopy(t.active_episode), complete=True, primitive_steps=count,
                chunks=count, corpus_refs=[dict(path=f"chunks/{i}.json", sha256="b"*64)
                                          for i in range(count)],
                episode_manifest_ref=dict(path="episodes/manifest.json", sha256="c"*64),
                reset_info=dict(actual_seed=7), success=False)


def test_new_profile_keeps_old_defaults_and_all_lineages_source_only():
    p = profile()
    assert p.arm == "action" and p.public_dim == 2062
    assert (SACConfig().batch_size, SACConfig().warmup_chunks, SACConfig().utd) == (256, 500, 20)
    learner = make_learner(p, lineage_specs()[0], device="cpu")
    assert (learner.config.batch_size, learner.config.warmup_chunks, learner.config.utd) == (32, 32, 4)
    assert learner.action_limit == 1 and learner.observation_dim == 2062
    assert len({s.learner_seed for s in lineage_specs()}) == 6
    assert [s.partition for s in lineage_specs()] == ["train"]*4+["validation"]*2
    with pytest.raises(ValueError, match="selected source"):
        replace(p, task="beat_block_hammer")
    assert p.contract()["teacher_label_selection"] == dict(
        calibration_minus=SOURCE_METHOD_KEYS[2], calibration_plus=SOURCE_METHOD_KEYS[3])


def test_crossing_episode_stays_uniform_then_exact_eligible_utd_at_boundary():
    t, sink = teacher()
    t.begin_episode(episode_spec(t.profile, t.lineage, 0))
    records = [transition(t, i, terminal=i == 35) for i in range(36)]
    assert t.learner.updates == 0 and all(r["source_warmup"] for r in records)
    assert all(r["optimizer_updates"] == 0 for r in records)
    assert t.eligible_chunks == 4  # chunks33..36 only, no retroactive warmup debt
    t.finish_episode(receipt(t, 36))
    assert t.learner.updates == 16 and t.learner.optimizer_steps == 48
    assert len(sink.blobs) == 2  # initial plus latest, not one replay blob/decision
    assert t.initial_recovery["retention"] == "immutable"
    t.begin_episode(episode_spec(t.profile, t.lineage, 1))
    metadata = transition(t, 36, terminal=True)
    assert not metadata["source_warmup"] and metadata["optimizer_updates"] == 16
    np.testing.assert_array_equal(t.learner.replay[-1][1], metadata["steering_coefficients"])
    assert metadata["policy_version"].startswith("sac:update:16:")


def test_full_pending_recovery_preserves_rng_replay_and_refuses_fresh_reset():
    t, sink = teacher()
    t.begin_episode(episode_spec(t.profile, t.lineage, 0))
    transition(t, 0)
    t.act(np.zeros(2062, np.float32))
    saved = deepcopy(sink.blobs["latest.pt"])
    clone, _ = teacher()
    clone.load_state_dict(saved, recovery_ref=t.last_recovery)
    assert _state_digest(clone.learner.state_dict()) == _state_digest(t.learner.state_dict())
    np.testing.assert_array_equal(clone.pending["coefficients"], t.pending["coefficients"])
    assert len(clone.learner.replay) == 1
    with pytest.raises(RuntimeError, match="pending"):
        clone.begin_episode(episode_spec(t.profile, t.lineage, 1))
    with pytest.raises(RuntimeError, match="inside a physical"):
        clone.finish_updates()
    np.testing.assert_array_equal(clone.learner.sample_warmup(np.zeros(2062)),
                                  t.learner.sample_warmup(np.zeros(2062)))


def test_partial_optimizer_cycle_is_preserved_and_not_silently_restarted():
    t, sink = teacher()
    t.pending_update_cycles, t.eligible_chunks = 1, 1
    def broken_update(**kwargs):
        raise RuntimeError("simulated partial optimizer failure")
    t.learner.update = broken_update
    with pytest.raises(RuntimeError, match="partial optimizer failure"):
        t.finish_updates()
    saved = sink.blobs["latest.pt"]
    assert saved["updating"] and saved["pending_update_cycles"] == 1
    clone, _ = teacher()
    with pytest.raises(RuntimeError, match="Interrupted partial"):
        clone.load_state_dict(saved, recovery_ref=t.last_recovery)


def test_source_val_cases_are_fixed_disjoint_and_cross_bank_is_24_natural_cases():
    p = profile()
    source, validation = set(), set()
    for lineage in lineage_specs():
        source.update(episode_spec(p, lineage, i)["reset_seed"] for i in range(48))
        before = [episode_spec(p, lineage, 0, phase="source_validation", case=i, milestone=0)
                  for i in range(2)]
        after = [episode_spec(p, lineage, 24, phase="source_validation", case=i, milestone=24)
                 for i in range(2)]
        assert [(s["reset_seed"], s["motor_seed"]) for s in before] == [
            (s["reset_seed"], s["motor_seed"]) for s in after]
        validation.update(s["reset_seed"] for s in before)
    cross = cross_evaluation_specs(p, "one_fixed_context_blind")
    assert len(cross) == 24 and len({s["reset_seed"] for s in cross}) == 4
    assert not source & validation and not (source | validation) & {s["reset_seed"] for s in cross}
    assert all(s["task"] in ("place_phone_stand", "place_object_stand") for s in cross)


def test_same_canonical_ledger_and_whole_episode_stop_precede_reset():
    t, _ = teacher()
    p = t.profile
    _, pair = make_pair()
    distribution = CalibrationDistribution(pair)
    control = SimpleNamespace(active=False, post_command_transform=distribution,
        condition_contract=distribution.contract(), interface_sha256=p.interface_sha256,
        episode_limit_for_task=lambda _: 400, set_coefficients=lambda *args: None,
        mapping_sha256=lambda: p.mapping_sha256)
    budget = SimpleNamespace(header=dict(total_actions=300000, phase_caps=dict(PHASE_CAPS),
        source_caps=p.source_caps(), contract=dict(scope_sha256=p.scope_sha256)), remaining=lambda **kwargs: 399)
    validations = []
    def validate(ref, specs):
        assert ref["retention"] == "immutable"
        validations.append(specs)
        return dict(path="validation.json", sha256="f"*64, complete=True,
                    phase="source_validation", episode_ids=[s["id"] for s in specs])
    def forbidden(*args, **kwargs):
        raise AssertionError("No reset/episode launch without full source headroom")
    result = collect_lineage(t, control=control, budget=budget, run_id="test",
        episode_runner=forbidden, post_command_transform=distribution,
        validate_source=validate, publish_prefix=forbidden)
    assert result["status"] == "shared_action_cap_stop" and t.active_episode is None
    assert len(validations) == 1 and t.validation_done == [0]
    assert t.validation_receipts["0"]["input_checkpoint"]["retention"] == "immutable"
    budget.header["source_caps"] = {}
    with pytest.raises(ValueError, match="ONE canonical"):
        p.validate_budget(budget)


def test_balanced_round_robin_uses_each_actual_independent_lineage(monkeypatch):
    p, order = profile(), []
    teachers = {s.method: SimpleNamespace(completed_episodes=[], lineage=s, profile=p,
                learner=SimpleNamespace(seed=s.learner_seed)) for s in lineage_specs()}
    def collect(t, **kwargs):
        key = next(k for k, item in teachers.items() if item is t)
        order.append(key)
        t.completed_episodes.append({})
        return dict(status="episode_committed")
    monkeypatch.setattr("fluffy_bassoon.meta_rl.steering_condition_collection.collect_lineage", collect)
    result = collect_round_robin(teachers)
    assert order == list(SOURCE_METHOD_KEYS)*24
    assert result["completed"] == dict.fromkeys(SOURCE_METHOD_KEYS, 24)
    teachers[SOURCE_METHOD_KEYS[1]].learner = teachers[SOURCE_METHOD_KEYS[0]].learner
    with pytest.raises(ValueError, match="six independent"):
        collect_round_robin(teachers)


def test_dpt_cannot_use_an_algebraic_but_unqualified_inverse():
    p = profile()
    policy = SimpleNamespace(action_limit=1., act=lambda obs: np.zeros(4, np.float32))
    with pytest.raises(ValueError, match="actual source-only"):
        QualifiedCoefficientTeacher(policy, qualification=dict(
            status="algebraic_inverse_only", condition_id="calibration_minus"),
            profile=p, condition_id="calibration_minus")


def test_dpt_qualified_fixed_policy_is_independent_of_flat_behavior_and_rejects_wrong_mapping():
    p = profile()
    coefficients = np.asarray([.4, 0., 0., 0.], np.float32)
    policy = SimpleNamespace(action_limit=1., coefficients=coefficients)
    reference = dict(path="actual-source-proof.json", sha256="a"*64)
    q = dict(status="physically_qualified_source_coefficient_teacher",
        mapping_sha256=p.mapping_sha256, motor_sha256=p.motor_sha256,
        condition_id="calibration_minus", source_only=True, new_target_data_used=False,
        coefficient_dim=4, coefficient_bound=1., same_mapping=True,
        qualification_receipt=reference, policy_checkpoint=reference,
        validation_case_manifest=reference, cost_receipt=reference,
        label_sampling_mode="fixed_coefficients", policy_sha256=_state_digest(coefficients))
    qualified = QualifiedCoefficientTeacher(policy, qualification=q, profile=p,
                                            condition_id="calibration_minus")
    policy.coefficients[:] = 0  # Labeler owns an independent frozen copy.
    np.testing.assert_array_equal(qualified.act(np.zeros(2062)), np.asarray([.4, 0., 0., 0.], np.float32))
    with pytest.raises(ValueError, match="actual source-only"):
        QualifiedCoefficientTeacher(policy, qualification=dict(q, mapping_sha256="f"*64),
                                    profile=p, condition_id="calibration_minus")


def test_actual_recovery_adapter_bounds_slots_and_protects_immutable_refs(tmp_path):
    from fluffy_bassoon.meta_rl.vla_history import file_sha256
    t, _ = teacher()
    sink = BoundedTeacherRecovery(tmp_path/"teacher", guard=lambda: None)
    initial = sink("initial", t.state_dict(), immutable=True)
    initial_sha = file_sha256(initial["path"])
    for i in range(5):
        latest = sink(f"pending-{i}", t.state_dict(), immutable=False)
    assert len(list((tmp_path/"teacher").glob("*.pt"))) == 3
    assert file_sha256(initial["path"]) == initial_sha
    loaded = sink.load(latest)
    clone, _ = teacher()
    clone.load_state_dict(loaded["collector"], recovery_ref=latest)
    assert _state_digest(clone.learner.state_dict()) == _state_digest(t.learner.state_dict())
    with pytest.raises(ValueError, match="bytes differ"):
        sink.load(dict(latest, sha256="0"*64))
