"""Frozen24-episode controlled A2 cross, called by the admitted source CLI.

No allocation, simulator loop, corpus, ledger or learner is created here.
Source/condition collection remains owned by steering_condition_collection.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .robotwin_history_corpus import parameter_digest
from .robotwin_steering_screen import run_complete_episode, reset_identity, paired_reset_fields
from .vla_noise_sac import NoiseSAC
from .steering_condition_collection import (
    SOURCE_METHOD_KEYS, QualifiedCoefficientTeacher, _state_digest, cross_evaluation_specs,
)
from fluffy_bassoon.rl.robotwin_corpus import _atomic_json
from fluffy_bassoon.rl.steering_actuator_condition import (
    CalibrationDistribution, contract_sha256, verify_same_initial_public_observation,
)


TRAIN1 = SOURCE_METHOD_KEYS[2:4]
CONDITIONS = ("calibration_minus", "calibration_plus")
PAIRED_GAP = .25
AGGREGATE_BLIND_GAP = .125
ACTOR_SEED = 1930000000
SCENES = 4
FORMAT = "source_only_frozen_condition_cross_v1"


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _artifact(reference):
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("Require a committed artifact path and SHA")
    digest = reference.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Require an actual artifact SHA256")
    path = Path(reference["path"])
    if not path.is_absolute():
        if "root" not in reference:
            raise ValueError("Relative artifact paths need an explicit canonical root")
        root = Path(reference["root"]).resolve()
        path = (root/path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact escapes its canonical root")
    actual = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            actual.update(block)
    if actual.hexdigest() != digest:
        raise ValueError("Committed artifact bytes changed")
    return path


def _policy(teacher, profile):
    """Use an existing source policy read-only; never call a collecting teacher."""
    if isinstance(teacher, QualifiedCoefficientTeacher):
        q = teacher.qualification
        if (q.get("mapping_sha256") != profile.mapping_sha256 or q.get("motor_sha256") != profile.motor_sha256
                or q.get("status") != "physically_qualified_source_coefficient_teacher"
                or q.get("same_mapping") is not True or q.get("source_only") is not True
                or q.get("new_target_data_used") is not False):
            raise ValueError("Constant/qualified controller differs from the physically qualified source mapping")
        if teacher.fixed is not None:
            raise ValueError("The declared cross uses actual TRAIN1 SAC controllers; no oracle relabeling")
        if q.get("label_sampling_mode") != "sampled_gaussian":
            raise ValueError("Use the declared stochastic source-validation/teacher sampling mode")
        teacher = teacher.policy
    policy = getattr(teacher, "learner", teacher)
    if (not isinstance(policy, NoiseSAC)
            or getattr(policy, "observation_dim", None) != 2062 or getattr(policy, "action_dim", None) != 4
            or getattr(policy, "action_limit", None) != 1. or not hasattr(policy, "actor")
            or not callable(getattr(policy, "state_dict", None))):
        raise ValueError("Use the actual existing source TRAIN1 SAC policy")
    return policy


def _frozen_identity(policy):
    return dict(full_state_sha256=_state_digest(policy.state_dict(include_replay=True)),
                policy_sha256=parameter_digest(policy.actor.state_dict()), optimizer_updates=int(policy.updates))


class _FrozenEpisodeController:
    """Same source actor distribution, separate paired RNG; observe never learns."""
    def __init__(self, policy, method, reference, condition_id, guard, seed):
        self.policy, self.method, self.reference = policy, method, reference
        self.condition_id, self.guard = condition_id, guard
        self.seed, self.last_rng = seed, None
        self.rng = torch.Generator(device=policy.device).manual_seed(seed)

    def act(self, observation):
        self.guard()
        value = np.asarray(observation, np.float32)
        if value.shape != (2062,) or not np.isfinite(value).all():
            raise ValueError("Frozen cross controllers receive native2062 public features only")
        before = self.rng.get_state().cpu().clone()
        with torch.no_grad():
            result, _ = self.policy.actor(torch.as_tensor(value[None], device=self.policy.device),
                                          generator=self.rng, deterministic=False)
        result = result[0].cpu().numpy()
        self.last_rng = dict(before=before, after=self.rng.get_state().cpu().clone())
        result = np.asarray(result, np.float32)
        if result.shape != (4,) or not np.isfinite(result).all() or np.any(np.abs(result) > 1):
            raise ValueError("Frozen controller left the declared coefficient support")
        return result.copy()

    def decision_metadata(self):
        return dict(policy_version=f"frozen-cross:{self.method}:{self.reference['policy_sha256']}",
            steering_lineage_id=self.method, steering_condition_id=self.condition_id,
            steering_partition="validation", optimizer_updates=self.reference["optimizer_updates"],
            source_warmup=False, frozen_controller_ref=deepcopy(self.reference), deployment_sgd=False,
            cross_actor_sampling_mode=self.reference["sampling_mode"], cross_actor_innovation_seed=self.seed,
            cross_private_actor_rng=deepcopy(self.last_rng))

    def observe(self, *transition):
        pass  # No replay, observation counter, optimizer or teacher update.

    def snapshot(self):
        return dict(seed=self.seed, sampling_mode=self.reference["sampling_mode"],
                    private_rng=self.rng.get_state().cpu().clone())


def summarize_cross(records, context_blind_method):
    if len(records) != 24 or len({r["spec"]["id"] for r in records}) != 24:
        raise ValueError("Summaries require the exact COMPLETE24-episode grid")
    cells = {(r["spec"]["condition_id"], r["spec"]["case"], r["spec"]["method"]): r for r in records}
    expected = {(condition, case, method) for condition in CONDITIONS for case in range(SCENES)
                for method in (*TRAIN1, context_blind_method)}
    if set(cells) != expected or not all(r["complete"] for r in records):
        raise ValueError("Missing/duplicate cells or partial episodes cannot qualify the cross")
    directions = {}
    for index, condition in enumerate(CONDITIONS):
        matched, mismatched = TRAIN1[index], TRAIN1[1-index]
        rows = {role: [cells[condition, case, method] for case in range(SCENES)]
                for role, method in (("matched", matched), ("mismatched", mismatched), ("blind", context_blind_method))}
        counts = {role: sum(bool(r["success"]) for r in values) for role, values in rows.items()}
        paired = {}
        for comparator in ("mismatched", "blind"):
            wins = sum(m["success"] and not c["success"] for m, c in zip(rows["matched"], rows[comparator]))
            losses = sum(not m["success"] and c["success"] for m, c in zip(rows["matched"], rows[comparator]))
            paired[comparator] = dict(wins=int(wins), losses=int(losses), ties=SCENES-int(wins)-int(losses),
                                      success_gap=(counts["matched"]-counts[comparator])/SCENES)
        directions[condition] = dict(successes=counts, episodes_per_controller=SCENES, paired=paired,
            actual_actions={role: sum(r["primitive_steps"] for r in values) for role, values in rows.items()},
            matched_mismatch_pass=paired["mismatched"]["success_gap"] >= PAIRED_GAP)
    matched = sum(row["successes"]["matched"] for row in directions.values())
    blind = sum(row["successes"]["blind"] for row in directions.values())
    aggregate_gap = (matched-blind)/(2*SCENES)
    both_directions = all(row["matched_mismatch_pass"] for row in directions.values())
    return dict(complete=True, predeclared_gap=PAIRED_GAP, directions=directions,
        both_directions_pass=both_directions,
        aggregate_blind=dict(matched_successes=matched, blind_successes=blind, paired_cases=2*SCENES,
                             success_gap=aggregate_gap, predeclared_gap=AGGREGATE_BLIND_GAP),
        criterion_pass=both_directions and aggregate_gap >= AGGREGATE_BLIND_GAP,
        scene_transfer=[dict(case=case, outcomes={condition:{method:bool(cells[condition, case, method]["success"])
            for method in (*TRAIN1, context_blind_method)} for condition in CONDITIONS}) for case in range(SCENES)],
        actual_actions=sum(r["primitive_steps"] for r in records), source_sgd=0,
        scope="Controlled four-scene source pilot; privileged matched-controller selection, not a population/natural-A1/ICRL/reward-causal claim")


def run_condition_cross(control, budget, profile, distribution, teachers, *, run_id, output, guard,
                        controller_refs, context_blind_method, blind_selection_ref, source_reset_identities,
                        checkpoint, episode_runner=run_complete_episode, state=None, should_start_episode=None):
    """Source CLI supplies already-admitted runtime, frozen policies and one sink.

    checkpoint is the existing (event,payload,immutable=...) full-recovery sink.
    State can resume a closed boundary only; failures/pending episodes reject.
    The one blind policy is selected on source validation BEFORE this bank.
    should_start_episode is the owner's optional pre-RPC admission predicate;
    only an explicit False at a closed boundary saves a resumable cursor.
    """
    profile.validate_budget(budget)
    methods = (*TRAIN1, context_blind_method)
    if len(set(methods)) != 3 or set(teachers) != set(methods) or set(controller_refs) != set(methods):
        raise ValueError("Bind two fixed TRAIN1 controllers and ONE separately named fixed blind controller")
    if (not isinstance(distribution, CalibrationDistribution) or control.active
            or control.post_command_transform is not distribution
            or contract_sha256(control.condition_contract) != contract_sha256(distribution.contract())
            or control.interface_sha256 != profile.interface_sha256
            or control.episode_limit_for_task(profile.task) != profile.episode_limit or profile.reset_action_bound != 0):
        raise ValueError("Installed source calibration/interface/native horizon or complete-episode boundary differs")
    control.set_coefficients(profile.arm, [0.]*4)
    if control.mapping_sha256() != profile.mapping_sha256 or any(
            s.mapping_sha256 != profile.mapping_sha256 or s.motor_sha256 != profile.motor_sha256
            for s in distribution.specs.values()):
        raise ValueError("Actual selected mapping/motor differs from the fixed source profile")
    policies = {method:_policy(teachers[method], profile) for method in methods}
    frozen = {method:_frozen_identity(policy) for method, policy in policies.items()}
    for method, reference in controller_refs.items():
        _artifact(reference)
        if (reference.get("policy_sha256") != frozen[method]["policy_sha256"]
                or reference.get("optimizer_updates") != frozen[method]["optimizer_updates"]
                or reference.get("sampling_mode") != "sampled_gaussian"
                or (method in TRAIN1 and reference.get("source_method") != method)):
            raise ValueError("Controller ref does not bind actual frozen TRAIN1/selected weights and sampling")
    blind_path = _artifact(blind_selection_ref)
    blind = json.loads(blind_path.read_text())
    if (blind.get("selected_method") != context_blind_method or blind.get("selected_from") != "source_validation"
            or blind.get("cross_cases_opened") is not False
            or blind.get("controller_ref") != controller_refs[context_blind_method] or not blind.get("source_validation_refs")):
        raise ValueError("ONE blind controller must be source-VAL-selected and bound before opening cross cases")
    scores = blind.get("final_validation_successes", {})
    if set(scores) != set(TRAIN1) or any(type(v) is not int or not 0 <= v <= 2 for v in scores.values()):
        raise ValueError("Blind selection requires both actual final TRAIN1 validation success counts out of2")
    selected = max(TRAIN1, key=lambda method:scores[method])  # Stable minus-first tie rule.
    if (blind.get("selected_source_method") != selected
            or controller_refs[context_blind_method] != controller_refs[selected]):
        raise ValueError("The ONE blind must use max final source-VAL successes/2, ties minus TRAIN1 first")
    for reference in blind["source_validation_refs"]:
        _artifact(reference)
    if not source_reset_identities or any(any(k not in row for k in (
            "actual_seed", "observation_sha256", "features_sha256")) for row in source_reset_identities):
        raise ValueError("Bind actual source/validation reset identities to establish fresh cross scenes")
    specs = cross_evaluation_specs(profile, context_blind_method)
    if len(specs) != 24 or len({s["id"] for s in specs}) != 24:
        raise ValueError("The inherited cross plan is not24unique episodes")
    for case in range(SCENES):
        group = [s for s in specs if s["case"] == case]
        if len(group) != 6 or len({s["reset_seed"] for s in group}) != 1 or len({s["motor_seed"] for s in group}) != 1:
            raise ValueError("Each fresh scene must share ONE reset/noise seed across all six cells")
    plan = dict(format=FORMAT, profile=profile.contract(), specs=specs, controller_refs=deepcopy(controller_refs),
        blind_selection_ref=deepcopy(blind_selection_ref), source_reset_identities=deepcopy(source_reset_identities),
        frozen=frozen, paired_gap=PAIRED_GAP, aggregate_blind_gap=AGGREGATE_BLIND_GAP,
        gap_required="matched minus mismatched >=.25 EACH condition; conditional minus fixed blind >=.125 over8cases",
        actor_sampling="same sampled_gaussian source actor, private paired innovations; no source RNG mutation",
        actor_seeds_by_scene=[ACTOR_SEED+case*997 for case in range(SCENES)],
        condition_contract=distribution.contract(), maximum_actions=24*(profile.episode_limit+profile.reset_action_bound),
        source_sgd=0, condition_blind_method=context_blind_method)
    plan_sha = _json_sha(plan)
    current = dict(format=FORMAT, plan_sha256=plan_sha, status="running", records=[], initial_resets={}, pending=None)
    if state is not None:
        if (state.get("format") != FORMAT or state.get("plan_sha256") != plan_sha
                or state.get("status") != "boundary_stop" or state.get("pending") is not None):
            raise ValueError("Only a hash-bound closed boundary can resume; diagnose failed/partial crosses")
        current = deepcopy(state); current["status"] = "running"
    done = len(current["records"])
    if [r["spec"]["id"] for r in current["records"]] != [s["id"] for s in specs[:done]]:
        raise ValueError("Resume records are not the exact complete-grid prefix")
    if set(current["initial_resets"]) != {str(r["spec"]["case"]) for r in current["records"]}:
        raise ValueError("Resume reset bank must bind exactly the completed scene prefix")
    for receipt, spec in zip(current["records"], specs):
        if (receipt.get("complete") is not True or any(receipt["spec"].get(k) != v for k,v in spec.items())
                or not receipt.get("corpus_refs")):
            raise ValueError("Resume must preserve complete original-grid cells and canonical refs")
        _artifact(receipt["episode_manifest_ref"])
        for ref in receipt["corpus_refs"]: _artifact(ref)
        verify_same_initial_public_observation(current["initial_resets"][str(spec["case"])], receipt["reset_info"])
    if budget.snapshot()["unresolved_decisions"] or budget.remaining(phase="condition_cross", method=methods[0]) < (24-done)*(profile.episode_limit+profile.reset_action_bound):
        raise RuntimeError("Shared ledger cannot admit the complete remaining cross; no case opened")
    output = Path(output)
    guard()
    output.mkdir(parents=True, exist_ok=state is not None)
    _atomic_json(output/"plan.json", plan, replace=state is not None)

    def save(event, *, immutable=False):
        return checkpoint(event, deepcopy(current), immutable=immutable)

    def check_frozen():
        if any(_frozen_identity(policies[m]) != frozen[m] for m in methods):
            raise RuntimeError("Frozen source controller/replay/optimizer/RNG changed during the cross")

    save("condition_cross_plan_locked", immutable=True)
    try:
        for index, original in enumerate(specs[done:], start=done):
            guard(); check_frozen()
            if control.active:
                raise RuntimeError("Never switch condition inside an active physical episode")
            if should_start_episode is not None:
                allowed = should_start_episode()
                if type(allowed) is not bool:
                    raise ValueError("Episode admission predicate must return an explicit bool")
                if not allowed:
                    check_frozen()
                    current.update(status="boundary_stop", pending=None, active_actor_rng=None,
                                   canonical_actions=budget.snapshot())
                    recovery = save("condition_cross_boundary_stop", immutable=True)
                    path = output/"boundary-stop.json"
                    _atomic_json(path, current, replace=True)
                    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        status="boundary_stop", complete=False, phase="condition_cross", recovery_ref=recovery,
                        episode_ids=[r["spec"]["id"] for r in current["records"]],
                        actual_actions=sum(r["primitive_steps"] for r in current["records"]))
            spec = deepcopy(original)
            key = str(spec["case"])
            if key in current["initial_resets"]:
                baseline = current["initial_resets"][key]
                spec.update(paired_reset_fields(dict(reset_info=baseline, reset_identity=reset_identity(baseline))))
            distribution.select(-1 if spec["condition_id"] == CONDITIONS[0] else 1)
            actor = _FrozenEpisodeController(policies[spec["method"]], spec["method"],
                controller_refs[spec["method"]], spec["condition_id"], guard, ACTOR_SEED+spec["case"]*997)
            current["pending"] = dict(spec=spec, ordinal=index)
            current["active_actor_rng"] = actor.snapshot()
            save("condition_cross_before_episode")

            def before_step(active_control, active_spec):
                guard()
                actual = deepcopy(active_control.last_reset_info)
                verify_same_initial_public_observation(actual, actual)
                if key in current["initial_resets"]:
                    verify_same_initial_public_observation(current["initial_resets"][key], actual)
                else:
                    forbidden = [*source_reset_identities, *current["initial_resets"].values()]
                    if any(any(actual[k] == prior[k] for k in ("actual_seed", "observation_sha256", "features_sha256"))
                           for prior in forbidden):
                        raise ValueError("A cross scene overlaps source/validation or another actual cross scene")
                    current["initial_resets"][key] = actual
                save("condition_cross_reset_bound")

            receipt = episode_runner(control, spec, budget, run_id=run_id, phase="condition_cross",
                controller=actor, before_step=before_step, on_chunk=lambda _: guard())
            if (receipt.get("complete") is not True or control.active or receipt.get("spec", {}).get("id") != spec["id"]
                    or not receipt.get("episode_manifest_ref") or not receipt.get("corpus_refs")):
                raise RuntimeError("Require the actual complete canonical episode and manifest/chunk refs")
            manifest_path = _artifact(receipt["episode_manifest_ref"])
            manifest = json.loads(manifest_path.read_text())
            if (manifest["pending_decision"] is not None or manifest["completed_chunks"] != receipt["chunks"]
                    or manifest["executed_primitive_steps"] != receipt["primitive_steps"]):
                raise ValueError("Canonical episode counters differ from the completed cross receipt")
            for reference in receipt["corpus_refs"]:
                _artifact(reference)
            verify_same_initial_public_observation(current["initial_resets"][key], receipt["reset_info"])
            check_frozen()
            compact = {k:deepcopy(receipt[k]) for k in ("spec", "success", "primitive_steps", "chunks", "complete",
                       "reset_info", "corpus_refs", "episode_manifest_ref")}
            current["records"].append(compact); current["pending"] = None
            current["active_actor_rng"] = actor.snapshot()
            save("condition_cross_episode_committed")
        _artifact(blind_selection_ref); check_frozen()
        current["active_actor_rng"] = None
        current.update(status="complete", summary=summarize_cross(current["records"], context_blind_method),
                       canonical_actions=budget.snapshot())
        recovery = save("condition_cross_complete", immutable=True)
        _atomic_json(output/"result.json", current)
        return dict(path=str(output/"result.json"), sha256=hashlib.sha256((output/"result.json").read_bytes()).hexdigest(),
            complete=True, phase="condition_cross", episode_ids=[s["id"] for s in specs],
            actual_actions=current["summary"]["actual_actions"], recovery_ref=recovery,
            both_directions_pass=current["summary"]["both_directions_pass"],
            criterion_pass=current["summary"]["criterion_pass"])
    except BaseException as error:
        current.update(status="blocked_failure", failure=dict(type=type(error).__name__, message=str(error)))
        try:
            save("condition_cross_interrupted_preserved", immutable=True)
        except BaseException:
            pass  # The preceding pending snapshot and canonical raw intents remain protected.
        raise
