"""Allocated source orchestration over the existing scalar steering collector.

No scheduler, second learner, simulator loop, corpus schema or ledger is added.
Selection/qualification and resource tickets are supplied by the sole owner.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from .steering_condition_collection import (
    BoundedTeacherRecovery, CoefficientTeacher, PHASE_CAPS, SourceProfile,
    collect_round_robin, lineage_specs, make_learner,
)
from .robotwin_steering_screen import (
    read_bound, restricted_summary, run_complete_episode, sha, verify_source_conventions,
)
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter, _atomic_json
from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv, load_benchmark_episode_limits
from fluffy_bassoon.rl.robotwin_steering import CoefficientSteering, TASKS, validate_interface
from fluffy_bassoon.rl.steering_actuator_condition import CalibrationDistribution, calibration_pair


FORMAT = "frozen_motus_source_orchestration_v1"
SELECTION_RULE_SHA256 = "87c8af53bcea7ee09ea55f11380b4a310d94e6ff2fc01e6f7b648b39299086b5"


class BoundaryStop(RuntimeError):
    """No next whole episode fits; completed data and actual charges remain."""


def validate_selection(selection, qualification, a1_result, profile, interface, *, interface_sha, selection_rule):
    """Verify the owner's natural-A1 selection; do not invent a ranking rule."""
    if (selection.get("schema") != "frozen_motus_source_selection_v1"
            or selection.get("status") != "selected_from_completed_A1"
            or selection.get("selection_basis") != "natural_A1_predeclared_criteria"
            or selection.get("condition_outcomes_used") is not False
            or selection.get("task") != profile.task or selection.get("arm") != profile.arm
            or selection.get("mapping_sha256") != profile.mapping_sha256
            or selection.get("interface_sha256") != interface_sha
            or profile.interface_sha256 != interface_sha
            or selection.get("motor_sha256") != profile.motor_sha256
            or profile.motor_sha256 != interface["motor_sha256"]
            or selection.get("episode_limit") != profile.episode_limit
            or selection.get("task_horizon_sha256") != profile.task_horizon_sha256):
        raise ValueError("Source selection differs from natural A1, selected task, mapping or motor FILE identity")
    if (qualification.get("status") != "passed" or not qualification.get("zero_kv_exact")
            or not qualification.get("zero_fixed_kv_exact") or not qualification.get("zero_action_exact")
            or qualification.get("frozen_before") != qualification.get("frozen_after")
            or not qualification.get("frozen_after") or qualification.get("actual_actions", 0) < 1
            or qualification.get("motor_gradients") != 0
            or a1_result.get("status") != "complete_restricted_development_search"
            or a1_result.get("frozen_motor_sha256") != qualification["frozen_after"]
            or a1_result.get("interface_sha256") != interface_sha):
        raise ValueError("Need actual completed A1 motor/episode qualification and unchanged frozen tensor identity")
    qualified = a1_result.get("records", {}).get(qualification.get("qualification_episode"), {})
    if (qualification.get("native_parity", {}).get("passed") is not True
            or not qualification.get("kv_gradient_norm", 0) > 0
            or qualification.get("complete_real_episode") != qualification.get("qualification_episode")
            or qualified.get("complete") is not True
            or qualified.get("primitive_steps") != qualification["actual_actions"]):
        raise ValueError("Actual A1 parity/gradient/complete real episode receipt is absent")
    summary = restricted_summary(a1_result["plan"], a1_result["records"])
    if (summary != a1_result.get("restricted_summary") or summary["selected_source_task"] != profile.task
            or selection.get("selection_rule", {}).get("sha256") != SELECTION_RULE_SHA256
            or selection_rule.get("format") != "frozen_motus_A1_operational_selection_v1"
            or selection_rule.get("tie_order") != ["noise", "kv", "action"]):
        raise ValueError("Bind the pre-outcome operational rule and original A1 summary/selected source task")
    counts = {arm: sum(row["repeated_recovery"] is True for row in summary["development_case_results"]
                       if row["arm"] == arm) for arm in ("noise", "kv", "action")}
    best = max(counts, key=counts.get)  # Dict order is the fixed predeclared tie break.
    if counts[best] < 1 or profile.arm != best:
        raise ValueError("No eligible selected arm: need greatest strict confirmed-rescue count, fixed ties noise/kv/action")
    ids = selection.get("evidence_episode_ids", [])
    records = a1_result.get("records", {})
    if (not ids or len(ids) != len(set(ids)) or any(identity not in records for identity in ids)
            or any(records[i].get("complete") is not True
                or records[i].get("primitive_steps", 0) < 1
                or records[i].get("spec", {}).get("task") != profile.task
                or records[i].get("spec", {}).get("arm") not in (profile.arm, "native") for i in ids)):
        raise ValueError("Selection must cite actual complete selected-task/interface A1 episodes")
    for row in summary["development_case_results"]:
        if row["arm"] != best or row["repeated_recovery"] is not True:
            continue
        confirms = [r for r in records.values() if r["spec"]["stage"] == "confirmation"
                    and r["spec"]["task"] == profile.task and r["spec"]["case"] == row["case"]
                    and r["spec"]["arm"] in (best, "native")]
        if len(confirms) != 4 or not {r["spec"]["id"] for r in confirms} <= set(ids):
            raise ValueError("Cite all actual 2/2 winner confirmations and 0/2 matched native episodes")
        motors = {r["spec"]["motor_seed"] for r in confirms}
        if len(motors) != 2:
            raise ValueError("Repeated rescue requires two independent paired motor draws")
        for seed in motors:
            pair = [r for r in confirms if r["spec"]["motor_seed"] == seed]
            if len(pair) != 2 or pair[0]["reset_identity"] != pair[1]["reset_identity"]:
                raise ValueError("Actual accepted initial public observations must match within each paired confirmation")


def validate_source_admission(scope, allocation, profile, *, source, hostname, output, profile_sha):
    limits = dict(cumulative_gpu_seconds=28800, all_in_simulator_actions=300000,
                  outputs_gib=64, fresh_cpu_qualification_worker_seconds=1800)
    if (scope.get("schema") != "frozen_pi05_endtoend_scope_v1"
            or any(scope.get("phase_limits", {}).get(k) != v for k, v in limits.items())):
        raise ValueError("Shared 8h/300k/64GiB/1800CPU pilot limits differ")
    if (allocation.get("schema") != "frozen_motus_source_allocation_v1"
            or allocation.get("scope_sha256") != profile.scope_sha256
            or allocation.get("profile_sha256") != profile_sha
            or allocation.get("source_commit") != source or allocation.get("hostname") != hostname
            or allocation.get("gpu_count") != 1 or not allocation.get("job_id")
            or Path(allocation["output"]).resolve() != Path(output).resolve()
            or not Path(output).resolve().is_relative_to(Path(allocation["phase_output_root"]).resolve())):
        raise ValueError("Actual owner allocation must bind exact source/profile/host/GPU/fresh shared output")
    costs = [allocation[k] for k in ("reserved_gpu_seconds", "prior_phase_gpu_seconds",
                                    "other_reserved_phase_gpu_seconds")]
    if any(type(v) is not int or v < 0 for v in costs) or costs[0] < 1 or sum(costs) > 28800:
        raise ValueError("No fresh child GPU allowance or forecast refund")
    if (not math.isfinite(allocation["driver_deadline_unix"])
            or allocation["driver_deadline_unix"] > allocation["allocation_started_unix"]+costs[0]
            or allocation["cleanup_reserve_seconds"] < 60
            or any(not math.isfinite(allocation[k]) for k in
                ("allocation_started_unix", "maximum_episode_seconds", "maximum_episode_update_seconds"))
            or allocation["maximum_episode_seconds"] <= 0 or allocation["maximum_episode_update_seconds"] < 0):
        raise ValueError("Finite original clock, whole-episode/update bounds and cleanup reserve required")
    ledger = allocation["action_ledger"]
    profile.validate_budget(type("BoundHeader", (), {"header": ledger})())
    if (ledger["campaign_id"] != scope["phase_id"]
            or ledger["contract"].get("phase_id") != scope["phase_id"]
            or ledger["contract"].get("motor_sha256") != profile.motor_sha256
            or not (Path(ledger["root"])/"events/0000000000.json").is_file()):
        raise ValueError("Reuse the already-created canonical phase ledger, not a new source ledger")
    # Ten versions per lineage cover seven immutable initial/milestone/prefix/
    # final versions and two rolling slots plus one in-progress replacement.
    full = allocation["max_full_recovery_bytes"]
    reserved = allocation["recovery_output_reservation_bytes"]
    if (type(full) is not int or full < 1 or type(reserved) is not int or reserved < 60*full
            or reserved+allocation["other_future_output_reservation_bytes"] > 64 << 30):
        raise ValueError("Admit actual recovery-size/retention plus other phase output within64GiB")


class RuntimeGuard:
    def __init__(self, allocation):
        self.allocation, self.stopping = allocation, False
        self.maximum_episode_seconds = float(allocation["maximum_episode_seconds"])

    def storage(self):
        used = 0
        for directory, dirs, files in os.walk(self.allocation["phase_output_root"], followlinks=False):
            paths = [Path(directory)/name for name in (*dirs, *files)]
            if any(p.is_symlink() for p in paths):
                raise RuntimeError("Canonical shared output rejects symlinks")
            used += sum((Path(directory)/name).stat().st_size for name in files)
        if used+self.allocation["max_full_recovery_bytes"] > 64 << 30:
            raise RuntimeError("Shared output cannot retain the next full recovery safely")
        return used

    def __call__(self):
        self.storage()
        if time.time() >= self.allocation["driver_deadline_unix"]-10:
            raise RuntimeError("Original source allocation clock exhausted; latest recoveries preserved")

    def admit_episode(self):
        self()
        required = (self.maximum_episode_seconds+self.allocation["maximum_episode_update_seconds"]
                    +self.allocation["cleanup_reserve_seconds"])
        if self.stopping or self.allocation["driver_deadline_unix"]-time.time() <= required:
            raise BoundaryStop("No next complete episode/update/recovery fits the finite remaining clock")


class FrozenSourceBehavior:
    """Validation-only restored behavior; no source observe/update or random NN warmup."""
    def __init__(self, checkpoint, profile, lineage, spec, *, archive_actor, guard, device="cuda"):
        self.learner = make_learner(profile, lineage, device=device)
        self.learner.load_state_dict(checkpoint["learner"], require_replay=True)
        self.profile, self.lineage, self.archive_actor, self.guard = profile, lineage, archive_actor, guard
        # Fixed private innovations pair each actual validation case across
        # milestones; do not consume or overwrite the source actor RNG.
        seed = (spec["reset_seed"]+3000017) % 2**32
        self.learner.rng = np.random.default_rng(seed)
        self.learner.torch_rng.manual_seed(seed)
        self.metadata = None

    def act(self, features):
        self.guard()
        warmup = self.learner.updates == 0
        before_numpy = deepcopy(self.learner.rng.bit_generator.state)
        before_torch = self.learner.torch_rng.get_state().cpu().numpy().copy()
        action = (self.learner.sample_warmup(features) if warmup else
                  self.learner.act(features, deterministic=False))
        acting = (dict(policy_version="bounded_uniform_warmup", policy_checkpoint=None,
            policy_sha256=None, policy_checkpoint_sha256=None) if warmup else self.archive_actor(self.learner))
        self.metadata = dict(acting, steering_coefficients=action.copy(),
            proposed_steering_coefficients=action.copy(), source_warmup=warmup,
            optimizer_updates=self.learner.updates, steering_lineage_id=self.lineage.method,
            steering_condition_id=self.lineage.condition_id, steering_partition=self.lineage.partition,
            steering_mapping_sha256=self.profile.mapping_sha256,
            steering_interface_sha256=self.profile.interface_sha256,
            interaction_numpy_rng=before_numpy, interaction_torch_rng=before_torch)
        return action

    def decision_metadata(self):
        return deepcopy(self.metadata)

    def observe(self, *args, **kwargs):
        # The common runner has committed the raw transition/ledger. Validation
        # never adds replay, changes source counters or runs optimizer steps.
        pass


def existing_actor_archive(output):
    """Load the existing archive class; do not copy or alter the old launcher."""
    path = Path(__file__).resolve().parents[3]/"scripts/diagnostics/robotwin_noise_compare.py"
    spec = importlib.util.spec_from_file_location("_source_existing_behavior_archive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BehaviorPolicyArchive(output)


def read_reference(reference):
    return read_bound(Path(reference["path"]), reference["sha256"])


def restore_global_rng(saved):
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"].cpu())
    if torch.cuda.is_initialized():
        states = saved["cuda_rng"]
        if states is None or len(states) != torch.cuda.device_count():
            raise ValueError("Closed source resume requires all actual allocation CUDA RNG states")
        torch.cuda.set_rng_state_all(states)
    elif saved["cuda_rng"] is not None:
        raise ValueError("Cannot silently discard captured CUDA RNG state")


def paired_validation_spec(spec, baseline):
    """Use the common paired-reset API; never rewrite text/features locally."""
    first = baseline.get("spec", {})
    instruction = baseline.get("reset_info", {}).get("instruction")
    identity = baseline.get("reset_identity", {})
    if (baseline.get("complete") is not True or first.get("milestone") != 0
            or first.get("phase") != "source_validation"
            or any(first.get(k) != spec.get(k) for k in
                ("task", "case", "method", "reset_seed", "motor_seed"))
            or not isinstance(instruction, list) or len(instruction) != 1
            or not isinstance(instruction[0], str) or not instruction[0].strip()
            or set(identity) != {"actual_seed", "observation_sha256", "features_sha256"}):
        raise ValueError("Fixed validation must bind its actual complete baseline case and original instruction")
    return dict(deepcopy(spec), expected_reset=deepcopy(identity), paired_instruction=deepcopy(instruction))


def run_source_collection(*, control, budget, profile, distribution, output, guard,
                          archive, run_id, student_profiles, control_qualification,
                          source_commit, device="cuda", prior=None, prefix_builder=None, after_source=None):
    """Wire six teachers, fixed physical validation and immutable prefix receipts."""
    output = Path(output)
    teachers, published, publications = {}, [], {}
    balanced, publishing = set(), [False]
    global_sink = BoundedTeacherRecovery(output/"orchestration-recovery", guard=guard, control=control)
    if prefix_builder is None:
        from .steering_history_prefix import build_prefix
        prefix_builder = build_prefix

    def publication(teacher, final_ref):
        return dict(lineage=asdict(teacher.lineage), condition_id=teacher.lineage.condition_id,
            mapping_sha256=profile.mapping_sha256, interface_sha256=profile.interface_sha256,
            complete_episode_manifests=[r["episode_manifest_ref"] for r in teacher.completed_episodes],
            actual_resets=[r["reset_info"] for r in teacher.completed_episodes],
            initial_full_state=teacher.initial_recovery, final_full_state=final_ref,
            actual_optimizer_updates=teacher.learner.updates,
            actual_primitive_steps=teacher.learner.primitive_steps,
            source_validation_receipts=deepcopy(teacher.validation_receipts))

    def publish_balanced():
        if publishing[0] or len(teachers) != 6:
            return
        counts = {len(t.completed_episodes) for t in teachers.values()}
        if len(counts) != 1:
            return
        count = counts.pop()
        if (count not in (12, 24) or count in balanced
                or any(t.active_episode is not None or t.pending is not None or t.updating
                    or t.pending_update_cycles or count not in t.validation_done for t in teachers.values())):
            return
        publishing[0] = True
        try:
            cohort = {}
            for key, teacher in teachers.items():
                cached = publications.get((key, count))
                if cached is None:  # VAL behavior is also a complete held-out lineage.
                    ref = teacher._persist("balanced_prefix_full_state", immutable=True)
                    cached = publication(teacher, ref)
                cohort[key] = cached
            path = output/"prefixes"/f"balanced-{count:03d}.json"
            prefix_builder(student_profiles[count], cohort,
                control_qualification=control_qualification, output=path,
                expected_hostname=socket.gethostname())
            published.append(dict(path=str(path), sha256=sha(path),
                objective_admission="pending actual separate learning/teacher qualification"))
            balanced.add(count)
        finally:
            publishing[0] = False

    def teacher_guard():
        guard()
        publish_balanced()
    def identify(learner):
        guard()
        return archive.identify(learner, "sac")

    for lineage in lineage_specs(profile.learner_seed):
        sink = BoundedTeacherRecovery(output/"recoveries"/lineage.method, guard=guard, control=control)
        def checkpoint(event, payload, *, immutable, sink=sink):
            ref = sink(event, payload, immutable=immutable)
            maximum = getattr(guard, "allocation", {}).get("max_full_recovery_bytes")
            if maximum is not None and ref["size_bytes"] > maximum:
                raise RuntimeError("Actual full state exceeds admitted recovery-size/retention estimate")
            return ref
        learner = make_learner(profile, lineage, device=device)
        teacher = CoefficientTeacher(learner, profile, lineage, checkpoint=checkpoint,
                                     archive_actor=identify, guard=teacher_guard)
        if prior is not None:
            ref = prior["teachers"][lineage.method]
            teacher.load_state_dict(BoundedTeacherRecovery.load(ref)["collector"], recovery_ref=ref)
            if teacher.active_episode is not None or teacher.pending is not None:
                raise ValueError("Only closed physical boundaries resume; preserve partial work for owner reconciliation")
        teachers[lineage.method] = teacher

    def episode(current, spec, ledger, **kwargs):
        try:
            guard.admit_episode()
        except BoundaryStop:
            # collect_lineage saved episode_begin but no reset or command has
            # happened. Explicitly clear only this provably unstarted episode.
            teacher = kwargs.get("controller")
            if isinstance(teacher, CoefficientTeacher):
                if teacher.decision_count or teacher.pending is not None:
                    raise RuntimeError("Cannot clear a started physical prefix")
                teacher.active_episode = None
                teacher._persist("unstarted_boundary_stop", immutable=True)
            raise
        extra_before, extra_chunk = kwargs.pop("before_step", None), kwargs.pop("on_chunk", None)
        def before(*args):
            guard()
            if extra_before is not None:
                extra_before(*args)
        def chunk(row):
            guard()
            if extra_chunk is not None:
                extra_chunk(row)
        record = run_complete_episode(current, spec, ledger, **kwargs, on_chunk=chunk, before_step=before)
        guard.maximum_episode_seconds = max(guard.maximum_episode_seconds, 1.5*record["wall_seconds"])
        compact = {k: v for k, v in record.items() if k != "control_recovery"}
        _atomic_json(output/"episodes"/(spec["id"]+".json"), compact)
        return record

    def validate(checkpoint_ref, specs):
        saved = BoundedTeacherRecovery.load(checkpoint_ref)["collector"]
        lineage = teachers[specs[0]["method"]].lineage
        baseline = None
        if specs[0]["milestone"] != 0:
            baseline_ref = teachers[lineage.method].validation_receipts.get("0")
            if baseline_ref is None:
                raise ValueError("Later validation cannot sample a new unpaired baseline instruction")
            baseline = read_reference(baseline_ref)["results"]
        records = []
        for index, spec in enumerate(specs):
            if baseline is not None:
                spec = paired_validation_spec(spec, baseline[index])
            if control.active:
                raise RuntimeError("Condition cannot switch during a physical episode")
            distribution.select(lineage.sign)
            frozen = FrozenSourceBehavior(saved, profile, lineage, spec,
                archive_actor=identify, guard=guard, device=device)
            records.append(episode(control, spec, budget, run_id=run_id,
                                   phase="source_validation", controller=frozen))
        path = output/"validation"/(specs[0]["id"].rsplit(":", 1)[0]+".json")
        _atomic_json(path, dict(complete=True, phase="source_validation",
            episode_ids=[s["id"] for s in specs], source_checkpoint=checkpoint_ref,
            behavior="bounded uniform iff completed optimizer updates0; otherwise frozen sampled actual actor",
            results=[{k: v for k, v in r.items() if k != "control_recovery"} for r in records]))
        return dict(path=str(path), sha256=sha(path), complete=True, phase="source_validation",
                    episode_ids=[s["id"] for s in specs])

    def prefix(record):
        count = len(record["complete_episode_manifests"])
        path = output/"prefixes"/f"{record['lineage']['method']}-{count:03d}.json"
        _atomic_json(path, dict(record, source_profile=profile.contract(), role="actual_source_prefix_gate_pending"))
        publications[(record["lineage"]["method"], count)] = deepcopy(record)

    status, error, cross_result = "running", None, None
    post_entered, post_completed = False, False
    try:
        result = collect_round_robin(teachers, control=control, budget=budget, run_id=run_id,
            episode_runner=episode, post_command_transform=distribution,
            validate_source=validate, publish_prefix=prefix, episode_target=24)
        status = result["status"]
        publish_balanced()
        if status == "initial_prefix_complete" and after_source is not None:
            post_entered = True
            cross_result = after_source(control=control, budget=budget, profile=profile, distribution=distribution,
                teachers=teachers, run_id=run_id, output=output, guard=guard, archive=archive, episode_runner=episode)
            post_completed = True
    except BoundaryStop as exc:
        status, error = "boundary_stop", str(exc)
    except BaseException as exc:
        status, error = "blocked_failure_preserved", repr(exc)
        raise
    finally:
        references = {}
        for key, teacher in teachers.items():
            try:
                references[key] = (teacher.last_recovery if teacher.last_recovery is not None
                    and teacher.last_recovery.get("retention") == "immutable" else
                    teacher._persist("orchestration_final_preserved", immutable=True))
            except BaseException:
                references[key] = teacher.last_recovery
        closed = (not control.active and all(t.active_episode is None and t.pending is None and not t.updating
                      for t in teachers.values()) and not budget.snapshot()["unresolved_decisions"])
        state = dict(format=FORMAT, status=status, error=error, source_commit=source_commit,
            profile=profile.contract(), teachers=references, closed_physical_boundary=closed,
            complete_episodes={k: len(t.completed_episodes) for k, t in teachers.items()},
            actual_optimizer_cycles={k: t.learner.updates for k, t in teachers.items()},
            actual_optimizer_steps={k: t.learner.optimizer_steps for k, t in teachers.items()},
            immutable_prefixes=published, actions=budget.snapshot(),
            post_source_entered=post_entered, post_source_completed=post_completed,
            condition_cross_result=cross_result,
            maximum_episode_seconds=guard.maximum_episode_seconds,
            scientific_claim="collection only; AD improvement and DPT competence require separate measured gates")
        try:
            state["global_recovery"] = global_sink("orchestration_boundary", state, immutable=True)
        finally:
            _atomic_json(output/"source-state.json", state)
    return status


def run_source_condition_cross(*, control, budget, profile, distribution, teachers,
                               run_id, output, guard, archive, episode_runner):
    """Only bind actual source controllers/receipts to the owner's existing A2 helper."""
    from .steering_condition_cross import run_condition_cross
    from .steering_condition_collection import SOURCE_METHOD_KEYS

    keys = SOURCE_METHOD_KEYS[2:4]
    final, scores, resets, validation_refs = {}, {}, [], []
    for key in keys:
        teacher = teachers[key]
        if teacher.learner.updates < 1 or teacher.last_recovery.get("retention") != "immutable":
            path = Path(output)/"condition-cross-not-admitted.json"
            _atomic_json(path, dict(status="not_admitted", reason="No completed trained source actor; do not substitute untrained NN behavior",
                                   new_physical_actions=0, source_method=key))
            return dict(path=str(path), sha256=sha(path), criterion_pass=False, status="not_admitted")
        descriptor = archive.identify(teacher.learner, "sac")
        final[key] = dict(teacher.last_recovery, policy_sha256=descriptor["policy_sha256"],
            optimizer_updates=teacher.learner.updates, source_method=key, sampling_mode="sampled_gaussian")
        ref = teacher.validation_receipts["24"]
        panel = read_reference(ref)
        if panel.get("complete") is not True or len(panel.get("results", [])) != 2:
            raise ValueError("Blind selection needs both actual final source-validation outcomes")
        scores[key] = sum(r["success"] for r in panel["results"])
        validation_refs.append(ref)
    for teacher in teachers.values():
        resets.extend(r["reset_identity"] for r in teacher.completed_episodes)
        for ref in teacher.validation_receipts.values():
            resets.extend(r["reset_identity"] for r in read_reference(ref)["results"])
    selected = max(keys, key=lambda key: scores[key])  # Fixed minusTRAIN1-first tie.
    alias = "source_validation_selected_blind"
    path = Path(output)/"condition-cross-blind-selection.json"
    _atomic_json(path, dict(selected_method=alias, selected_source_method=selected,
        selected_from="source_validation", cross_cases_opened=False, controller_ref=final[selected],
        final_validation_successes=scores, source_validation_refs=validation_refs))
    controllers = {key: teachers[key] for key in keys}
    controllers[alias] = teachers[selected]
    final[alias] = deepcopy(final[selected])
    sink = BoundedTeacherRecovery(Path(output)/"condition-cross-recovery", guard=guard, control=control)
    return run_condition_cross(control, budget, profile, distribution, controllers,
        run_id=run_id, output=Path(output)/"condition-cross", guard=guard,
        controller_refs=final, context_blind_method=alias,
        blind_selection_ref=dict(path=str(path), sha256=sha(path)), source_reset_identities=resets,
        checkpoint=sink, episode_runner=episode_runner)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scope", "allocation", "interface", "joint-conventions", "selection", "profile"):
        p.add_argument(f"--{name}", type=Path, required=True)
        p.add_argument(f"--{name}-sha256", required=True)
    for name in ("output", "model-path", "norm-stats", "rlinf-root", "robotwin-root", "gpu-probe"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-sha256")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux":
        raise RuntimeError("Tests/models/simulators require the actual allocated Linux runtime")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Require exact clean owner-deployed Git commit")
    scope, allocation = read_bound(args.scope, args.scope_sha256), read_bound(args.allocation, args.allocation_sha256)
    interface = validate_interface(read_bound(args.interface, args.interface_sha256))
    profile = SourceProfile(**read_bound(args.profile, args.profile_sha256))
    selection = read_bound(args.selection, args.selection_sha256)
    qualification = read_reference(selection["qualification_receipt"])
    a1_result = read_reference(selection["a1_result"])
    if (profile.scope_sha256 != args.scope_sha256 or profile.source_selection_sha256 != args.selection_sha256):
        raise ValueError("Profile did not bind exact shared scope and completed A1 selection")
    validate_selection(selection, qualification, a1_result, profile, interface,
        interface_sha=args.interface_sha256, selection_rule=read_reference(selection["selection_rule"]))
    validate_source_admission(scope, allocation, profile, source=source, hostname=socket.gethostname(),
                              output=args.output, profile_sha=args.profile_sha256)
    cpu = read_reference(allocation["CPU_qualification_receipt"])
    if cpu.get("status") != "passed" or cpu.get("source_commit") != source:
        raise ValueError("Need actual allocated CPU qualification of this exact entry-point commit")
    student_profiles = {int(k): read_reference(ref) for k, ref in allocation["student_profiles"].items()}
    if (set(student_profiles) != {12, 24} or any(p.get("prefix_episodes") != n
            or p.get("source_task") != profile.task or p.get("scope_sha256") != profile.scope_sha256
            or p.get("interface", {}).get("mapping_sha256") != profile.mapping_sha256
            or p.get("interface", {}).get("control_interface_sha256") != profile.interface_sha256
            for n, p in student_profiles.items())):
        raise ValueError("Bind the two actual student prefix profiles before source collection")
    control_qualification = allocation["control_qualification"]
    read_reference(control_qualification)
    # Resolve integrated software before model loading, not halfway through a
    # paid source allocation. These are the existing owners' components.
    from .steering_history_prefix import build_prefix
    if allocation.get("run_condition_cross", False):
        from .steering_condition_cross import run_condition_cross
    probe = json.loads(args.gpu_probe.read_text())
    if (probe.get("status") != "passed" or probe.get("hostname") != socket.gethostname()
            or len(probe.get("devices", [])) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("Require actual exclusive one-GPU allocation/CUDA proof")
    if sha(args.model_path/"model.safetensors") != profile.motor_sha256 or sha(args.norm_stats) != interface["normalization_sha256"]:
        raise ValueError("Motor FILE or normalization changed")
    if interface["joint_convention_sha256"] != args.joint_conventions_sha256:
        raise ValueError("Interface lacks actual source joint convention binding")
    verify_source_conventions(read_bound(args.joint_conventions, args.joint_conventions_sha256), interface,
                              dict(robotwin=args.robotwin_root, rlinf=args.rlinf_root))
    limits = load_benchmark_episode_limits(args.robotwin_root, TASKS)
    if limits[profile.task] != profile.episode_limit or sha(args.robotwin_root/"task_config/_eval_step_limit.yml") != profile.task_horizon_sha256:
        raise ValueError("Actual native horizon differs from the selected profile")
    prior = None
    if args.resume:
        prior = read_bound(args.resume, args.resume_sha256)
        if (prior.get("format") != FORMAT or prior.get("status") != "boundary_stop"
                or prior.get("source_commit") != source or prior.get("closed_physical_boundary") is not True
                or prior.get("profile") != profile.contract()):
            raise ValueError("Resume only exact closed source boundaries; failures require owner reconciliation")
        if prior.get("post_source_entered") and not prior.get("post_source_completed"):
            raise ValueError("Preserve the existing A2 cursor/recovery; no automatic restarted cross panel")
        captured = BoundedTeacherRecovery.load(prior["global_recovery"])
        if captured["collector"].get("teachers") != prior["teachers"]:
            raise ValueError("Captured global recovery does not bind this exact six-teacher boundary")
    guard = RuntimeGuard(allocation)
    guard.admit_episode()
    args.output.mkdir(parents=True, exist_ok=False)
    if os.statvfs(args.output).f_bavail*os.statvfs(args.output).f_frsize < 16 << 30:
        raise RuntimeError("Source setup needs actual16GiB free and its bounded shared64GiB reservation")
    probe_path = args.output/".write-probe"
    with probe_path.open("xb") as target:
        target.write(b"source-write-fsync-probe\n"); target.flush(); os.fsync(target.fileno())
    probe_path.unlink()
    _atomic_json(args.output/"admission.json", dict(source_commit=source, scope_sha256=args.scope_sha256,
        profile_sha256=args.profile_sha256, allocation_sha256=args.allocation_sha256,
        selection_sha256=args.selection_sha256, gpu_probe_sha256=sha(args.gpu_probe)))
    signal.signal(signal.SIGTERM, lambda *_: setattr(guard, "stopping", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(guard, "stopping", True))
    ledger, control, env = allocation["action_ledger"], None, None
    try:
        with CampaignBudget(ledger["root"], campaign_id=ledger["campaign_id"], total_actions=300000,
                source_caps=ledger["source_caps"], phase_caps=ledger["phase_caps"], contract=ledger["contract"],
                run_id=args.run_id, source_commit=source) as budget:
            profile.validate_budget(budget)
            if budget.snapshot()["unresolved_decisions"]:
                raise ValueError("Unresolved physical charges prohibit new model/reset execution")
            env = RobotwinNoiseEnv(model_path=args.model_path, norm_stats=args.norm_stats,
                robotwin_root=args.robotwin_root, rlinf_root=args.rlinf_root,
                output_dir=args.output/"runtime", action_horizon=32, num_steps=5, task_episode_limits=limits)
            # Basis is fixed by the same interface seed as A1; establish its
            # identity before recording a static both-sign condition contract.
            control = CoefficientSteering(env, interface, interface_sha256=args.interface_sha256)
            control.set_coefficients(profile.arm, np.zeros(4, np.float32))
            if control.mapping_sha256() != profile.mapping_sha256:
                raise ValueError("Actual selected fixed mapping differs from A1")
            pair = calibration_pair(physical_basis=control.physical_basis(),
                physical_basis_sha256=control.config()["action"]["basis_sha256"],
                arm_joint_bound=interface["arm_joint_bound"], selected_arm=profile.arm,
                command_lower=interface["joint_lower"], command_upper=interface["joint_upper"],
                range_receipt_sha256=args.joint_conventions_sha256, motor_sha256=profile.motor_sha256,
                mapping_sha256=profile.mapping_sha256)
            distribution = CalibrationDistribution(pair)
            # Install exactly once before corpus registration/first reset.
            control.condition_contract = distribution.contract()
            control.post_command_transform = env.backend.post_command_transform = distribution
            env.backend.contract = dict(env.backend.contract, coefficient_steering=control.config())
            env.contract = env.backend.contract
            writer = RobotwinCorpusWriter(args.output/"corpus", dict(schema=FORMAT, source_commit=source,
                motor_sha256=profile.motor_sha256, interface_sha256=args.interface_sha256,
                profile_sha256=args.profile_sha256, role="source_condition_changing_policy_and_fixed_validation"))
            control.attach_corpus(writer)
            if env.frozen_fingerprint() != qualification["frozen_after"]:
                raise ValueError("Loaded frozen tensors differ from completed A1 qualification")
            archive = existing_actor_archive(args.output/"actors")
            if prior is not None:
                restore_global_rng(captured)
            run_source_collection(control=control, budget=budget, profile=profile,
                distribution=distribution, output=args.output, guard=guard, archive=archive,
                run_id=args.run_id, prior=prior, source_commit=source, student_profiles=student_profiles,
                control_qualification=control_qualification, prefix_builder=build_prefix,
                after_source=run_source_condition_cross if allocation.get("run_condition_cross", False) else None)
            if env.frozen_fingerprint() != qualification["frozen_after"]:
                raise RuntimeError("Frozen motor changed during source collection")
    finally:
        closed, close_error = False, None
        try:
            if control is not None:
                control.close()
            elif env is not None:
                env.close()
            closed = True
        except BaseException as exc:
            close_error = repr(exc)
            raise
        finally:
            _atomic_json(args.output/"cleanup.json", dict(control_closed=closed, cleanup_error=close_error,
                actual_GPU_cost="owner scheduler closure required; reservation is not consumption"))


if __name__ == "__main__":
    main()
