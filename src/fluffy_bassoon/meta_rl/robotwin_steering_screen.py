"""Bounded A1 source-only restricted search on the exact frozen Motus motor.

Resource admission, scheduler submission and terminal GPU-cost settlement are
owned by the existing resource owner. This driver never schedules a job.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
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
import xml.etree.ElementTree as ET

import numpy as np
import torch

from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter, _atomic_json, _fsync_directory
from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv, load_benchmark_episode_limits, RLINF_COMMIT, ROBOTWIN_COMMIT
from fluffy_bassoon.rl.robotwin_steering import ARMS, TASKS, CoefficientSteering, validate_interface, normalize_arm
from fluffy_bassoon.rl.pi05_residual import ReactiveKVResidual, velocity_with_kv_residual


FORMAT = "frozen_motus_a1_restricted_search_v1"
SEED = 20261031
CANDIDATES, REPEATS = 3, 2
PHASE_CAPS = dict(a1=30000, source=120000, source_validation=30000,
                  condition_cross=15000, student_evaluation=105000)
SOURCE_METHODS = {f"sac_calibration_{condition}_{partition}{seed}"
    for condition in ("minus", "plus") for partition, seed in (("train", 0), ("train", 1), ("validation", 2))}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bound(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Input SHA mismatch: {path}")
    return json.loads(Path(path).read_text())


def build_plan(limits):
    if set(limits) != set(TASKS) or any(type(v) is not int or v < 1 for v in limits.values()):
        raise ValueError("Require both task horizons from the verified native source")
    discovery = 8 * sum(limits.values())
    # Each failure case: 3 arms * (3 candidates + 2 winner confirmations),
    # plus TWO independent, matching native confirmation episodes =17.
    per_case = (len(ARMS)*(CANDIDATES+REPEATS)+REPEATS) * max(limits.values())
    maximum_cases = min(3, (30000-discovery)//per_case)
    if maximum_cases < 1:
        raise ValueError("This native-horizon grid cannot fit a complete search case within30000")
    dev = [dict(id=f"dev-{task}-{case:02d}", task=task, case=case,
                reset_seed=310310000+task_index*10000+997*case,
                motor_seed=SEED+1000*task_index+case, arm="native", coefficients=[0.]*4,
                stage="development", horizon=limits[task])
           for task_index, task in enumerate(TASKS) for case in range(8)]
    return dict(format=FORMAT, seed=SEED, tasks=list(TASKS), limits=dict(limits), development=dev,
                candidate_count=CANDIDATES, confirmation_repeats=REPEATS, maximum_failure_cases=maximum_cases,
                discovery_maximum_actions=discovery, maximum_actions_per_failure_case=per_case,
                worst_case_actions=discovery+maximum_cases*per_case, stage_action_ceiling=30000,
                reset_policy_actions=0, qualification_episode="reuses the first planned complete native dev episode",
                task_selection="first task in fixed task order with a native dev failure",
                case_selection="first at most maximum_failure_cases failed cases in fixed index order",
                candidate_distribution="independent fixed uniform[-1,1]^4; three candidates, no further generations",
                unused_capacity="not permission for more cases, candidates or retries", locked_queries_opened=False)


def proposals(task, case, arm):
    return np.random.default_rng(np.random.SeedSequence([SEED, TASKS.index(task), case, ARMS.index(arm)])).uniform(
        -1., 1., (CANDIDATES, 4)).astype(np.float32).tolist()


def next_episode(plan, records):
    """Deterministic cursor derived only from complete development/search rows."""
    for spec in plan["development"]:
        if spec["id"] not in records:
            return deepcopy(spec)
    selected = next((task for task in TASKS if any(not records[s["id"]]["success"]
                    for s in plan["development"] if s["task"] == task)), None)
    if selected is None:
        return None
    cases = [s for s in plan["development"] if s["task"] == selected and not records[s["id"]]["success"]
             ][:plan["maximum_failure_cases"]]
    for case in cases:
        for arm in ARMS:
            for index, coefficients in enumerate(proposals(selected, case["case"], arm)):
                spec = dict(case, id=f"search-{selected}-{case['case']:02d}-{arm}-{index}", arm=arm,
                            coefficients=coefficients, stage="search", expected_reset=records[case["id"]]["reset_identity"])
                if spec["id"] not in records:
                    return spec
    for case in cases:
        winners = {}
        for arm in ARMS:
            rows = [records[f"search-{selected}-{case['case']:02d}-{arm}-{i}"] for i in range(CANDIDATES)]
            best = max(rows, key=lambda r: (r["success"], -r["primitive_steps"] if r["success"] else 0))
            if best["success"]:
                winners[arm] = best["spec"]["coefficients"]
        if not winners:
            continue  # An all-failure search is inconclusive; no invented teacher.
        for repeat in range(REPEATS):
            for arm, coefficients in (("native", [0.]*4), *winners.items()):
                spec = dict(case, id=f"confirm-{selected}-{case['case']:02d}-{arm}-{repeat}", arm=arm,
                            coefficients=coefficients, stage="confirmation",
                            motor_seed=SEED+100000+100*case["case"]+repeat,
                            expected_reset=records[case["id"]]["reset_identity"])
                if spec["id"] not in records:
                    return spec
    return None


def reset_identity(info):
    return {key: info[key] for key in ("actual_seed", "observation_sha256", "features_sha256")}


def restricted_summary(plan, records):
    selected = next((task for task in TASKS if any(not records[s["id"]]["success"]
        for s in plan["development"] if s["task"] == task and s["id"] in records)), None)
    rows = []
    for case in range(8):
        native = [r for r in records.values() if r["spec"]["stage"] == "confirmation"
                  and r["spec"]["case"] == case and r["spec"]["arm"] == "native"]
        for arm in ARMS:
            searched = [r for r in records.values() if r["spec"]["stage"] == "search"
                        and r["spec"]["case"] == case and r["spec"]["arm"] == arm]
            if not searched: continue
            confirmation = [r for r in records.values() if r["spec"]["stage"] == "confirmation"
                            and r["spec"]["case"] == case and r["spec"]["arm"] == arm]
            rows.append(dict(case=case, arm=arm, search_successes=sum(r["success"] for r in searched),
                searched_episodes=len(searched), confirmation_successes=sum(r["success"] for r in confirmation),
                confirmation_episodes=len(confirmation), native_confirmation_successes=sum(r["success"] for r in native),
                native_confirmation_episodes=len(native),
                repeated_recovery=(len(confirmation) == len(native) == REPEATS
                    and all(r["success"] for r in confirmation) and not any(r["success"] for r in native))))
    return dict(selected_source_task=selected, development_case_results=rows,
                complete_episodes=len(records), actual_actions=sum(r["primitive_steps"] for r in records.values()),
                query_cases_opened=False, automatically_admitted_teacher=False,
                meaning="Restricted development reachability; no universal ceiling, held-out efficacy or adaptation claim")


def run_complete_episode(control, spec, budget, *, run_id, phase="a1", choose=None,
                         on_chunk=None, on_episode=None, before_step=None, controller=None):
    """Common scalar loop for A1 and A2; no learner update inside this loop.

    Admission reserves space for the FULL native episode, then journals every
    chunk before execution. No action-budget or artificial episode cut is used.
    on_chunk receives public current/next features, actual requested4D action,
    observed reward, term/trunc flags and duration; hidden actuator audit is not
    supplied as student features. on_episode runs only after a complete episode.
    """
    spec = dict(spec, arm=normalize_arm(spec["arm"]))
    horizon = control.episode_limit_for_task(spec["task"])
    if isinstance(control, CoefficientSteering) and control.env.corpus_writer is None:
        raise RuntimeError("The common real episode loop requires its canonical raw corpus writer")
    method = spec.get("method", spec["arm"])
    if budget.snapshot()["unresolved_decisions"]:
        raise RuntimeError("A repeated/uncertain physical intent blocks new episode execution")
    if horizon != spec["horizon"] or budget.remaining(phase=phase, method=method) < horizon:
        raise RuntimeError("Insufficient full native-episode admission; no reset or action issued")
    control.set_recording_context(dict(phase=phase, method=method, trial_id=spec["id"],
                                      policy_version=spec.get("policy_version", spec["id"])))
    control.set_coefficients(spec["arm"], spec["coefficients"])
    features = control.reset(spec["task"], spec["reset_seed"], motor_seed=spec["motor_seed"])
    if control.last_reset_info.get("reset_policy_actions") != 0:
        raise RuntimeError("Unreserved reset motor commands are not admitted")
    identity = reset_identity(control.last_reset_info)
    if spec.get("expected_reset", identity) != identity:
        raise RuntimeError("Actual seed/raw observation/public features differ from the paired development reset")
    if before_step is not None:
        before_step(control, spec)
    started, chunks, steps, success, corpus_refs = time.monotonic(), 0, 0, False, []
    while control.active:
        if choose is not None and controller is not None:
            raise ValueError("Choose either a controller or a coefficient callback")
        coefficients = np.asarray(controller.act(features.copy()) if controller is not None else
            (spec["coefficients"] if choose is None else choose(features.copy())), np.float32)
        requested = min(control.native_horizon, horizon-control.elapsed_steps)
        identity_key = f"{run_id}:{spec['id']}:{chunks}"
        acting = {} if controller is None else controller.decision_metadata()
        policy_version = acting.get("policy_version", spec.get("policy_version", spec["id"]))
        reservation = budget.reserve(identity_key, requested, phase=phase, method=method,
            task=spec["task"], episode=spec["id"], chunk=chunks,
            policy_version=policy_version)
        if not reservation.execute or reservation.granted != requested:
            raise RuntimeError("A repeated/uncertain/short physical intent must not execute")
        control.set_recording_context(dict(policy_version=policy_version))
        control.record_decision_metadata(dict(acting, campaign_decision_id=identity_key,
            campaign_reserved_actions=requested, behavior_version=policy_version,
            steering_lineage_id=acting.get("steering_lineage_id", "a1-development"),
            steering_condition_id=acting.get("steering_condition_id", "natural"),
            steering_partition=acting.get("steering_partition", "validation"),
            source_warmup=acting.get("source_warmup", False),
            optimizer_updates=acting.get("optimizer_updates", 0)))
        following, reward, end, trunc, info = control.step(coefficients)
        actual = info["primitive_steps"]
        budget.resolve(identity_key, actual)
        if info.get("budget_truncated", False):
            raise RuntimeError("Unexpected external episode cut")
        corpus_ref = None
        if info.get("corpus_chunk") is not None:
            root = control.env.corpus_writer.root
            path = root/info["corpus_chunk"]
            corpus_ref = dict(root=str(root), path=info["corpus_chunk"], sha256=sha(path))
            corpus_refs.append(corpus_ref)
        if controller is not None and actual > 0:
            controller.observe(features.copy(), control.coefficients.copy(), reward, following.copy(),
                               end, trunc, actual, corpus_ref)
        if on_chunk is not None and actual > 0:
            on_chunk(dict(observations=features.copy(), next_observations=following.copy(),
                proposed_coefficients=coefficients.copy(), executed_coefficients=control.coefficients.copy(),
                reward=reward, terminated=end, truncated=trunc, duration=actual, chunk=chunks))
        features = following
        steps += actual
        chunks += 1
        success |= bool(info["success"])
    record = dict(spec=deepcopy(spec), success=success, primitive_steps=steps, chunks=chunks,
                  reset_identity=identity, reset_info=deepcopy(control.last_reset_info),
                  wall_seconds=time.monotonic()-started, complete=True, control_recovery=control.snapshot())
    record["corpus_refs"] = corpus_refs
    record["episode_manifest_ref"] = None
    writer = control.env.corpus_writer
    if writer is not None:
        relative = Path("episodes")/control.last_reset_info["corpus_episode_id"]/"manifest.json"
        path = writer.root/relative
        manifest = json.loads(path.read_text())
        if (manifest["pending_decision"] is not None or manifest["completed_chunks"] != chunks
                or manifest["executed_primitive_steps"] != steps or len(manifest["chunks"]) != chunks):
            raise RuntimeError("Completed episode and canonical manifest counters disagree")
        last = json.loads((path.parent/manifest["chunks"][-1]["path"]).read_text())
        if (last["status"] != "complete" or not (last["learning_result"]["terminated"]
                or last["learning_result"]["truncated"])):
            raise RuntimeError("The actual final raw chunk does not close the episode")
        record["episode_manifest_ref"] = dict(root=str(writer.root), path=str(relative), sha256=sha(path))
    if on_episode is not None:
        on_episode(record)
    return record


def qualify_real_motor(control, spec, output):
    """First actual observation/noise: decoded zero parity and frozen suffix gradient."""
    backend, model = control.env.backend, control.env.backend.model
    before = backend.frozen_fingerprint()
    control.set_coefficients("native", np.zeros(4, np.float32))
    rng = np.random.default_rng(spec["motor_seed"])
    noise = rng.standard_normal((32, 32)).astype(np.float32)
    native_parity = control.env.check_noise_parity(noise)
    reference = backend.decode(noise)
    cache_copy = tuple(tuple(t.detach().clone() for t in pair) for pair in backend._kv_cache)
    key = backend._kv_cache[-1][0]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(SEED)
        kv = ReactiveKVResidual(public_dim=2062, kv_heads=key.shape[-2], head_dim=key.shape[-1], rank=4).to(model.device).eval()
    try:
        backend.kv_adapter, backend.action_residual = kv, None
        kv_actions = backend.decode(noise)
        backend.kv_adapter, backend.action_residual = control.kv, None
        fixed_kv_actions = backend.decode(noise)
        backend.kv_adapter, backend.action_residual = None, control.action
        action_actions = backend.decode(noise)
        if (not np.array_equal(reference, kv_actions) or not np.array_equal(reference, fixed_kv_actions)
                or not np.array_equal(reference, action_actions)):
            raise RuntimeError("Actual decoded zero-KV/zero-action commands differ from native")
        noisy = torch.from_numpy(noise)[None].to(model.device)
        with torch.enable_grad():
            velocity = velocity_with_kv_residual(model, backend._prepared, noisy,
                torch.full((1,), .5, device=model.device), backend._kv_cache, backend._prefix_mask,
                backend._public_features, kv)
            loss = velocity.float().square().mean()
            gradients = torch.autograd.grad(loss, tuple(kv.output_bases.parameters()))
        norm = sum(float(g.detach().float().square().sum()) for g in gradients)**.5
        if not math.isfinite(norm) or norm <= 0 or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Actual frozen suffix did not isolate a nonzero KV adapter gradient")
        if any(not torch.equal(t, old) for pair, old_pair in zip(backend._kv_cache, cache_copy) for t, old in zip(pair, old_pair)):
            raise RuntimeError("Qualification mutated native cache tensors")
        after = backend.frozen_fingerprint()
        if before != after:
            raise RuntimeError("Frozen motor hash changed during qualification")
        result = dict(status="decoded_and_gradient_pass_episode_pending", native_parity=native_parity,
            zero_kv_exact=True, zero_fixed_kv_exact=True, zero_action_exact=True, kv_gradient_norm=norm, motor_gradients=0,
            frozen_before=before, frozen_after=after, cache_shapes=[list(pair[0].shape) for pair in backend._kv_cache],
            qualification_episode=spec["id"], actual_actions=0)
        _atomic_json(output/"gpu-qualification.json", result, replace=True)
        return result
    finally:
        control.set_coefficients(spec["arm"], spec["coefficients"])


def validate_admission(scope, allocation, *, scope_sha, source, output, hostname, interface):
    if scope.get("schema") != "frozen_pi05_endtoend_scope_v1":
        raise ValueError("Wrong shared phase scope")
    limits = scope["phase_limits"]
    if (limits["cumulative_gpu_seconds"] != 28800 or limits["all_in_simulator_actions"] != 300000
            or limits["outputs_gib"] != 64 or limits["fresh_cpu_qualification_worker_seconds"] != 1800):
        raise ValueError("Shared cumulative phase caps changed")
    if (scope["first_stage_limits_within_phase"]["gpu_seconds"] != 7200
            or scope["first_stage_limits_within_phase"]["all_in_simulator_actions"] != 30000):
        raise ValueError("First stage limits changed")
    if (allocation.get("scope_sha256") != scope_sha or allocation.get("source_commit") != source
            or allocation.get("hostname") != hostname or allocation.get("gpu_count") != 1
            or Path(allocation["output"]).resolve() != output.resolve()):
        raise ValueError("Actual allocation ticket does not bind scope, source, host, GPU and fresh output")
    values = [allocation[k] for k in ("reserved_gpu_seconds", "prior_phase_gpu_seconds",
              "other_reserved_phase_gpu_seconds", "prior_a1_gpu_seconds")]
    if any(type(v) is not int or v < 0 for v in values) or values[0] < 1:
        raise ValueError("Invalid cumulative allocation costs")
    if sum(values[:3]) > 28800 or values[0]+values[3] > 7200:
        raise ValueError("Allocation exceeds shared8h or cumulative first-stage2h cap")
    ledger = allocation["action_ledger"]
    if (ledger["total_actions"] != 300000 or ledger["phase_caps"] != PHASE_CAPS
            or set(ledger["source_caps"]) != SOURCE_METHODS
            or any(type(v) is not int or v < 1 for v in ledger["source_caps"].values())):
        raise ValueError("Wrong shared action ledger; no child allowance or old-ledger borrowing")
    if (ledger["campaign_id"] != scope["phase_id"] or ledger["contract"].get("scope_sha256") != scope_sha
            or ledger["contract"].get("phase_id") != scope["phase_id"]
            or ledger["contract"].get("motor_sha256") != interface["motor_sha256"]
            or ledger["contract"].get("candidate_tasks") != list(TASKS)):
        raise ValueError("Canonical ledger is not bound to this fresh shared phase")
    if not output.resolve().is_relative_to(Path(allocation["phase_output_root"]).resolve()):
        raise ValueError("Output is outside the shared64GiB root")


def verify_source_conventions(receipt, interface, roots):
    if (receipt.get("arm_joint_units") != "radians" or receipt.get("gripper_indices") != [6, 13]
            or receipt.get("gripper_mode") != "absolute" or receipt.get("maximum_arm_delta", 0) < interface["arm_joint_bound"]
            or receipt.get("joint_lower") != interface["joint_lower"] or receipt.get("joint_upper") != interface["joint_upper"]
            or receipt.get("rlinf_commit") != RLINF_COMMIT or receipt.get("robotwin_commit") != ROBOTWIN_COMMIT
            or receipt.get("reset_policy_action_bound") != 0
            or not receipt.get("source_files")):
        raise ValueError("Missing reviewed source-only joint/gripper convention and delta-bound evidence")
    for row in receipt["source_files"]:
        root = Path(roots[row["root"]]).resolve()
        path = (root/row["path"]).resolve()
        if not path.is_relative_to(root) or sha(path) != row["sha256"]:
            raise ValueError("Source convention evidence differs from deployed source")
    records = receipt.get("joint_limit_records", [])
    arms = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12}
    if len(records) != 12 or {r["action_index"] for r in records} != arms:
        raise ValueError("Require12 exact decoded arm-index to source joint-limit bindings")
    for row in records:
        root = Path(roots[row["root"]]).resolve()
        path = (root/row["path"]).resolve()
        if not path.is_relative_to(root) or sha(path) != row["sha256"]:
            raise ValueError("Joint-limit source SHA or containment mismatch")
        matches = [joint for joint in ET.parse(path).getroot().iter("joint") if joint.get("name") == row["joint"]]
        if len(matches) != 1 or matches[0].get("type") != "revolute" or matches[0].find("limit") is None:
            raise ValueError("Decoded arm range must bind a unique limited revolute source joint")
        limit, index = matches[0].find("limit"), row["action_index"]
        actual = [float(limit.get("lower", "nan")), float(limit.get("upper", "nan"))]
        expected = [interface["joint_lower"][index], interface["joint_upper"][index]]
        if any(not math.isfinite(v) for v in actual) or not np.allclose(actual, expected, rtol=0, atol=1e-7):
            raise ValueError("Declared decoded joint ranges differ from the bound source URDF")


def save_recovery(path, state, control=None):
    data = dict(format=FORMAT, state=deepcopy(state), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        control=None if control is None else control.snapshot(), optimizer=None,
        interpretation="fixed random search; no learner/optimizer updates; physics resumes only at closed episodes")
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as target:
        torch.save(data, target); target.flush(); os.fsync(target.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scope", "allocation", "interface", "joint-conventions"):
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
        raise RuntimeError("Actual allocated Linux only; do not run motors or tests on macOS")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Require the exact clean deployed commit")
    scope = read_bound(args.scope, args.scope_sha256)
    allocation = read_bound(args.allocation, args.allocation_sha256)
    interface = validate_interface(read_bound(args.interface, args.interface_sha256))
    validate_admission(scope, allocation, scope_sha=args.scope_sha256, source=source,
                       output=args.output, hostname=socket.gethostname(), interface=interface)
    if (not math.isfinite(allocation["driver_deadline_unix"]) or allocation["driver_deadline_unix"]-time.time() < 180):
        raise RuntimeError("Allocation's original clock cannot admit startup and recovery")
    probe = json.loads(args.gpu_probe.read_text())
    if probe.get("status") != "passed" or probe.get("hostname") != socket.gethostname() or len(probe.get("devices", [])) != 1:
        raise ValueError("Require actual one-GPU allocation CUDA probe")
    if sha(args.model_path/"model.safetensors") != interface["motor_sha256"] or sha(args.norm_stats) != interface["normalization_sha256"]:
        raise ValueError("Motor or normalization differs from exact interface binding")
    if interface["joint_convention_sha256"] != args.joint_conventions_sha256:
        raise ValueError("Interface did not bind the reviewed convention receipt")
    verify_source_conventions(read_bound(args.joint_conventions, args.joint_conventions_sha256), interface,
                              dict(robotwin=args.robotwin_root, rlinf=args.rlinf_root))
    limits = load_benchmark_episode_limits(args.robotwin_root, TASKS)
    plan = build_plan(limits)
    if any(v > 48*max(limits.values()) for v in allocation["action_ledger"]["source_caps"].values()):
        raise ValueError("A lineage cap exceeds48 complete native episodes; reset has zero policy actions")
    args.output.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.output/"plan.json", dict(plan, source_commit=source, scope_sha256=args.scope_sha256,
        interface_sha256=args.interface_sha256, allocation_sha256=args.allocation_sha256,
        episode_limit_file_sha256=sha(args.robotwin_root/"task_config/_eval_step_limit.yml")))
    state = dict(status="running", plan=plan, records={}, pending=None, source_commit=source,
                 scope_sha256=args.scope_sha256, interface_sha256=args.interface_sha256, prior_outputs=[],
                 action_ledger=allocation["action_ledger"], allocation_sha256=args.allocation_sha256)
    if args.resume:
        if not args.resume_sha256 or sha(args.resume) != args.resume_sha256:
            raise ValueError("Resume requires an exact closed-boundary recovery hash")
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        prior = saved["state"]
        if (saved["format"] != FORMAT or prior["status"] != "boundary_stop" or prior["pending"] is not None
                or prior["scope_sha256"] != args.scope_sha256 or prior["interface_sha256"] != args.interface_sha256
                or prior["source_commit"] != source or prior["plan"] != plan):
            raise ValueError("Only an exact same-code closed operational boundary may resume; diagnose real failures separately")
        state = deepcopy(prior)
        state.update(status="running", prior_outputs=prior["prior_outputs"]+[str(args.resume.parent)])
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
    if not args.resume:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    ledger = allocation["action_ledger"]
    control = None
    qualification = None
    maximum_episode_seconds = float(allocation.get("initial_episode_seconds", 180))
    if not math.isfinite(maximum_episode_seconds) or maximum_episode_seconds <= 0:
        raise ValueError("Require a reviewed complete-episode wall bound")

    def storage_guard():
        root = Path(allocation["phase_output_root"])
        used = 0
        for directory, dirs, files in os.walk(root, followlinks=False):
            if any((Path(directory)/name).is_symlink() for name in (*dirs, *files)):
                raise RuntimeError("Shared output accounting rejects symlinks")
            used += sum((Path(directory)/name).stat().st_size for name in files)
        if used + (64 << 20) > 64 << 30:
            raise RuntimeError("Shared64GiB output allowance cannot admit another native chunk")
        return used

    try:
        with CampaignBudget(ledger["root"], campaign_id=ledger["campaign_id"], total_actions=300000,
                source_caps=ledger["source_caps"], phase_caps=ledger["phase_caps"], contract=ledger["contract"],
                run_id=args.run_id, source_commit=source) as budget:
            if budget.snapshot()["unresolved_decisions"]:
                raise RuntimeError("Canonical physical ledger has unresolved charges; no model/reset/retry admitted")
            storage_guard()
            writer = RobotwinCorpusWriter(args.output/"corpus", dict(schema=FORMAT, source_commit=source,
                motor_sha256=interface["motor_sha256"], interface_sha256=args.interface_sha256, role="source_A1_development_only"))
            env = RobotwinNoiseEnv(model_path=args.model_path, norm_stats=args.norm_stats,
                robotwin_root=args.robotwin_root, rlinf_root=args.rlinf_root, output_dir=args.output/"runtime",
                action_horizon=32, num_steps=5, task_episode_limits=limits)
            control = CoefficientSteering(env, interface, interface_sha256=args.interface_sha256)
            control.attach_corpus(writer)
            if args.resume:
                random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
                torch.set_rng_state(saved["torch_rng"])
                if saved["cuda_rng"] is None or len(saved["cuda_rng"]) != torch.cuda.device_count():
                    raise ValueError("Complete GPU RNG recovery is absent or device count differs")
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
                recovered = saved["control"]
                for name, actual in (("noise_basis", control.noise_basis), ("physical_basis", control.physical_basis())):
                    if not np.array_equal(recovered[name], actual):
                        raise ValueError("Fixed search basis changed across recovery")
                if not torch.equal(recovered["kv_basis"], control.kv.basis.cpu()):
                    raise ValueError("Fixed KV basis changed across recovery")
                control.kv.to(env.backend.model.device)
            _atomic_json(args.output/"mapping.json", control.config())
            frozen = env.frozen_fingerprint()
            state["frozen_motor_sha256"] = frozen
            if args.resume and prior["frozen_motor_sha256"] != frozen:
                raise RuntimeError("Resumed frozen motor differs")

            def before_step(current, spec):
                nonlocal qualification
                if qualification is None:
                    qualification = qualify_real_motor(current, spec, args.output)
                storage_guard()
                if time.time() >= allocation["driver_deadline_unix"]-30:
                    raise RuntimeError("Original allocation deadline reached inside an admitted episode")

            while (spec := next_episode(plan, state["records"])) is not None:
                if stop[0] or allocation["driver_deadline_unix"]-time.time() <= maximum_episode_seconds+60:
                    state["status"] = "boundary_stop"
                    break
                storage_guard()
                state["pending"] = spec
                save_recovery(args.output/"recovery.pt", state, control)
                record = run_complete_episode(control, spec, budget, run_id=args.run_id,
                    before_step=before_step, on_chunk=lambda row: storage_guard())
                maximum_episode_seconds = max(maximum_episode_seconds, 1.5*record["wall_seconds"])
                state["maximum_episode_seconds"] = maximum_episode_seconds
                # Store a small immutable episode summary; tensors/RNG remain in recovery/raw corpus.
                small = {k: v for k, v in record.items() if k != "control_recovery"}
                _atomic_json(args.output/(spec["id"]+".json"), small)
                state["records"][spec["id"]] = small
                state["pending"] = None
                state["actions"] = budget.snapshot()
                if qualification["status"] != "passed":
                    qualification.update(status="passed", actual_actions=record["primitive_steps"],
                                         complete_real_episode=spec["id"])
                    _atomic_json(args.output/"gpu-qualification.json", qualification, replace=True)
                save_recovery(args.output/"recovery.pt", state, control)
            else:
                state["status"] = "complete_restricted_development_search"
            if env.frozen_fingerprint() != frozen:
                raise RuntimeError("Frozen motor hash changed")
            state["actions"] = budget.snapshot()
            state["restricted_summary"] = restricted_summary(plan, state["records"])
            state["storage_bytes"] = storage_guard()
            state["interpretation"] = "A1 reachability only; selected maxima are biased, no held-out/query, RL or ICRL claim"
    except BaseException as error:
        state.update(status="blocked_failure", failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        try:
            if control is not None:
                control.close()
        except BaseException as error:
            state.update(status="blocked_cleanup_failure", cleanup_error=str(error))
            raise
        finally:
            save_recovery(args.output/"recovery.pt", state, control)
            _atomic_json(args.output/"result.json", state, replace=True)


if __name__ == "__main__":
    main()
