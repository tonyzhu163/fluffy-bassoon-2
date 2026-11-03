"""Finite one-node frozen-steering PPO with complete-bank recovery.

The scheduler launcher supplies a reviewed immutable profile and eight-device
probe. Persistent workers collect physical histories; only this process updates
the small steering/value network. Every request and SGD bank is retained.
"""
import argparse
from dataclasses import asdict
import json
import fcntl
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .joint_steering_ppo import (
    DEV_ROUNDS, UPDATES, ROLLOUT_LANES, TRAIN_EPISODES, JointActionJournal, boundary_state, case_key, cases_for,
    chronological_records, make_policy, normalize_first_bank, qualify_optimizer_restore, restore_boundary,
    sha_file, spec_arguments, summarize_episodes, write_json, select_scale,
)
from .joint_steering_ppo_worker import _save
from .joint_steering_metrics import MetricPublisher, resource_snapshot
from .ppo import generalized_advantages
from .residual_ppo import stack_records, update
from .rlinf_native_ppo import checked_checkout, cpu_tree, tensor_hash


APPROVED_ARM_GPU_HOURS = 60


def events():
    result = [("calibrate", 0), ("test_before", 0), ("dev", 0)]
    for iteration in range(1, UPDATES + 1):
        result.append(("train", iteration))
        if iteration in DEV_ROUNDS:
            result.append(("dev", iteration))
    return [*result, ("test_after", UPDATES)]


def _same_request(left, right):
    # Request publication is deterministic. A complete bank from a crashed
    # master is consumed once, rather than replayed in the simulator.
    if tensor_hash(left) != tensor_hash(right):
        raise ValueError("Existing request differs from the complete-boundary recovery")


def evaluation_case_root(profile):
    """Both arms and attempt folders must resolve the same new scene bank."""
    explicit = profile.get("evaluation_case_root")
    if explicit is not None:
        root = Path(explicit)
    else:
        group = profile.get("comparison_id") or profile.get("wandb", {}).get("group")
        if not isinstance(group, str) or not group or Path(group).name != group or group in (".", ".."):
            raise ValueError("Paired evaluation requires an explicit shared case root or comparison_id")
        root = Path(profile["canonical_root"]).parents[1] / (group + "-paired-eval-v2")
    if not root.is_absolute() or root == Path(profile["canonical_root"]):
        raise ValueError("Paired evaluation case root must be shared and distinct from the old arm-local bank")
    return str(root)


def dispatch(run, profile, policy, specification, *, sequence, phase, iteration,
             children, deadline):
    cases = cases_for("train", 1)[:8] if phase == "calibrate" else cases_for(phase, iteration)
    state = cpu_tree(policy.state_dict())
    paired_root = evaluation_case_root(profile)
    paired = phase in ("dev", "test_before", "test_after")
    requests = []
    for rank in range(8):
        directory = run / "workers" / f"rank-{rank:02d}"
        request = dict(sequence=sequence, phase=phase, iteration=iteration,
            policy_iteration=iteration - 1 if phase == "train" else iteration,
            cases=cases[rank::8], policy_config=asdict(policy.config), policy=state,
            spec=specification, lanes=profile["lanes_per_worker"], canonical_root=profile["canonical_root"],
            evaluation_protocol="fixed_batches_shared_scenes_v2", evaluation_case_root=paired_root,
            batch_mode="fixed_waves_v1" if paired else "async", strict_pairing=paired,
            **{key: profile[key] for key in ("model_directory", "norm_path", "source_commit", "runtime_commit")},
            deterministic=phase != "train", zero_steering=phase == "test_before")
        if not request["cases"]:
            raise ValueError("Every admitted device must have a finite nonempty case bank")
        path = directory / f"request-{sequence:06d}.pt"
        if path.exists():
            _same_request(torch.load(path, map_location="cpu", weights_only=False), request)
        else:
            _save(path, request)
        requests.append(path)
    results = {}
    while len(results) < 8:
        for rank, request_path in enumerate(requests):
            if rank in results:
                continue
            directory = request_path.parent
            error = directory / "workererror.json"
            if error.exists():
                raise RuntimeError(f"Worker {rank} failed; diagnose {error} before any new attempt")
            path = directory / f"result-{sequence:06d}.pt"
            if path.exists():
                result = torch.load(path, map_location="cpu", weights_only=False)
                if (not result["completed"] or result["rank"] != rank or result["sequence"] != sequence
                        or result["phase"] != phase or result["iteration"] != iteration
                        or result["request_sha256"] != sha_file(request_path)):
                    raise ValueError("Worker result does not bind this exact policy/case bank")
                signature = result["native_model_frozen_signature"]
                if (signature["model_sha256"] != profile["model_sha256"]
                        or signature["model_config_sha256"] != profile["model_config_sha256"]
                        or signature["norm_sha256"] != profile["norm_sha256"]
                        or not signature["all_parameters_frozen"]):
                    raise ValueError("Frozen initializer differs from the approved checkpoint")
                expected_cases = torch.load(request_path, map_location="cpu", weights_only=False)["cases"]
                if phase != "calibrate" and ([case_key(row["case"]) for row in result["episodes"]]
                                              != [case_key(case) for case in expected_cases]):
                    raise ValueError("Complete episode identities differ from the requested bank")
                results[rank] = result
            elif children[rank].poll() is not None:
                raise RuntimeError(f"Worker {rank} exited before publishing its complete bank")
        if len(results) < 8:
            if time.time() >= deadline:
                raise TimeoutError("Finite allocation deadline; preserve completed/partial evidence")
            time.sleep(.5)
    return [results[rank] for rank in range(8)]


def _paired_readout(before, after, *, development=False):
    from scipy.stats import binomtest
    def comparison(tasks, metric):
        left = {case_key(row["case"]): row for row in before if row["case"]["task"] in tasks}
        right = {case_key(row["case"]): row for row in after if row["case"]["task"] in tasks}
        if set(left) != set(right) or not left:
            raise ValueError("Final evaluation requires the unchanged paired case bank")
        wins = sum(bool(right[key][metric]) and not left[key][metric] for key in left)
        losses = sum(bool(left[key][metric]) and not right[key][metric] for key in left)
        # All tasks have50cases, so pooled case gain equals the task macro gain.
        gain = (wins - losses) / len(left)
        discordant = wins + losses
        p = float(binomtest(wins, discordant, .5).pvalue) if discordant else 1.
        generator = np.random.default_rng(921007)
        bootstrap = np.zeros(10_000)
        for task in tasks:
            keys = [key for key in left if key[0] == task]
            delta = np.asarray([int(bool(right[key][metric])) - int(bool(left[key][metric])) for key in keys])
            bootstrap += delta[generator.integers(len(delta), size=(len(bootstrap), len(delta)))].mean(1) / len(tasks)
        return dict(cases=len(left), before=sum(bool(row[metric]) for row in left.values()),
            after=sum(bool(row[metric]) for row in right.values()), after_only=wins, before_only=losses,
            gain=gain, exact_two_sided_mcnemar_p=p,
            paired_task_stratified_bootstrap_95ci=np.quantile(bootstrap, [.025, .975]).tolist(),
            bootstrap_scope="Fixed task set, resampling paired cases within task; not held-out-task uncertainty")
    from .joint_steering_ppo import TASKS, OOD_TASKS
    return dict(ID={metric: comparison(TASKS, metric) for metric in ("success_once", "success_at_end")},
        OOD={} if development else {metric: comparison(OOD_TASKS, metric)
                                   for metric in ("success_once", "success_at_end")},
        per_task={task: comparison((task,), "success_once")
                  for task in (TASKS if development else (*TASKS, *OOD_TASKS))},
        per_task_at_end={task: comparison((task,), "success_at_end")
                        for task in (TASKS if development else (*TASKS, *OOD_TASKS))},
        primary="ID macro first-success gain", fixed_final_checkpoint=not development,
        exploratory_monitoring=development,
        multiple_arms="Two primary arm comparisons; report both raw and Holm-adjusted p-values across arms")


def start_workers(run, profile, deadline):
    from fluffy_bassoon.baige_gpu import mapping
    devices = mapping()
    if len(devices) != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError("Require exactly eight actual attested allocated devices")
    children, streams = [], []
    try:
        for rank, device in enumerate(devices):
            directory = run / "workers" / f"rank-{rank:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            stream = (directory / f"console-{time.time_ns()}.log").open("x")
            streams.append(stream)
            environment = dict(os.environ, **profile["native_env"])
            environment.update(CUDA_VISIBLE_DEVICES=device["visibility_token"],
                LYCO_BAIGE_ISOLATED_UUID=device["uuid"], LYCO_BAIGE_GPU_MAP=os.environ["LYCO_BAIGE_GPU_MAP"],
                MUJOCO_EGL_DEVICE_ID=str(device["allocation_ordinal"]),
                ROBOCASA_NATIVE_OUTPUT=str(directory / "native-runtime"),
                ROBOCASA_PI05_TORCH_MODEL=profile["model_directory"],
                ROBOCASA_PI05_NATIVE_NORM=profile["norm_path"], RLINF_PATH=profile["runtime_path"],
                PYTHONPATH=os.pathsep.join((str(Path(profile["code"]) / "src"), profile["runtime_path"],
                    environment.get("PYTHONPATH", ""))),
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            children.append(subprocess.Popen([sys.executable, "-B", "-m",
                "fluffy_bassoon.meta_rl.joint_steering_ppo_driver",
                "--run", str(run), "--profile", str(run / "PROFILE.json"),
                "--deadline-unix", str(deadline), "--worker-rank", str(rank)],
                env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
        return children, streams
    except BaseException:
        stop_workers(run, children, streams)
        raise


def stop_workers(run, children, streams):
    (run / "STOP").touch()
    end = time.time() + 15
    while any(child.poll() is None for child in children) and time.time() < end:
        time.sleep(.2)
    # These are exact child process groups created by this driver. No other
    # scheduler job or user process is ever signalled here.
    for child in children:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    for stream in streams:
        stream.close()
    release = dict(hostname=socket.gethostname(), all_children_released=True,
        child_exit_codes={str(child.pid): child.returncode for child in children},
        profile_sha256=tensor_hash(json.loads((run / "PROFILE.json").read_text())), ended_unix=time.time())
    write_json(run / f"released-{time.time_ns()}.json", release)
    write_json(run / "RELEASED.json", release)


def run_master(run, profile, *, deadline, resume=False):
    run = Path(run)
    if (profile["lanes_per_worker"] != ROLLOUT_LANES
            or profile["episodes_per_update"] != TRAIN_EPISODES or profile["updates"] != UPDATES):
        raise ValueError("Profile differs from the verified50-update/384-episode/48-lane recipe")
    reference = profile["admission_reference"]
    if sha_file(reference["path"]) != reference["sha256"]:
        raise ValueError("Qualified production admission receipt changed")
    admission = json.loads(Path(reference["path"]).read_text())
    if not admission["completed"] or admission["worker_lanes"] != ROLLOUT_LANES:
        raise ValueError("Actual48-lane steering/optimizer qualification is incomplete")
    if any(admission[key] != profile[key] for key in
           ("source_commit", "runtime_commit", "model_sha256", "model_config_sha256", "norm_sha256")):
        raise ValueError("Admission does not bind this exact source/runtime/frozen initializer")
    qualified = admission["arms"][profile["arm"]]
    if not qualified["passed"]:
        raise ValueError("This independent steering arm is not qualified")
    arm_budget_seconds = APPROVED_ARM_GPU_HOURS * 3600
    if (not 0 <= profile["prior_conservative_gpu_seconds"] < arm_budget_seconds
            or not math.isfinite(profile["submitted_unix"]) or profile["submitted_unix"] > time.time()):
        raise ValueError("Require actual nonnegative prior-attempt GPU accounting")
    remaining_gpu_seconds = arm_budget_seconds - profile["prior_conservative_gpu_seconds"]
    hard_end = profile["submitted_unix"] + remaining_gpu_seconds / 8
    if (remaining_gpu_seconds <= 0 or not math.isfinite(deadline)
            or deadline > hard_end - 120 or deadline <= time.time()):
        raise ValueError(
            f"Finite deadline must fit the remaining{APPROVED_ARM_GPU_HOURS}GPUh arm escrow "
            "with release headroom")
    checked_checkout(Path(profile["code"]), profile["source_commit"])
    checked_checkout(Path(profile["runtime_path"]), profile["runtime_commit"])
    if (sha_file(Path(profile["model_directory"]) / "model.safetensors") != profile["model_sha256"]
            or sha_file(Path(profile["model_directory"]) / "config.json") != profile["model_config_sha256"]
            or sha_file(profile["norm_path"]) != profile["norm_sha256"]):
        raise ValueError("Approved original checkpoint/normalization identity differs")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "RESULT.json").exists() and json.loads((run / "RESULT.json").read_text()).get("completed"):
        raise ValueError("A scientifically completed run must not be restarted")
    profile_path = run / "PROFILE.json"
    if profile_path.exists():
        if json.loads(profile_path.read_text()) != profile:
            raise ValueError("Existing run needs explicit same-contract complete-boundary recovery")
        if not resume and ((run / "workers").exists() or (run / "checkpoint-latest.pt").exists()):
            raise ValueError("A previously started run needs explicit recovery")
    else:
        if resume:
            raise ValueError("Recovery needs an existing immutable profile and checkpoint")
        write_json(profile_path, profile)
    if (run / "STOP").exists():
        if not resume:
            raise ValueError("A stopped run requires explicit recovery")
        release = json.loads((run / "RELEASED.json").read_text())
        if (release["hostname"] != socket.gethostname() or not release["all_children_released"]
                or release["profile_sha256"] != tensor_hash(profile)
                or any(value is None for value in release["child_exit_codes"].values())):
            raise ValueError("Resume requires proved child release in the same allocation; otherwise migrate to a fresh attempt")
        (run / "STOP").replace(run / f"stop-consumed-{time.time_ns()}")
        marker = run / "CHECKPOINT_AND_EXIT"
        if marker.exists():
            marker.replace(run / f"checkpoint-exit-consumed-{time.time_ns()}")
    arm = profile["arm"]
    policy = make_policy(arm, device="cuda:0", seed=profile["seed"],
                         optimizer_config=qualified["optimizer_config"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=policy.config.learning_rate)
    generator = torch.Generator().manual_seed(profile["seed"] + 1)
    specification, worker_states, cursor, iteration = qualified["interface_spec"], {}, 0, 0
    if {key: value for key, value in specification.items() if key != "fraction"} != {
            key: value for key, value in spec_arguments(arm).items() if key != "fraction"}:
        raise ValueError("Qualified interface differs from the agreed arm")
    journal = JointActionJournal(run / "accounting", rank=99, category="train", scope="master_read")
    profile_hash = tensor_hash(profile)
    measured_seconds = {}
    if resume:
        saved = torch.load(run / "checkpoint-latest.pt", map_location="cpu", weights_only=False)
        if saved["profile_sha256"] != profile_hash:
            raise ValueError("Cannot change source/runtime/initializer or resources on recovery")
        specification = saved["spec"]
        iteration, cursor, worker_states = restore_boundary(saved, policy, optimizer, generator,
            arm=arm, interface_spec=specification)
        measured_seconds = saved["measured_seconds"]
        optimizer_progress = run / "optimizer-progress.json"
        if optimizer_progress.exists():
            optimizer_progress.replace(run / f"optimizer-progress-prior-{time.time_ns()}.json")
        if saved["escrow"] != journal.snapshot():
            # Outstanding collection after the last complete checkpoint may
            # legitimately add charged actions. It must not refund old ones.
            current = journal.snapshot()
            if any(current["charged"][key] < saved["escrow"]["charged"][key] for key in ("train", "eval")):
                raise ValueError("Recovery cannot discard consumed or unresolved controls")
    children = streams = []
    publisher = MetricPublisher(run, profile)
    try:
        children, streams = start_workers(run, profile, deadline)
        schedule = events()
        for event_index, (phase, requested_iteration) in enumerate(schedule[cursor:], cursor):
            # Stop only at a complete bank/update boundary when an operator
            # requests it. A finite deadline remains the hard resource bound.
            if (run / "CHECKPOINT_AND_EXIT").exists():
                write_json(run / "PAUSED.json", dict(completed=False, reason="operator_checkpoint_exit",
                    iteration=iteration, sequence=cursor, checkpoint=str(run / "checkpoint-latest.pt")))
                return
            estimate_phase = "test_before" if phase == "test_after" else phase
            estimate = max(measured_seconds.get(estimate_phase, [30.])) * 1.25 + 30
            if time.time() + estimate >= deadline:
                write_json(run / "PAUSED.json", dict(completed=False, reason="finite_resource_boundary",
                    iteration=iteration, sequence=cursor, next_phase=phase,
                    checkpoint=str(run / "checkpoint-latest.pt"), no_automatic_retry=True))
                return
            started = time.time()
            charged_before = sum(journal.snapshot()["charged"].values())
            results = dispatch(run, profile, policy, specification, sequence=event_index + 1,
                phase=phase, iteration=requested_iteration, children=children, deadline=deadline)
            if len({row["native_model_frozen_signature"]["cuda_uuid"] for row in results}) != 8:
                raise ValueError("Eight worker ranks did not use eight distinct actual devices")
            status = dict(phase=phase, sequence=event_index + 1, iteration=requested_iteration,
                          source_commit=profile["source_commit"], last_check_unix=time.time())
            if phase == "calibrate":
                calibration = select_scale(results, selected_fraction=specification["fraction"])
                write_json(run / "CALIBRATION.json", calibration)
                status["calibrated_fraction"] = specification["fraction"]
            else:
                episodes = [episode for result in results for episode in result["episodes"]]
                status.update(summary=summarize_episodes(episodes))
                if phase == "train":
                    records = chronological_records(results, cases_for(phase, requested_iteration))
                    if requested_iteration == 1:
                        records = normalize_first_bank(policy, records)
                    rollout = stack_records(records, device="cuda:0")
                    if "behavior_mean" not in rollout or "behavior_std" not in rollout:
                        raise ValueError("Joint steering PPO requires actual retained behavior Gaussian parameters")
                    advantages, returns = generalized_advantages(rollout["rewards"], rollout["values"],
                        rollout["ends"], gamma=policy.config.gamma, gae_lambda=policy.config.gae_lambda)
                    _save(run / "banks" / f"round-{requested_iteration:03d}-before-sgd.pt", dict(
                        policy_version=requested_iteration - 1, policy=policy.state_dict(),
                        interface_spec=specification, records=records, rollout=rollout,
                        advantages=advantages, returns=returns, episodes=episodes,
                        behavior_result_refs=[dict(path=str(run / "workers" / f"rank-{rank:02d}" /
                            f"result-{event_index + 1:06d}.pt"), sha256=sha_file(run / "workers" /
                            f"rank-{rank:02d}" / f"result-{event_index + 1:06d}.pt")) for rank in range(8)]))
                    status["ppo"] = update(policy, optimizer, rollout, generator=generator,
                        epoch_callback=lambda row: write_json(run / "optimizer-progress.json",
                            dict(iteration=requested_iteration, **row)))
                    status["ppo"].update(coefficient_rms=float(rollout["controls"].square().mean().sqrt()),
                        coefficient_saturation_fraction=float((rollout["controls"].tanh().abs() > .95).float().mean()),
                        positive_reward_count=int((rollout["rewards"] > 0).sum()),
                        negative_reward_count=int((rollout["rewards"] < 0).sum()))
                    if status["ppo"]["optimizer_steps"] < 1:
                        raise ValueError("No optimizer update occurred; diagnose behavior likelihoods before advancing the round")
                    iteration = requested_iteration
                    if requested_iteration == 1:
                        status["gpu_recovery_qualification"] = qualify_optimizer_restore(policy, optimizer,
                            generator, rollout["features"].reshape(-1, policy.config.feature_dim),
                            arm=arm, interface_spec=specification)
                        write_json(run / "GPU-ADAM-RECOVERY.json", status["gpu_recovery_qualification"])
                else:
                    _save(run / "evaluations" / f"{phase}-{requested_iteration:03d}.pt",
                          dict(episodes=episodes, summary=status["summary"], policy_iteration=iteration))
                    if phase == "dev":
                        baseline = episodes if requested_iteration == 0 else torch.load(
                            run / "evaluations/dev-000.pt", map_location="cpu", weights_only=False)["episodes"]
                        status["paired_development"] = _paired_readout(baseline, episodes, development=True)
            escrow = journal.snapshot()
            if escrow["pending"]:
                raise ValueError("Complete worker bank left unresolved physical-call reservations")
            worker_states = {rank: dict(worker_rng=result["worker_rng"], sequence=result["sequence"],
                frozen_signature=result["native_model_frozen_signature"]) for rank, result in enumerate(results)}
            cursor = event_index + 1
            elapsed = time.time() - started
            measured_seconds.setdefault(phase, []).append(elapsed)
            with torch.no_grad():
                std = policy.log_std.clamp(-5, 2).exp()
                status.update(steering_std_mean=float(std.mean()), steering_std_min=float(std.min()),
                    steering_std_max=float(std.max()), learning_rate=optimizer.param_groups[0]["lr"])
            saved = boundary_state(policy, optimizer, generator, arm=arm, iteration=iteration,
                sequence=cursor, worker_states=worker_states, escrow=escrow, interface_spec=specification)
            saved.update(profile_sha256=profile_hash, last_event=phase, measured_seconds=measured_seconds)
            path = run / "checkpoints" / f"event-{cursor:03d}-{phase}-{iteration:03d}.pt"
            reference = _save(path, saved)
            _save(run / "checkpoint-latest.pt", saved)
            status.update(checkpoint=reference, escrow=escrow, elapsed_seconds=elapsed,
                          completed_events=cursor, total_events=len(schedule), resources=resource_snapshot(run))
            status["collection_and_learning_seconds"] = elapsed
            status["elapsed_seconds"] = time.time() - started
            measured_seconds[phase][-1] = status["elapsed_seconds"]
            status["physical_controls_this_event"] = sum(escrow["charged"].values()) - charged_before
            status["controls_per_second"] = status["physical_controls_this_event"] / max(status["elapsed_seconds"], 1e-9)
            status["remaining_gpu_hours"] = max(0., APPROVED_ARM_GPU_HOURS -
                (profile["prior_conservative_gpu_seconds"] + 8 * (time.time() - profile["submitted_unix"])) / 3600)
            if phase == "train":
                future = schedule[cursor:]
                def estimate_phase(name):
                    values = measured_seconds.get("test_before" if name == "test_after" else name, [30.])
                    # Round1 includes one-time normalization/recovery work.
                    return float(np.mean(values[1:] if name == "train" and len(values) > 1 else values))
                projected_remaining = sum(estimate_phase(name) for name, _ in future)
                status["projected_remaining_seconds"] = projected_remaining
                status["projected_all_attempt_gpu_hours"] = (profile["prior_conservative_gpu_seconds"] +
                    8 * (time.time() - profile["submitted_unix"] + projected_remaining + 120)) / 3600
            write_json(run / "metrics" / f"event-{cursor:03d}.json", status)
            write_json(run / "PROGRESS.json", status)
            publisher.publish(status)
            if (phase == "train" and iteration >= 2
                    and status["projected_all_attempt_gpu_hours"] > APPROVED_ARM_GPU_HOURS):
                write_json(run / "PAUSED.json", dict(completed=False, reason="measured_budget_decision",
                    iteration=iteration, sequence=cursor, checkpoint=reference,
                    projected_all_attempt_gpu_hours=status["projected_all_attempt_gpu_hours"],
                    approved_arm_gpu_hours=APPROVED_ARM_GPU_HOURS, no_automatic_retry=True))
                return
        before = torch.load(run / "evaluations/test_before-000.pt", map_location="cpu", weights_only=False)
        after = torch.load(run / "evaluations" / f"test_after-{UPDATES:03d}.pt", map_location="cpu", weights_only=False)
        result = _paired_readout(before["episodes"], after["episodes"])
        result.update(completed=True, arm=arm, iterations=iteration, escrow=journal.snapshot(),
                      completed_unix=time.time(), source_commit=profile["source_commit"])
        write_json(run / "RESULT.json", result)
    except BaseException:
        write_json(run / f"master-error-{time.time_ns()}.json", dict(error=traceback.format_exc(),
            iteration=iteration, sequence=cursor, escrow=journal.snapshot(), retry=False))
        raise
    finally:
        try:
            stop_workers(run, children, streams)
        finally:
            publisher.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--worker-rank", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if args.worker_rank is not None:
        if not 0 <= args.worker_rank < 8:
            parser.error("Worker rank is outside the approved eight devices")
        from .joint_steering_ppo_worker import run_worker
        run_worker(args.run, args.worker_rank, Path(profile["code"]), args.deadline_unix)
    else:
        args.run.mkdir(parents=True, exist_ok=True)
        with (args.run / ".master.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_master(args.run, profile, deadline=args.deadline_unix, resume=args.resume)


if __name__ == "__main__":
    main()
