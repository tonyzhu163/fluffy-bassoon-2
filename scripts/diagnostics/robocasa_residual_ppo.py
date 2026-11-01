#!/usr/bin/env python
"""Allocated-Linux RoboCasa binding: frozen native pi0.5, matched residual PPO.

Run a prospectively bound headroom bank, a training arm or fixed evaluation.
Flora runs in a separate process in the same allocation. A missing label leaves pending.pt intact
with no GAE/update, rather than collecting a replacement on-policy batch.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from fluffy_bassoon.meta_rl.residual_ppo import (
    ResidualActorCritic, ResidualPPOConfig, history_trial, stack_records, update,
)
from fluffy_bassoon.meta_rl.robocasa_residual import (
    FrozenRoboCasaMotor, RoboCasaResidualEnvironment, collect_pending,
    install_terminal_labels, native_environment_factory, restore_checkpoint, save_checkpoint,
)
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import (
    FLORA_CODE, FLORA_MODEL, digest, flora_jobs, parse_outputs, sha, write_new,
)
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_monitor import ProgressLog
from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal, restore_rng, snapshot_rng
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import (
    TASKS, arm_stopping, case_bank, protocol, require_calibration_permission, rollout_migration_contract,
)
from fluffy_bassoon.meta_rl.vla_history import save_dataset
from fluffy_bassoon.rl.action_residual import ResidualBound, action_std_from_norm_stats


def annotation_failure(profile):
    for name in profile.get("annotation_error_paths", []):
        path = Path(name)
        if path.exists():
            raise RuntimeError(f"Owned annotation service failed; preserve complete pending state: {path}")


def wait_for_annotation_engines(profile, report):
    """Bind real scorer warmups before a colocated policy allocates its motor."""
    bindings = profile.get("annotation_engines")
    if bindings is None:
        if profile["arm"] == "F" and profile.get("parallel_controller_cpu_offload"):
            raise ValueError("Colocated F requires all eight actual scorer warmups before rollout")
        return
    if (profile["arm"] != "F" or not profile.get("parallel_controller_cpu_offload")
            or len(bindings) != 8 or len(profile["rollout_devices"]) != 8):
        raise ValueError("Eight colocated scorers and an offloaded F controller required")
    expected = [read_ref(row["profile"]) for row in bindings]
    paths = [Path(row["ready_path"]) for row in bindings]
    offload_path = Path(profile["output"]) / "controller-motor-offload.json"
    offload_ref = dict(path=str(offload_path), sha256=sha(offload_path))
    limit = min(profile["deadline_unix"] - profile["closure_seconds"],
                time.time() + profile["annotation_startup_wall_ceiling_seconds"])
    while not all(path.exists() for path in paths):
        annotation_failure(profile)
        if time.time() >= limit:
            raise TimeoutError("Scorer warmups incomplete; no policy rollout dispatched")
        report("waiting_for_eight_scorer_warmups")
        time.sleep(1)
    references = []
    for rank, (path, scorer) in enumerate(zip(paths, expected)):
        ready = json.loads(path.read_text())
        device = profile["rollout_devices"][rank]
        required = scorer["minimum_free_mib_after_engine_warmup"]
        memory = ready["memory"]
        if (scorer["annotation_rank"] != rank or scorer["annotation_world"] != 8
                or scorer["source_commit"] != profile["source_commit"]
                or scorer["F_output"] != profile["output"] or scorer["task"] != profile["task"]
                or scorer["baseline_validation"] != profile["baseline_validation"]
                or scorer.get("calibration_exception") != profile.get("calibration_exception")
                or scorer["engine_ready_path"] != str(path) or scorer["expected_gpu_uuid"] != device["uuid"]
                or ready.get("format") != "robocasa_flora_tp1_engine_ready_v1"
                or ready.get("completed") is not True or ready["annotation_rank"] != rank
                or ready["annotation_world"] != 8 or ready["source_commit"] != profile["source_commit"]
                or ready["profile_sha256"] != digest(scorer) or ready["gpu_uuid"] != device["uuid"]
                or ready["gpu_probe"] != scorer["actual_gpu_probe"]
                or ready["code_commit"] != FLORA_CODE or ready["model_revision"] != FLORA_MODEL
                or ready["model_receipt"] != scorer["model_receipt"]
                or ready["controller_offload"] != offload_ref or ready["tensor_parallel_size"] != 1
                or ready["batch_size"] != scorer["batch_size"]
                or ready["gpu_memory_utilization"] != scorer["gpu_memory_utilization"]
                or type(ready["engine_process_pid"]) is not int or ready["engine_process_pid"] <= 0
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                           for v in (required, memory["free_mib"], memory["used_mib"], memory["total_mib"]))
                or required <= 0 or memory["minimum_free_required_mib"] != required
                or memory["free_mib"] < required or not 0 <= memory["used_mib"] < memory["total_mib"]
                or memory["free_mib"] > memory["total_mib"]):
            raise ValueError("Exact eight source/model/device/profile/memory warmups required before rollout")
        references.append(dict(path=str(path), sha256=sha(path)))
    annotation_failure(profile)
    write_new(Path(profile["output"]) / "annotation-engines-bound.json", dict(
        completed=True, engines=references, controller_offload=offload_ref,
        qualification="Actual warmup and observed free-memory barrier; no new physical episode"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one actually allocated Linux GPU required")
    profile = json.loads(Path(args.profile).read_text())
    code = Path(__file__).resolve().parents[2]
    head = lambda path: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if (head(code) != profile["source_commit"] or profile["mode"] not in ("headroom", "common_before", "train", "evaluate")
            or profile["arm"] not in ("S", "F") or profile["task"] not in TASKS
            or not 0 < profile["scheduler_seconds"] < 8 * 3600 or profile["closure_seconds"] < 180
            or time.time() >= profile["deadline_unix"] - profile["closure_seconds"]):
        raise ValueError("Exact source/native task/arm and finite resource profile required")
    if profile["task"] == "CloseFridge":
        raise ValueError("The fixed OOD task cannot be selected for calibration or training")
    read_ref(profile["actual_gpu_probe"])
    if os.environ.get("LYCO_BAIGE_GPU_MAP"):
        from fluffy_bassoon.baige_gpu import verify_selected_cuda
        verify_selected_cuda(torch)
    qualified = read_ref(profile["loading_qualification"])
    if (not qualified["completed"] or not qualified["converted_norm_identical"]
            or len(qualified["actual_loader"]) != 1 or qualified["has_value_head"]
            or any(qualified["actual_loader"][0][key] for key in ("missing_keys", "unexpected_keys"))):
        raise ValueError("Actual original-checkpoint strict load and native normalization qualification required")
    # The profile binds the derived hash to the original c484 checkpoint and
    # the actual conversion receipt. No unrelated ready motor may be substituted.
    initializer = profile["initializer"]
    if (initializer["repository"] != "robocasa/robocasa365_checkpoints"
            or initializer["revision"] != "c484448aba1a9b60a04c9b0ca117241518ea69f3"
            or initializer["prefix"] != "pi05_pretrain_human300/multitask_learning/75000"
            or qualified["normalization_sha256"] != initializer["normalization_sha256"]):
        raise ValueError("Use the same released Human30075000 motor/norm for both arms")
    norm_path = Path(initializer["norm_path"])
    if sha(norm_path) != initializer["normalization_sha256"]:
        raise ValueError("Bound motor normalization changed")
    for reference in profile["core_qualification"]:
        read_ref(reference)
    output = Path(profile["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "profile.json", profile)
    parallel = profile.get("rollout_devices")
    if profile.get("parallel_controller_cpu_offload") and not parallel:
        raise ValueError("Controller motor offload requires separate native rollout workers")
    if parallel:
        from fluffy_bassoon.meta_rl.robocasa_residual_parallel import ParallelActionJournal, ParallelResidualEnvironment
        if len(parallel) != 8 or len({row["uuid"] for row in parallel}) != 8:
            raise ValueError("Eight distinct allocated GPUs required for parallel rollouts")
        journal = ParallelActionJournal(profile["accounting_root"], len(parallel))
    else:
        journal = ActionJournal(profile["accounting_root"], rank=0, world=1, attempt=profile["attempt"])
    progress = ProgressLog(output / "progress", profile["arm"], contract_sha256=digest(profile)).start()
    last_report = [0.]

    def report(phase, **fields):
        now = time.time()
        if now < last_report[0] + 20:
            return
        last_report[0] = now
        elapsed_gpu_seconds = sum(max(0., now - started) for started in
            profile.get("gpu_allocation_starts_unix", [profile["allocated_start_unix"]] * (len(parallel) if parallel else 1)))
        progress.emit(phase=phase, physical_actions=journal.charged(),
            aggregate_gpu_seconds=elapsed_gpu_seconds,
            this_open_job_elapsed_gpu_seconds=elapsed_gpu_seconds,
            remaining_wall_seconds=profile["deadline_unix"] - now, **fields)

    report("frozen_motor_loading")
    # A stricter ticket may stop below 1M; no old common-action credit is reused.
    action_limit = int(profile["maximum_all_in_actions"])
    if not 0 < action_limit <= 1_000_000:
        raise ValueError("Finite all-in physical action limit required")
    from hydra import compose, initialize_config_dir
    from rlinf.models.embodiment.openpi import get_model
    import imageio.v2 as imageio
    with initialize_config_dir(config_dir=str(code / "configs/native_ppo"), version_base=None):
        cfg = compose(config_name="robocasa365_pi05_eval")
    if head(os.environ["RLINF_PATH"]) != profile["rlinf_execution_commit"]:
        raise ValueError("Pinned native RLinf derivative required")
    cfg.rollout.model.model_path = initializer["model_directory"]
    cfg.rollout.model.openpi_data.norm_stats_path = str(norm_path)
    model = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to("cuda").eval().requires_grad_(False)
    stats = json.loads(norm_path.read_text())
    stats = stats.get("norm_stats", stats)
    motor = FrozenRoboCasaMotor(model, stats, seed=profile["motor_seed"])
    if profile.get("parallel_controller_cpu_offload"):
        # The parallel controller uses only the motor's private seed generator;
        # every prediction runs in an allocated rollout worker's own model.
        model.to("cpu")
        if any(t.is_cuda for t in model.parameters()) or any(t.is_cuda for t in model.buffers()):
            raise ValueError("The unused controller motor must release all CUDA tensors")
        torch.cuda.empty_cache()
    bound = ResidualBound.from_std(action_std_from_norm_stats(norm_path), indices=range(6), fraction=.25, action_dim=12)
    torch.manual_seed(42)
    config = ResidualPPOConfig(**profile["ppo"])
    if config.feature_dim != 2088 or config.residual_dim != 6:
        raise ValueError("Public proprio16 + first12 + mean12 + native pooled2048, six arm controls")
    policy = ResidualActorCritic(config).to("cuda").eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    generators = {"action": torch.Generator(device="cuda").manual_seed(profile["residual_seed"]),
                  "update": torch.Generator().manual_seed(profile["update_seed"])}
    contract = dict(initializer=initializer, bound=asdict(bound), ppo=asdict(config), arm=profile["arm"],
        task=profile["task"], source_commit=profile["source_commit"],
        motor_seed=profile["motor_seed"], residual_seed=profile["residual_seed"], update_seed=profile["update_seed"],
        features="normalized proprio16+base first12+mean12+same-prefix masked mean2048/RMS; fixed no fit")
    task = profile["task"]
    environments = []

    def environment(cases, name):
        if parallel:
            env = ParallelResidualEnvironment(cases, horizon=TASKS[cases[0]["task"]][0],
                output=output / name / "videos", monitoring_root=Path(profile["monitoring_root"]) / name,
                profile=profile, code=code, report=report)
        else:
            env = RoboCasaResidualEnvironment(native_environment_factory(cfg.env.eval), cases,
                horizon=TASKS[cases[0]["task"]][0], output=output / name / "videos",
                monitoring_root=Path(profile["monitoring_root"]) / name, journal=journal,
                writer_factory=imageio.get_writer, observer=lambda: report("native_" + name))
        environments.append(env)
        return env

    def reserve(episodes, horizon):
        if journal.charged() + episodes * (horizon + 128) > action_limit:
            raise RuntimeError("Cannot fit another complete episode bank in physical-action ticket")
        if time.time() + profile["closure_seconds"] >= profile["deadline_unix"]:
            raise TimeoutError("Stop at closed boundary before finite deadline")

    def truth(episodes, directory):
        return {row["id"]: int(json.loads((Path(profile["monitoring_root"]) / directory / (row["id"] + ".json")).read_text())["simulator_final_success"]) for row in episodes}

    def panel(phase, iteration):
        target = "CloseFridge" if "OOD" in phase else task
        reserve(8, TASKS[target][0])
        name = f"{phase}-{iteration:03d}"
        env = environment(case_bank(target, phase), name)
        global_rng, motor_rng = snapshot_rng(), motor.state_dict()
        motor.seed_rng = np.random.default_rng(profile["evaluation_motor_seeds"][phase])
        try:
            records, episodes = collect_pending(policy, bound, motor, env, episodes=8,
                deterministic=True, generator=generators["action"], verify_zero=profile["mode"] == "headroom")
        finally:
            motor.load_state_dict(motor_rng)
            restore_rng(global_rng)
            env.close()
        labels = truth(episodes, name)
        result = dict(phase=phase, iteration=iteration, episodes=8, successes=sum(labels.values()),
            physical_actions=journal.charged(), policy_checkpoint=str(output / "final.pt") if phase in ("ID", "OOD") else None)
        write_new(output / name / "evaluation.json", result)
        return records, episodes, result

    try:
        if profile["mode"] == "headroom":
            if profile["headroom_cases"] != case_bank(task, "baseline"):
                raise ValueError("Native constructor cases must be frozen before outcomes")
            records, episodes, result = panel("baseline", 0)
            write_new(output / "headroom.json", dict(result, native_headroom=1 <= result["successes"] <= 5,
                zero_residual_verified_chunks=sum(row["zero_residual_verified"] for row in records),
                qualification="first planned episode replays every native chunk seed before executing it once; zero residual; exploratory8, not population success"))
            write_new(output / "episodes.json", episodes)
            return
        if profile["mode"] == "common_before":
            for phase in ("initial_DEV", "before_ID", "before_OOD"):
                panel(phase, 0)
            return
        if profile["mode"] == "evaluate":
            if sha(profile["checkpoint"]) != profile["checkpoint_sha256"]:
                raise ValueError("Final checkpoint bytes changed before ID/OOD evaluation")
            env = environment(profile["training_cases"], "resume-state")
            restore_checkpoint(profile["checkpoint"], policy=policy, optimizer=optimizer, bound=bound, motor=motor,
                environment=env, generators=generators, contract=contract)
            panel("ID", int(profile["iteration"]))
            panel("OOD", int(profile["iteration"]))
            return
        gate = read_ref(profile["baseline_validation"])
        authorization = read_ref(profile["calibration_exception"]) if profile.get("calibration_exception") else None
        permission = require_calibration_permission(gate, protocol(task),
            calibration_sha256=profile["baseline_validation"]["sha256"], authorization=authorization)
        contract.update(calibration=profile["baseline_validation"], calibration_permission=permission,
                        calibration_exception=profile.get("calibration_exception"))
        cases = [case for iteration in range(1, 11) for round_index in range(4)
                 for case in case_bank(task, "train", iteration=iteration, round_index=round_index)]
        if profile["training_cases"] != cases:
            raise ValueError("Prospectively frozen native TRAIN bank required")
        train_env = environment(cases, "train")
        iteration = 0
        pending = None
        if profile.get("resume"):
            if sha(profile["resume"]) != profile["resume_sha256"]:
                raise ValueError("Producing checkpoint bytes changed before recovery")
            resume_contract = contract
            if profile.get("rollout_migration"):
                migration = read_ref(profile["rollout_migration"])
                if not parallel:
                    raise ValueError("Only an explicit complete-state rollout execution migration is allowed")
                resume_contract = rollout_migration_contract(contract, migration,
                                                            checkpoint_sha256=profile["resume_sha256"])
            restored = restore_checkpoint(profile["resume"], policy=policy, optimizer=optimizer, bound=bound,
                motor=motor, environment=train_env, generators=generators, contract=resume_contract)
            iteration = restored["iteration"]
            if restored["stage"] == "pending_labels":
                started = Path(profile["resume"]).parent / f"UPDATE-STARTED-{iteration + 1:03d}.json"
                if started.exists():
                    review = read_ref(profile["update_recovery_review"])
                    if review.get("completed") is not True or review.get("pending_checkpoint_sha256") != profile["resume_sha256"]:
                        raise ValueError("Prior update began: bind reviewed full-state recovery before replay")
                pending = restored
        else:
            save_checkpoint(output / "boundary-000.pt", policy=policy, optimizer=optimizer, bound=bound, motor=motor,
                environment=train_env, generators=generators, contract=contract, iteration=0, stage="boundary")
        if profile.get("parallel_controller_cpu_offload"):
            # Publish only after the learner/RNG and initial recovery are fully
            # created, so vLLM profiles a stable controller CUDA footprint.
            write_new(output / "controller-motor-offload.json", dict(
                completed=True, device="cpu", predictions="allocated rollout workers only",
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved()))
        wait_for_annotation_engines(profile, report)
        history = []
        dev_evaluations = ([read_ref(profile["initial_dev_evaluation"])]
                          if profile.get("independent_stopping") else [])
        if profile.get("prior_dev_evaluations"):
            dev_evaluations.extend(read_ref(reference) for reference in profile["prior_dev_evaluations"])
        iteration_times = []
        stop_reason = "fixed10"
        final_actions = 8 * (TASKS[task][0] + 128) + 8 * (900 + 128)
        while iteration < 10:
            # The first iteration is retained training, never a disposable
            # benchmark. Subsequent admission uses its actual full-batch time.
            estimate = (max(iteration_times) * 1.25 if iteration_times
                        else profile["first_iteration_wall_ceiling_seconds"])
            if iteration + 1 in (2, 4, 6, 8, 10):
                estimate += profile["dev_evaluation_wall_ceiling_seconds"]
            if (pending is None and (journal.charged() + 64 * (TASKS[task][0] + 128)
                    + (8 * (TASKS[task][0] + 128) if iteration + 1 in (2, 4, 6, 8, 10) else 0)
                    + final_actions > action_limit or time.time() + estimate
                    + profile["final_evaluation_wall_reserve_seconds"] + profile["closure_seconds"]
                    >= profile["deadline_unix"])):
                stop_reason = "finite_complete_iteration_capacity"
                break
            iteration_start = time.monotonic()
            if pending is None:
                reserve(64, TASKS[task][0])
                records, episodes = collect_pending(policy, bound, motor, train_env, episodes=64, generator=generators["action"])
                path = output / f"pending-{iteration + 1:03d}.pt"
                checkpoint_sha = save_checkpoint(path, policy=policy, optimizer=optimizer, bound=bound, motor=motor,
                    environment=train_env, generators=generators, contract=contract, iteration=iteration,
                    stage="pending_labels", records=records, episodes=episodes)
            else:
                records, episodes = pending["pending_records"], pending["pending_episodes"]
                path = Path(profile["resume"])
                checkpoint_sha = sha(path)
                pending = None
            request = output / f"labels-{iteration + 1:03d}"
            request.mkdir()
            jobs = flora_jobs(episodes, protocol(task))
            write_new(request / "request.json", dict(pending_checkpoint=dict(path=str(path), sha256=checkpoint_sha),
                jobs=jobs, jobs_sha256=digest(jobs), episodes=episodes, protocol=protocol(task)))
            if profile["arm"] == "S":
                labels = truth(episodes, "train")
            else:
                ready_path = request / "ready.json"
                while not ready_path.exists():
                    annotation_failure(profile)
                    if time.time() + profile["closure_seconds"] >= profile["deadline_unix"]:
                        raise TimeoutError("Entire pending batch saved; Flora labels incomplete; no update")
                    report("waiting_for_complete_Flora_labels", iteration=iteration + 1)
                    time.sleep(1)
                ready = json.loads(ready_path.read_text())
                if (ready["pending_checkpoint_sha256"] != checkpoint_sha or ready["jobs_sha256"] != digest(jobs)
                        or ready["code_commit"] != FLORA_CODE or ready["model_revision"] != FLORA_MODEL):
                    raise ValueError("Stale or different Flora batch/weights/code")
                labels = parse_outputs(episodes, read_ref(ready["outputs"]), protocol(task))
            labeled = install_terminal_labels(records, episodes, labels, arm=profile["arm"])
            write_new(output / f"UPDATE-STARTED-{iteration + 1:03d}.json", dict(
                producing_pending_sha256=checkpoint_sha, effective_iteration=iteration + 1))
            metrics = update(policy, optimizer, stack_records(labeled, device="cuda"), generator=generators["update"])
            progress.emit(phase="PPO_update_complete", iteration=iteration + 1, physical_actions=journal.charged(),
                policy_updates=iteration + 1, episodes_complete=(iteration + 1) * 64, labels_complete=64,
                ppo_diagnostics=metrics)
            iteration += 1
            history.append((iteration - 1, labeled))
            checkpoint = output / f"boundary-{iteration:03d}.pt"
            save_checkpoint(checkpoint, policy=policy, optimizer=optimizer, bound=bound, motor=motor,
                environment=train_env, generators=generators, contract=contract, iteration=iteration, stage="boundary")
            write_new(output / f"update-{iteration:03d}.json", dict(iteration=iteration, metrics=metrics,
                physical_actions=journal.charged(), checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint))))
            progress.emit(phase="PPO_checkpoint_complete", iteration=iteration, policy_updates=iteration,
                last_full_recovery=dict(path=str(checkpoint), sha256=sha(checkpoint)),
                physical_actions=journal.charged(), ppo_diagnostics=metrics)
            if iteration in (2, 4, 6, 8, 10):
                _, _, evaluation = panel("DEV", iteration)
                if profile.get("independent_stopping"):
                    dev_evaluations.append(evaluation)
                    decision = arm_stopping(dev_evaluations)
                    write_new(output / f"stop-{iteration:03d}.json", decision)
                    if decision["stop"]:
                        stop_reason = decision["reason"]
                        iteration_times.append(time.monotonic() - iteration_start)
                        break
                    iteration_times.append(time.monotonic() - iteration_start)
                    continue
                # The owner combines complete same-iteration DEV panels only;
                # final ID/OOD never enter this decision.
                stop_path = Path(profile["paired_stop_root"]) / f"{iteration:03d}.json"
                while not stop_path.exists():
                    if time.time() + profile["closure_seconds"] >= profile["deadline_unix"]:
                        break
                    time.sleep(1)
                if not stop_path.exists() or json.loads(stop_path.read_text())["stop"]:
                    stop_reason = (json.loads(stop_path.read_text())["reason"] if stop_path.exists()
                                   else "peer_or_finite_clock_boundary")
                    iteration_times.append(time.monotonic() - iteration_start)
                    break
            iteration_times.append(time.monotonic() - iteration_start)
        save_checkpoint(output / "final.pt", policy=policy, optimizer=optimizer, bound=bound, motor=motor,
            environment=train_env, generators=generators, contract=contract, iteration=iteration, stage="boundary", final=True)
        write_new(output / "FINAL.json", dict(completed=True, iteration=iteration, stop_reason=stop_reason,
            measured_iteration_seconds=iteration_times, checkpoint=str(output / "final.pt"),
            sha256=sha(output / "final.pt"), contract=contract, includes=["policy", "critic", "Adam", "global/private RNG", "motor sampler", "case cursor"]))
        if history:
            save_dataset(output / "history.npz", [history_trial(history)], objective="ad" if len(history) > 1 else "behavior_cloning",
                metadata=dict(contract=contract, final_checkpoint_sha256=sha(output / "final.pt"), evaluation_data=False))
        panel("ID", iteration)
        panel("OOD", iteration)
    finally:
        for env in environments:
            env.close()
        progress.close()


if __name__ == "__main__":
    main()
