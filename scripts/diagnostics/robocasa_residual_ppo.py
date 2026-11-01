#!/usr/bin/env python
"""Allocated-Linux RoboCasa binding: frozen native pi0.5, matched residual PPO.

Run a prospectively bound headroom bank, a training arm or fixed evaluation.
Flora runs in its separate allocation. A missing label leaves pending.pt intact
with no GAE/update, rather than collecting a replacement on-policy batch.
"""
import argparse
from dataclasses import asdict
import json
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
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS, case_bank, protocol
from fluffy_bassoon.meta_rl.vla_history import save_dataset
from fluffy_bassoon.rl.action_residual import ResidualBound, action_std_from_norm_stats


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
    journal = ActionJournal(profile["accounting_root"], rank=0, world=1, attempt=profile["attempt"])
    progress = ProgressLog(output / "progress", profile["arm"], contract_sha256=digest(profile)).start()
    last_report = [0.]

    def report(phase, **fields):
        now = time.time()
        if now < last_report[0] + 20:
            return
        last_report[0] = now
        progress.emit(phase=phase, physical_actions=journal.charged(),
            this_open_job_elapsed_gpu_seconds=now - profile["allocated_start_unix"],
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
        if gate.get("go") is not True or gate.get("protocol") != protocol(task):
            raise ValueError("Native headroom and same-bank Flora AUROC gate required before learning")
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
            restored = restore_checkpoint(profile["resume"], policy=policy, optimizer=optimizer, bound=bound,
                motor=motor, environment=train_env, generators=generators, contract=contract)
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
        history = []
        while iteration < 10:
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
                ppo_diagnostics=metrics)
            iteration += 1
            history.append((iteration - 1, labeled))
            checkpoint = output / f"boundary-{iteration:03d}.pt"
            save_checkpoint(checkpoint, policy=policy, optimizer=optimizer, bound=bound, motor=motor,
                environment=train_env, generators=generators, contract=contract, iteration=iteration, stage="boundary")
            write_new(output / f"update-{iteration:03d}.json", dict(iteration=iteration, metrics=metrics,
                physical_actions=journal.charged(), checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint))))
            if iteration in (2, 4, 6, 8, 10):
                panel("DEV", iteration)
                # The owner combines complete same-iteration DEV panels only;
                # final ID/OOD never enter this decision.
                stop_path = Path(profile["paired_stop_root"]) / f"{iteration:03d}.json"
                while not stop_path.exists():
                    if time.time() + profile["closure_seconds"] >= profile["deadline_unix"]:
                        break
                    time.sleep(1)
                if not stop_path.exists() or json.loads(stop_path.read_text())["stop"]:
                    break
        save_checkpoint(output / "final.pt", policy=policy, optimizer=optimizer, bound=bound, motor=motor,
            environment=train_env, generators=generators, contract=contract, iteration=iteration, stage="boundary", final=True)
        write_new(output / "FINAL.json", dict(completed=True, iteration=iteration, checkpoint=str(output / "final.pt"),
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
