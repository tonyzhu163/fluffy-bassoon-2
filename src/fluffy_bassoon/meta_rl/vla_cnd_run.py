"""Allocated LIBERO CND proof of concept from an immutable DSRL checkpoint.

This run uses the already qualified frozen SAC visual bottleneck. Shared VLA
prefix features need separate numerical qualification. This same-task POC does
not establish semantic task OOD adaptation.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import numpy as np

from .vla_history import load_dataset
from .vla_libero import (FrozenOpenPI, OfficialSAC, SimulatorClient, append_json,
    evaluate, qualify_recovery, run_episode, sha256, structure_digest, train_history, write_json)
from .vla_noise_distillation import collect_noise_distillation


def validate_scene_banks(train, validation, calibration, evaluation, *, initial_states):
    banks = [set(bank) for bank in (train, validation, calibration, evaluation)]
    if (any(not bank or len(bank) != len(set(bank)) for bank in (train, validation, calibration, evaluation))
            or any(type(value) is not int or not 0 <= value < initial_states for bank in banks for value in bank)
            or any(a & b for i, a in enumerate(banks) for b in banks[i + 1:])):
        raise ValueError("CND train/validation/calibration/evaluation scene banks must be valid and disjoint")


def compatible_prior(prior, current):
    """Allow readable metadata additions, never task/motor/noise changes."""
    for key in ("motor", "noise", "dsrl_commit", "chunk", "horizon", "simulator"):
        before, after = prior[key], current[key]
        if key == "simulator":
            after = {name: value for name, value in after.items()
                     if name in before or name not in ("bddl_file", "problem_folder")}
        if before != after:
            raise ValueError(f"CND prior migration changed the scientific interface: {key}")


def learner_fingerprint(learner):
    from flax import serialization
    return structure_digest(dict(learning=serialization.to_state_dict(learner.agent._save_dict),
        updates=learner.updates, actor_rng=np.asarray(learner.agent._rng)))


def qualify_teacher(args, sim, motor, learner, checkpoint):
    """Fixed deterministic teacher, on a predeclared training calibration bank."""
    before_state = learner_fingerprint(learner)
    episodes, physical = [], 0
    for state in args.calibration_states:
        before = time.monotonic()
        _, receipt = run_episode(sim, motor, lambda obs: learner.action(obs, deterministic=True),
            initial_state=state, scene_seed=args.seed + state * 1000, chunk=args.chunk,
            bounded=True, record_features=False)
        physical += receipt["primitive_steps"]
        row = dict(**receipt, method="teacher_mean", split="calibration", attempt=1,
            trial_seed=args.seed + state * 1000, completed=True,
            elapsed_seconds=time.monotonic() - before, cumulative_train_steps=0)
        append_json(args.output / "teacher-calibration-episodes.jsonl", row)
        print(json.dumps(dict(teacher_calibration=row)), flush=True)
        episodes.append(row)
    successes = sum(row["success"] for row in episodes)
    after_state = learner_fingerprint(learner)
    result = dict(checkpoint_sha256=sha256(checkpoint), successes=successes,
        completed_episodes=len(episodes), success_rate=successes / len(episodes),
        primitive_steps=physical, deterministic=True, states=args.calibration_states,
        minimum_success_rate=args.min_teacher_success,
        qualified=successes / len(episodes) >= args.min_teacher_success,
        teacher_state_sha256=before_state, after_teacher_state_sha256=after_state,
        teacher_unchanged=before_state == after_state,
        settling_steps=len(episodes) * sim.metadata.get("settle_steps", 0),
        scope="teacher-training calibration scenes; held-out deployment scenes excluded")
    write_json(args.output / "teacher-qualification.json", result)
    if before_state != after_state:
        raise RuntimeError("Deterministic teacher changed during competence qualification")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--suite", default="libero_90")
    p.add_argument("--task-id", type=int, default=57)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--chunk", type=int, default=20)
    p.add_argument("--openpi-root", type=Path, required=True)
    p.add_argument("--dsrl-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-config", choices=("pi0_libero", "pi05_libero"), default="pi0_libero")
    p.add_argument("--feature-interface", choices=("sac",), default="sac")
    p.add_argument("--prior-recovery", type=Path, required=True)
    p.add_argument("--prior-contract", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--utd", type=int, default=20)
    p.add_argument("--train-states", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--validation-states", type=int, nargs="+", default=[36, 37])
    p.add_argument("--calibration-states", type=int, nargs="+", default=list(range(24, 32)))
    p.add_argument("--eval-states", type=int, nargs="+", default=[40, 41, 42, 43])
    p.add_argument("--min-teacher-success", type=float, default=.7)
    p.add_argument("--min-episodes", type=int, default=4)
    p.add_argument("--max-episodes", type=int, default=6)
    p.add_argument("--anneal-decisions", type=int, nargs=2, default=[35, 55])
    p.add_argument("--anneal-powers", type=float, nargs="+", default=[.5, 1., 2.])
    p.add_argument("--max-collection-steps", type=int, default=14400)
    p.add_argument("--history-updates", type=int, default=500)
    p.add_argument("--history-device", default="cuda")
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--online-dsrl", action="store_true")
    p.add_argument("--qualify-only", action="store_true",
                   help="Measure fixed teacher competence and stop before CND collection/training")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Run CND robot computation on allocated Linux only")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    if (not math.isfinite(args.min_teacher_success) or not 0 <= args.min_teacher_success <= 1
            or len(args.calibration_states) < 8 or args.attempts < 1 or args.history_updates < 1):
        raise ValueError("Require predeclared teacher qualification and positive train/evaluation budgets")
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (source != args.source_commit or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require exact clean reviewed CND source")
    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    lock = (args.output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    np.random.seed(args.seed)
    random.seed(args.seed)
    sim, started = SimulatorClient(args.port), time.monotonic()
    stage = "initializing"
    try:
        validate_scene_banks(args.train_states, args.validation_states, args.calibration_states,
                             args.eval_states, initial_states=sim.metadata["initial_states"])
        if (args.suite != sim.metadata["suite"] or args.task_id != sim.metadata["task_id"]
                or args.horizon != sim.metadata["horizon"]):
            raise ValueError("Requested CND task/horizon differs from the actual simulator")
        prior = json.loads(args.prior_contract.read_text())
        motor = FrozenOpenPI(args)
        contract = dict(prior, source_commit=source, chunk=args.chunk, horizon=args.horizon,
            simulator={key: value for key, value in sim.metadata.items() if key != "pid"}, motor=motor.metadata)
        compatible_prior(prior, contract)
        contract["cnd_poc"] = dict(seed=args.seed, train_states=args.train_states, validation_states=args.validation_states,
            calibration_states=args.calibration_states, evaluation_states=args.eval_states,
            teacher_minimum_success=args.min_teacher_success, min_episodes=args.min_episodes,
            max_episodes=args.max_episodes, anneal_decisions=args.anneal_decisions,
            anneal_powers=args.anneal_powers, max_collection_steps=args.max_collection_steps,
            history_updates=args.history_updates, evaluation_attempts=args.attempts,
            evaluation_online_dsrl=args.online_dsrl, online_batch_size=args.batch_size, online_utd=args.utd,
            qualification_only=args.qualify_only)
        learner = OfficialSAC(args.dsrl_root, args.seed)
        sampler = np.random.default_rng(args.seed)
        progress = learner.restore(args.prior_recovery, sampler, prior)
        prior_cost = int(progress["primitive_steps"])
        migration = dict(original_source_commit=prior["source_commit"], current_source_commit=source,
            original_checkpoint_sha256=sha256(args.prior_recovery),
            original_contract_sha256=sha256(args.prior_contract), prior_primitive_steps=prior_cost,
            previous_migration=progress.get("migration"), weights_unchanged=True)
        learner.save(args.output / "source-recovery.pkl", sampler, contract, **dict(progress, migration=migration))
        write_json(args.output / "prior-migration.json", migration)
        write_json(args.output / "contract.json", contract)
        stage = "source_recovery_qualification"
        obs = sim.reset(args.calibration_states[0], args.seed + 100000)["observation"]
        # Allocate/attest actual motor context before recovery checks and rollouts.
        motor.infer(obs, sim.metadata["instruction"], np.zeros((motor.horizon, 32), np.float32))
        write_json(args.output / "source-recovery-check.json", qualify_recovery(args, learner, contract, obs))
        write_json(args.output / "feature-parity.json", learner.qualify_features(obs))
        stage = "teacher_qualification"
        teacher = qualify_teacher(args, sim, motor, learner, args.output / "source-recovery.pkl")
        if args.qualify_only:
            write_json(args.output / "summary.json", dict(completed=True, qualification_only=True,
                teacher_qualified=teacher["qualified"], teacher=teacher, prior_training_steps=prior_cost,
                elapsed_seconds=time.monotonic() - started,
                scope="fixed teacher competence only; no CND histories or student training"))
            return 0 if teacher["qualified"] else 2
        if not teacher["qualified"]:
            write_json(args.output / "summary.json", dict(completed=False, stage=stage,
                reason="fixed teacher below predeclared competence threshold; no CND data/training ran",
                teacher=teacher, prior_training_steps=prior_cost,
                elapsed_seconds=time.monotonic() - started))
            print(json.dumps(dict(teacher_not_qualified=teacher)), flush=True)
            return 2
        if motor.feature_encoder is not None:
            write_json(args.output / "common-feature-parity.json",
                motor.feature_encoder.qualify(obs, sim.metadata["instruction"]))
            feature_contract = motor.feature_encoder.contract
        else:
            feature_contract = dict(kind="fixed qualified SAC actor bottleneck50 + public robot state8",
                                    teacher_checkpoint_sha256=sha256(args.output / "source-recovery.pkl"))
        contract.update(feature_contract=feature_contract,
                        feature_dim=58 if motor.feature_encoder is None else 4104,
                        method="fixed-teacher CND steering proof of concept")
        # Save the exact unchanged source under the final local contract used by online references.
        learner.save(args.output / "source-recovery.pkl", sampler, contract, **dict(progress, migration=migration))
        teacher["checkpoint_sha256"] = sha256(args.output / "source-recovery.pkl")
        write_json(args.output / "contract.json", contract)
        write_json(args.output / "teacher-qualification.json", teacher)
        scenes = [dict(trial_id=f"{split}-state-{state}", split=split, initial_state=state,
            scene_seed=args.seed + state * 1000) for split, states in
            (("train", args.train_states), ("validation", args.validation_states)) for state in states]
        def fingerprint():
            return learner_fingerprint(learner)
        stage = "cnd_collection"
        paths = collect_noise_distillation(teacher_action=lambda obs: learner.action(obs, deterministic=True),
            teacher_fingerprint=fingerprint, teacher_checkpoint=args.output / "source-recovery.pkl",
            teacher_steps=learner.updates, teacher_primitive_steps=prior_cost + teacher["primitive_steps"],
            teacher_evaluation=teacher, motor_sha256=contract["motor"]["receipt_sha256"],
            feature_contract=feature_contract, noise_contract=contract["noise"], scenes=scenes,
            output_dir=args.output / "cnd-data", min_episodes=args.min_episodes,
            max_episodes=args.max_episodes, episode_horizon=args.horizon, chunk=args.chunk,
            anneal_decisions=args.anneal_decisions, anneal_powers=args.anneal_powers, seed=args.seed,
            max_primitive_steps=args.max_collection_steps, heldout_initial_states=args.eval_states,
            min_teacher_success_rate=args.min_teacher_success, min_teacher_evaluation_episodes=8,
            sim=sim, motor=motor, row_feature_encoder=(
                None if motor.feature_encoder is not None else lambda row: learner.features(row["sac_observation"])))
        _, dataset = load_dataset(paths["train"], expected_sha256=sha256(paths["train"]))
        total_source = dataset["metadata"]["total_source_primitive_steps"]
        stage = "student_training"
        args.history_objective = "cnd"
        student = train_history(args, learner, contract, paths=paths, artifact_prefix="cnd-history", objective="cnd")
        stage = "paired_robot_evaluation"
        evaluation = evaluate(args, sim, motor, ["native", "dsrl"], learner=learner, train_steps=prior_cost)
        if args.online_dsrl:
            evaluation.update(evaluate(args, sim, motor, ["dsrl_online"], learner=learner,
                                       train_steps=prior_cost, contract=contract))
        evaluation.update(evaluate(args, sim, motor, ["history", "reset", "static"],
            learner=learner, student=student, train_steps=total_source, label_prefix="cnd_"))
        write_json(args.output / "summary.json", dict(completed=True, evaluation=evaluation,
            prior_training_steps=prior_cost, teacher_calibration_steps=teacher["primitive_steps"],
            total_source_steps=total_source, elapsed_seconds=time.monotonic() - started,
            motor_calls=motor.calls, scope="same-task held-out scenes; not semantic task OOD"))
        from .vla_performance_report import load_episodes, plot_curves, report
        rows = load_episodes(args.output / "episodes.jsonl")
        write_json(args.output / "performance.json", report(rows))
        plot_curves(rows, args.output / "success-by-attempt.png")
        return 0
    except Exception as error:
        write_json(args.output / "failure.json", dict(stage=stage, error=repr(error),
            hostname=socket.gethostname(), elapsed_seconds=time.monotonic() - started))
        raise
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
