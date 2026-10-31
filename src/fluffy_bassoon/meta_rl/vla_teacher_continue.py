"""Continue a complete DSRL teacher in a fresh immutable LIBERO run.

Reuse the original source learner, optimizer, replay, sampler and RNG. The VLA
motor remains frozen. This is extra teacher training cost, not student ICRL.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

from .vla_cnd_run import compatible_prior, qualify_teacher
from .vla_libero import (FrozenOpenPI, OfficialSAC, SimulatorClient, qualify_recovery,
                         sha256, source_train, write_json)


def continuation_target(prior_steps, additional_max, horizon):
    """Reserve a whole final episode so rounding cannot exceed the new budget."""
    if (any(type(value) is not int for value in (prior_steps, additional_max, horizon))
            or prior_steps < 0 or horizon < 1 or additional_max < horizon):
        raise ValueError("Require nonnegative prior cost and room for a complete final episode")
    return prior_steps + additional_max - horizon + 1


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
    p.add_argument("--model-config", choices=("pi0_libero",), default="pi0_libero")
    p.add_argument("--feature-interface", choices=("sac",), default="sac")
    p.add_argument("--prior-recovery", type=Path, required=True)
    p.add_argument("--prior-contract", type=Path, required=True)
    p.add_argument("--additional-max-steps", type=int, default=20000)
    p.add_argument("--calibration-states", type=int, nargs="+", default=list(range(24, 32)))
    p.add_argument("--min-teacher-success", type=float, default=.7)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Teacher continuation requires allocated Linux resources")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require exact clean reviewed continuation source")
    continuation_target(0, args.additional_max_steps, args.horizon)
    if (len(args.calibration_states) < 8 or len(set(args.calibration_states)) != len(args.calibration_states)
            or not 0 <= args.min_teacher_success <= 1):
        raise ValueError("Require predeclared teacher competence on at least eight calibration scenes")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "traces").mkdir()
    import fcntl
    lock = (args.output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started, stage = time.monotonic(), "initializing"
    sim = SimulatorClient(args.port)
    try:
        prior = json.loads(args.prior_contract.read_text())
        if (args.seed != prior["seed"] or sim.metadata["suite"] != args.suite or sim.metadata["task_id"] != args.task_id
                or sim.metadata["horizon"] != args.horizon
                or any(not 0 <= state < prior["train_states"] for state in args.calibration_states)):
            raise ValueError("Continuation task/horizon/calibration bank differs from original teacher interface")
        # Keep the original optimization and sampling contract exactly.
        args.train_states, args.warmup_chunks = prior["train_states"], prior["warmup_chunks"]
        args.batch_size, args.utd = prior["batch_size"], prior["utd"]
        motor = FrozenOpenPI(args)
        contract = dict(prior, source_commit=args.source_commit, chunk=args.chunk, horizon=args.horizon,
            simulator={key: value for key, value in sim.metadata.items() if key != "pid"}, motor=motor.metadata)
        compatible_prior(prior, contract)
        learner, sampler = OfficialSAC(args.dsrl_root, args.seed), np.random.default_rng(args.seed)
        progress = learner.restore(args.prior_recovery, sampler, prior)
        prior_steps = int(progress["primitive_steps"])
        args.max_steps = continuation_target(prior_steps, args.additional_max_steps, args.horizon)
        migration = dict(original_source_commit=prior["source_commit"], current_source_commit=args.source_commit,
            original_checkpoint_sha256=sha256(args.prior_recovery), original_contract_sha256=sha256(args.prior_contract),
            prior_primitive_steps=prior_steps, prior_episodes=progress["episodes"], prior_updates=learner.updates,
            additional_primitive_step_ceiling=args.additional_max_steps,
            stop_at_complete_episode_after_total_steps=args.max_steps,
            optimizer_replay_sampler_rng_preserved=True, previous_migration=progress.get("migration"))
        contract["continuation"] = migration
        args.resume = args.output / "source-recovery.pkl"
        learner.save(args.resume, sampler, contract, **dict(progress, migration=migration))
        write_json(args.output / "contract.json", contract)
        write_json(args.output / "prior-migration.json", migration)
        stage = "source_recovery_qualification"
        obs = sim.reset(args.calibration_states[0], args.seed + 100000)["observation"]
        motor.infer(obs, sim.metadata["instruction"], np.zeros((motor.horizon, 32), np.float32))
        write_json(args.output / "source-recovery-check.json", qualify_recovery(args, learner, contract, obs))
        stage = "teacher_continuation"
        # source_train constructs its own learner and restores our migrated full
        # checkpoint before any scene sampling or gradient update.
        learner = source_train(args, sim, motor, contract)
        final = json.loads((args.output / "progress.jsonl").read_text().splitlines()[-1])
        total_steps = int(final["primitive_steps_total"])
        if not 0 < total_steps - prior_steps <= args.additional_max_steps:
            raise RuntimeError("Teacher continuation exceeded its explicit interaction ceiling")
        write_json(args.output / "feature-parity.json", learner.qualify_features(obs))
        stage = "teacher_qualification"
        teacher = qualify_teacher(args, sim, motor, learner, args.resume)
        summary = dict(completed=True, teacher_qualified=teacher["qualified"], teacher=teacher,
            prior_training_steps=prior_steps, additional_training_steps=total_steps - prior_steps,
            total_teacher_training_steps=total_steps, teacher_calibration_steps=teacher["primitive_steps"],
            source_updates=learner.updates, source_checkpoint_sha256=sha256(args.resume),
            source_optimizer=contract["dsrl_commit"], elapsed_seconds=time.monotonic() - started,
            scope="ordinary DSRL teacher continuation; frozen VLA; no held-out deployment data")
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return 0 if teacher["qualified"] else 2
    except Exception as error:
        write_json(args.output / "failure.json", dict(stage=stage, error=repr(error),
            hostname=socket.gethostname(), elapsed_seconds=time.monotonic() - started))
        raise
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
