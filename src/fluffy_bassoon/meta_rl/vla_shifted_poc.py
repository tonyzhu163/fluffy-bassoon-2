"""Parallel-suite development experiment with the frozen RLinf LIBERO-130 motor.

Native/no-history/search and ordinary online DSRL controls run on predeclared
development variants. No untrained model is presented as a meta-controller.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

from .vla_libero import (OfficialSAC, SimulatorClient, append_json, run_episode,
                         sha256, write_json)


METHODS = ("native", "zero", "tied_gaussian", "search", "dsrl_online")


def validate_development_case(metadata, benchmark, tasks, horizon):
    origin = metadata.get("origin_base", {})
    key = origin.get("train_task_key")
    if (metadata.get("benchmark_kind") != benchmark or metadata.get("development") is not True
            or not metadata.get(f"{benchmark}_case") or key not in tasks
            or tasks[key]["split"] != "train" or not tasks[key]["teacher_collection_allowed"]
            or metadata.get("horizon") != horizon or metadata.get("initial_states", 0) < 1
            or not metadata.get("instruction") or not metadata.get("bddl_sha256")):
        raise ValueError("Require a predeclared development variant of a source-training family")


def replay_rows(rows):
    return [dict(observations=row["sac_observation"], next_observations=row["sac_next_observation"],
        actions=row["executed_noise"][None], next_actions=row["executed_noise"][None],
        rewards=np.float32(row["reward"]), masks=np.float32(not row["success"]),
        discount=np.float32(.999 ** row["primitive_steps"])) for row in rows]


class SearchController:
    """Try a frozen arm order until success, then repeat that successful arm."""
    def __init__(self, seed, attempts):
        rng = np.random.default_rng(seed)
        self.arms = np.concatenate((np.zeros((1, 32), np.float32),
            rng.uniform(-1., 1., (max(0, attempts - 1), 32)).astype(np.float32)))
        self.index, self.solved = 0, False

    def action(self, _):
        return self.arms[self.index].copy()

    def observe_episode(self, success):
        self.solved = self.solved or bool(success)
        if not self.solved:
            self.index = min(self.index + 1, len(self.arms) - 1)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--benchmark", choices=("plus", "pro"), required=True)
    p.add_argument("--ports", type=int, nargs="+", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rlinf-root", type=Path, required=True)
    p.add_argument("--openpi-root", type=Path, required=True)
    p.add_argument("--dsrl-root", type=Path, required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--utd", type=int, default=20)
    p.add_argument("--warmup-chunks", type=int, default=32)
    p.add_argument("--max-primitive-steps", type=int, default=11200)
    p.add_argument("--frozen-split", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Run robot computation and tests only on allocated Linux")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    if (not args.ports or len(set(args.ports)) != len(args.ports)
            or min(args.horizon, args.chunk, args.attempts, args.batch_size, args.utd) < 1
            or args.warmup_chunks < 0 or not 0 <= args.seed < 2**32
            or len(args.ports) * len(METHODS) * args.attempts * args.horizon > args.max_primitive_steps):
        raise ValueError("Invalid complete-episode experiment or explicit interaction budget")
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require a clean, exact reviewed run source")
    from .vla_cnd_corpus import load_frozen_split
    split_path = args.frozen_split or root / "outputs/libero-pi05-corpus-20261029-r1/split.json"
    split, tasks = load_frozen_split(split_path)
    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    lock = (args.output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    clients, episodes = [], []
    started, stage, primitive_steps, settling_steps = time.monotonic(), "connecting", 0, 0
    try:
        for port in args.ports:
            sim = SimulatorClient(port)
            clients.append(sim)
            validate_development_case(sim.metadata, args.benchmark, tasks, args.horizon)
        args.model_config, args.feature_interface = "rlinf_pi05_libero130", "none"
        from .vla_rlinf_libero import FrozenRLinfPi05Libero
        stage = "loading_rlinf130_motor"
        motor = FrozenRLinfPi05Libero(args)
        if args.chunk > motor.horizon:
            raise ValueError("Executed chunk exceeds the released motor horizon")
        contract = dict(source_commit=args.source_commit, benchmark=args.benchmark,
            frozen_split_sha256=split["metadata_sha256"], motor=motor.metadata,
            seed=args.seed, horizon=args.horizon, chunk=args.chunk, attempts=args.attempts,
            methods=list(METHODS), batch_size=args.batch_size, utd=args.utd,
            warmup_chunks=args.warmup_chunks, max_primitive_steps=args.max_primitive_steps,
            cases=[{key: value for key, value in sim.metadata.items() if key != "pid"} for sim in clients],
            source_policy="Official DSRL PixelSACLearner; fresh independent learner for each development case",
            deployment="Online DSRL updates between whole attempts; search uses only its own earlier success",
            scope="development variants only; no final test score and no meta-controller performance claim")
        write_json(args.output / "contract.json", contract)
        for case_index, sim in enumerate(clients):
            case_dir = args.output / f"case-{case_index:02d}"
            case_dir.mkdir()
            scene_seed = args.seed + case_index * 1000
            stage = f"case_{case_index}_motor_parity"
            observation = sim.reset(0, scene_seed)["observation"]
            settling_steps += int(sim.metadata.get("settle_steps", 0))
            zeros = np.zeros((motor.horizon, 32), np.float32)
            first = motor.infer(observation, sim.metadata["instruction"], zeros)
            second = motor.infer(observation, sim.metadata["instruction"], zeros.copy())
            delta = float(np.max(np.abs(first - second)))
            write_json(case_dir / "motor-noise-parity.json", dict(maximum_delta=delta, passed=delta <= 1e-5))
            if not np.isfinite(delta) or delta > 1e-5:
                raise RuntimeError("The released motor did not reproduce the same initial-noise input")
            for method in METHODS:
                stage = f"case_{case_index}_{method}"
                method_dir = case_dir / method
                method_dir.mkdir()
                rng = np.random.default_rng(scene_seed + 279_011)
                search = SearchController(scene_seed, args.attempts)
                learner = OfficialSAC(args.dsrl_root, args.seed) if method == "dsrl_online" else None
                sampler = np.random.default_rng(scene_seed)
                local_steps = 0
                if method == "search":
                    write_json(method_dir / "arms.json", dict(arms=search.arms.tolist(),
                        rule="zero then seeded uniform; after own success retain that arm"))
                for attempt in range(1, args.attempts + 1):
                    before = time.monotonic()
                    updates_before = 0 if learner is None else learner.updates
                    warming = learner is not None and len(learner.replay) < args.warmup_chunks
                    if method == "search":
                        choose = search.action
                    elif method in ("zero", "native"):
                        choose = lambda _: np.zeros(32, np.float32)
                    elif learner is not None and not warming:
                        choose = learner.action
                    else:
                        choose = lambda _: rng.standard_normal(32).astype(np.float32)
                    rows, result = run_episode(sim, motor, choose, initial_state=0,
                        scene_seed=scene_seed, chunk=args.chunk, native=method == "native", rng=rng,
                        bounded=method not in ("native", "tied_gaussian") and not warming,
                        record_features=False)
                    local_steps += int(result["primitive_steps"])
                    primitive_steps += int(result["primitive_steps"])
                    settling_steps += int(sim.metadata.get("settle_steps", 0))
                    metrics = {}
                    if learner is not None:
                        learner.replay.extend(replay_rows(rows))
                        if len(learner.replay) > args.warmup_chunks:
                            for _ in range(len(rows) * args.utd):
                                metrics = learner.update(args.batch_size, sampler)
                        learner.save(method_dir / "recovery.pkl", sampler, contract,
                            case_index=case_index, method=method, episodes=attempt, primitive_steps=local_steps,
                            rollout_noise_rng=rng.bit_generator.state)
                    if method == "search":
                        search.observe_episode(result["success"])
                    trace = method_dir / f"episode-{attempt:03d}.pkl.gz"
                    with gzip.open(trace, "xb", compresslevel=1) as stream:
                        pickle.dump(dict(rows=rows, summary=result), stream, protocol=5)
                    receipt = dict(result, method=method, split="development", attempt=attempt,
                        trial_seed=scene_seed, case_index=case_index, benchmark=args.benchmark,
                        cumulative_train_steps=0, cumulative_deployment_steps=local_steps,
                        source_updates_before=updates_before,
                        source_updates_after=0 if learner is None else learner.updates,
                        warmup=warming, metrics=metrics, trace_sha256=sha256(trace),
                        elapsed_seconds=time.monotonic() - before, completed=True)
                    episodes.append(receipt)
                    append_json(args.output / "episodes.jsonl", receipt)
                    write_json(args.output / "progress.json", dict(completed_episodes=len(episodes),
                        expected_episodes=len(clients) * len(METHODS) * args.attempts,
                        primitive_steps=primitive_steps, settling_steps=settling_steps, last_episode=receipt))
                    print(json.dumps(receipt), flush=True)
                    if primitive_steps > args.max_primitive_steps:
                        raise RuntimeError("Development experiment exceeded its declared interaction ceiling")
        write_json(args.output / "summary.json", dict(completed=True, benchmark=args.benchmark,
            episodes=len(episodes), primitive_steps=primitive_steps, settling_steps=settling_steps,
            total_simulator_steps=primitive_steps + settling_steps,
            elapsed_seconds=time.monotonic() - started, motor_calls=motor.calls,
            success_by_case_method={f"{index}/{method}": dict(
                successes=sum(bool(row["success"]) for row in episodes
                    if row["case_index"] == index and row["method"] == method), episodes=args.attempts)
                for index in range(len(clients)) for method in METHODS},
            scope=contract["scope"]))
        return 0
    except Exception as error:
        write_json(args.output / "failure.json", dict(stage=stage, error=repr(error),
            traceback=traceback.format_exc(), hostname=socket.gethostname(),
            completed_episodes=len(episodes), primitive_steps=primitive_steps))
        raise
    finally:
        for sim in clients:
            try:
                sim.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
