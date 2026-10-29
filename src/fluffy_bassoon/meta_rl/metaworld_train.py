"""Bounded state-based Meta-World PPO, with measured GPU-hour curves."""

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from fluffy_bassoon.evaluation_stages import write_json
from .checkpoint import restore_checkpoint, save_checkpoint
from .metaworld_env import benchmark_pools, make_lanes, task_digest
from .metaworld_policy import make_policy, match_policy
from .metaworld_workers import make_worker_lanes
from .ppo import collect_trials, replay_windows, update_ppo
from .screen import allocation, clean_source


def core_settings(core):
    if core == "persistent_ctm":
        return dict(ticks=1, history_length=32, synapse_depth=8,
                    bounded_synchronisation=True, decay_per_observation=True)
    if core == "transformer_xl":
        return dict(ticks=1, memory_length=64, layers=2, heads=4)
    if core == "gru":
        return dict(ticks=1)
    raise ValueError("Unknown benchmark core")


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


@contextmanager
def preserve_random_streams():
    from .checkpoint import rng_state, restore_rng
    saved = rng_state()
    try:
        yield
    finally:
        restore_rng(saved)


@torch.no_grad()
def replay_error(policy, rollout, replay_chunk_length=0):
    data, state, _ = replay_windows(rollout, replay_chunk_length)
    log_prob, _, values = policy.evaluate_sequence(data["observations"],
        data["raw_actions"], data["resets"], initial_state=state)
    return dict(log_probability=float((log_prob - data["log_probs"]).abs().max()),
                value=float((values - data["values"]).abs().max()))


@torch.no_grad()
def evaluate(policy, pools, benchmark, *, attempts, horizon, lane_count,
             conditions=("normal",), seed_start=1_000_000_000, training_attempts=None,
             env_workers=0, allocated_cpus=16):
    records = {}
    with preserve_random_streams():
        for split in ("train_eval", "test"):
            factory = make_worker_lanes if env_workers else make_lanes
            worker_options = dict(workers=env_workers, allocated_cpus=allocated_cpus) if env_workers else {}
            envs = factory(pools[split], lane_count, benchmark=benchmark,
                           attempts=attempts, horizon=horizon, **worker_options)
            try:
                records[split] = {}
                for condition in conditions:
                    collect_trials(policy, envs, np.arange(seed_start, seed_start + len(envs)),
                                   deterministic=True, reset_each_attempt=condition == "memory_reset",
                                   reward_blind=condition == "reward_blind",
                                   clock_scale=(None if training_attempts is None else
                                                np.array([max(training_attempts - 1, 1), horizon])))
                    outcomes = np.asarray([env.outcomes for env in envs], dtype=bool)
                    returns = np.asarray([env.native_returns for env in envs])
                    families = sorted({env.family for env in envs})
                    by_family = {name: outcomes[[e.family == name for e in envs]].mean(0).tolist()
                                 for name in families}
                    curve = np.asarray(list(by_family.values())).mean(0)
                    records[split][condition] = dict(success=float(curve.mean()),
                        success_by_episode=curve.tolist(), per_family=by_family,
                        outcomes=outcomes.tolist(), native_returns=returns.tolist(),
                        family_by_lane=[env.family for env in envs], trials=len(envs),
                        environment_steps=len(envs) * attempts * horizon)
            finally:
                for env in envs:
                    env.close()
    return records


def parser(*, core_choices=("persistent_ctm", "gru", "transformer_xl")):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=("ml1-push", "ml10"), required=True)
    p.add_argument("--core", choices=core_choices, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--benchmark-seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=50)
    p.add_argument("--attempts", type=int, default=5)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--updates", type=int, default=80)
    p.add_argument("--parameter-budget", type=int, default=250_000)
    p.add_argument("--ctm-ticks", type=int, default=1)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--minibatch-trials", type=int, default=10)
    p.add_argument("--replay-chunk-length", type=int, default=0,
                   help="Zero: full-trial gradients. Positive divisor of trial length: cached-state TBPTT.")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-attempts", type=int, default=25)
    p.add_argument("--eval-lanes", type=int, default=10)
    p.add_argument("--max-elapsed-seconds", type=float, default=900.)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--env-workers", type=int, default=0, help="Zero: serial simulator reference; positive: CPU worker groups.")
    p.add_argument("--allocated-cpus", type=int, default=16)
    p.add_argument("--resume", type=Path)
    p.add_argument("--qualify-one-update", action="store_true",
                   help="Throughput qualification: exactly one complete update without evaluation.")
    return p


def main(argv=None, *, learner=None):
    argument_parser = (parser(core_choices=(learner["core_name"],))
                       if learner is not None and "core_name" in learner else parser())
    if learner is not None:
        argument_parser.set_defaults(**learner.get("defaults", {}))
    args = argument_parser.parse_args(argv)
    if args.qualify_one_update and (args.updates != 1 or args.resume is not None):
        raise ValueError("Throughput qualification requires one update and no resume")
    evidence = allocation()
    source = clean_source(args.source_commit)
    positive = (args.num_envs, args.attempts, args.horizon, args.updates, args.parameter_budget,
                args.ppo_epochs, args.minibatch_trials, args.eval_every, args.eval_attempts,
                args.eval_lanes, args.threads, args.ctm_ticks, args.allocated_cpus)
    if (any(v < 1 for v in positive) or args.horizon > 500
            or args.replay_chunk_length < 0 or args.env_workers < 0
            or args.replay_chunk_length and args.attempts * args.horizon % args.replay_chunk_length
            or not np.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0
            or not 0 <= args.seed < 1_000_000_000 or not 0 <= args.benchmark_seed < 1_000_000):
        raise ValueError("Invalid bounded benchmark settings")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pools = benchmark_pools(args.benchmark, args.benchmark_seed)
    factory = make_worker_lanes if args.env_workers else make_lanes
    worker_options = dict(workers=args.env_workers, allocated_cpus=args.allocated_cpus) if args.env_workers else {}
    envs = factory(pools["train"], args.num_envs, benchmark=args.benchmark,
                   attempts=args.attempts, horizon=args.horizon, **worker_options)
    settings = core_settings(args.core) if learner is None else dict(learner["settings"])
    if learner is None and args.core == "persistent_ctm":
        settings["ticks"] = args.ctm_ticks
    matcher = match_policy if learner is None else learner["match"]
    config, size = (matcher(args.core, settings, envs[0], args.parameter_budget) if learner is None else
                    matcher(settings, envs[0], args.parameter_budget))
    # Width search must not choose the actual learner's initialization stream.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = (make_policy(args.core, config, envs[0]) if learner is None else
              learner["make"](config, envs[0])).to(args.device)
    optimizer_options = dict(lr=3e-4, eps=1e-8)
    if learner is not None:
        optimizer_options.update(learner.get("optimizer", {}))
    optimizer = torch.optim.Adam(policy.parameters(), **optimizer_options)
    sampler = np.random.default_rng(args.seed)
    optimization = dict(epochs=args.ppo_epochs, minibatch_trials=args.minibatch_trials,
                        clip=.2, gamma=.99, gae_lambda=.95, entropy_coef=0., value_coef=.5,
                        max_grad_norm=.5, normalize_advantages=True,
                        replay_chunk_length=args.replay_chunk_length)
    if learner is not None:
        optimization.update(learner.get("optimization", {}))
    protocol = ("metaworld_state_cached_chunk_ppo_v2" if args.replay_chunk_length else
                "metaworld_state_full_trial_ppo_v1")
    if learner is not None:
        protocol = learner.get("protocol", protocol)
    contract = dict(protocol=protocol, source_commit=source,
        benchmark=args.benchmark, benchmark_seed=args.benchmark_seed, seed=args.seed,
        pool_hashes={key: task_digest(pool[1]) for key, pool in pools.items()},
        goal_visible=args.benchmark == "ml10", resample_instance=args.benchmark == "ml10",
        reward="native dense reward multiplied by 0.1; success scored separately",
        observation="state, previous action/reward, episode boundary, normalized public clock",
        normalization="running state moments updated only after each complete PPO update",
        actor_critic="current normalized state plus context; separate tanh MLPs 64,64",
        core=args.core, core_config=config, parameter_budget=args.parameter_budget,
        attempts=args.attempts, horizon=args.horizon, num_envs=args.num_envs,
        optimization=optimization, device=args.device, dtype="float32",
        optimizer=dict(name="Adam", **optimizer_options),
        evaluation=dict(attempts=args.eval_attempts, lanes=args.eval_lanes,
                        seed_start=1_000_000_000, action_selection="deterministic",
                        clock_scale=[max(args.attempts - 1, 1), args.horizon],
                        success="any success within episode; average episodes and families equally"),
        throughput_qualification_only=args.qualify_one_update,
        versions={name: importlib.metadata.version(name) for name in
                  ("torch", "numpy", "gymnasium", "mujoco", "metaworld")})
    if learner is not None:
        method = learner["contract"](config)
        for key in ("normalization", "actor_critic", "action_distribution", "observation", "reward"):
            if key in method:
                contract[key] = method[key]
        contract["method"] = dict(name=learner["name"], **method)
    update = update_ppo if learner is None else learner.get("update", update_ppo)
    counters = dict(environment_steps=0, optimizer_steps=0, evaluation_steps=0,
                    training_seconds=0., evaluation_seconds=0., cumulative_elapsed_seconds=0.)
    iteration = 0
    if args.resume:
        iteration, counters = restore_checkpoint(args.resume, policy=policy, optimizer=optimizer,
                                                 environment_rng=sampler, contract=contract)
    parent_elapsed = counters["cumulative_elapsed_seconds"]
    if iteration >= args.updates:
        raise ValueError("Target updates must exceed the restored iteration")
    write_json(args.output_dir / "config.json", dict(contract=contract, allocation=evidence,
        **size, target_updates=args.updates,
        execution=dict(env_workers=args.env_workers, allocated_cpus=args.allocated_cpus,
                       worker_backend=envs[0]._worker_batch.manifest if args.env_workers else "serial"),
        target_environment_steps=args.updates * args.num_envs * args.attempts * args.horizon,
        training_transitions_per_collection=args.num_envs * args.attempts * args.horizon,
        max_elapsed_seconds=args.max_elapsed_seconds,
        eval_every=args.eval_every, threads=args.threads,
        resume=str(args.resume) if args.resume else None,
        accounting="one allocated GPU; all active run wall time, including CPU simulation, is charged"))
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    def snapshot():
        synchronize(args.device)
        counters["cumulative_elapsed_seconds"] = parent_elapsed + time.monotonic() - started
        return save_checkpoint(args.output_dir / "latest.pt", policy=policy, optimizer=optimizer,
                               environment_rng=sampler, iteration=iteration,
                               counters=counters, contract=contract)

    def evaluation(conditions=("normal",)):
        before = time.monotonic()
        result = evaluate(policy, pools, args.benchmark, attempts=args.eval_attempts,
                          horizon=args.horizon, lane_count=args.eval_lanes, conditions=conditions,
                          training_attempts=args.attempts, env_workers=args.env_workers,
                          allocated_cpus=args.allocated_cpus)
        synchronize(args.device)
        counters["evaluation_seconds"] += time.monotonic() - before
        counters["evaluation_steps"] += sum(v["environment_steps"] for split in result.values()
                                             for v in split.values())
        return result

    def record(log, **values):
        synchronize(args.device)
        elapsed = parent_elapsed + time.monotonic() - started
        row = dict(iteration=iteration, environment_steps=counters["environment_steps"],
                   elapsed_seconds=elapsed, gpu_hours=elapsed / 3600 if args.device == "cuda" else None,
                   training_gpu_hours=counters["training_seconds"] / 3600 if args.device == "cuda" else None,
                   **values)
        log.write(json.dumps(row, allow_nan=False) + "\n")
        # Raw evaluation arrays remain in the file, keeping console output small.
        compact = {key: value for key, value in row.items() if key != "evaluation"}
        if "evaluation" in row:
            compact["success"] = {key: {c: x["success"] for c, x in split.items()}
                                  for key, split in row["evaluation"].items()}
        print(json.dumps(compact, allow_nan=False), flush=True)

    last_evaluation = None
    try:
        snapshot()
        with (args.output_dir / "metrics.jsonl").open("x", buffering=1) as log:
            if not args.qualify_one_update:
                last_evaluation = evaluation()
                record(log, evaluation=last_evaluation)
            while (iteration < args.updates and not stop
                   and time.monotonic() - started < args.max_elapsed_seconds):
                before = time.monotonic()
                seeds = sampler.integers(0, 1_000_000_000, size=args.num_envs)
                collection_breakdown = {}
                rollout = collect_trials(policy, envs, seeds, replay_chunk_length=args.replay_chunk_length,
                                         timing=collection_breakdown)
                synchronize(args.device)
                collection = time.monotonic() - before
                # Reconstruct one full trial before the first optimizer step.
                # Normalization and recurrence must match behavior likelihoods.
                error = None
                if iteration == 0:
                    first = (rollout if args.replay_chunk_length else
                             {key: value[:, :1] for key, value in rollout.items()})
                    error = replay_error(policy, first, args.replay_chunk_length)
                    if max(error.values()) > 2e-5:
                        raise RuntimeError(f"Behavior replay differs: {error}")
                update_started = time.monotonic()
                metrics = update(policy, optimizer, rollout, **optimization)
                policy.core.observe_after_update(rollout["observations"])
                synchronize(args.device)
                update_seconds = time.monotonic() - update_started
                counters["training_seconds"] += time.monotonic() - before
                counters["environment_steps"] += rollout["rewards"].numel()
                counters["optimizer_steps"] += metrics["optimizer_steps"]
                iteration += 1
                # Every completed update has a full optimizer/sampler/RNG snapshot.
                snapshot()
                result = dict(training=metrics, behavior_replay_error=error,
                    rollout_success=float(np.mean([env.outcomes for env in envs])),
                    timing=dict(rollout_seconds=collection, update_seconds=update_seconds,
                                collection_breakdown=collection_breakdown))
                if not args.qualify_one_update and (iteration % args.eval_every == 0 or iteration == args.updates):
                    last_evaluation = evaluation()
                    result["evaluation"] = last_evaluation
                record(log, **result)
            if not args.qualify_one_update:
                last_evaluation = evaluation(("normal", "memory_reset", "reward_blind"))
                record(log, final=True, evaluation=last_evaluation)
        digest = snapshot()
        write_json(args.output_dir / "summary.json", dict(completed=iteration == args.updates,
            iteration=iteration, source_commit=source, counters=counters,
            checkpoint_sha256=digest, last_evaluation=last_evaluation,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0))
    except BaseException as exc:
        # Preserve the preceding complete update; do not label partial optimizer work resumable.
        write_json(args.output_dir / "failure.json", dict(error=type(exc).__name__, message=str(exc),
            last_complete_iteration=iteration, checkpoint=str(args.output_dir / "latest.pt")))
        raise
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
