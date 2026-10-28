"""DPT-inspired chronological pretraining: imperfect contexts, separate expert labels."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from gymnasium.spaces.utils import flatten

from fluffy_bassoon.evaluation_stages import sha256, write_json
from .checkpoint import restore_checkpoint, save_checkpoint
from .cores import build_policy
from .maniskill_rotation import (CachedRotationTrial, OutcomeCache, PROTOCOL, evaluate_policy,
                                validate_cpu_execution)
from .teachers import FeedbackTeacher, TEACHER_VERSION, DPT_LABEL_VERSION, expert_action, rotation_teacher_rates
from .train import BANDITS, DEFAULT_HORIZONS, evaluate_trials, make_env
from .warmstart import COMPACT, CACHED, export_warmstart


def generate_histories(factory, task, seeds, *, arms=2, rotation_rates=None, objective="dpt"):
    """Private task state is used only for labels. One column is a full causal trial."""
    if objective not in ("dpt", "behavior_cloning"):
        raise ValueError("Unknown supervised objective")
    observations, actions, labels, rewards = [], [], [], []
    for seed in seeds:
        env = factory()
        try:
            teacher = FeedbackTeacher(task, seed=int(seed) + 2**32, arms=arms,
                                      rotation_rates=rotation_rates)
            behavior_rng = np.random.default_rng(int(seed) + 2**33)
            label_rng = np.random.default_rng(int(seed) + 2**34)
            # Fixed per trial, not annealed with the attempt clock.
            random_fraction = float(behavior_rng.uniform())
            observation, _ = env.reset(seed=int(seed))
            trial_obs, trial_actions, trial_labels, trial_rewards = [], [], [], []
            for step in range(env.attempts * env.horizon):
                encoded = dict(observation, clock=observation["clock"] / np.maximum(
                    env.observation_space["clock"].high, 1))
                trial_obs.append(flatten(env.observation_space, encoded).astype(np.float32))
                action = teacher.action(observation)
                if behavior_rng.random() < random_fraction:
                    action = (behavior_rng.uniform(-.999, .999, 2).astype(np.float32) if task == "reach"
                              else int(behavior_rng.integers(env.action_space.n)))
                teacher.last_action = action
                label = (expert_action(env, task, observation, rng=label_rng, rotation_rates=rotation_rates)
                         if objective == "dpt" else action)
                trial_actions.append(np.arctanh(action) if task == "reach" else action)
                trial_labels.append(np.arctanh(label) if task == "reach" else label)
                observation, reward, done, truncated, _ = env.step(action)
                trial_rewards.append(reward)
                if truncated or done != (step == env.attempts * env.horizon - 1):
                    raise ValueError("Teacher collection requires complete fixed-length trials")
            observations.append(trial_obs)
            actions.append(trial_actions)
            labels.append(trial_labels)
            rewards.append(trial_rewards)
        finally:
            env.close()
    return dict(observations=np.asarray(observations, dtype=np.float32).swapaxes(0, 1),
                raw_actions=np.asarray(actions, dtype=np.float32 if task == "reach" else np.int64).swapaxes(0, 1),
                target_actions=np.asarray(labels, dtype=np.float32 if task == "reach" else np.int64).swapaxes(0, 1),
                rewards=np.asarray(rewards, dtype=np.float32).swapaxes(0, 1), seeds=np.asarray(seeds))


def supervised_loss(policy, data, lanes):
    device = next(policy.parameters()).device
    observations = torch.as_tensor(data["observations"][:, lanes], device=device)
    actions = torch.as_tensor(data["target_actions"][:, lanes], device=device)
    # The policy replays from fresh state and retains the graph across every attempt.
    logs, _, _ = policy.evaluate_sequence(observations, actions)
    return -logs.mean()


def update_supervised(policy, optimizer, data, lanes):
    policy.eval()  # Keep recurrent dropout off, with autograd on.
    loss = supervised_loss(policy, data, lanes)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite teacher imitation loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    constrain = getattr(policy.core, "constrain_parameters", None)
    if constrain:
        constrain()
    if any(not p.isfinite().all() for p in policy.parameters()):
        raise RuntimeError("Non-finite warm-start parameter")
    return dict(loss=float(loss.detach()), gradient_norm=float(norm))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=(*BANDITS, "reach", "darkroom", "keydoor", "cached_rotation"), required=True)
    p.add_argument("--core", choices=("persistent_ctm", "transformer_xl", "gru"), default="persistent_ctm")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--ticks", type=int)
    p.add_argument("--memory-length", type=int, default=16)
    p.add_argument("--attempts", type=int, default=8)
    p.add_argument("--horizon", type=int)
    p.add_argument("--arms", type=int, default=2)
    p.add_argument("--objective", choices=("dpt", "behavior_cloning"), default="dpt")
    p.add_argument("--cache", type=Path)
    p.add_argument("--cache-sha256")
    p.add_argument("--train-trials", type=int, default=2048)
    p.add_argument("--validation-trials", type=int, default=128)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--batch-trials", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-trials", type=int, default=128)
    p.add_argument("--max-elapsed-seconds", type=float, default=1800.)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--expected-hostname", help="Exact allocated Baige scheduler hostname for CPU execution")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, help="Reuse immutable train.npz/validation.npz/dataset.json")
    p.add_argument("--resume", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Warm-start computation requires allocated Linux resources")
    validate_cpu_execution(os.environ, os.uname().nodename, args.expected_hostname)
    if any(getattr(args, key) < 1 for key in ("width", "attempts", "train_trials", "validation_trials",
            "updates", "batch_trials", "eval_every", "eval_trials", "threads")):
        raise ValueError("Require positive training and evaluation sizes")
    if (not 0 <= args.seed < 2**32 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0):
        raise ValueError("Invalid seed or optimization limits")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Require one allocated visible GPU")
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Require the exact clean reviewed source")
    args.horizon = DEFAULT_HORIZONS.get(args.task, 1) if args.horizon is None else args.horizon
    args.ticks = (4 if args.core == "persistent_ctm" else 1) if args.ticks is None else args.ticks
    core_config = dict(width=args.width, ticks=args.ticks)
    if args.core == "persistent_ctm":
        core_config.update(history_length=10, synapse_depth=8, bounded_synchronisation=True, decay_per_observation=True)
    elif args.core == "transformer_xl":
        core_config.update(memory_length=args.memory_length, layers=2, heads=4)
    cache, rates = None, None
    if args.task == "cached_rotation":
        if args.cache is None or args.cache_sha256 is None or args.horizon != 1:
            raise ValueError("Cached rotation needs its pinned outcomes cache and horizon=1")
        cache = OutcomeCache(args.cache, expected_sha256=args.cache_sha256)
        rates = rotation_teacher_rates(cache)
        factory = lambda: CachedRotationTrial(cache, split="train", attempts=args.attempts)
        family = CACHED
    else:
        if args.cache is not None or args.cache_sha256 is not None:
            raise ValueError("Cache arguments apply only to cached_rotation")
        factory = lambda: make_env(args.task, args.attempts, args.horizon, args.arms)
        family = COMPACT
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    sampler = np.random.default_rng(args.seed)
    env = factory()
    policy = build_policy(args.core, core_config, env).to(args.device).eval()
    env.close()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset_contract = dict(behavior=TEACHER_VERSION, labels=DPT_LABEL_VERSION, objective=args.objective,
        context="actual observations/actions/rewards; per-trial uniform random-action fraction",
        task=args.task, attempts=args.attempts, horizon=args.horizon,
        arms=args.arms if args.task in BANDITS else None, encoding="clock_by_declared_max_v1",
        training_seed_start=0, training_trials=args.train_trials,
        validation_seed_start=800_000_000, validation_trials=args.validation_trials,
        cache=None if cache is None else cache.describe(),
        rotation_likelihoods=None if rates is None else rates.tolist())
    if args.train_trials >= 800_000_000 or args.validation_trials >= 200_000_000:
        raise ValueError("Training, validation and evaluation seed ranges must remain disjoint")
    dataset_dir = args.dataset_dir or args.output_dir
    if args.dataset_dir:
        data_manifest = json.loads((dataset_dir / "dataset.json").read_text())
        if data_manifest["contract"] != dataset_contract:
            raise ValueError("Saved teacher dataset contract differs")
        for name in ("train", "validation"):
            if sha256(dataset_dir / f"{name}.npz") != data_manifest[f"{name}_sha256"]:
                raise ValueError("Teacher dataset checksum differs")
        train = dict(np.load(dataset_dir / "train.npz", allow_pickle=False))
        validation = dict(np.load(dataset_dir / "validation.npz", allow_pickle=False))
    else:
        train = generate_histories(factory, args.task, np.arange(args.train_trials), arms=args.arms,
                                   rotation_rates=rates, objective=args.objective)
        # Cached histories validate imitation on fresh TRAIN scenes. Behavioral evaluation uses held-out scenes.
        validation = generate_histories(factory, args.task, np.arange(800_000_000, 800_000_000 + args.validation_trials),
                                        arms=args.arms, rotation_rates=rates, objective=args.objective)
        for name, dataset in (("train", train), ("validation", validation)):
            np.savez_compressed(dataset_dir / f"{name}.npz", **dataset)
        data_manifest = dict(contract=dataset_contract,
            train_sha256=sha256(dataset_dir / "train.npz"), validation_sha256=sha256(dataset_dir / "validation.npz"),
            teacher_environment_steps=int(train["rewards"].size + validation["rewards"].size),
            teacher_train_mean_return=float(train["rewards"].sum(0).mean()),
            teacher_validation_mean_return=float(validation["rewards"].sum(0).mean()))
        write_json(dataset_dir / "dataset.json", data_manifest)
    contract = dict(source_commit=source, phase="supervised_history_warmstart", objective=args.objective, task=args.task,
        attempts=args.attempts, horizon=args.horizon, arms=args.arms if args.task in BANDITS else None,
        observation_encoding="clock_by_declared_max_v1", core=args.core, core_config=core_config,
        dtype="float32", device=args.device, seed=args.seed, dataset=data_manifest,
        batch_trials=args.batch_trials, learning_rate=args.learning_rate,
        evaluation=dict(seed_start=1_000_000_000, trials=args.eval_trials))
    if cache:
        contract.update(protocol=PROTOCOL, cache=cache.describe())
    counters = dict(teacher_environment_steps=data_manifest["teacher_environment_steps"],
                    update_forward_samples=0, optimizer_steps=0, evaluation_steps=0)
    iteration = 0
    if args.resume:
        iteration, counters = restore_checkpoint(args.resume, policy=policy, optimizer=optimizer,
                                                  environment_rng=sampler, contract=contract)
    if iteration >= args.updates:
        raise ValueError("Target must exceed the restored iteration")
    write_json(args.output_dir / "config.json", dict(contract=contract, target_updates=args.updates,
               dataset_dir=str(dataset_dir.resolve()), hostname=os.uname().nodename))
    started, stop = time.monotonic(), False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        signal.signal(getattr(signal, name), request_stop)
    def checkpoint():
        return save_checkpoint(args.output_dir / "latest.pt", policy=policy, optimizer=optimizer,
            environment_rng=sampler, iteration=iteration, counters=counters, contract=contract)
    @torch.no_grad()
    def evaluation():
        losses = [float(supervised_loss(policy, validation, np.arange(start, min(start + args.batch_trials,
                  args.validation_trials)))) for start in range(0, args.validation_trials, args.batch_trials)]
        sizes = [min(args.batch_trials, args.validation_trials - start)
                 for start in range(0, args.validation_trials, args.batch_trials)]
        seeds = np.arange(1_000_000_000, 1_000_000_000 + args.eval_trials)
        if cache:
            behavior, raw = evaluate_policy(policy, cache, seeds, attempts=args.attempts)
            np.savez_compressed(args.output_dir / f"evaluation-{iteration}.npz", **raw)
            counters["evaluation_steps"] += 6 * len(seeds) * args.attempts
        else:
            behavior = {}
            for deterministic in (True, False):
                for name, reset, blind in (("normal", False, False), ("memory_wiped", True, False),
                                            ("reward_blind", False, True)):
                    key = name if deterministic else "sampled_" + name
                    behavior[key], raw = evaluate_trials(policy, args.task, args.attempts, args.horizon, seeds,
                        arms=args.arms, reset_each_attempt=reset, reward_blind=blind, deterministic=deterministic)
                    counters["evaluation_steps"] += behavior[key]["environment_steps"]
                    np.savez_compressed(args.output_dir / f"evaluation-{iteration}-{key}.npz", **raw)
        return dict(validation_nll=float(np.average(losses, weights=sizes)), behavior=behavior)
    checkpoint()
    with (args.output_dir / "metrics.jsonl").open("x", buffering=1) as log:
        result = evaluation()
        log.write(json.dumps(dict(iteration=iteration, evaluation=result), allow_nan=False) + "\n")
        while iteration < args.updates and not stop and time.monotonic() - started < args.max_elapsed_seconds:
            lanes = sampler.integers(args.train_trials, size=args.batch_trials)
            metrics = update_supervised(policy, optimizer, train, lanes)
            iteration += 1
            counters["optimizer_steps"] += 1
            counters["update_forward_samples"] += args.batch_trials * args.attempts * args.horizon
            row = dict(iteration=iteration, **metrics, elapsed_seconds=time.monotonic() - started)
            if iteration % args.eval_every == 0:
                result = evaluation()
                row["evaluation"] = result
                checkpoint()
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if iteration % args.eval_every == 0:
                print(json.dumps(row, allow_nan=False), flush=True)
        result = evaluation()
        resume_digest = checkpoint()
        digest = export_warmstart(args.output_dir / "warmstart.pt", family=family, policy=policy,
            contract=contract, iteration=iteration, counters=counters, data=data_manifest, evaluation=result) if iteration else None
        write_json(args.output_dir / "summary.json", dict(completed=iteration == args.updates,
            iteration=iteration, counters=counters, evaluation=result, warmstart_sha256=digest,
            resume_checkpoint_sha256=resume_digest, elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
