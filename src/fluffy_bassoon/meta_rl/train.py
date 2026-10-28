"""Run compact full-trial recurrent PPO inside a Linux Slurm step."""

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

from fluffy_bassoon.evaluation_stages import sha256, write_json
from fluffy_bassoon.gpu_execution import validate_execution
from .checkpoint import restore_checkpoint, save_checkpoint, initialize_from_motor_checkpoint
from .cores import CORE_NAMES, build_policy, core_config_from_args, policy_size
from .environments import (BernoulliBandit, DarkRoom, HiddenGainReach, HiddenRewardBandit, KeyToDoor,
                           NonstationaryBernoulliBandit)
from .ppo import collect_trials, update_ppo


BANDITS = {"bandit": HiddenRewardBandit, "bernoulli": BernoulliBandit,
           "nonstationary": NonstationaryBernoulliBandit}
GRIDS = {"darkroom": DarkRoom, "keydoor": KeyToDoor}
DEFAULT_HORIZONS = {"reach": 8, "darkroom": 20, "keydoor": 50}


def make_env(task, attempts, horizon, arms=2):
    if task in BANDITS:
        if horizon != 1:
            raise ValueError("Bandits have exactly one decision per attempt")
        return BANDITS[task](attempts=attempts, arms=arms)
    if task == "reach":
        return HiddenGainReach(attempts=attempts, horizon=horizon)
    if task in GRIDS:
        return GRIDS[task](attempts=attempts, horizon=horizon)
    raise ValueError("Unknown compact RL task")


@torch.no_grad()
def evaluate_trials(policy, task, attempts, horizon, seeds, *, arms=2, reset_each_attempt=False,
                    deterministic=True, policy_seed=2_000_000_000, normalize_clock=True,
                    reward_blind=False):
    if type(policy_seed) is not int or not 0 <= policy_seed < 2**32:
        raise ValueError("Policy evaluation seed must be an integer in [0, 2**32)")
    envs = [make_env(task, attempts, horizon, arms) for _ in seeds]
    try:
        reference = next(policy.parameters())
        devices = [reference.device.index] if reference.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(policy_seed)
            rollout = collect_trials(policy, envs, seeds, reset_each_attempt=reset_each_attempt,
                                      deterministic=deterministic, normalize_clock=normalize_clock,
                                      reward_blind=reward_blind)
        successes = rollout["rewards"].reshape(attempts, horizon, len(envs)).sum(1)
        trial_data = dict(rewards=successes.T.cpu().numpy(),
                          raw_actions=rollout["raw_actions"].transpose(0, 1).cpu().numpy())
        common = dict(trial_reward=float(successes.sum(0).mean()), trials=len(envs),
                      environment_steps=int(rollout["action_mask"].sum()),
                      policy_steps=rollout["rewards"].numel(), settling_steps=int(rollout["settling_steps"].sum()),
                      reset_each_attempt=reset_each_attempt,
                      deterministic=deterministic, policy_seed=None if deterministic else policy_seed,
                      normalize_clock=normalize_clock, reward_blind=reward_blind)
        if task in ("bernoulli", "nonstationary"):
            # Latent probabilities score evaluation only; PPO uses sampled rewards and public observations.
            means = np.stack([env._mean_history if task == "nonstationary"
                              else np.broadcast_to(env._means, (attempts, env._means.size)) for env in envs])
            actions = rollout["raw_actions"].cpu().numpy().T.astype(np.int64)
            expected = np.take_along_axis(means, actions[..., None], 2)[..., 0]
            regret = means.max(2) - expected
            trial_data["pseudo_regret"] = regret
            return dict(**common, reward_by_attempt=successes.mean(1).cpu().tolist(),
                        mean_reward=float(successes.mean()),
                        expected_regret_by_attempt=regret.mean(0).tolist(),
                        cumulative_expected_regret=float(regret.sum(1).mean())), trial_data
        if task in GRIDS:
            # Dense grid rewards sum to a per-attempt return, not a success label.
            return dict(**common, reward_by_attempt=successes.mean(1).cpu().tolist(),
                        mean_reward=float(successes.mean())), trial_data
        return dict(**common, success_by_attempt=successes.mean(1).cpu().tolist(),
                    mean_success=float(successes.mean())), trial_data
    finally:
        for env in envs:
            env.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=(*BANDITS, "reach", *GRIDS), default="bandit")
    p.add_argument("--core", choices=CORE_NAMES,
                   default="gru", help="ctm is the CMM port without memory blocks; persistent_ctm is the robot CTM")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--arms", type=int, default=2, help="Number of arms for bandit tasks")
    p.add_argument("--horizon", type=int)
    p.add_argument("--ticks", type=int, default=1)
    p.add_argument("--stm-length", type=int, default=10)
    p.add_argument("--ltm-tokens", type=int, default=32)
    p.add_argument("--synapse-depth", type=int, default=8)
    p.add_argument("--transformer-memory-length", type=int, default=64)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--looped-memory-tokens", type=int, default=8)
    p.add_argument("--bounded-synchronisation", action="store_true",
                   help="persistent_ctm only: alpha/beta with RMS norm instead of alpha/sqrt(beta)")
    p.add_argument("--decay-per-observation", action="store_true",
                   help="persistent_ctm only: apply each learned decay once per observation, not per tick")
    p.add_argument("--mode", choices=("meta", "episode"), default="meta")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--value-coef", type=float, default=.5, help="Weight of the shared critic fitting loss")
    p.add_argument("--reward-offset", type=float, default=0.,
                   help="Constant subtracted for return/value fitting; raw feedback and evaluation stay unchanged")
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-trials", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-trials", type=int, default=128)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--max-elapsed-seconds", type=float, default=1800.)
    p.add_argument("--bounded-action-entropy", action="store_true",
                   help="Use a reparameterized entropy estimate after action squashing (reaching only)")
    p.add_argument("--motor-warmstart",type=Path)
    p.add_argument("--motor-warmstart-sha256")
    p.add_argument("--resume", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Model computation requires allocated Linux resources")
    execution = os.environ.get("FLUFFY_GPU_EXECUTION", "slurm")
    if execution not in ("slurm", "baige"):
        raise ValueError("Training requires allocated Slurm or Baige execution")
    validate_execution(execution, os.environ)
    if (args.motor_warmstart is None)!=(args.motor_warmstart_sha256 is None):
        raise ValueError("Motor warmstart requires both a path and its declared SHA256")
    for name in ("updates", "num_envs", "width", "attempts", "ticks", "eval_every",
                 "eval_trials", "checkpoint_every", "ppo_epochs", "minibatch_trials"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if (not 0 <= args.seed < 2**32 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0
            or not math.isfinite(args.reward_offset) or not 0 <= args.reward_offset <= 1
            or not math.isfinite(args.value_coef) or args.value_coef < 0):
        raise ValueError("Invalid seed, learning rate, time budget, reward offset or value coefficient")
    if args.bounded_action_entropy and args.task != "reach":
        raise ValueError("Bounded action entropy requires continuous reaching actions")
    if (args.bounded_synchronisation or args.decay_per_observation) and args.core != "persistent_ctm":
        raise ValueError("Synchronization options apply only to the persistent_ctm core")
    args.horizon = DEFAULT_HORIZONS.get(args.task, 1) if args.horizon is None else args.horizon
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Training requires the exact clean source commit")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("The initial learner requires exactly one allocated visible GPU")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    environment_rng = np.random.default_rng(args.seed)
    envs = [make_env(args.task, args.attempts, args.horizon, args.arms) for _ in range(args.num_envs)]
    core_config = core_config_from_args(args)
    policy = build_policy(args.core, core_config, envs[0]).to(args.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.)
    size = policy_size(policy)
    optimization = dict(epochs=args.ppo_epochs, minibatch_trials=args.minibatch_trials,
                        clip=.2, gamma=1., gae_lambda=1., entropy_coef=.01,
                        value_coef=args.value_coef, max_grad_norm=1., episode_returns=args.mode == "episode",
                        reward_offset=args.reward_offset, normalize_advantages=True,
                        bounded_entropy=args.bounded_action_entropy)
    contract = dict(source_commit=source, task=args.task, attempts=args.attempts, horizon=args.horizon,
                    observation_encoding="clock_by_declared_max_v1",
                    arms=args.arms if args.task in BANDITS else None,
                    num_envs=args.num_envs, seed=args.seed, core=args.core, core_config=core_config,
                    mode=args.mode, device=args.device, dtype="float32", optimization=optimization,
                    optimizer=dict(name="AdamW", learning_rate=args.learning_rate, weight_decay=0.,
                                   betas=[.9, .999], eps=1e-8),
                    evaluation=dict(seed_start=1_000_000_000, trials=args.eval_trials,
                                    action_selection=["greedy", "stochastic"], policy_seed=2_000_000_000))
    if args.motor_warmstart:
        contract['initialization']=initialize_from_motor_checkpoint(args.motor_warmstart,
            args.motor_warmstart_sha256,policy=policy,contract=contract)
    counters = dict(environment_steps=0, settling_steps=0, rollout_forward_samples=0, update_forward_samples=0,
                    active_action_update_samples=0, optimizer_steps=0, evaluation_steps=0,
                    evaluation_policy_steps=0, evaluation_settling_steps=0)
    iteration = 0
    if args.resume:
        iteration, counters = restore_checkpoint(args.resume, policy=policy, optimizer=optimizer,
                                                  environment_rng=environment_rng, contract=contract)
    if iteration >= args.updates:
        raise ValueError("Target updates must exceed the restored iteration")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "config.json", dict(contract=contract, target_updates=args.updates,
               max_elapsed_seconds=args.max_elapsed_seconds, eval_every=args.eval_every,
               checkpoint_every=args.checkpoint_every, **size,
               motor_warmstart_path=None if args.motor_warmstart is None else str(args.motor_warmstart),
               resumed_from=None if args.resume is None else dict(path=str(args.resume), sha256=sha256(args.resume)),
               execution=execution, job_id=os.environ.get("SLURM_JOB_ID", os.uname().nodename),
               hostname=os.uname().nodename,
               runtime=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version)))
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    for name in ("SIGUSR1", "SIGTERM", "SIGINT"):
        signal.signal(getattr(signal, name), request_stop)
    started = time.monotonic()
    checkpoint_digest = None
    last_evaluation = None
    def checkpoint():
        return save_checkpoint(args.output_dir / "latest.pt", policy=policy, optimizer=optimizer,
                               environment_rng=environment_rng, iteration=iteration,
                               counters=counters, contract=contract)
    def evaluation():
        seeds = np.arange(1_000_000_000, 1_000_000_000 + args.eval_trials)
        result = {}
        for name, deterministic in (("normal", True), ("stochastic", False)):
            normal, _ = evaluate_trials(policy, args.task, args.attempts, args.horizon, seeds,
                                        arms=args.arms, reset_each_attempt=args.mode == "episode",
                                        deterministic=deterministic)
            reset = normal if args.mode == "episode" else evaluate_trials(
                policy, args.task, args.attempts, args.horizon, seeds, arms=args.arms,
                reset_each_attempt=True, deterministic=deterministic)[0]
            for evaluated in ([normal] if args.mode == "episode" else [normal, reset]):
                counters["evaluation_steps"] += evaluated["environment_steps"]
                counters["evaluation_policy_steps"] += evaluated["policy_steps"]
                counters["evaluation_settling_steps"] += evaluated["settling_steps"]
            result[name] = normal
            result["reset_between_attempts" if deterministic else "stochastic_reset_between_attempts"] = reset
        return result
    checkpoint_digest = checkpoint()
    with (args.output_dir / "metrics.jsonl").open("x", buffering=1) as log:
        last_evaluation = evaluation()
        log.write(json.dumps(dict(iteration=iteration, evaluation=last_evaluation,
                                   counters=dict(counters), elapsed_seconds=time.monotonic() - started), allow_nan=False) + "\n")
        while iteration < args.updates and not stop and time.monotonic() - started < args.max_elapsed_seconds:
            seeds = environment_rng.integers(0, 1_000_000_000, size=args.num_envs)
            rollout_started = time.monotonic()
            rollout = collect_trials(policy, envs, seeds, reset_each_attempt=args.mode == "episode")
            rollout_seconds = time.monotonic() - rollout_started
            update_started = time.monotonic()
            metrics = update_ppo(policy, optimizer, rollout, **optimization)
            update_seconds = time.monotonic() - update_started
            iteration += 1
            steps = rollout["rewards"].numel()
            counters["environment_steps"] += int(rollout["action_mask"].sum())
            counters["settling_steps"] += int(rollout["settling_steps"].sum())
            counters["rollout_forward_samples"] += steps
            counters["update_forward_samples"] += metrics["forward_samples"]
            counters["optimizer_steps"] += metrics["optimizer_steps"]
            counters["active_action_update_samples"] += metrics["active_action_samples"]
            record = dict(iteration=iteration, training=metrics,
                          trial_reward=float(rollout["rewards"].sum(0).mean()),
                          timing=dict(rollout_seconds=rollout_seconds, update_seconds=update_seconds))
            if iteration % args.eval_every == 0 or iteration == args.updates:
                evaluation_started = time.monotonic()
                last_evaluation = evaluation()
                record["timing"]["evaluation_seconds"] = time.monotonic() - evaluation_started
                record["evaluation"] = last_evaluation
            if iteration % args.checkpoint_every == 0:
                checkpoint_digest = checkpoint()
            record.update(counters=dict(counters), elapsed_seconds=time.monotonic() - started)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
        checkpoint_digest = checkpoint()
    for env in envs:
        env.close()
    summary = dict(completed=iteration == args.updates, iteration=iteration,
                   source_commit=source, counters=counters, elapsed_seconds=time.monotonic() - started,
                   checkpoint=dict(path=str(args.output_dir / "latest.pt"), sha256=checkpoint_digest),
                   last_evaluation=last_evaluation)
    write_json(args.output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
