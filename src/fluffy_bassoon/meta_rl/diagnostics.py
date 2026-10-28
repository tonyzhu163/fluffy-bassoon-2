"""History-headroom checks before training architecture comparisons."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from fluffy_bassoon.evaluation_stages import write_json
from fluffy_bassoon.gpu_execution import validate_execution
from .environments import BernoulliBandit, HiddenGainReach


def bandit_baseline(env, seed, mode, rng):
    if mode not in ("oracle", "random", "win_stay", "thompson"):
        raise ValueError("Unknown bandit baseline")
    env.reset(seed=int(seed))
    alpha, beta = np.ones(env.action_space.n), np.ones(env.action_space.n)
    previous, previous_reward = 0, 1.
    rewards, regret = [], []
    for _ in range(env.attempts):
        if mode == "oracle":
            action = int(env._means.argmax())
        elif mode == "random":
            action = int(rng.integers(env.action_space.n))
        elif mode == "win_stay":
            action = previous if previous_reward else (previous + 1) % env.action_space.n
        else:
            action = int(rng.beta(alpha, beta).argmax())
        _, reward, _, _, _ = env.step(action)
        alpha[action] += reward
        beta[action] += 1 - reward
        previous, previous_reward = action, reward
        rewards.append(reward)
        # Private means are used only to score the chosen action, never by non-oracle policies.
        regret.append(float(env._means.max() - env._means[action]))
    return np.array(rewards), np.array(regret)


def gain_baseline(env, seed, mode):
    if mode not in ("oracle", "identify_carry", "identify_reset"):
        raise ValueError("Unknown gain baseline")
    if not np.array_equal(np.sort(env.gains), [-1., 1.]):
        raise ValueError("This identification diagnostic is specific to signed unit gains")
    observation, _ = env.reset(seed=int(seed))
    gain = float(env._gain) if mode == "oracle" else None
    rewards = []
    for _ in range(env.attempts * env.horizon):
        state = observation["state"]
        delta = state[2:] - state[:2]
        if gain is not None:
            action = np.clip(delta / (env.step_size * gain), -1, 1).astype(np.float32)
        else:
            residual = np.maximum(np.abs(delta) - (env.horizon - 1) * env.step_size, 0)
            if np.linalg.norm(residual) < env.tolerance - 1e-5:
                action = np.zeros(2, dtype=np.float32)
                action[np.abs(delta).argmin()] = .01
            else:
                action = np.clip(delta / env.step_size, -1, 1).astype(np.float32)
        returned, reward, _, _, info = env.step(action)
        if gain is None:
            displacement = returned["transition_state"][:2] - state[:2]
            evidence = float(np.dot(displacement, action))
            if abs(evidence) < 1e-9:
                raise RuntimeError("The probe did not identify an actuator direction")
            gain = 1. if evidence > 0 else -1.
        if info["attempt_end"]:
            rewards.append(reward)
            if mode == "identify_reset":
                gain = None
        observation = returned
    return np.array(rewards)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=1024)
    parser.add_argument("--seed-start", type=int, default=1_000_000_000)
    parser.add_argument("--arms", type=int, default=6)
    parser.add_argument("--bandit-attempts", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--gain-horizons", type=int, nargs="+", default=[4, 8])
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Run environment qualification on allocated Linux resources")
    validate_execution("slurm", os.environ)
    if args.trials < 1 or args.seed_start < 0 or args.arms < 2:
        raise ValueError("Invalid trial count, seed or arm count")
    if any(value < 1 for value in args.bandit_attempts + args.gain_horizons):
        raise ValueError("Attempt counts and horizons must be positive")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Require exact clean source")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results = dict(source_commit=source, trials=args.trials, seed_start=args.seed_start,
                   arms=args.arms, bandit={}, gain={}, job_id=os.environ["SLURM_JOB_ID"],
                   hostname=os.uname().nodename)
    seeds = range(args.seed_start, args.seed_start + args.trials)
    for attempts in args.bandit_attempts:
        condition = {}
        for mode in ("oracle", "random", "win_stay", "thompson"):
            env = BernoulliBandit(attempts=attempts, arms=args.arms)
            rng = np.random.default_rng(717)
            trials = [bandit_baseline(env, seed, mode, rng) for seed in seeds]
            rewards, regret = (np.stack([trial[index] for trial in trials]) for index in (0, 1))
            env.close()
            np.savez_compressed(args.output_dir / f"bernoulli-{attempts}-{mode}.npz", rewards=rewards, regret=regret)
            condition[mode] = dict(mean_reward=float(rewards.mean()), reward_by_attempt=rewards.mean(0).tolist(),
                                   cumulative_expected_regret=float(regret.sum(1).mean()),
                                   cumulative_regret_curve=np.cumsum(regret.mean(0)).tolist(),
                                   environment_steps=args.trials * attempts)
            print(f"bernoulli K{attempts} {mode}: reward={rewards.mean():.4f}, regret={regret.sum(1).mean():.4f}", flush=True)
        results["bandit"][str(attempts)] = condition
    for horizon in args.gain_horizons:
        condition = {}
        for mode in ("oracle", "identify_carry", "identify_reset"):
            env = HiddenGainReach(horizon=horizon)
            rewards = np.stack([gain_baseline(env, seed, mode) for seed in seeds])
            env.close()
            np.savez_compressed(args.output_dir / f"gain-{horizon}-{mode}.npz", rewards=rewards)
            condition[mode] = dict(mean_success=float(rewards.mean()), success_by_attempt=rewards.mean(0).tolist(),
                                   environment_steps=args.trials * horizon * rewards.shape[1])
            print(f"gain H{horizon} {mode}: success={rewards.mean():.4f}", flush=True)
        results["gain"][str(horizon)] = condition
    results["elapsed_seconds"] = time.monotonic() - started
    write_json(args.output_dir / "summary.json", results)
    return results


if __name__ == "__main__":
    main()
