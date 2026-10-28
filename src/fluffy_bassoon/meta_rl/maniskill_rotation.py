"""CPU proof of concept for reward-only A1 adaptation on a frozen motor cache."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import sys
import time

import gymnasium as gym
from gymnasium.spaces.utils import flatdim
import numpy as np
import torch

from fluffy_bassoon.evaluation_stages import write_json
from fluffy_bassoon.gpu_execution import validate_execution
from .checkpoint import restore_checkpoint, save_checkpoint
from .environments import _TrialEnv
from .ppo import GRUCore, PersistentCTMVectorCore, RecurrentActorCritic, collect_trials, update_ppo
from .transformer_xl import TransformerXLCore


ANGLES = (-45., 0., 45.)
NET_ANGLES = (-90., -45., 0., 45., 90.)
PROTOCOL = "cached_rotation_a1_plumbing_v2"
SCENE_FIELDS = ("object", "overlay", "position", "orientation", "episode_id", "scene_index")


class OutcomeCache:
    """Validate a complete exact-angle table; discard all graded outcomes."""

    def __init__(self, path, expected_sha256=None):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and expected_sha256 != self.sha256:
            raise ValueError("Outcome cache checksum differs")
        self.scenes, self.outcomes = {}, defaultdict(list)
        seen = set()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        for row in rows:
            scene = tuple(row[key] for key in SCENE_FIELDS)
            if any(type(value) is not int or value < 0 for value in scene):
                raise ValueError("Scene identifiers must be nonnegative integers")
            index = row["scene_index"]
            if index in self.scenes and self.scenes[index] != scene:
                raise ValueError("A scene index maps to different scene identities")
            self.scenes[index] = scene
            angle, repeat = row["net_angle_degrees"], row["repeat"]
            if (isinstance(angle, bool) or not isinstance(angle, (int, float)) or not np.isfinite(angle)
                    or type(repeat) is not int or repeat < 0 or type(row["success_once"]) is not bool):
                raise ValueError("Invalid angle, repeat or binary success")
            identity = (index, float(angle), repeat)
            if identity in seen:
                raise ValueError("Duplicate cached scene/angle/repeat")
            seen.add(identity)
            self.outcomes[index, float(angle)].append(float(row["success_once"]))
        if not self.scenes:
            raise ValueError("Empty outcome cache")
        self.split_scenes = {
            "train": sorted(index for index in self.scenes if index % 4 != 3),
            "validation": sorted(index for index in self.scenes if index % 4 == 3),
        }
        if any(not scenes for scenes in self.split_scenes.values()):
            raise ValueError("Both scene splits must be nonempty")
        for index in self.scenes:
            if any((index, angle) not in self.outcomes for angle in NET_ANGLES):
                raise ValueError("Every scene requires all exact composed net angles")
        train_objects = {self.scenes[index][0] for index in self.split_scenes["train"]}
        validation_objects = {self.scenes[index][0] for index in self.split_scenes["validation"]}
        if train_objects != validation_objects:
            raise ValueError("This pilot requires shared objects and disjoint scenes")
        # Episode identities cannot cross the declared scene split.
        train_episodes = {self.scenes[index][4] for index in self.split_scenes["train"]}
        if train_episodes & {self.scenes[index][4] for index in self.split_scenes["validation"]}:
            raise ValueError("Episode identity crosses scene splits")
        self.rows = len(rows)

    def describe(self):
        return dict(sha256=self.sha256, rows=self.rows, scenes=len(self.scenes),
                    splits={key: len(value) for key, value in self.split_scenes.items()},
                    theta_degrees=list(ANGLES), phi_degrees=list(ANGLES),
                    net_angle_degrees=list(NET_ANGLES), reward_field="success_once",
                    split_rule="scene_index modulo 4: validation=3; train=0,1,2")


class CachedRotationTrial(_TrialEnv):
    """K one-action attempts with fixed hidden theta and fresh cached scenes."""

    def __init__(self, cache, *, split="train", attempts=8, reward_blind=False):
        if split not in cache.split_scenes:
            raise ValueError("Unknown cache scene split")
        if type(reward_blind) is not bool:
            raise ValueError("Reward blinding must be boolean")
        if len(cache.split_scenes[split]) < attempts:
            raise ValueError("Need at least K distinct scenes in the requested split")
        self.cache, self.split, self.reward_blind = cache, split, reward_blind
        super().__init__(attempts=attempts, horizon=1, state_dim=1, action_space=gym.spaces.Discrete(3))

    def _sample_hidden(self):
        self._theta = float(self.np_random.choice(ANGLES))
        self._trial_scenes = self.np_random.choice(self.cache.split_scenes[self.split], self.attempts,
                                                  replace=False)

    def _start_attempt(self):
        return np.zeros(1, dtype=np.float32)

    def _transition(self, action):
        net_angle = self._theta + ANGLES[int(action)]
        outcomes = self.cache.outcomes[int(self._trial_scenes[self._attempt]), net_angle]
        reward = outcomes[int(self.np_random.integers(len(outcomes)))]
        return self._state.copy(), bool(reward)

    def _observation(self, transition_state, action, reward, attempt_end):
        observation = super()._observation(transition_state, action, reward, attempt_end)
        if self.reward_blind:
            observation["previous_reward"][:] = 0
        return observation


def baseline_trials(cache, seeds, attempts=8):
    results, raw = {}, {}
    for mode in ("zero", "oracle", "thompson"):
        rewards = []
        for seed in seeds:
            env = CachedRotationTrial(cache, split="validation", attempts=attempts)
            env.reset(seed=int(seed))
            rng = np.random.default_rng(int(seed) + 17)
            alpha, beta = np.ones(3), np.ones(3)
            trial = []
            for _ in range(attempts):
                action = 1 if mode == "zero" else (ANGLES.index(-env._theta) if mode == "oracle"
                                                    else int(rng.beta(alpha, beta).argmax()))
                _, reward, *_ = env.step(action)
                alpha[action] += reward
                beta[action] += 1 - reward
                trial.append(reward)
            rewards.append(trial)
            env.close()
        rewards = np.asarray(rewards)
        raw[mode] = rewards
        results[mode] = dict(mean_reward=float(rewards.mean()),
                             reward_by_attempt=rewards.mean(0).tolist(),
                             trial_reward=float(rewards.sum(1).mean()), trials=len(seeds))
    return results, raw


@torch.no_grad()
def evaluate_policy(policy, cache, seeds, attempts=8):
    results, raw = {}, {}
    for deterministic in (True, False):
        for lesion in ("normal", "memory_wiped", "reward_blind"):
            envs = [CachedRotationTrial(cache, split="validation", attempts=attempts,
                                       reward_blind=lesion == "reward_blind") for _ in seeds]
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(2_000_000_000)
                    rollout = collect_trials(policy, envs, seeds, reset_each_attempt=lesion == "memory_wiped",
                                              deterministic=deterministic)
                rewards = rollout["rewards"].cpu().numpy().T
                name = ("greedy_" if deterministic else "sampled_") + lesion
                raw[name] = rewards
                raw[name + "_actions"] = rollout["raw_actions"].cpu().numpy().T
                results[name] = dict(mean_reward=float(rewards.mean()),
                                     reward_by_attempt=rewards.mean(0).tolist(),
                                     trial_reward=float(rewards.sum(1).mean()), trials=len(seeds))
            finally:
                for env in envs:
                    env.close()
    return results, raw


def validate_cpu_execution(environ, hostname, expected_hostname=None):
    """Accept an actual Slurm step or the exact named scheduled CPU container."""
    if environ.get("SLURM_JOB_ID"):
        validate_execution("slurm", environ)
        return "slurm"
    if (not expected_hostname or hostname != expected_hostname
            or not re.fullmatch(r"job-[a-z0-9]+-master-0", hostname)
            or not environ.get("KUBERNETES_SERVICE_HOST")):
        raise ValueError("Require the exact scheduled Baige container or a Slurm step")
    return "baige"


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-hostname", help="Exact authorized allocated Linux host if not using Slurm")
    parser.add_argument("--core", choices=("persistent_ctm", "transformer_xl", "gru"), default="persistent_ctm")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--ticks", type=int, help="Default CTM T4; Transformer-XL and GRU T1")
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=.01)
    parser.add_argument("--reward-offset", type=float, default=.5)
    parser.add_argument("--eval-trials", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-elapsed-seconds", type=float, default=1800.)
    parser.add_argument("--resume", type=Path)
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Run the cached robot pilot only on allocated Linux resources")
    execution = validate_cpu_execution(os.environ, socket.gethostname(), args.expected_hostname)
    # This pilot is CPU-only; do not initialize CUDA RNG state on a shared GPU host.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.ticks = (4 if args.core == "persistent_ctm" else 1) if args.ticks is None else args.ticks
    if any(getattr(args, key) < 1 for key in ("width", "ticks", "attempts", "updates", "num_envs",
                                            "ppo_epochs", "eval_trials", "eval_every", "threads")):
        raise ValueError("Counts must be positive")
    if (not 0 <= args.seed < 2**32 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not np.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0
            or not np.isfinite(args.value_coef) or args.value_coef < 0
            or not np.isfinite(args.reward_offset) or not 0 <= args.reward_offset <= 1):
        raise ValueError("Invalid seed, learning rate, time budget or PPO value/reward settings")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Require the exact clean source commit")
    cache = OutcomeCache(args.cache, args.cache_sha256)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    envs = [CachedRotationTrial(cache, attempts=args.attempts) for _ in range(args.num_envs)]
    inputs = int(flatdim(envs[0].observation_space))
    core_config = dict(width=args.width, ticks=args.ticks)
    if args.core == "persistent_ctm":
        core_config.update(history_length=10, synapse_depth=8,
                           bounded_synchronisation=True, decay_per_observation=True)
    elif args.core == "transformer_xl":
        core_config.update(memory_length=16, layers=2, heads=4)
    factory = {"persistent_ctm": PersistentCTMVectorCore, "transformer_xl": TransformerXLCore,
               "gru": GRUCore}[args.core]
    policy = RecurrentActorCritic(factory(inputs, **core_config), envs[0].action_space)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.)
    optimization = dict(epochs=args.ppo_epochs, minibatch_trials=min(16, args.num_envs), clip=.2,
                        gamma=1., gae_lambda=1., entropy_coef=.01, value_coef=args.value_coef,
                        max_grad_norm=1., reward_offset=args.reward_offset, normalize_advantages=True)
    contract = dict(protocol=PROTOCOL, source_commit=source,
                    cache=cache.describe(), core=args.core, core_config=core_config,
                    attempts=args.attempts, num_envs=args.num_envs, seed=args.seed,
                    observation_encoding="clock_by_declared_max_v1", optimization=optimization,
                    learning_rate=args.learning_rate, device="cpu", dtype="float32")
    counters = dict(environment_steps=0, optimizer_steps=0, evaluation_steps=0)
    iteration = 0
    if args.resume:
        iteration, counters = restore_checkpoint(args.resume, policy=policy, optimizer=optimizer,
                                                  environment_rng=rng, contract=contract)
    if iteration >= args.updates:
        raise ValueError("Target updates must exceed checkpoint iteration")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = policy.core.initial_state(1)
    tensors = state.values() if isinstance(state, dict) else (state,)
    write_json(args.output_dir / "config.json", dict(contract=contract, updates=args.updates,
               parameters=sum(parameter.numel() for parameter in policy.parameters()),
               retained_state_bytes_per_lane=sum(value.numel() * value.element_size() for value in tensors),
               cache_path=str(args.cache), execution=execution, hostname=socket.gethostname(),
               limitation="existing headroom table; shared objects/angles; no AD warm start or live validation"))
    seeds = np.arange(1_000_000_000, 1_000_000_000 + args.eval_trials)
    baselines, baseline_raw = baseline_trials(cache, seeds, args.attempts)
    np.savez_compressed(args.output_dir / "baselines.npz", **baseline_raw)
    write_json(args.output_dir / "baselines.json", baselines)
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        signal.signal(getattr(signal, name), request_stop)
    started = time.monotonic()
    def checkpoint():
        return save_checkpoint(args.output_dir / "latest.pt", policy=policy, optimizer=optimizer,
                               environment_rng=rng, iteration=iteration, counters=counters, contract=contract)
    def evaluate():
        result, arrays = evaluate_policy(policy, cache, seeds, args.attempts)
        counters["evaluation_steps"] += 6 * args.attempts * args.eval_trials
        np.savez_compressed(args.output_dir / f"evaluation-{iteration}.npz", **arrays)
        return result
    checkpoint()
    evaluation = evaluate()
    with (args.output_dir / "metrics.jsonl").open("x", buffering=1) as log:
        log.write(json.dumps(dict(iteration=iteration, evaluation=evaluation)) + "\n")
        while iteration < args.updates and not stop and time.monotonic() - started < args.max_elapsed_seconds:
            rollout = collect_trials(policy, envs, rng.integers(0, 1_000_000_000, size=args.num_envs))
            metrics = update_ppo(policy, optimizer, rollout, **optimization)
            iteration += 1
            counters["environment_steps"] += args.num_envs * args.attempts
            counters["optimizer_steps"] += metrics["optimizer_steps"]
            record = dict(iteration=iteration, training=metrics,
                          trial_reward=float(rollout["rewards"].sum(0).mean()),
                          elapsed_seconds=time.monotonic() - started)
            if iteration % args.eval_every == 0 or iteration == args.updates:
                evaluation = evaluate()
                record["evaluation"] = evaluation
                checkpoint()
            log.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
    if iteration % args.eval_every and iteration != args.updates:
        evaluation = evaluate()
    digest = checkpoint()
    write_json(args.output_dir / "summary.json", dict(completed=iteration == args.updates,
               iteration=iteration, checkpoint_sha256=digest, evaluation=evaluation, baselines=baselines,
               counters=counters, elapsed_seconds=time.monotonic() - started))
    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
