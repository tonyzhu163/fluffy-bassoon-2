"""Bounded original-Can DSRL-NA and recurrent noise-policy comparison.

Run only inside an allocated Linux GPU job. External checkpoints and packages
are explicit CLI inputs, not downloaded during a training run.
"""
from __future__ import annotations

import argparse
from functools import partial
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
import torch

from .can_dsrl import (ACTION_DIM, CHUNK, HORIZON, NOISE_DIM, OBS_DIM, CanChunkEnv,
                       collect_trial, fingerprint, load_motor, rng_state, restore_rng,
                       sha256, write_json)
from .ppo import GRUCore, RecurrentActorCritic, update_ppo


class TrainingBudgetReached(Exception):
    """A controlled stop at a complete replay/simulator boundary."""


def validate_resume(saved, current):
    for key in ("source_commit", "method", "checkpoint_sha256", "normalization_sha256", "seed", "envs", "attempts",
                "core", "width", "layers", "heads", "memory_length"):
        if saved[key] != current[key]:
            raise ValueError(f"Recovery contract changed: {key}")


def make_policy(width=128, device="cuda", *, core="gtrxl", layers=2, heads=4, memory_length=300):
    inputs = OBS_DIM + NOISE_DIM + 4
    if core == "gtrxl":
        from .gtrxl import GatedTransformerXLCore
        recurrence = GatedTransformerXLCore(inputs, width=width, layers=layers, heads=heads,
                                            memory_length=memory_length, ticks=1)
    elif core == "gru":
        recurrence = GRUCore(inputs, width)
    else:
        raise ValueError(f"Unknown recurrent core: {core}")
    return RecurrentActorCritic(recurrence, gym.spaces.Box(-1., 1., (NOISE_DIM,), np.float32)).to(device)


def policy_from_args(args, device="cuda"):
    return make_policy(args.width, device, core=args.core, layers=args.layers, heads=args.heads,
                       memory_length=args.memory_length)


def dsrl_class():
    from stable_baselines3 import DSRL

    class RecoverableDSRL(DSRL):
        def __init__(self, *args, diffusion_act_dim=(CHUNK, ACTION_DIM), **kwargs):
            # SB3 load constructs an empty learner before restoring its saved
            # attributes; upstream DSRL otherwise indexes a None default here.
            super().__init__(*args, diffusion_act_dim=diffusion_act_dim, **kwargs)
            self._after_train = None

        def train(self, *args, **kwargs):
            super().train(*args, **kwargs)
            if self._after_train is not None:
                self._after_train()

        def _get_torch_save_params(self):
            states, variables = super()._get_torch_save_params()
            return states + ["critic_noise.optimizer"], variables

        def _excluded_save_params(self):
            return super()._excluded_save_params() + ["diffusion_policy", "_after_train"]

    return RecoverableDSRL


def make_dsrl(env, motor, *, width=2048, batch_size=256, utd=20, seed=1, buffer_size=100_000):
    # Identical actor/critic construction to the authors' train_dsrl.py.
    return dsrl_class()("MlpPolicy", env, learning_rate=3e-4, buffer_size=buffer_size,
                        learning_starts=1, batch_size=batch_size, tau=.005, gamma=.99,
                        train_freq=1, gradient_steps=utd, ent_coef="auto", target_entropy=0.,
                        target_update_interval=1, seed=seed, device="cuda", verbose=0,
                        policy_kwargs=dict(net_arch=dict(pi=[width] * 3, qf=[width] * 3),
                                           activation_fn=torch.nn.Tanh, log_std_init=0.,
                                           post_linear_modules=[torch.nn.LayerNorm], n_critics=2),
                        diffusion_policy=motor, diffusion_act_dim=(CHUNK, ACTION_DIM),
                        noise_critic_grad_steps=10, critic_backup_combine_type="min")


def vec_env(args):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    return SubprocVecEnv([partial(CanChunkEnv, args.dsrl_root, args.normalization, seed=args.seed + i,
                                 retries=args.attempts if args.method == "dsrl-na-retry" else 1)
                          for i in range(args.envs)], start_method="spawn")


def scalar_evaluation(actor, envs, seeds, *, attempts, native=False, motor=None, seed=1):
    """Frozen DSRL/native evaluation on the same complete retry trials as PPO."""
    rows = []
    from .can_dsrl_compare import seed_scene
    for scene_seed in seeds:
        seed_scene(scene_seed, seed)
        batch, lanes = [scene_seed], envs[:1]
        successes, release_successes, hashes = [], [], []
        for _ in range(attempts):
            obs = np.stack([e.reset(seed=int(s))[0] for e, s in zip(lanes, batch, strict=True)])
            import hashlib
            hashes.append([hashlib.sha256(o.tobytes()).hexdigest() for o in obs])
            for _ in range(HORIZON // CHUNK):
                with torch.no_grad():
                    if native:
                        noise = torch.randn(len(batch), CHUNK, ACTION_DIM, device="cuda")
                        actions = motor(torch.as_tensor(obs, device="cuda"), noise)
                    else:
                        actions, _ = actor.predict_diffused(obs, deterministic=False)
                result = [e.step(a) for e, a in zip(lanes, actions, strict=True)]
                obs = np.stack([r[0] for r in result])
            successes.append([r[4]["success_once"] for r in result])
            release_successes.append([r[4]["release_logged_success"] for r in result])
        rows.append(dict(scene_seeds=list(map(int, batch)), successes=np.asarray(successes).T.tolist(),
                         release_logged_successes=np.asarray(release_successes).T.tolist(),
                         initial_observation_sha256=hashes, attempts=attempts,
                         primitive_steps=len(batch) * attempts * HORIZON))
    return rows


def save_ppo(path, policy, optimizer, sampler, updates, steps, contract):
    payload = dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                   sampler=sampler.bit_generator.state, rng=rng_state(), updates=updates,
                   primitive_steps=steps, contract=contract, boundary="complete trial")
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def qualify_ppo_recovery(args, policy, optimizer, sampler, motor, envs, contract, *, attempts=2):
    """Verify the next sampled trial and optimizer update from a full recovery."""
    from .can_dsrl_compare import _hash_structure
    path = args.output / "ppo-recovery.pt"
    save_ppo(path, policy, optimizer, sampler, 1, len(envs) * attempts * HORIZON, contract)
    saved = torch.load(path, weights_only=False)
    validate_resume(saved["contract"], contract)
    restored = policy_from_args(args, device=next(policy.parameters()).device)
    restored.load_state_dict(saved["policy"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=3e-4)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_sampler = np.random.default_rng()
    restored_sampler.bit_generator.state = saved["sampler"]
    if (fingerprint(restored) != fingerprint(policy)
            or _hash_structure(restored_optimizer.state_dict()) != _hash_structure(optimizer.state_dict())):
        raise RuntimeError("PPO recovery changed weights or Adam state")
    restore_rng(saved["rng"])
    expected_seeds = sampler.integers(0, 1_000_000, size=len(envs)).tolist()
    expected, expected_row = collect_trial(policy, motor, envs, expected_seeds, attempts=attempts)
    expected_trial_rng = rng_state()
    expected_metrics = update_ppo(policy, optimizer, expected, epochs=1, minibatch_trials=len(envs))
    expected_update_rng = rng_state()
    restore_rng(saved["rng"])
    actual_seeds = restored_sampler.integers(0, 1_000_000, size=len(envs)).tolist()
    actual, actual_row = collect_trial(restored, motor, envs, actual_seeds, attempts=attempts)
    if (actual_seeds != expected_seeds or actual_row != expected_row
            or _hash_structure(restored_sampler.bit_generator.state) != _hash_structure(sampler.bit_generator.state)):
        raise RuntimeError("PPO recovery changed the next sampled trial")
    for name in expected:
        if not torch.equal(expected[name], actual[name]):
            raise RuntimeError(f"PPO recovery changed next trial {name}")
    if _hash_structure(rng_state()) != _hash_structure(expected_trial_rng):
        raise RuntimeError("PPO recovery changed next-trial RNG state")
    actual_metrics = update_ppo(restored, restored_optimizer, actual, epochs=1, minibatch_trials=len(envs))
    if (fingerprint(restored) != fingerprint(policy)
            or _hash_structure(restored_optimizer.state_dict()) != _hash_structure(optimizer.state_dict())
            or _hash_structure(rng_state()) != _hash_structure(expected_update_rng)
            or actual_metrics != expected_metrics):
        raise RuntimeError("PPO recovery changed the next optimizer update")
    return dict(passed=True, recovery_sha256=sha256(path), next_scene_seeds=expected_seeds,
                next_trial_actions_identical=True, next_trial_observations_rewards_identical=True,
                next_optimizer_state_identical=True, next_rng_identical=True)


def evaluate_ppo(policy, motor, envs, seeds, attempts, seed=1, *, reset_each_attempt=False):
    from .can_dsrl_compare import seed_scene
    state = rng_state()
    results = {}
    try:
        for name in ("normal", "memory_reset", "reward_blind"):
            restore_rng(state)
            rows = []
            for scene_seed in seeds:
                seed_scene(scene_seed, seed)
                clear_memory = reset_each_attempt or name == "memory_reset"
                _, row = collect_trial(policy, motor, envs[:1], [scene_seed], attempts=attempts,
                                       reset_memory=clear_memory, reward_blind=name == "reward_blind")
                row["reset_each_attempt"] = clear_memory
                rows.append(row)
            results[name] = rows
        return results
    finally:
        restore_rng(state)


def train_ppo(args, motor, contract, deadline):
    envs = [CanChunkEnv(args.dsrl_root, args.normalization, seed=args.seed + i) for i in range(args.envs)]
    policy = policy_from_args(args)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    sampler = np.random.default_rng(args.seed)
    steps, update = 0, 0
    if args.resume:
        saved = torch.load(args.resume / "recovery.pt", weights_only=False)
        validate_resume(saved["contract"], contract)
        policy.load_state_dict(saved["policy"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.bit_generator.state = saved["sampler"]
        steps, update = saved["primitive_steps"], saved["updates"]
        restore_rng(saved["rng"])
    try:
        while steps + args.envs * args.attempts * HORIZON <= args.steps and time.monotonic() < deadline:
            seeds = sampler.integers(0, 1_000_000, size=args.envs).tolist()
            rollout, row = collect_trial(policy, motor, envs, seeds, attempts=args.attempts,
                                         reset_memory=args.method == "episode-reset")
            with torch.no_grad():
                replay, _, _ = policy.evaluate_sequence(rollout["observations"], rollout["raw_actions"], rollout["resets"])
                error = float((replay - rollout["log_probs"]).abs().max())
            if error > 1e-4:
                raise RuntimeError(f"Old-policy likelihood replay failed: {error}")
            metrics = update_ppo(policy, optimizer, rollout, epochs=4, minibatch_trials=args.envs,
                                 gamma=1., gae_lambda=1.)
            steps += row["primitive_steps"]
            update += 1
            record = dict(update=update, primitive_steps=steps, replay_error=error,
                          success=float(np.mean(row["successes"])), **metrics)
            with (args.output / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
            save_ppo(args.output / "recovery.pt", policy, optimizer, sampler, update, steps, contract)
        # Evaluation is always on disjoint reset seeds and never used for training.
        training_motor_samples = motor.samples
        evaluation = evaluate_ppo(policy, motor, envs, list(range(2_000_000, 2_000_000 + args.eval_trials)),
                                  args.attempts, args.seed, reset_each_attempt=args.method == "episode-reset")
        write_json(args.output / "evaluation.json", evaluation)
        return dict(updates=update, primitive_steps=steps, parameters=sum(p.numel() for p in policy.parameters()),
                    training_motor_samples=training_motor_samples, evaluation_motor_samples=motor.samples - training_motor_samples,
                    optimizer_steps=4 * update,
                    evaluation_primitive_steps=3 * args.eval_trials * args.attempts * HORIZON,
                    budget_reached=steps < args.steps)
    finally:
        for env in envs:
            env.close()


def save_dsrl(args, agent, env, contract):
    """Save every learner/simulator state; the two payloads share a manifest."""
    import shutil
    directory = args.output / "checkpoints" / f"chunks{agent.num_timesteps:09d}-updates{agent._n_updates:09d}"
    if directory.exists():
        manifest = json.loads((args.output / "recovery-manifest.json").read_text())
        if (manifest["directory"] == str(directory.relative_to(args.output))
                and all(sha256(directory / name) == digest for name, digest in manifest["files"].items())):
            return
        raise RuntimeError("Conflicting recovery generation")
    directory.mkdir(parents=True, exist_ok=False)
    agent.save(directory / "recovery")
    agent.save_replay_buffer(directory / "replay.pkl")
    torch.save(dict(rng=rng_state(), simulator=env.env_method("snapshot"),
                    last_obs=agent._last_obs, last_episode_starts=agent._last_episode_starts,
                    num_timesteps=agent.num_timesteps, updates=agent._n_updates,
                    warmup_primitive_steps=getattr(agent, "can_warmup_primitive_steps", 0),
                    phase="collection after completed optimizer update", contract=contract), directory / "runtime.pt")
    files = ["recovery.zip", "replay.pkl", "runtime.pt"]
    manifest = dict(directory=str(directory.relative_to(args.output)),
                    files={name: sha256(directory / name) for name in files})
    pending = args.output / "recovery-manifest.tmp"
    write_json(pending, manifest)
    pending.replace(args.output / "recovery-manifest.json")
    # A new, complete generation is published before superseded payloads are
    # pruned. Configs, logs and evaluation artifacts are retained separately.
    for old in directory.parent.iterdir():
        if old != directory and old.is_dir():
            shutil.rmtree(old)


def train_dsrl(args, motor, config, contract, deadline):
    from stable_baselines3.common.callbacks import BaseCallback
    env = vec_env(args)
    if args.resume:
        manifest = json.loads((args.resume / "recovery-manifest.json").read_text())
        directory = args.resume / manifest["directory"]
        if not all(sha256(directory / name) == digest for name, digest in manifest["files"].items()):
            raise ValueError("Recovery artifact hash differs")
        runtime = torch.load(directory / "runtime.pt", weights_only=False)
        validate_resume(runtime["contract"], contract)
        agent = dsrl_class().load(directory / "recovery.zip", env=env, device="cuda", diffusion_policy=motor)
        agent.load_replay_buffer(directory / "replay.pkl")
    else:
        agent = make_dsrl(env, motor, seed=args.seed)
    # Initialize SB3's logger and collection state without running any training.
    agent._setup_learn(total_timesteps=args.steps // CHUNK, callback=None)
    obs = env.reset()
    warmup = int(config.train.init_rollout_steps)
    warmup_steps = 0
    try:
        if args.resume:
            for lane, state in enumerate(runtime["simulator"]):
                env.env_method("restore", state, indices=lane)
            obs = runtime["last_obs"]
            warmup_steps = runtime["warmup_primitive_steps"]
            agent.num_timesteps, agent._n_updates = runtime["num_timesteps"], runtime["updates"]
            agent._last_obs, agent._last_episode_starts = obs, runtime["last_episode_starts"]
            restore_rng(runtime["rng"])
        # Authors' 1501 parallel steps = 24,016 primitive steps at four lanes.
        for _ in range(0 if args.resume else warmup):
            if time.monotonic() >= deadline:
                raise TimeoutError("Budget exhausted before original DSRL warmup completed")
            noise = torch.randn(args.envs, CHUNK, ACTION_DIM, device="cuda")
            actions = motor(torch.as_tensor(obs, device="cuda"), noise)
            next_obs, rewards, dones, infos = env.step(actions)
            actions_store = actions.reshape(args.envs, -1)
            # Use SB3's normal terminal-observation replacement for exact buffer semantics.
            next_for_buffer = next_obs.copy()
            for i, done in enumerate(dones):
                if done and "terminal_observation" in infos[i]:
                    next_for_buffer[i] = infos[i]["terminal_observation"]
            agent.replay_buffer.add(obs, next_for_buffer, actions_store, rewards, dones, infos)
            obs = next_obs
            warmup_steps += args.envs * CHUNK
        if not args.resume:
            agent.replay_buffer.final_offline_step()
        agent.can_warmup_primitive_steps = warmup_steps
        # Continue from the current simulator state, without an uncounted reset.
        agent._last_obs = obs
        if not args.resume:
            agent._last_episode_starts = np.zeros(args.envs, dtype=bool)

        class Progress(BaseCallback):
            def _on_step(self):
                if self.n_calls % (HORIZON // CHUNK) == 0:
                    record = dict(primitive_steps=warmup_steps + self.model.num_timesteps * CHUNK,
                                  learner_updates=self.model._n_updates,
                                  metrics={k: float(v) for k, v in self.model.logger.name_to_value.items()
                                           if k.startswith("train/") and np.isscalar(v)})
                    with (args.output / "updates.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    print(json.dumps(record), flush=True)
                return True

        def after_train():
            # SB3's rollout-end callback precedes its pending optimizer update.
            # This hook runs after the original update, at a simulator boundary.
            total_vec_steps = warmup_steps // (args.envs * CHUNK) + agent.num_timesteps // args.envs
            if total_vec_steps % (HORIZON // CHUNK) == 0:
                if time.monotonic() >= deadline:
                    raise TrainingBudgetReached()
                if total_vec_steps % 300 == 0:
                    save_dsrl(args, agent, env, contract)

        agent._after_train = after_train

        remaining = max(0, args.steps - warmup_steps - agent.num_timesteps * CHUNK)
        if remaining:
            try:
                agent.learn(total_timesteps=remaining // CHUNK, callback=Progress(), reset_num_timesteps=False)
            except TrainingBudgetReached:
                pass
        save_dsrl(args, agent, env, contract)
        training_motor_samples = motor.samples
        evaluation_envs = [CanChunkEnv(args.dsrl_root, args.normalization) for _ in range(args.envs)]
        try:
            evaluation = scalar_evaluation(agent, evaluation_envs,
                                           list(range(2_000_000, 2_000_000 + args.eval_trials)), attempts=args.attempts, seed=args.seed)
            write_json(args.output / "evaluation.json", {"frozen": evaluation})
        finally:
            for e in evaluation_envs:
                e.close()
        from .can_dsrl_compare import _optimizers, _optimizer_steps
        optimized = {id(p): p for optimizer in _optimizers(agent).values()
                     for group in optimizer.param_groups for p in group["params"]}
        return dict(primitive_steps=warmup_steps + agent.num_timesteps * CHUNK,
                    warmup_primitive_steps=warmup_steps, learner_updates=agent._n_updates,
                    evaluation_primitive_steps=args.eval_trials * args.attempts * HORIZON,
                    budget_reached=warmup_steps + agent.num_timesteps * CHUNK < args.steps,
                    training_motor_samples=training_motor_samples, evaluation_motor_samples=motor.samples - training_motor_samples,
                    optimizer_steps=_optimizer_steps(agent), trainable_parameters=sum(p.numel() for p in optimized.values()))
    finally:
        env.close()


def qualify(args, motor, config, contract):
    env = CanChunkEnv(args.dsrl_root, args.normalization)
    try:
        policy = policy_from_args(args)
        frozen = fingerprint(motor.base_policy)
        initial, _ = env.reset(seed=2_000_001)
        initial = torch.as_tensor(initial[None], device="cuda")
        noise = torch.randn(1, CHUNK, ACTION_DIM, device="cuda")
        with torch.no_grad():
            wrapped = motor(initial, noise, return_numpy=False)
            direct = motor.base_policy(cond={"state": initial, "noise_action": noise}, deterministic=True).trajectories
        sampler_error = float((wrapped - direct).abs().max())
        if sampler_error > 1e-6:
            raise RuntimeError("Released sampler and motor wrapper differ")
        torch.cuda.synchronize()
        small_started = time.monotonic()
        rollout, row = collect_trial(policy, motor, [env], [2_000_001], attempts=2)
        torch.cuda.synchronize()
        small_trial_seconds = time.monotonic() - small_started
        hashes = row["initial_observation_sha256"]
        if hashes[0] != hashes[1]:
            raise RuntimeError("Retry reset did not reproduce the initial observation")
        with torch.no_grad():
            replay = policy.evaluate_sequence(rollout["observations"], rollout["raw_actions"], rollout["resets"])[0]
        error = float((replay - rollout["log_probs"]).abs().max())
        if error > 1e-4:
            raise RuntimeError("Likelihood replay failed")
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        metrics = update_ppo(policy, optimizer, rollout, epochs=1, minibatch_trials=1)
        if not metrics["gradient_norm"] > 0:
            raise RuntimeError("Zero PPO learning gradient")
        ppo_recovery = qualify_ppo_recovery(args, policy, optimizer, np.random.default_rng(args.seed),
                                           motor, [env], contract)
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec = DummyVecEnv([lambda: CanChunkEnv(args.dsrl_root, args.normalization)])
        try:
            agent = make_dsrl(vec, motor)
            agent._setup_learn(total_timesteps=1, callback=None)
            obs = vec.reset()
            for _ in range(HORIZON // CHUNK):
                actions = motor(torch.as_tensor(obs, device="cuda"), torch.randn(1, CHUNK, ACTION_DIM, device="cuda"))
                following, reward, done, infos = vec.step(actions)
                agent.replay_buffer.add(obs, following, actions.reshape(1, -1), reward, done, infos)
                obs = following
            agent._last_obs = obs
            agent._last_episode_starts = np.ones(1, dtype=bool)
            agent.train(gradient_steps=1, batch_size=8)
            if agent._n_updates != 1:
                raise RuntimeError("DSRL did not advance its optimizer")
            save_dsrl(args, agent, vec, contract)
            save_params, _ = agent._get_torch_save_params()
            if "critic_noise.optimizer" not in save_params:
                raise RuntimeError("Missing noise-critic optimizer recovery")
            manifest = json.loads((args.output / "recovery-manifest.json").read_text())
            saved_dir = args.output / manifest["directory"]
            restored = dsrl_class().load(saved_dir / "recovery.zip", env=vec, device="cuda", diffusion_policy=motor)
            restored.load_replay_buffer(saved_dir / "replay.pkl")
            if (fingerprint(restored.policy) != fingerprint(agent.policy)
                    or fingerprint(restored.critic_noise) != fingerprint(agent.critic_noise)
                    or not restored.critic_noise.optimizer.state):
                raise RuntimeError("Incomplete DSRL recovery roundtrip")
            runtime = torch.load(saved_dir / "runtime.pt", weights_only=False)
            # An episode-start snapshot is sufficient because a deterministic
            # reset reinitializes MuJoCo controllers, clocks and observation caches.
            vec.env_method("restore", runtime["simulator"][0])
            restored._last_obs = runtime["last_obs"]
            restored._last_episode_starts = runtime["last_episode_starts"]
            saved_rng = runtime["rng"]
            restore_rng(saved_rng)
            expected_actions, _ = agent.predict_diffused(obs, deterministic=False)
            expected_transition = vec.step(expected_actions)
            vec.env_method("restore", runtime["simulator"][0])
            restore_rng(saved_rng)
            actual_actions, _ = restored.predict_diffused(obs, deterministic=False)
            actual_transition = vec.step(actual_actions)
            if not np.array_equal(actual_actions, expected_actions):
                raise RuntimeError("Recovery changed next sampled actions")
            if any(not np.array_equal(a, b) for a, b in zip(actual_transition[:3], expected_transition[:3], strict=True)):
                raise RuntimeError("Recovery changed next simulator transition")
            restored.set_logger(agent.logger)
            restore_rng(saved_rng)
            agent.train(gradient_steps=1, batch_size=8)
            restore_rng(saved_rng)
            restored.train(gradient_steps=1, batch_size=8)
            if (fingerprint(restored.policy) != fingerprint(agent.policy)
                    or fingerprint(restored.critic_noise) != fingerprint(agent.critic_noise)
                    or not torch.equal(restored.log_ent_coef, agent.log_ent_coef)):
                raise RuntimeError("Recovery changed next optimizer update")
            del restored
        finally:
            vec.close()
        if fingerprint(motor.base_policy) != frozen:
            raise RuntimeError("Frozen motor changed")
        from .can_dsrl_parity import qualify_environment
        environment_parity = qualify_environment(args, motor)
        # Measure the released batch size and complete 20+10 update schedule,
        # rather than extrapolating from the small recovery fixture above.
        torch.cuda.synchronize()
        full_update_started = time.monotonic()
        samples_before = motor.samples
        agent.train(gradient_steps=20, batch_size=256)
        torch.cuda.synchronize()
        full_update_seconds = time.monotonic() - full_update_started
        full_update_samples = motor.samples - samples_before
        del agent
        # Measure the exact training geometry used by the pilot, including its
        # whole-trial likelihood replay and all four PPO epochs. No truncated
        # sequence or detached attention cache substitutes for this update.
        if args.envs != 4 or args.attempts != 4:
            raise ValueError("Qualification requires the default four lanes and four attempts")
        full_envs = [env]
        try:
            for _ in range(3):
                full_envs.append(CanChunkEnv(args.dsrl_root, args.normalization))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            full_started = time.monotonic()
            full_rollout, full_row = collect_trial(policy, motor, full_envs,
                                                  [2_000_020 + i for i in range(4)], attempts=4)
            torch.cuda.synchronize()
            rollout_seconds = time.monotonic() - full_started
            replay_started = time.monotonic()
            with torch.no_grad():
                full_replay = policy.evaluate_sequence(full_rollout["observations"], full_rollout["raw_actions"],
                                                        full_rollout["resets"])[0]
            full_error = float((full_replay - full_rollout["log_probs"]).abs().max())
            torch.cuda.synchronize()
            replay_seconds = time.monotonic() - replay_started
            if full_error > 1e-4:
                raise RuntimeError("Full-trial old-policy likelihood replay failed")
            update_started = time.monotonic()
            full_metrics = update_ppo(policy, optimizer, full_rollout, epochs=4, minibatch_trials=4,
                                      gamma=1., gae_lambda=1.)
            torch.cuda.synchronize()
            ppo_update_seconds = time.monotonic() - update_started
            total_seconds = time.monotonic() - full_started
            peak_memory = torch.cuda.max_memory_allocated() / 2**30
            if not full_metrics["gradient_norm"] > 0 or not all(np.isfinite(v) for v in full_metrics.values()):
                raise RuntimeError("Full-trial PPO did not produce a finite nonzero update")
        finally:
            for lane in full_envs[1:]:
                lane.close()
        return dict(passed=True, real_trial=row, ppo=metrics, likelihood_replay_max_error=error,
                    dsrl_optimizer_updates=22, motor_unchanged=True, sampler_parity_max_error=sampler_error,
                    released_batch_update_seconds=full_update_seconds,
                    released_batch_update_motor_samples=full_update_samples,
                    ppo_recovery_roundtrip=ppo_recovery, ppo_small_trial_seconds=small_trial_seconds,
                    ppo_full_trial_geometry=dict(lanes=4, attempts=4, chunks=300, epochs=4, minibatch_trials=4),
                    ppo_full_trial_rollout_seconds=rollout_seconds, ppo_full_trial_replay_seconds=replay_seconds,
                    ppo_full_trial_update_seconds=ppo_update_seconds, ppo_full_trial_total_seconds=total_seconds,
                    ppo_full_trial_peak_cuda_memory_gib=peak_memory, ppo_full_trial_metrics=full_metrics,
                    ppo_full_trial_likelihood_replay_max_error=full_error, ppo_full_real_trial=full_row,
                    environment_parity=environment_parity,
                    dsrl_recovery_roundtrip=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["qualify", "native", "dsrl-na", "dsrl-na-retry", "dsrl-online", "rl2", "episode-reset"], required=True)
    parser.add_argument("--dsrl-root", type=Path, required=True)
    parser.add_argument("--robomimic-root", type=Path, required=True,
                        help="Clean upstream MuJoCo import compatibility source")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--normalization-sha256", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=48_000)
    parser.add_argument("--core", choices=["gru", "gtrxl"], default="gtrxl")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--memory-length", type=int, default=300)
    parser.add_argument("--eval-trials", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=3000)
    parser.add_argument("--resume", type=Path, help="Prior complete recovery directory; keep its outputs immutable")
    parser.add_argument("--dsrl-reference", type=Path, help="Trained DSRL output for paired online adaptation")
    parser.add_argument("--gpu-attestation", type=Path, help="Fresh parent allocation probe for an isolated worker")
    args = parser.parse_args()
    from fluffy_bassoon.gpu_execution import validate_execution
    if sys.platform != "linux":
        raise RuntimeError("Model work requires allocated Linux")
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "slurm"), os.environ)
    if args.envs != 4 and args.method in ("dsrl-na", "dsrl-na-retry"):
        raise ValueError("Original DSRL reproduction uses four environments")
    if args.method in ("dsrl-na", "dsrl-na-retry") and args.steps % (args.envs * HORIZON):
        raise ValueError("DSRL target steps must end at a complete episode boundary")
    if args.method in ("dsrl-na", "dsrl-na-retry") and args.steps <= 1501 * args.envs * CHUNK:
        raise ValueError("DSRL training target must exceed the released 24,016-step warmup")
    if min(args.envs, args.attempts, args.steps, args.eval_trials, args.width,
           args.layers, args.heads, args.memory_length, args.max_seconds) <= 0:
        raise ValueError("Counts and time budget must be positive")
    if args.core == "gtrxl" and args.width % args.heads:
        raise ValueError("GTrXL width must be divisible by its number of attention heads")
    root = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit:
        raise ValueError("Run source commit differs")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Run worktree must be clean")
    if sha256(args.checkpoint) != args.checkpoint_sha256 or sha256(args.normalization) != args.normalization_sha256:
        raise ValueError("Frozen input artifact hash differs")
    compatibility_commit = "8df6e8e482458befaf1decfcd9d635d68a7afcae"
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.robomimic_root, text=True).strip() != compatibility_commit:
        raise ValueError("RoboMimic compatibility source commit differs")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=args.robomimic_root, text=True).strip():
        raise ValueError("RoboMimic compatibility source must stay clean")
    from importlib.util import find_spec
    installed_env_source = Path(find_spec("robomimic").origin).parent / "envs/env_robosuite.py"
    source_env = args.robomimic_root / "robomimic/envs/env_robosuite.py"
    expected_env_sha256 = "43b0f018be42f776eb76fca3e5a7a40318c112883be53ae200b82c45da1c7183"
    if sha256(installed_env_source) != expected_env_sha256 or sha256(source_env) != expected_env_sha256:
        raise ValueError("Installed RoboMimic environment differs from the pinned compatibility source")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    try:
        if args.gpu_attestation:
            from .can_dsrl_gpu import validate_prequalified_worker
            validate_prequalified_worker(args.gpu_attestation, args.source_commit, args.output / "gpu-probe.json")
        else:
            subprocess.run([sys.executable, str(root / "scripts/[train]gpu_probe.py"), "--expected-count", "1",
                            "--output", str(args.output / "gpu-probe.json"), "--purpose", "training",
                            "--baige-container"], check=True, timeout=90)
            if args.method == "qualify":
                from .can_dsrl_gpu import validate_prequalified_worker
                validate_prequalified_worker(args.output / "gpu-probe.json", args.source_commit,
                                             args.output / "gpu-worker-attestation.json")
        motor, config = load_motor(args.dsrl_root, args.checkpoint, args.normalization)
        contract = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        contract.update(frozen_motor_sha256=fingerprint(motor.base_policy), job=socket.gethostname(),
                        robomimic_commit=compatibility_commit, robomimic_env_sha256=expected_env_sha256,
                        runtime={name: version(name) for name in ("torch", "numpy", "robomimic", "robosuite", "mujoco")},
                        reward="original sum(raw_success - 1) per four-step chunk", dynamics="unchanged Can",
                        trial="same initial scene retried across attempts; no hidden-physics variation",
                        baseline=f"authors DSRL-NA; {args.core} whole-trial PPO; method comparison, not capacity matched")
        write_json(args.output / "contract.json", contract)
        if args.method == "qualify":
            result = qualify(args, motor, config, contract)
        elif args.method in ("rl2", "episode-reset"):
            result = train_ppo(args, motor, contract, started + args.max_seconds)
        elif args.method in ("dsrl-na", "dsrl-na-retry"):
            result = train_dsrl(args, motor, config, contract, started + args.max_seconds)
        elif args.method == "dsrl-online":
            if not args.dsrl_reference:
                raise ValueError("Online adaptation requires --dsrl-reference")
            from .can_dsrl_compare import evaluate_online
            result = evaluate_online(args, motor, started + args.max_seconds)
        else:
            envs = [CanChunkEnv(args.dsrl_root, args.normalization) for _ in range(args.envs)]
            try:
                rows = scalar_evaluation(None, envs, list(range(2_000_000, 2_000_000 + args.eval_trials)),
                                         attempts=args.attempts, native=True, motor=motor, seed=args.seed)
                write_json(args.output / "evaluation.json", {"native": rows})
                result = dict(primitive_steps=args.eval_trials * args.attempts * HORIZON, training=False)
            finally:
                for e in envs:
                    e.close()
        if fingerprint(motor.base_policy) != contract["frozen_motor_sha256"]:
            raise RuntimeError("Frozen motor changed")
        result.setdefault("completed", not result.get("budget_reached", False))
        result.update(elapsed_seconds=time.monotonic() - started,
                      motor_batch_calls=motor.calls, motor_samples=motor.samples,
                      peak_cuda_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(args.output / "summary.json", result)
        print(json.dumps(result, allow_nan=False), flush=True)
    except BaseException as error:
        write_json(args.output / "failure.json", dict(type=type(error).__name__, error=str(error),
                                                      elapsed_seconds=time.monotonic() - started))
        raise


if __name__ == "__main__":
    main()
