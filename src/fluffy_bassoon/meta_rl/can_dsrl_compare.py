"""Held-out Can retry trials for the original DSRL-NA online learner.

Each scene starts from the same immutable training recovery. Only its own four
attempts can update that copy of the learner. Imports from the Can driver stay
inside functions so the driver may call this module without a circular import.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np


def scene_rng_seed(scene_seed, seed=1):
    """Public pairing rule; frozen evaluators should apply this before a scene."""
    return (int(scene_seed) + int(seed)) % (2**32)


def seed_scene(scene_seed, seed=1):
    import torch
    value = scene_rng_seed(scene_seed, seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    return value


def flatten_replay_buffer(source, *, extra_transitions, buffer_class=None):
    """Copy each occupied (time, lane) once into a single-lane replay.

    DSRL-NA stores decoded motor actions, rather than its chosen noise. Preserve
    all arrays and timeout masks, including optional noise actions. Uniform
    sampling has the same transition distribution after flattening. Spare room
    prevents the release's full-ring sampling bug during a bounded retry trial.
    """
    if source.optimize_memory_usage or source.offline_mix_ratio != -1:
        raise ValueError("Require ordinary DSRL replay with offline_mix_ratio=-1")
    if extra_transitions < 0:
        raise ValueError("extra_transitions must be nonnegative")
    if buffer_class is None:
        from stable_baselines3.common.buffers import ReplayBuffer
        buffer_class = ReplayBuffer
    if source.full:
        indices = (np.arange(source.buffer_size) + source.pos) % source.buffer_size
    else:
        indices = np.arange(source.pos)
    count = len(indices) * source.n_envs
    if count == 0:
        raise ValueError("Online comparison requires nonempty pretrained replay")
    capacity = max(source.buffer_size * source.n_envs, count + extra_transitions + 1)
    target = buffer_class(capacity, source.observation_space, source.action_space,
                          device=source.device, n_envs=1, optimize_memory_usage=False,
                          handle_timeout_termination=source.handle_timeout_termination,
                          offline_mix_ratio=-1)
    names = ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]
    if hasattr(source, "noise_actions"):
        names.append("noise_actions")
    for name in names:
        array = getattr(source, name)
        flat = array[indices].reshape(count, 1, *array.shape[2:])
        getattr(target, name)[:count] = flat
    target.pos, target.full = count, False
    target.use_noise_actions = getattr(source, "use_noise_actions", False)
    offline_steps = getattr(source, "offline_steps", -1)
    target.offline_steps = min(count, offline_steps * source.n_envs) if offline_steps >= 0 else -1
    return target


def _hash_structure(value):
    """Fingerprint nested model/optimizer payloads without pickle metadata."""
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda entry: (str(type(entry)), str(entry))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for entry in item:
                visit(entry)
        elif torch.is_tensor(item):
            digest.update(str(item.dtype).encode() + str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str(item.dtype).encode() + str(item.shape).encode() + item.tobytes())
        else:
            digest.update(type(item).__name__.encode() + repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _optimizers(agent):
    optimizers = dict(actor=agent.actor.optimizer, critic=agent.critic.optimizer,
                      noise_critic=agent.critic_noise.optimizer)
    if agent.ent_coef_optimizer is not None:
        optimizers["entropy"] = agent.ent_coef_optimizer
    return optimizers


def _optimizer_steps(agent):
    # Adam's `step` is per parameter. Its maximum is the optimizer invocation
    # count here; freshly initialized, unused parameters contribute zero.
    return {name: max((int(state.get("step", 0)) for state in optimizer.state.values()), default=0)
            for name, optimizer in _optimizers(agent).items()}


def _learner_hashes(agent):
    from .can_dsrl import fingerprint
    hashes = {name: fingerprint(getattr(agent, name))
              for name in ("actor", "critic", "critic_target", "critic_noise")}
    hashes["optimizers"] = _hash_structure({name: optimizer.state_dict()
                                            for name, optimizer in _optimizers(agent).items()})
    if agent.log_ent_coef is not None:
        hashes["log_ent_coef"] = _hash_structure(agent.log_ent_coef)
    return hashes


def run_online_trial(agent, env, *, scene_seed, attempts, deadline, seed=1,
                     clock=time.monotonic):
    """Adapt within one scene, stopping at a resumable episode boundary.

    A started episode completes even if it crosses the deadline. The maximum
    scheduling overrun is one episode, including its online optimizer work.
    """
    import torch
    from .can_dsrl import CHUNK, HORIZON
    if attempts < 1 or agent.gradient_steps != 20 or agent.noise_critic_grad_steps != 10:
        raise ValueError("Comparison requires K>=1 and released DSRL-NA 20+10 update schedule")
    if agent.actor_gradient_steps >= 0:
        raise ValueError("Comparison requires an actor step for every critic step")
    paired_rng_seed = seed_scene(scene_seed, seed)
    started = clock()
    updates_before = _optimizer_steps(agent)
    learner_updates_before = int(agent._n_updates)
    steps, chunks, train_calls = 0, 0, 0
    rows = []
    stop_reason = None
    for attempt in range(attempts):
        if clock() >= deadline:
            stop_reason = "deadline_before_attempt"
            break
        obs, _ = env.reset(seed=int(scene_seed))
        initial_hash = hashlib.sha256(obs.tobytes()).hexdigest()
        if rows and initial_hash != rows[0]["initial_observation_sha256"]:
            raise RuntimeError("Retry reset did not reproduce the initial scene")
        row = dict(attempt=attempt + 1, success=False, primitive_steps=0, chunks=0,
                   episode_complete=False, adaptation_complete=False,
                   initial_observation_sha256=initial_hash)
        rows.append(row)
        for _ in range(HORIZON // CHUNK):
            with torch.no_grad():
                decoded, _ = agent.predict_diffused(obs[None], deterministic=False)
            action = np.asarray(decoded).reshape(1, -1)
            following, reward, terminated, truncated, info = env.step(action[0])
            primitive = int(info["primitive_steps"])
            if primitive != CHUNK:
                raise RuntimeError("Can chunk changed its primitive-step contract")
            done = bool(terminated or truncated)
            replay_info = dict(info, **{"TimeLimit.truncated": bool(truncated and not terminated)})
            agent.replay_buffer.add(obs[None], following[None], action, np.array([reward]),
                                    np.array([done]), [replay_info])
            obs = following
            chunks, steps = chunks + 1, steps + primitive
            row["chunks"] += 1
            row["primitive_steps"] += primitive
            row["success"] = bool(info["success_once"])
            row["episode_complete"] = done
            # `train()` uses saved training progress for its learning-rate
            # schedule. Do not restart that schedule with `_setup_learn()`.
            agent.num_timesteps += 1
            agent._last_obs = obs[None]
            agent._last_episode_starts = np.array([done])
            agent.train(gradient_steps=20, batch_size=agent.batch_size)
            train_calls += 1
            if done:
                if row["primitive_steps"] != HORIZON:
                    raise RuntimeError("Can episode ended before the unchanged 300-step horizon")
                row["adaptation_complete"] = True
                break
        if not row["adaptation_complete"]:
            raise RuntimeError("Can did not complete its declared 300-step episode")
    updates_after = _optimizer_steps(agent)
    optimizer_updates = {name: updates_after[name] - value for name, value in updates_before.items()}
    expected = {name: train_calls * (10 if name == "noise_critic" else 20) for name in updates_before}
    if optimizer_updates != expected or int(agent._n_updates) - learner_updates_before != 20 * train_calls:
        raise RuntimeError(f"DSRL optimizer schedule changed: {optimizer_updates}, expected {expected}")
    complete = len(rows) == attempts and all(row["adaptation_complete"] for row in rows)
    finished = clock()
    return dict(scene_seed=int(scene_seed), rng_seed=paired_rng_seed, attempts=attempts,
                attempt_records=rows,
                successes=[row["success"] if row["episode_complete"] else None for row in rows],
                initial_observation_sha256=[row["initial_observation_sha256"] for row in rows],
                primitive_steps=steps, chunks=chunks, adaptation_train_calls=train_calls,
                adaptation_learner_updates=int(agent._n_updates) - learner_updates_before,
                adaptation_optimizer_updates=optimizer_updates,
                adaptation_optimizer_steps_total=sum(optimizer_updates.values()),
                completed=complete, stop_reason=stop_reason, next_attempt_index=len(rows),
                elapsed_seconds=finished - started,
                deadline_overrun_seconds=max(0., finished - deadline))


def _reference_files(reference):
    from .can_dsrl import sha256
    reference = Path(reference).resolve()
    manifest_path = reference / "recovery-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    directory = (reference / manifest["directory"]).resolve()
    if not directory.is_relative_to(reference):
        raise ValueError("Recovery directory must stay inside reference output")
    files = {name: directory / name for name in ("recovery.zip", "replay.pkl", "runtime.pt")}
    for name, path in files.items():
        if sha256(path) != manifest["files"][name]:
            raise ValueError(f"Reference recovery hash mismatch: {name}")
    hashes = {name: manifest["files"][name] for name in files}
    hashes["recovery-manifest.json"] = sha256(manifest_path)
    return files, hashes


def _save_partial(directory, agent, row):
    import torch
    from .can_dsrl import rng_state, write_json
    directory.mkdir(parents=True, exist_ok=False)
    agent.save(directory / "recovery")
    agent.save_replay_buffer(directory / "replay.pkl")
    # The next action is a deterministic scene reset. No mid-episode simulator
    # snapshot is claimed to restore robosuite's hidden controller state.
    torch.save(dict(boundary="before_next_attempt_reset", scene_seed=row["scene_seed"],
                    next_attempt_index=row["next_attempt_index"], rng=rng_state(), row=row,
                    last_obs=agent._last_obs, last_episode_starts=agent._last_episode_starts),
               directory / "runtime.pt")
    write_json(directory / "trial.json", row)


def _load_fresh_learner(files, env, motor, extra_transitions):
    from stable_baselines3.common.logger import configure
    from .can_dsrl_run import dsrl_class
    agent = dsrl_class().load(files["recovery.zip"], env=env, device="cuda", diffusion_policy=motor)
    agent.load_replay_buffer(files["replay.pkl"])
    source = agent.replay_buffer
    agent.replay_buffer = flatten_replay_buffer(source, extra_transitions=extra_transitions)
    agent.set_logger(configure(folder=None, format_strings=[]))
    return agent


def evaluate_online(args, motor, deadline):
    """Evaluate original DSRL adaptation, bounded by an absolute monotonic time.

    Required args: dsrl_reference, output, dsrl_root, normalization, eval_trials,
    attempts, seed. `dsrl_reference` is an existing DSRL-NA training output. This
    function writes online-evaluation.json incrementally and returns its payload.
    It never rewrites the reference recovery or the frozen motor checkpoint.
    """
    import torch
    from .can_dsrl import (CHUNK, HORIZON, CanChunkEnv, fingerprint, restore_rng,
                           rng_state, write_json)
    if args.eval_trials < 1 or args.attempts < 1:
        raise ValueError("Positive held-out trial and attempt counts required")
    files, source_hashes = _reference_files(args.dsrl_reference)
    original_rng = rng_state()
    motor_hash = fingerprint(motor.base_policy)
    runtime = torch.load(files["runtime.pt"], map_location="cpu", weights_only=False)
    if runtime["contract"]["frozen_motor_sha256"] != motor_hash:
        raise ValueError("Reference DSRL and comparison use different frozen motors")
    result = dict(method="original_dsrl_na_online_adaptation", reference=str(Path(args.dsrl_reference).resolve()),
                  reference_sha256=source_hashes, frozen_motor_sha256=motor_hash,
                  training_source_contract=runtime["contract"],
                  training_num_timesteps=int(runtime["num_timesteps"]),
                  training_learner_updates=int(runtime["updates"]),
                  scene_mode="retry", requested_trials=args.eval_trials, attempts=args.attempts,
                  scenes=[], primitive_steps=0, adaptation_learner_updates=0,
                  adaptation_optimizer_steps_total=0, completed=False,
                  rng_rule="(scene_seed + training_seed) modulo 2**32, applied after learner load",
                  update_schedule=dict(gradient_steps_per_chunk=20, noise_critic_steps_per_chunk=10,
                                       learning_rate_schedule="preserved saved training progress"))
    output = Path(args.output) / "online-evaluation.json"
    expected_hashes = None
    try:
        for scene in range(2_000_000, 2_000_000 + args.eval_trials):
            if time.monotonic() >= deadline:
                result["stop_reason"] = "deadline_before_scene"
                break
            env = CanChunkEnv(args.dsrl_root, args.normalization)
            agent = None
            try:
                # Loading all saved optimizers and replay anew is deliberate:
                # no held-out scene may inherit another scene's adaptations.
                agent = _load_fresh_learner(files, env, motor, args.attempts * (HORIZON // CHUNK))
                if (int(agent.num_timesteps) != int(runtime["num_timesteps"])
                        or int(agent._n_updates) != int(runtime["updates"])):
                    raise RuntimeError("Reference agent and runtime counters disagree")
                fresh_hashes = _learner_hashes(agent)
                if expected_hashes is None:
                    expected_hashes = fresh_hashes
                    result["starting_learner_sha256"] = fresh_hashes
                elif fresh_hashes != expected_hashes:
                    raise RuntimeError("A held-out scene did not start from the same learner and optimizers")
                start_count = int(agent.replay_buffer.pos)
                row = run_online_trial(agent, env, scene_seed=scene, attempts=args.attempts,
                                       deadline=deadline, seed=args.seed)
                row.update(starting_learner_sha256=fresh_hashes,
                           starting_replay_transitions=start_count,
                           final_learner_sha256=_learner_hashes(agent))
                result["scenes"].append(row)
                for name in ("primitive_steps", "adaptation_learner_updates", "adaptation_optimizer_steps_total"):
                    result[name] += row[name]
                if not row["completed"]:
                    partial = Path(args.output) / "online-partial" / f"scene{scene}"
                    _save_partial(partial, agent, row)
                    row["partial_recovery"] = str(partial)
                    result["stop_reason"] = row["stop_reason"]
                write_json(output, result)
                print(json.dumps(dict(online_scene=scene, completed=row["completed"],
                                      primitive_steps=row["primitive_steps"],
                                      optimizer_steps=row["adaptation_optimizer_steps_total"])), flush=True)
                if not row["completed"]:
                    break
            finally:
                if agent is not None:
                    del agent
                env.close()
        result["completed"] = len(result["scenes"]) == args.eval_trials and all(
            row["completed"] for row in result["scenes"])
        if fingerprint(motor.base_policy) != motor_hash:
            raise RuntimeError("Online adaptation changed the frozen diffusion motor")
        _, final_hashes = _reference_files(args.dsrl_reference)
        if final_hashes != source_hashes:
            raise RuntimeError("Online evaluation changed the pretrained DSRL recovery")
        result["reference_unchanged"] = True
        result["motor_unchanged"] = True
        write_json(output, result)
        return result
    finally:
        restore_rng(original_rng)
