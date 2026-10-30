"""Opt-in complete-trial PPO collection with shared batched policy inference.

The environment owns isolated simulator lanes and the shared frozen motor.
This module owns only each lane's causal controller history. Returned trials
use the scalar collector schema and enter the unchanged PPO objective.
"""
from __future__ import annotations

import numpy as np
import torch

from .vla_noise_ppo import observation_hash, policy_inputs


class CollectionNotStarted(RuntimeError):
    """A stop request arrived before any new simulator interaction."""


@torch.no_grad()
def collect_trials_batched(learner, vector_env, tasks, episode_seeds_per_lane, *,
                           deterministic=False, reset_memory=None, reward_blind=False,
                           primitive_budget=None, should_stop=None, episode_callback=None,
                           controller_microbatch=0, decision_callback=None):
    """Collect one complete trial per lane without updating the shared policy.

    ``reset_batch(tasks,seeds)`` returns ``[B,D]`` features. ``reset_lanes``
    accepts selected indices/tasks/seeds and returns only their feature rows.
    ``step_batch(noises,active_mask=...)`` receives ``[B,*noise_shape]`` and
    returns features/rewards/terminated/truncated plus per-lane exact
    ``primitive_steps`` and ``success`` arrays. Inactive lanes execute zero
    commands. The environment must enforce each lane's episode horizon.

    A supplied primitive budget reserves the complete batch's worst case
    before reset; no partial or padded trial is returned. Stop requests after
    collection starts drain this complete batch, leaving a resumable learner
    boundary. Callbacks receive ``(lane_index, episode_receipt)``.

    ``controller_microbatch=1`` keeps separate lane states and evaluates only
    active lanes through the original B1 controller kernels, in lane order.
    The expensive frozen motor and simulators remain batched. This preserves
    scalar replay numerics without changing motor/TF32 precision. The default
    zero uses the full controller batch and requires its own numerical gate.
    """
    tasks = tuple(tasks)
    seeds = tuple(tuple(row) for row in episode_seeds_per_lane)
    size = len(tasks)
    c = learner.config
    if type(controller_microbatch) is not int or controller_microbatch not in (0, 1):
        raise ValueError("controller_microbatch must be zero (full batch) or one (scalar kernels)")
    if decision_callback is not None and controller_microbatch != 1:
        raise ValueError("Audited production decisions require scalar controller kernels")
    if (not size or len(seeds) != size or any(not row for row in seeds)
            or any(type(seed) is not int or seed < 0 for row in seeds for seed in row)
            or any(not isinstance(task, str) or not task for task in tasks)):
        raise ValueError("Require one task and nonempty nonnegative seed sequence per lane")
    if (tuple(vector_env.noise_shape) != learner.noise_shape
            or int(vector_env.observation_dim) != learner.observation_dim):
        raise ValueError("Vector environment and learner interfaces differ")
    limits = [(vector_env.episode_limit_for_task(task) if hasattr(vector_env, "episode_limit_for_task")
               else c.episode_limit) for task in tasks]
    if any(type(limit) is not int or not 1 <= limit <= c.episode_limit for limit in limits):
        raise ValueError("Native task episode limits must fit the policy's declared horizon bound")
    worst_case = sum(len(row)*limit for row, limit in zip(seeds, limits))
    if primitive_budget is not None and (type(primitive_budget) is not int
                                        or primitive_budget < worst_case):
        raise ValueError("Budget must reserve every complete trial before collection")
    if should_stop is not None and should_stop():
        raise CollectionNotStarted("Stopped at a complete-batch boundary before reset")
    reset_memory = c.reset_each_episode if reset_memory is None else bool(reset_memory)
    learner.policy.eval()
    versions = tuple(parameter._version for parameter in learner.policy.parameters())
    state = None
    lane_states = [None] * size
    previous_noise = np.zeros((size, learner.action_dim), np.float32)
    previous_reward = np.zeros(size, np.float32)
    episode = np.zeros(size, np.int64)
    primitive = np.zeros(size, np.int64)
    chunks = np.zeros(size, np.int64)
    reward_sum = np.zeros(size, np.float64)
    success = np.zeros(size, bool)
    active = np.ones(size, bool)
    keys = ("observations", "raw_actions", "log_probs", "values", "rewards",
            "trial_ends", "attempt_ends", "resets")
    records = [{key: [] for key in keys} for _ in tasks]
    record_limits = [[] for _ in tasks]
    episodes = [[] for _ in tasks]
    stop_requested = False

    def features_array(value, rows):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (rows, learner.observation_dim) or not np.isfinite(value).all():
            raise FloatingPointError("Invalid vector frozen-motor features")
        return value

    features = features_array(vector_env.reset_batch(tasks, [row[0] for row in seeds]), size)
    initial_hash = [observation_hash(row) for row in features]
    while active.any():
        if should_stop is not None:
            stop_requested |= bool(should_stop())
        inputs = np.stack([policy_inputs(features[lane], previous_noise[lane],
            previous_reward[lane], not c.reset_each_episode and episode[lane] > 0 and chunks[lane] == 0,
            episode[lane], primitive[lane], train_episodes=c.train_episodes,
            episode_limit=limits[lane], reward_blind=reward_blind) for lane in range(size)])
        tensor = torch.as_tensor(inputs, device=learner.device)
        resets = torch.as_tensor(active & (chunks == 0) & reset_memory, device=learner.device)
        if controller_microbatch == 1:
            rows = {}
            controller_streams = {}
            for lane in np.flatnonzero(active):
                # Scalar collection/replay starts from a fresh B1 allocation.
                # Clone the row to retain zero storage offset/alignment as well
                # as shape; D need not be divisible by a GEMM alignment size.
                if decision_callback is not None:
                    controller_streams[int(lane)] = dict(
                        torch_before=torch.get_rng_state().clone(),
                        cuda_before=torch.cuda.get_rng_state_all() if learner.device.type == "cuda" else [])
                result = learner.policy.step(tensor[lane:lane+1].clone(), lane_states[lane],
                    reset=resets[lane:lane+1], deterministic=deterministic)
                lane_states[lane] = result[-1]
                rows[int(lane)] = (result[0][0], result[1][0], result[2][0], result[4][0])
                if decision_callback is not None:
                    controller_streams[int(lane)].update(
                        torch_after=torch.get_rng_state().clone(),
                        cuda_after=torch.cuda.get_rng_state_all() if learner.device.type == "cuda" else [])
            template = next(iter(rows.values()))
            action, raw, log_prob, value = (torch.stack([
                rows[lane][field] if lane in rows else torch.zeros_like(template[field])
                for lane in range(size)]) for field in range(4))
        else:
            action, raw, log_prob, _, value, state = learner.policy.step(
                tensor, state, reset=resets, deterministic=deterministic)
        noises = action.cpu().numpy() * c.action_limit
        if decision_callback is not None:
            for lane in np.flatnonzero(active):
                decision_callback(int(lane), dict(
                    episode_index=int(episode[lane]), chunk_index=int(chunks[lane]),
                    episode_limit=limits[lane], history_index=len(records[lane]["rewards"]),
                    policy_update=int(learner.updates),
                    pre_squash_actions=raw[lane].detach().cpu().numpy().copy(),
                    old_log_prob=log_prob[lane].detach().cpu().numpy().copy(),
                    old_value=value[lane].detach().cpu().numpy().copy(),
                    policy_input=inputs[lane].copy(),
                    recurrent_reset_mask=resets[lane].detach().cpu().numpy().copy(),
                    previous_noise=previous_noise[lane].copy(),
                    previous_reward=float(previous_reward[lane]),
                    episode_boundary=bool(not c.reset_each_episode and episode[lane] > 0 and chunks[lane] == 0),
                    reset_episode_feedback=bool(c.reset_each_episode), reward_blind=bool(reward_blind),
                    controller_microbatch=1, collector_lane=int(lane),
                    controller_rng=controller_streams[int(lane)]))
        following, rewards, terminated, truncated, info = vector_env.step_batch(
            noises.reshape(size, *learner.noise_shape), active_mask=active.copy())
        following = features_array(following, size)
        rewards = np.asarray(rewards)
        terminated, truncated = np.asarray(terminated), np.asarray(truncated)
        counts = np.asarray(info["primitive_steps"])
        successes = np.asarray(info.get("success", np.zeros(size, bool)))
        cuts = np.asarray(info.get("budget_truncated", np.zeros(size, bool)))
        if (any(array.shape != (size,) for array in
                (rewards, terminated, truncated, counts, successes, cuts))
                or counts.dtype.kind not in "iu" or not np.isfinite(rewards).all()):
            raise ValueError("Invalid per-lane simulator outcome arrays")
        if (np.any(counts[~active] != 0) or np.any(counts[active] < 1)
                or np.any(counts[active] > int(vector_env.native_horizon))):
            raise ValueError("Require exact positive active-lane counts and zero inactive counts")
        if np.any(cuts[active]):
            raise RuntimeError("Complete-trial collection cannot accept an external budget cut")
        reset_lanes = []
        for lane in np.flatnonzero(active):
            primitive[lane] += int(counts[lane])
            if primitive[lane] > limits[lane]:
                raise RuntimeError("Simulator exceeded a lane's episode horizon")
            done = bool(terminated[lane] or truncated[lane])
            if primitive[lane] >= limits[lane] and not done:
                raise RuntimeError("Simulator must truncate at its episode horizon")
            trial_end = done and episode[lane] == len(seeds[lane])-1
            values = dict(observations=tensor[lane], raw_actions=raw[lane],
                log_probs=log_prob[lane], values=value[lane],
                rewards=tensor.new_tensor(float(rewards[lane])),
                trial_ends=torch.tensor(trial_end, device=learner.device),
                attempt_ends=torch.tensor(done, device=learner.device), resets=resets[lane])
            for key, item in values.items():
                records[lane][key].append(item.detach().cpu())
            record_limits[lane].append(torch.tensor(limits[lane], dtype=torch.int64))
            previous_noise[lane] = noises[lane]
            previous_reward[lane] = float(rewards[lane])
            reward_sum[lane] += float(rewards[lane])
            success[lane] |= bool(successes[lane])
            chunks[lane] += 1
            if not done:
                continue
            row = dict(task=tasks[lane], episode=int(episode[lane])+1,
                seed=seeds[lane][episode[lane]], success=bool(success[lane]),
                return_=float(reward_sum[lane]), primitive_steps=int(primitive[lane]),
                chunks=int(chunks[lane]), initial_observation_sha256=initial_hash[lane],
                episode_limit=limits[lane],
                reset_memory=reset_memory, reward_blind=bool(reward_blind),
                reset_episode_feedback=c.reset_each_episode, budget_truncated=False)
            episodes[lane].append(row)
            if episode_callback is not None:
                episode_callback(int(lane), row)
            if trial_end:
                active[lane] = False
            else:
                episode[lane] += 1
                primitive[lane] = chunks[lane] = 0
                reward_sum[lane] = 0.
                success[lane] = False
                if c.reset_each_episode:
                    previous_noise[lane] = 0.
                    previous_reward[lane] = 0.
                reset_lanes.append(int(lane))
        features = following.copy()
        if reset_lanes:
            replacements = features_array(vector_env.reset_lanes(reset_lanes,
                [tasks[lane] for lane in reset_lanes],
                [seeds[lane][episode[lane]] for lane in reset_lanes]), len(reset_lanes))
            features[reset_lanes] = replacements
            for lane in reset_lanes:
                initial_hash[lane] = observation_hash(features[lane])
    if tuple(parameter._version for parameter in learner.policy.parameters()) != versions:
        raise RuntimeError("Shared behavior-policy weights changed during collection")
    output = []
    for lane in range(size):
        rollout = {key: torch.stack(items)[:, None] for key, items in records[lane].items()}
        rollout["episode_limits"] = torch.stack(record_limits[lane])[:, None]
        rollout["action_mask"] = torch.ones_like(rollout["trial_ends"], dtype=torch.bool)
        result = dict(task=tasks[lane], episodes=episodes[lane],
            primitive_steps=sum(row["primitive_steps"] for row in episodes[lane]),
            chunks=sum(row["chunks"] for row in episodes[lane]),
            budget_truncated=False, requested_episodes=len(seeds[lane]))
        if should_stop is not None:
            result["stop_requested"] = stop_requested
        output.append((rollout, result))
    return output
