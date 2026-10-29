"""Allocated-runtime trajectory parity with the released Can environment wrappers.

This is deliberately separate from the simulator-free contract tests. It runs
the authors' wrapper chain and the local adapter on the same decoded action
tape, including every primitive transition of complete 300-step episodes.
"""
from __future__ import annotations

import hashlib
import os
import sys
from types import SimpleNamespace


def qualify_environment(args, motor):
    """Return real-simulator parity evidence; require allocated Linux execution.

    Required args are ``dsrl_root``, ``normalization`` and ``seed``. Two sampled
    scenes and a repeat of the first exercise scene changes and retry resets.
    Decoder queries generate a reference tape only; both environments execute
    precisely that tape. The helper restores caller RNG state before returning.
    """
    if sys.platform != "linux":
        raise RuntimeError("Environment parity requires allocated Linux")
    from fluffy_bassoon.gpu_execution import validate_execution

    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "slurm"), os.environ)

    import numpy as np
    import torch

    from .can_dsrl import (ACTION_DIM, CHUNK, HORIZON, OBS_DIM, CanChunkEnv,
                           configure_external, restore_rng, rng_state)

    root = configure_external(args.dsrl_root)
    from env_utils import (ActionChunkWrapper, ObservationWrapperRobomimic,
                           make_robomimic_env)

    class RecordingLow:
        """Observe primitive results without changing the released low wrapper."""

        def __init__(self, low):
            self.low = low
            self.actions, self.states, self.rewards, self.successes = [], [], [], []

        def __getattr__(self, name):
            return getattr(self.low, name)

        def reset(self, *positional, **keywords):
            self.actions, self.states, self.rewards, self.successes = [], [], [], []
            return self.low.reset(*positional, **keywords)

        def step(self, action):
            result = self.low.step(action)
            observation, reward = result[:2]
            success = bool(self.low.env.is_success()["task"])
            if success != bool(reward > 0):
                raise RuntimeError("Can raw reward no longer identifies simulator success")
            self.actions.append(np.asarray(action, dtype=np.float32).copy())
            self.states.append(np.asarray(observation["state"], dtype=np.float32).reshape(OBS_DIM).copy())
            self.rewards.append(float(reward))
            self.successes.append(success)
            return result

    def make_low():
        return RecordingLow(make_robomimic_env(
            env="can", normalization_path=str(args.normalization), dppo_path=str(root / "dppo"),
            low_dim_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
        ))

    def actor_observation(value):
        # The release feeds float32 states to the motor and SB3's policy. Its
        # raw normalization can produce float64, whereas our adapter casts here.
        return np.asarray(value, dtype=np.float32).reshape(OBS_DIM)

    def digest(values, dtype):
        return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()

    def same(actual, expected, label):
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            raise RuntimeError(f"Released Can wrapper parity failed: {label}")

    saved_rng = rng_state()
    reference_low, adapter = None, None
    try:
        reference_low = make_low()
        # This is exactly train_dsrl.py's low-dimensional wrapper chain.
        reference = ActionChunkWrapper(
            ObservationWrapperRobomimic(reference_low, reward_offset=1),
            SimpleNamespace(act_steps=CHUNK, obs_dim=OBS_DIM), max_episode_steps=HORIZON,
        )
        adapter = CanChunkEnv(low_env=make_low())
        generator = np.random.default_rng(args.seed)
        distinct = (3_000_000 + generator.choice(1_000_000, size=2, replace=False)).tolist()
        scenes = distinct + [distinct[0]]
        initial_by_scene, rows = {}, []
        device = next(motor.base_policy.parameters()).device
        for scene in scenes:
            before_reference = rng_state()
            # Both released reset wrappers ignore reset(seed=...). Seeding the
            # global NumPy stream immediately before reset matches low.reset's
            # documented options={"seed": ...} mechanism used by the adapter.
            np.random.seed(scene)
            initial, _ = reference.reset()
            initial = actor_observation(initial)
            initial_hash = digest(initial, np.float32)
            if scene in initial_by_scene and initial_by_scene[scene] != initial_hash:
                raise RuntimeError("Released Can did not reproduce a scene after an intervening reset")
            initial_by_scene[scene] = initial_hash
            current, tape, reference_chunks = initial, [], []
            for chunk in range(HORIZON // CHUNK):
                noise = torch.randn(1, CHUNK, ACTION_DIM, device=device)
                decoded = np.asarray(motor(torch.as_tensor(current[None], device=device), noise), dtype=np.float32)
                if decoded.shape != (1, CHUNK, ACTION_DIM) or not np.isfinite(decoded).all():
                    raise RuntimeError("Can parity received invalid decoded motor actions")
                action = decoded[0].reshape(-1).copy()
                following, reward, terminated, truncated, _ = reference.step(action)
                expected_end = chunk == HORIZON // CHUNK - 1
                if bool(terminated) != expected_end or bool(truncated):
                    raise RuntimeError("Released Can changed its 300-step termination contract")
                current = actor_observation(following)
                tape.append(action)
                reference_chunks.append((current.copy(), float(reward), bool(terminated), bool(truncated)))
            after_reference = rng_state()
            # Isolate shared process RNG between the two real simulator runs.
            # Simulator state comes from each environment's own ordinary reset.
            restore_rng(before_reference)
            actual_initial, _ = adapter.reset(seed=scene)
            same(actual_initial, initial, "initial float32 policy observation")
            adapter_chunks = []
            for chunk, action in enumerate(tape):
                following, reward, terminated, truncated, info = adapter.step(action)
                expected_observation, expected_reward, expected_end, expected_truncated = reference_chunks[chunk]
                same(following, expected_observation, f"chunk {chunk} float32 policy observation")
                same([reward, terminated, truncated],
                     [expected_reward, expected_end, expected_truncated], f"chunk {chunk} reward/termination")
                start, end = chunk * CHUNK, (chunk + 1) * CHUNK
                same(info["raw_reward"], sum(reference_low.rewards[start:end]), f"chunk {chunk} raw reward")
                same(info["success_once"], any(reference_low.successes[:end]), f"chunk {chunk} true success")
                same(info["primitive_steps"], CHUNK, f"chunk {chunk} primitive step count")
                adapter_chunks.append(float(reward))
            same(adapter.low.actions, reference_low.actions, "all primitive actions")
            same(adapter.low.states, reference_low.states, "all primitive normalized observations")
            same(adapter.low.rewards, reference_low.rewards, "all primitive raw rewards")
            same(adapter.low.successes, reference_low.successes, "all primitive simulator successes")
            if len(adapter.low.states) != HORIZON or reference.count != HORIZON:
                raise RuntimeError("Can parity did not execute exactly 300 primitive steps per environment")
            try:
                adapter.step(tape[-1])
            except RuntimeError:
                pass
            else:
                raise RuntimeError("Can adapter accepted a step after its fixed horizon")
            rows.append(dict(
                scene_seed=int(scene), primitive_steps_per_environment=HORIZON,
                chunks=HORIZON // CHUNK, initial_observation_sha256=initial_hash,
                primitive_actions_sha256=digest(reference_low.actions, np.float32),
                primitive_observations_sha256=digest(reference_low.states, np.float32),
                primitive_rewards_sha256=digest(reference_low.rewards, np.float64),
                raw_reward_sum=float(sum(reference_low.rewards)), shifted_reward_sum=float(sum(adapter_chunks)),
                simulator_success_once=bool(any(reference_low.successes)),
                release_logged_success=bool(any(item[1] > -1 for item in reference_chunks)),
                exact_float32_observations=True, exact_primitive_actions=True,
                exact_raw_rewards=True, exact_chunk_rewards=True, exact_horizon=True,
                final_terminated=True, final_truncated=False,
            ))
            restore_rng(after_reference)
        return dict(
            passed=True, reference="released ObservationWrapperRobomimic + ActionChunkWrapper",
            reset_pairing="NumPy seed before released reset; seed option through local adapter",
            observation_comparison="exact float32 policy inputs; release normalization may start as float64",
            success_metrics="simulator success at any primitive; release logger requires chunk reward > -1",
            scenes=rows, distinct_scenes=len(distinct), repeated_scene=int(distinct[0]),
            primitive_steps=2 * len(scenes) * HORIZON,
            decoder_samples=len(scenes) * (HORIZON // CHUNK),
        )
    finally:
        if adapter is not None:
            adapter.close()
        if reference_low is not None:
            raw = reference_low.low.env
            if hasattr(raw, "env") and hasattr(raw.env, "close"):
                raw.env.close()
        restore_rng(saved_rng)
