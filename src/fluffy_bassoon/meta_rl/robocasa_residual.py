"""RoboCasa365 binding for the existing memoryless residual PPO learner.

The motor is RLinf's frozen Pi0Eval, not a second sampler. Its actual prefix
pass supplies the features. The environment is RLinf's native365 wrapper;
only accounting, raw video and terminal-truth isolation are added here.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .residual_ppo import config_record
from .rlinf_native_ppo import cpu_tree, restore_rng, sha, snapshot_rng
from .warmstart import atomic_save


PUBLIC_OBSERVATION_KEYS = ("main_images", "wrist_images", "extra_view_images", "states", "task_descriptions")


def public_observation(observation):
    return {key: observation[key] for key in PUBLIC_OBSERVATION_KEYS if key in observation}


class FrozenRoboCasaMotor:
    """One native H50/10-step draw, with a reproducible private seed per chunk.

    Features are normalized public proprio16, decoded first/mean action12,
    and the masked mean of the SAME prefix pass, RMS scaled without fitting.
    No task ID, simulator information, reward or history enters the learner.
    """
    def __init__(self, model, norm_stats, *, seed=42):
        if model.action_horizon != 50 or model.num_steps != 10 or model.action_env_dim != 12:
            raise ValueError("Require released native H50/10-step/12D Pi0Eval")
        self.model = model.eval().requires_grad_(False)
        self.seed_rng = np.random.default_rng(seed)
        self.stats = norm_stats
        self.last = None

    @torch.no_grad()
    def infer(self, observation, *, noise_seed=None):
        seed = int(self.seed_rng.integers(0, 2**63 - 1)) if noise_seed is None else int(noise_seed)
        chunks, features = self._infer(observation, [seed], explicit_noise=False)
        self.last = dict(base_chunk=chunks[0].copy(), noise_seed=seed)
        return chunks[0], features[0], seed

    @torch.no_grad()
    def infer_batch(self, observation, *, noise_seeds):
        """Share one native forward pass; retain each lane's original noise draw."""
        seeds = [int(seed) for seed in noise_seeds]
        if not seeds:
            raise ValueError("A complete nonempty lane seed list is required")
        chunks, features = self._infer(observation, seeds, explicit_noise=True)
        return chunks, features, seeds

    def _infer(self, observation, seeds, *, explicit_noise):
        batch = len(seeds)
        generators = [torch.Generator(device=self.model.device).manual_seed(seed) for seed in seeds]
        sampler = dict(noise=torch.cat([torch.randn(
            (1, self.model.action_horizon, self.model.action_dim),
            device=self.model.device, generator=generator) for generator in generators], 0)) if explicit_noise else dict(rng=generators[0])
        captured = []
        original = self.model.build_prefix_cache

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            prefix, mask, _ = result
            if mask.ndim != 2 or not mask.any(1).all():
                raise ValueError("Native prefix must have valid tokens")
            valid = mask[..., None]
            pooled = prefix.float().masked_fill(~valid, 0).sum(1) / mask.sum(1, keepdim=True)
            pooled = pooled / pooled.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            captured.append(pooled.detach().cpu().numpy())
            return result

        self.model.build_prefix_cache = capture
        try:
            chunk, _ = self.model.predict_action_batch(public_observation(observation), mode="eval", **sampler)
        finally:
            self.model.build_prefix_cache = original
        chunk = np.asarray(chunk.detach().cpu() if torch.is_tensor(chunk) else chunk, dtype=np.float32)
        if chunk.shape != (batch, 50, 12) or len(captured) != 1 or captured[0].shape[0] != batch:
            raise ValueError("One actual prefix pass and one decoded H50 chunk per lane required")
        raw = np.asarray(observation["states"], dtype=np.float32)
        if raw.shape != (batch, 25):
            raise ValueError("Require native public state25 before published16D selection")
        state = raw[:, [*range(11, 25), 7, 8]]
        state_stats, action_stats = self.stats["state"], self.stats["actions"]
        normalize = lambda x, stats: (x - np.asarray(stats["mean"])[:x.shape[-1]]) / (np.asarray(stats["std"])[:x.shape[-1]] + 1e-6)
        features = np.concatenate((normalize(state, state_stats), normalize(chunk[:, 0], action_stats),
            normalize(chunk.mean(1), action_stats), captured[0]), -1).astype(np.float32)
        if not np.isfinite(chunk).all() or not np.isfinite(features).all():
            raise ValueError("Nonfinite native motor result")
        return chunk, features

    def state_dict(self):
        return deepcopy(self.seed_rng.bit_generator.state)

    def load_state_dict(self, value):
        self.seed_rng.bit_generator.state = deepcopy(value)


class RoboCasaResidualEnvironment:
    """Single native lane; full horizon and fresh constructor seed per case.

    factory(case) returns a native Robocasa365Env whose child factories use
    counted_native_factory. writer_factory is imageio.get_writer in production.
    Simulator labels go ONLY to monitoring_root. Returned info is truth free.
    """
    def __init__(self, factory, cases, *, horizon, output, monitoring_root, journal, writer_factory, observer=None,
                 episode_id_prefix="", rank=0, episode_index_offset=0):
        from .rlinf_robocasa_ppo import TASKS
        if (not cases or any(case['task'] not in TASKS for case in cases)
                or horizon is not None and (horizon not in {row[0] for row in TASKS.values()}
                    or horizon % 50 or any(TASKS[case['task']][0]!=horizon for case in cases))):
            raise ValueError("Bind native full episode horizon and a finite case bank")
        self.factory, self.cases, self.horizon = factory, deepcopy(cases), horizon
        self.bound_horizon = horizon  # None means each case's exact registered native horizon.
        self.output, self.monitoring_root = Path(output), Path(monitoring_root)
        self.output.mkdir(parents=True, exist_ok=True)
        self.monitoring_root.mkdir(parents=True, exist_ok=True)
        self.journal, self.writer_factory = journal, writer_factory
        self.observer = observer
        self.episode_id_prefix, self.rank = episode_id_prefix, int(rank)
        if type(episode_index_offset) is not int or episode_index_offset < 0:
            raise ValueError('Preserve a nonnegative original case index')
        self.episode_index_offset = episode_index_offset
        self.cursor, self.active = 0, False
        self.native = None

    def reset(self):
        if self.active or self.cursor >= len(self.cases):
            raise ValueError("Never abandon/recollect an episode or extend the fixed bank")
        if self.native is not None:
            native, self.native = self.native, None
            native.close()
        case = self.cases[self.cursor]
        from .rlinf_robocasa_ppo import TASKS
        self.horizon = TASKS[case['task']][0]
        self.key = f"{self.episode_id_prefix}episode-{self.cursor+self.episode_index_offset:06d}"
        self.case = case
        intent = self.journal.begin(128, phase="native_constructor_reset")
        self.native = self.factory(case)
        if self.native.auto_reset or not self.native.ignore_terminations:
            raise ValueError("Both arms require full horizon without success termination")
        observation, _ = self.native.reset()
        counts = self._counts()
        self.journal.finish(intent, counts["reset_controls"])
        self.reset_counts = deepcopy(counts)
        self.fps = float(counts["clock"]["control_freq"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Actual native frame clock missing")
        self.timestamps = [counts["control_time"]]
        self.steps, self.active = 0, True
        from .robocasa_video import EpisodeVideos
        self.writer = EpisodeVideos(self.output, self.key, fps=self.fps, writer_factory=self.writer_factory)
        self._frame(observation)
        self.cursor += 1
        return public_observation(observation)

    def _counts(self):
        rows = self.native.env.get_env_attr("_flora_accounting")
        if len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError("Require actual observed native controller/physics counters")
        return deepcopy(rows[0])

    def _frame(self, observation):
        from .robocasa_video import observation_frames
        self.writer.append(observation_frames(observation))

    def step_chunk(self, chunk):
        chunk = np.asarray(chunk)
        if not self.active or chunk.shape != (50, 12) or not np.isfinite(chunk).all() or self.steps + 50 > self.horizon:
            raise ValueError("One full finite H50 chunk within the native episode required")
        intent = self.journal.begin(50, phase="policy_controls")
        for action in chunk:
            observation, _, terminations, truncations, info = self.native.step(action[None], auto_reset=False)
            self.steps += 1
            if np.asarray(terminations).any() or bool(np.asarray(truncations).any()) != (self.steps == self.horizon):
                raise ValueError("Native masks must end exactly at the full horizon")
            counts = self._counts()
            self.timestamps.append(counts["control_time"])
            self._frame(observation)
        self.journal.finish(intent, 50)
        if self.observer is not None:
            self.observer()
        if self.steps != self.horizon:
            return public_observation(observation), False, {}
        if counts["controls"] - counts["reset_controls"] != self.horizon or not np.allclose(
                np.diff(self.timestamps), 1 / self.fps, rtol=0, atol=1e-7):
            raise ValueError("Observed physical/frame clock differs from native full horizon")
        self.writer.close()
        self.active = False
        from .robocasa_video import REWARD_CAMERAS
        episode = dict(id=self.key, task=self.native.task_descriptions[0], camera_order=list(REWARD_CAMERAS),
            horizon=self.horizon, completed=True, success_termination_disabled=True, raw_rgb_only=True,
                info_on_video=False, extra_info_on_video=[], rank=self.rank,
                round=self.cursor - 1 + self.episode_index_offset, local_env=0,
                videos=self.writer.refs(),
            frame_clock=dict(frame_count=self.horizon + 1, native_fps=self.fps, encoded_fps=self.fps,
                resampled=False, physical_steps=self.horizon, initial_frame_included=True))
        from .rlinf_flora_ppo import digest, write_new
        episode["frame_clock"]["timestamp_sha256"] = digest(self.timestamps)
        write_new(self.output / (self.key + ".json"), episode)
        truth = bool(np.asarray(info["episode"]["success_at_end"])[0])
        write_new(self.monitoring_root / (self.key + ".json"), dict(id=self.key, case=self.case,
            simulator_final_success=truth, native_counts=counts,
            native_ep_meta=deepcopy(self.native.current_info_list[0].get("ep_meta")),
            video_refs_sha256=digest(episode["videos"])))
        return public_observation(observation), True, episode

    def state_dict(self):
        if self.active:
            raise ValueError("Only complete episode boundaries are resumable")
        state = dict(cases=deepcopy(self.cases), cursor=self.cursor, horizon=self.bound_horizon)
        if self.episode_index_offset:
            state['episode_index_offset'] = self.episode_index_offset
        return state

    def load_state_dict(self, state):
        if (self.active or state["cases"] != self.cases or state["horizon"] != self.bound_horizon
                or state.get('episode_index_offset', 0) != self.episode_index_offset):
            raise ValueError("Cannot change native case layout at recovery")
        self.cursor = state["cursor"]

    def close(self):
        if self.active:
            self.writer.close()  # Incomplete evidence is retained, never scored.
        if self.native is not None:
            native, self.native = self.native, None
            native.close()


def native_environment_factory(cfg):
    """No constructor runs until a charged environment.reset()."""
    def factory(case):
        from omegaconf import OmegaConf
        from rlinf.envs.sim.robocasa365.robocasa365_env import Robocasa365Env
        from rlinf.envs.sim.robocasa.venv import RobocasaSubprocEnv
        from .rlinf_robocasa_ppo_env import counted_native_factory

        class CountedNative(Robocasa365Env):
            def _init_env(self):
                self._refresh_task_context()
                horizon = int(self.task_horizons[0])
                self.env = RobocasaSubprocEnv([counted_native_factory(fn, physics_tick_limit=(horizon + 128) * 128)
                    for fn in self.get_env_fns()])

        config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        config.seed = int(case["native_seed"])
        config.seed_strategy = "same"
        config.auto_reset, config.ignore_terminations = False, True
        config.task_filter = [case["task"]]
        from .rlinf_robocasa_ppo import TASKS
        horizon, soup, mode, _ = TASKS[case["task"]]
        config.task_soup, config.task_mode, config.split = soup, mode, "target"
        config.max_episode_steps = config.max_steps_per_rollout_epoch = horizon
        return CountedNative(config, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
    return factory


def collect_pending(policy, bound, motor, environment, *, episodes, generator, deterministic=False, verify_zero=False,
                    planned_motor_seeds=None, planned_action_noise=None):
    """Complete batch FIRST. No simulator or Flora reward is read here."""
    if hasattr(environment, "collect"):
        return environment.collect(policy, bound, motor, episodes=episodes, generator=generator,
                                   deterministic=deterministic, verify_zero=verify_zero)
    if planned_motor_seeds is None and planned_action_noise is not None:
        raise ValueError("Explicit actor noise requires a complete motor seed plan")
    reference = next(policy.parameters())
    records, finished = [], []
    for episode_index in range(episodes):
        observation, done, first = environment.reset(), False, True
        if hasattr(policy, "begin_episode"):
            policy.begin_episode()
        while not done:
            planned = planned_motor_seeds is not None
            index = len(records)
            if planned and index >= len(planned_motor_seeds):
                raise ValueError("Rollout exceeded the frozen full-episode draw schedule")
            chunk, features, seed = (motor.infer(observation, noise_seed=int(planned_motor_seeds[index]))
                                     if planned else motor.infer(observation))
            tensor = torch.as_tensor(features, device=reference.device, dtype=reference.dtype)
            noise = {} if planned_action_noise is None else {"action_noise": planned_action_noise[index:index + 1]}
            control, log_prob, value = policy.act(tensor[None], deterministic=deterministic, generator=generator, **noise)
            control = control[0].cpu().numpy()
            executed = bound.apply(np.asarray(chunk), control)
            verified = verify_zero and episode_index == 0
            if verified:
                base_reference, _, _ = motor.infer(observation, noise_seed=seed)
                if np.any(control != 0) or not np.array_equal(executed, base_reference):
                    raise ValueError("Zero-residual native sampler replay differs before physical execution")
            observation, done, info = environment.step_chunk(executed)
            if hasattr(policy, "observe"):
                policy.observe(control.astype(np.float32), 0., bool(done))
            records.append(dict(features=np.array(features, dtype=np.float32, copy=True), noise_seed=seed,
                base_chunk=np.array(chunk, copy=True), executed_chunk=np.array(executed, copy=True),
                control=control.astype(np.float32), log_prob=float(log_prob[0]), value=float(value[0]),
                reward=0., done=bool(done), episode_start=first, zero_residual_verified=verified))
            first = False
            if done:
                finished.append(info)
    if len(finished) != episodes:
        raise ValueError("Incomplete batch cannot enter reward labeling")
    if planned_motor_seeds is not None and len(records) != len(planned_motor_seeds):
        raise ValueError("Full episode draw schedule was not consumed exactly")
    return records, finished


def install_terminal_labels(records, episodes, labels, *, arm):
    """All labels or none; this function cannot fall back to simulator truth."""
    ids = [episode["id"] for episode in episodes]
    endings = [index for index, row in enumerate(records) if row["done"]]
    if arm not in ("S", "F") or len(set(ids)) != len(ids) or set(labels) != set(ids) or len(endings) != len(ids):
        raise ValueError("Exact complete batch label identities required")
    if not records or not records[-1]["done"] or any(row["reward"] != 0 for row in records):
        raise ValueError("Only an unlabeled complete pending batch can be installed")
    for key in ids:
        value = labels[key]
        if not np.isfinite(value) or not 0 <= value <= 1 or (arm == "S" and value not in (0, 1)):
            raise ValueError("Invalid terminal reward")
    result = deepcopy(records)
    for index, key in zip(endings, ids):
        result[index]["reward"] = float(labels[key])
    return result


def save_checkpoint(path, *, policy, optimizer, bound, motor, environment, generators,
                    contract, iteration, stage, records=None, episodes=None, final=False):
    """Full recovery, including pending unprocessed data; never overwrite a boundary."""
    path = Path(path)
    if path.exists() or stage not in ("boundary", "pending_labels") or final and stage != "boundary":
        raise ValueError("Fresh complete boundary or pre-label pending checkpoint required")
    state = dict(format="robocasa_residual_ppo_v1", config=config_record(policy.config, bound),
        contract=deepcopy(contract), policy=cpu_tree(policy.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
        iteration=iteration, stage=stage, final=bool(final), rng=snapshot_rng(), motor=motor.state_dict(),
        environment=environment.state_dict(), generators={key: value.get_state() for key, value in generators.items()},
        pending_records=records, pending_episodes=episodes)
    if stage == "pending_labels" and (not records or not records[-1]["done"] or not episodes):
        raise ValueError("Preserve the entire completed on-policy batch before any labels/GAE/update")
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_save(path, state)


def restore_checkpoint(path, *, policy, optimizer, bound, motor, environment, generators, contract):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state["format"] != "robocasa_residual_ppo_v1" or state["contract"] != contract
            or state["config"] != config_record(policy.config, bound) or set(state["generators"]) != set(generators)):
        raise ValueError("Exact motor/norm/learner/case/RNG contract required on recovery")
    policy.load_state_dict(state["policy"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    motor.load_state_dict(state["motor"])
    environment.load_state_dict(state["environment"])
    for key, generator in generators.items():
        generator.set_state(state["generators"][key])
    restore_rng(state["rng"])
    return state
