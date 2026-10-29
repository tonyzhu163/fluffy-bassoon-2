"""Frozen pi0.5 RoboTwin noise interface for both recurrent PPO and DSRL.

The real backend requires an allocated Linux GPU.  No parameters, observations,
action conventions, or task rewards are modified.  The module imports only
NumPy until that backend is constructed, so accounting contracts can be checked
with an injected backend on allocated Linux CPU resources.
"""
from __future__ import annotations

from concurrent.futures import Future
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import numpy as np


SOURCE_TASKS = ("handover_mic", "pick_diverse_bottles", "place_a2b_left",
                "place_object_stand", "place_phone_stand", "scan_object")
HELDOUT_TASKS = ("adjust_bottle", "move_stapler_pad", "place_a2b_right",
                 "place_dual_shoes", "place_empty_cup", "stamp_seal")
RLINF_COMMIT = "c605b9d55ed93a180119b099e6f836abff293b08"
ROBOTWIN_COMMIT = "dca9ec682688821c42944eacd7f5dd9bfb15396f"
MOTUS_REVISION = "effc6e1a99bd4d0edf9447e8ba313def67649cfe"
FEATURE_DIM = 2048 + 14
RESET_ATTEMPTS = 32


class _SynchronousExecutor:
    """Future-compatible execution on the simulator scene's owning thread.

    With one physical environment there is no parallel work to dispatch.
    RoboTwin otherwise creates a scene on its executor worker, then destroys
    that scene on the calling thread when switching tasks. SAPIEN's native
    render/scene resources require a consistent lifecycle thread.
    """
    def __init__(self):
        self.owner_thread = threading.get_ident()
        self.closed = False

    def submit(self, function, /, *args, **kwargs):
        if self.closed:
            raise RuntimeError("Cannot submit to a shut down simulator executor")
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("RoboTwin scene operations must stay on their owning thread")
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.closed = True


def _install_synchronous_vector_executor(venv):
    """Install immediately after construction, before the first reset/step."""
    if venv.n_envs != 1:
        raise RuntimeError("Synchronous RoboTwin adapter requires one physical environment")
    if isinstance(venv.env_thread_pool, _SynchronousExecutor):
        return
    original = venv.env_thread_pool
    # VectorEnv.__init__ creates SubEnv/setup_task synchronously; its executor
    # has not received work yet. No native scene has crossed threads here.
    original.shutdown(wait=True, cancel_futures=True)
    venv.env_thread_pool = _SynchronousExecutor()


def _setup_task_with_retry_limit(subenv, class_decorator, max_attempts=RESET_ATTEMPTS):
    """Preserve RoboTwin setup's seed order but expose persistent failures.

    The pinned upstream setup_task silently catches every Exception forever.
    A missing object therefore creates hundreds of scenes instead of reporting
    its cause. Successful setup and the first max_attempts seeds are unchanged.
    """
    subenv.close()
    subenv.task = class_decorator(subenv.task_name)
    with subenv.global_lock:
        with subenv.lock:
            first_seed = subenv.env_seed
            last_error = None
            for trial_seed in range(first_seed, first_seed + max_attempts):
                task = class_decorator(subenv.task_name)
                try:
                    task.setup_demo(now_ep_num=trial_seed, seed=trial_seed,
                                    is_test=True, **subenv.args)
                    episode_info = task.get_info()
                except Exception as error:
                    last_error = error
                    task.close_env()
                    continue
                task.close_env()
                subenv.episode_info_list = [episode_info]
                return
            raise RuntimeError(
                f"RoboTwin {subenv.task_name} setup failed for {max_attempts} seeds "
                f"starting at {first_seed}; last error: {type(last_error).__name__}: "
                f"{last_error}. Check assets under {os.environ.get('ASSETS_PATH', '<unset>')} "
                "before retrying; no episode was collected."
            ) from last_error


def _reset_with_retry_limit(subenv, unstable_error, env_seed=None,
                            max_attempts=RESET_ATTEMPTS):
    """Match pinned reset semantics, with a bound on unstable-scene retries."""
    with subenv.global_lock:
        with subenv.lock:
            if subenv.task is not None:
                subenv.reset_count += 1
                should_clear = subenv.reset_count % subenv.clear_cache_freq == 0
                subenv.task.close_env(clear_cache=should_clear)
            if env_seed is not None:
                subenv.env_seed = env_seed
            subenv.instruction = subenv.create_instruction()
            subenv.args["instruction"] = subenv.instruction
            first_seed = subenv.env_seed
            last_error = None
            for trial_seed in range(first_seed, first_seed + max_attempts):
                try:
                    subenv.task.setup_demo(now_ep_num=trial_seed, seed=trial_seed,
                                          **subenv.args)
                    subenv.task.step_lim = subenv.args["step_lim"]
                    subenv.task.run_steps = 0
                    subenv.task.reward_step = 0
                    return
                except unstable_error as error:
                    last_error = error
                    subenv.task.close_env(clear_cache=True)
                except Exception:
                    subenv.task.close_env(clear_cache=True)
                    raise
            raise RuntimeError(
                f"RoboTwin {subenv.task_name} reset remained unstable for "
                f"{max_attempts} seeds starting at {first_seed}; last error: "
                f"{type(last_error).__name__}: {last_error}; no episode was collected."
            ) from last_error


def _install_reset_retry_limit():
    """Patch only this process; external pinned source files remain unchanged."""
    from robotwin.envs import vector_env

    if getattr(vector_env.SubEnv, "_fb2_reset_attempts", None) == RESET_ATTEMPTS:
        return

    def setup_task(subenv):
        return _setup_task_with_retry_limit(subenv, vector_env.class_decorator)

    def reset(subenv, env_seed=None):
        return _reset_with_retry_limit(subenv, vector_env.UnStableError, env_seed)

    vector_env.SubEnv.setup_task = setup_task
    vector_env.SubEnv.reset = reset
    vector_env.SubEnv._fb2_reset_attempts = RESET_ATTEMPTS


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _observation_hash(obs):
    """Hash public observations without adding seed/task metadata as features."""
    digest = hashlib.sha256()
    for key in ("main_images", "wrist_images", "states"):
        value = obs[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.ascontiguousarray(value)
        digest.update(key.encode() + str(value.dtype).encode() + str(value.shape).encode())
        digest.update(value.tobytes())
    digest.update(json.dumps(list(obs["task_descriptions"]), ensure_ascii=False).encode())
    return digest.hexdigest()


def model_config(rlinf_root, model_path, norm_stats, action_horizon=32, num_steps=5):
    """Use the checkpoint's own delta-joint convention and native horizon.

    Motus declares adapt_to_pi=True and use_delta_joint_actions=True. RLinf's
    ALOHA data config implements exactly those input/output transforms with
    extra_delta_transform=True: delta arm joints, absolute grippers, then
    inverse AbsoluteActions before the simulator receives joint targets.
    """
    from omegaconf import OmegaConf

    root = Path(rlinf_root)
    return OmegaConf.merge(
        OmegaConf.load(root / "examples/embodiment/config/model/pi0_5.yaml"),
        dict(model_path=str(model_path), precision=None, pi05=True,
             num_action_chunks=action_horizon, action_dim=14, num_steps=num_steps,
             add_value_head=False,
             openpi_data=dict(norm_stats_path=str(norm_stats), adapt_to_pi=True,
                              extra_delta_transform=True),
             openpi=dict(task="eval", config_name="pi05_aloha_robotwin",
                         num_images_in_input=3, action_horizon=action_horizon,
                         action_chunk=action_horizon, action_env_dim=14,
                         model_action_dim=32, num_steps=num_steps,
                         torch_compile=False, max_token_len=200)))


def environment_config(rlinf_root, robotwin_root, task_name, seed, episode_limit,
                       *, task_config_name="demo_clean"):
    from omegaconf import OmegaConf

    if task_config_name not in ("demo_clean", "demo_randomized"):
        raise ValueError("task_config_name must be demo_clean or demo_randomized")
    randomize_appearance = task_config_name == "demo_randomized"
    root = Path(rlinf_root)
    cfg = OmegaConf.load(root / "examples/embodiment/config/env/robotwin_adjust_bottle.yaml")
    return OmegaConf.merge(cfg, dict(
        total_num_envs=1, auto_reset=False, ignore_terminations=False,
        seed=int(seed), group_size=1, use_fixed_reset_state_ids=True,
        seeds_path=None, assets_path=str(robotwin_root), center_crop=False,
        # Full observations expose actual completed/partly executed actions.
        # The fast sparse path does not report their count on early success.
        skip_intermediate_renders=False, is_eval=False,
        max_episode_steps=episode_limit, max_steps_per_rollout_epoch=episode_limit,
        reward_coef=1.0, use_custom_reward=True, use_rel_reward=True,
        video_cfg=dict(save_video=False, info_on_video=False, video_base_dir="unused"),
        task_config=dict(task_name=task_name, planner_backend="mplib",
                         embodiment=["aloha-agilex"], step_lim=episode_limit,
                         render_freq=0, eval_video_log=False,
                         camera=dict(collect_wrist_camera=True),
                         domain_randomization=dict(
                             random_background=bool(randomize_appearance),
                             cluttered_table=bool(randomize_appearance),
                             clean_background_rate=.02 if randomize_appearance else 1.0,
                             random_head_camera_dis=0,
                             random_table_height=.03 if randomize_appearance else 0,
                             random_light=bool(randomize_appearance),
                             crazy_random_light_rate=.02 if randomize_appearance else 0))))


def _install_allocated_renderer(records):
    """Process-local binding; leaves installed SAPIEN and RoboTwin untouched."""
    import sapien
    import sapien.wrapper.engine as engine
    from fluffy_bassoon.baige_gpu import pci, selected_device

    expected = pci(selected_device()["pci_bus_id"])
    records = Path(records)
    records.mkdir(parents=True, exist_ok=True)
    if getattr(engine.Engine.create_scene, "_fb2_bound_pci", None) == expected:
        return expected
    counter = 0

    def create_scene(self, config=None):
        nonlocal counter
        if config is None:
            config = sapien.physx.PhysxSceneConfig()
        sapien.physx.set_scene_config(config)
        scene = engine.Scene([sapien.physx.PhysxCpuSystem(),
                              sapien.render.RenderSystem(device="pci:" + expected)])
        device = scene.render_system.device
        actual = pci(device.pci_string.removeprefix("pci:"))
        if actual != expected or not device.can_render():
            raise RuntimeError("RoboTwin rendering is not bound to the allocated GPU")
        counter += 1
        (records / f"renderer-{os.getpid()}-{counter}.json").write_text(json.dumps(dict(
            pid=os.getpid(), expected_cuda_pci=expected, actual_render_pci=actual,
            render_device=device.name, physics="native CPU PhysX")) + "\n")
        return scene

    create_scene._fb2_bound_pci = expected
    engine.Engine.create_scene = create_scene
    return expected


class _RoboTwinBackend:
    observation_dim = FEATURE_DIM

    def __init__(self, model_path, norm_stats, robotwin_root, rlinf_root,
                 device="cuda", episode_limit=400, action_horizon=32,
                 num_steps=5, task_config_name="demo_clean", output_dir=None):
        from fluffy_bassoon.gpu_execution import validate_execution

        mode = os.environ.get("FLUFFY_GPU_EXECUTION", "baige")
        validate_execution(mode, os.environ)
        if sys.platform != "linux" or mode != "baige":
            raise RuntimeError("This initial RoboTwin adapter requires an allocated Linux Baige job")
        for root, expected in ((rlinf_root, RLINF_COMMIT), (robotwin_root, ROBOTWIN_COMMIT)):
            actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            if actual != expected:
                raise RuntimeError(f"Unqualified source revision at {root}: {actual}")
            dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
            if dirty:
                raise RuntimeError(f"External source is modified: {root}")
        # Validate every sampled object variant before loading the 3.35B motor.
        # Incomplete assets otherwise enter RoboTwin's silent setup retry loop.
        from fluffy_bassoon.rl.robotwin_assets import validate_robotwin_assets

        self.asset_receipt = validate_robotwin_assets(
            robotwin_root, SOURCE_TASKS + HELDOUT_TASKS)
        sys.path[:0] = [str(Path(rlinf_root).resolve()), str(Path(robotwin_root).resolve())]
        import torch
        from rlinf.models import get_model

        if torch.cuda.device_count() != 1 or str(device) not in ("cuda", "cuda:0"):
            raise RuntimeError("Require exactly one allocated visible GPU")
        self.torch = torch
        self.robotwin_root = Path(robotwin_root).resolve()
        self.rlinf_root = Path(rlinf_root).resolve()
        self.episode_limit = episode_limit
        if task_config_name not in ("demo_clean", "demo_randomized"):
            raise ValueError("task_config_name must be demo_clean or demo_randomized")
        self.task_config_name = task_config_name
        self.noise_shape = (int(action_horizon), 32)
        self.native_horizon = int(action_horizon)
        self.output_dir = Path(output_dir) if output_dir else None
        records = os.environ.get("FB2_ROBOTWIN_RENDER_RECORDS")
        if records is None and self.output_dir is not None:
            records = str(self.output_dir / "renderer-bindings")
        if not records:
            raise RuntimeError("Set output_dir or FB2_ROBOTWIN_RENDER_RECORDS for actual renderer evidence")
        self.render_pci = _install_allocated_renderer(records)
        self.cfg = model_config(rlinf_root, model_path, norm_stats, action_horizon, num_steps)
        self.model = get_model(self.cfg).to(device).eval().requires_grad_(False)
        self._versions = {name: value._version for name, value in self.model.named_parameters()}
        self.env = None
        self.task_name = None
        self.prefix_seconds = self.motor_seconds = 0.0
        self.prefix_calls = self.motor_queries = 0
        self._prepared = self._prefix_mask = self._kv_cache = None
        self._raw_obs = None
        self.contract = dict(
            model_path=str(model_path), norm_stats_path=str(norm_stats),
            norm_stats_sha256=_sha256(norm_stats),
            rlinf_commit=RLINF_COMMIT, robotwin_commit=ROBOTWIN_COMMIT,
            noise_shape=list(self.noise_shape), native_horizon=self.native_horizon,
            observation_dim=self.observation_dim,
            features="masked mean frozen final VLM prefix (2048), normalized proprioception (14)",
            action_transform="ALOHA adapt_to_pi + delta joints, absolute grippers, inverse AbsoluteActions",
            task_config_name=task_config_name,
            assets=self.asset_receipt, reset_max_attempts=RESET_ATTEMPTS,
            simulator_execution="one physical environment; scene setup/reset/step/close on calling thread",
            appearance_randomization=task_config_name == "demo_randomized",
            step_accounting="RoboTwin full-observation valid_action_mask, not issued length or dense run_steps",
            actual_render_pci=self.render_pci)

    def _encode(self, raw):
        from rlinf.models.embodiment.openpi.modules.model import preprocess_observation

        torch = self.torch
        views = [raw["main_images"][0], raw["wrist_images"][0, 0], raw["wrist_images"][0, 1]]
        for view in views:
            image = view.detach().cpu().numpy() if hasattr(view, "detach") else np.asarray(view)
            if (image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8
                    or float(image.std()) <= 1.0):
                raise RuntimeError("Require three nonblank uint8 RGB views from the real simulator")
        tick = time.monotonic()
        with torch.no_grad():
            observation = self.model.env_obs_to_observation(raw)
            prepared = preprocess_observation(observation, train=False)
            prefix, mask, cache = self.model.build_prefix_cache(prepared)
            valid = mask[..., None].float()
            mean = (prefix.float() * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            features = torch.cat([mean, observation.state[:, :14].float()], dim=-1)
        torch.cuda.synchronize()
        self.prefix_seconds += time.monotonic() - tick
        self.prefix_calls += 1
        self._prepared, self._prefix_mask, self._kv_cache = prepared, mask, cache
        self._raw_obs = raw
        result = features[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if result.shape != (self.observation_dim,) or not np.isfinite(result).all():
            raise RuntimeError("Invalid public frozen-VLA steering features")
        return result

    def _check_frozen(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Frozen motor contract changed")
        if any(p._version != self._versions[n] for n, p in self.model.named_parameters()):
            raise RuntimeError("Frozen motor parameters were modified")

    def reset(self, task_name, seed):
        # RoboTwin seeds NumPy and Torch globally during setup_demo. Keep that
        # scene randomization independent of the steering learner's streams.
        torch = self.torch
        saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 torch.cuda.get_rng_state_all())
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return self._reset_scene(task_name, seed)
        finally:
            random.setstate(saved[0])
            np.random.set_state(saved[1])
            torch.set_rng_state(saved[2])
            torch.cuda.set_rng_state_all(saved[3])

    def _reset_scene(self, task_name, seed):
        from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

        # RoboTwin's task assets/configuration resolve paths relative to its root.
        os.chdir(self.robotwin_root)
        _install_reset_retry_limit()
        if self.env is not None and self.task_name != task_name:
            self.env.venv.close()
            self.env = None
        if self.env is None:
            cfg = environment_config(self.rlinf_root, self.robotwin_root, task_name, seed,
                                     self.episode_limit,
                                     task_config_name=self.task_config_name)
            self.env = RoboTwinEnv(cfg, 1, seed_offset=0, total_num_processes=1, worker_info=None)
            _install_synchronous_vector_executor(self.env.venv)
            self.task_name = task_name
        raw, _ = self.env.reset(env_seeds=[int(seed)])
        subenv = self.env.venv.envs[0]
        task = subenv.task
        actual_seed = int(task.ep_num)
        renderer = task.scene.render_system.device
        from fluffy_bassoon.baige_gpu import pci
        actual_pci = pci(renderer.pci_string.removeprefix("pci:"))
        if actual_pci != self.render_pci or not renderer.can_render():
            raise RuntimeError("Task's actual renderer does not match allocated GPU")
        self._check_frozen()
        features = self._encode(raw)
        return features, dict(task=task_name, requested_seed=int(seed), actual_seed=actual_seed,
                              instruction=list(raw["task_descriptions"]),
                              observation_sha256=_observation_hash(raw),
                              features_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
                              actual_render_pci=actual_pci, reset_policy_actions=0,
                              reset_internal_physics_steps="not separately instrumented")

    def decode(self, noise):
        torch = self.torch
        tick = time.monotonic()
        with torch.no_grad():
            tensor = torch.as_tensor(noise, dtype=torch.float32, device=self.model.device)[None]
            targets = self.model._sample_actions_from_prefix_cache(
                self._prepared, self._prefix_mask, self._kv_cache, noise=tensor)
            actions = self.model.decode_actions(targets, self._prepared.state)
        torch.cuda.synchronize()
        self.motor_seconds += time.monotonic() - tick
        self.motor_queries += 1
        self._check_frozen()
        result = actions[0].detach().float().cpu().numpy()
        if result.shape != (self.native_horizon, 14) or not np.isfinite(result).all():
            raise RuntimeError("Motor must decode finite native ALOHA joint targets")
        return result

    def step(self, noise, action_count):
        actions = self.decode(noise)[:action_count]
        torch = self.torch
        obs, rewards, terms, truncs, infos = self.env.chunk_step(torch.from_numpy(actions[None]))
        info = infos[-1]
        mask = info.get("valid_action_mask")
        if mask is None:
            raise RuntimeError("Missing exact RoboTwin executed-action mask")
        actual = int(torch.as_tensor(mask).sum().item())
        if not 0 <= actual <= action_count:
            raise RuntimeError("Invalid RoboTwin executed-action count")
        success = bool(torch.as_tensor(info["episode"]["success_once"]).any())
        terminal = bool(torch.as_tensor(terms).any())
        truncated = bool(torch.as_tensor(truncs).any())
        # The wrapper pads observations but the final public observation is real.
        raw = obs[-1]
        features = self._encode(raw)
        return features, float(torch.as_tensor(rewards).sum()), terminal, truncated, dict(
            primitive_steps=actual, issued_primitive_steps=action_count,
            success=success, motor_queries=1, prefix_calls=1,
            observation_sha256=_observation_hash(raw),
            actions_sha256=hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
            prefix_seconds=self.prefix_seconds, motor_seconds=self.motor_seconds)

    def check_noise_parity(self, noise, atol=2e-5):
        """Compare cached-prefix actions with the native model's explicit-noise API."""
        torch = self.torch
        noise = np.asarray(noise, dtype=np.float32).reshape(self.noise_shape)
        cached = self.decode(noise)
        with torch.no_grad():
            actions, _ = self.model.predict_action_batch(
                self._raw_obs, mode="eval",
                noise=torch.as_tensor(noise, dtype=torch.float32, device=self.model.device)[None])
        native = actions[0].detach().float().cpu().numpy()
        error = float(np.max(np.abs(native - cached)))
        self._check_frozen()
        if error > atol:
            raise RuntimeError(f"Explicit-noise/native sampler parity failed: {error}")
        return dict(max_abs_error=error, tolerance=atol, passed=True)

    def frozen_fingerprint(self):
        self._check_frozen()
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            tensor = value.detach().cpu().contiguous().reshape(-1).view(self.torch.uint8)
            digest.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def close(self):
        if self.env is not None:
            self.env.venv.close()
            self.env = None


class RobotwinNoiseEnv:
    """One physical simulator shared by serial logical trial lanes in a pilot.

    The controller owns recurrent memory. A reset here resets the simulator
    only; it does not know or mutate any policy memory or optimizer state.
    """

    def __init__(self, model_path=None, norm_stats=None, robotwin_root=None, rlinf_root=None,
                 device="cuda", episode_limit=400, action_horizon=32, num_steps=5,
                 task_config_name="demo_clean", output_dir=None, *, backend=None):
        if type(episode_limit) is not int or episode_limit < 1:
            raise ValueError("episode_limit must be a positive integer")
        self.backend = backend if backend is not None else _RoboTwinBackend(
            model_path, norm_stats, robotwin_root, rlinf_root, device, episode_limit,
            action_horizon, num_steps, task_config_name, output_dir)
        self.observation_dim = self.backend.observation_dim
        self.noise_shape = tuple(self.backend.noise_shape)
        self.native_horizon = int(self.noise_shape[0])
        self.episode_limit = episode_limit
        self.elapsed_steps = 0
        self.active = False
        self.last_reset_info = None
        self.contract = getattr(self.backend, "contract", {})

    def reset(self, task_name, seed):
        if task_name not in SOURCE_TASKS + HELDOUT_TASKS:
            raise ValueError("Task is outside the explicit VLRLA twelve-task inventory")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        features, info = self.backend.reset(task_name, seed)
        self.elapsed_steps = 0
        self.active = True
        self.last_reset_info = info
        return self._features(features)

    def _features(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (self.observation_dim,) or not np.isfinite(features).all():
            raise RuntimeError("Backend returned invalid features")
        return features

    def step(self, noise, *, max_primitive_steps=None):
        if not self.active:
            raise RuntimeError("reset() is required before stepping or after episode completion")
        noise = np.asarray(noise, dtype=np.float32)
        if noise.size != int(np.prod(self.noise_shape)) or not np.isfinite(noise).all():
            raise ValueError("Require finite noise with the motor's native noise_shape")
        allowance = min(self.native_horizon, self.episode_limit - self.elapsed_steps)
        if max_primitive_steps is not None:
            if type(max_primitive_steps) is not int or max_primitive_steps < 1:
                raise ValueError("max_primitive_steps must be a positive integer")
            allowance = min(allowance, max_primitive_steps)
        features, reward, terminated, truncated, info = self.backend.step(
            noise.reshape(self.noise_shape), allowance)
        actual = info.get("primitive_steps")
        if type(actual) is not int or not 0 <= actual <= allowance:
            raise RuntimeError("Backend must report an exact valid executed-action count")
        if actual == 0 and not (terminated or truncated):
            raise RuntimeError("A live simulator step made no progress")
        self.elapsed_steps += actual
        natural_episode_end = bool(terminated or truncated or self.elapsed_steps >= self.episode_limit)
        budget_truncated = (max_primitive_steps is not None and actual >= max_primitive_steps
                            and not natural_episode_end)
        truncated = bool(truncated or self.elapsed_steps >= self.episode_limit or budget_truncated)
        if not np.isfinite(reward):
            raise RuntimeError("Nonfinite environment reward")
        self.active = not (terminated or truncated)
        info = dict(info, episode_primitive_steps=self.elapsed_steps,
                    budget_truncated=bool(budget_truncated))
        return self._features(features), float(reward), bool(terminated), truncated, info

    def check_noise_parity(self, noise):
        return self.backend.check_noise_parity(noise)

    def frozen_fingerprint(self):
        return self.backend.frozen_fingerprint()

    def close(self):
        self.backend.close()
        self.active = False
