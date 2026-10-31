"""Frozen pi0.5 RoboTwin noise interface for both recurrent PPO and DSRL.

The real backend requires an allocated Linux GPU. No parameters, action
conventions, or task rewards are modified. An opt-in observation-capture mode
defers camera reads to the end of each chunk; image equivalence requires a
separate allocated-runtime check. The module imports only
NumPy until that backend is constructed, so accounting contracts can be checked
with an injected backend on allocated Linux CPU resources.
"""
from __future__ import annotations

from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from copy import deepcopy
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


def validate_paired_instruction(value):
    """Canonical scalar-episode language binding; never execute or normalize text."""
    if (not isinstance(value, (list, tuple)) or len(value) != 1
            or not isinstance(value[0], str) or not value[0].strip()):
        raise ValueError("paired_instruction must contain one nonempty actual instruction")
    return tuple(value)


def bind_paired_instruction(raw, instruction):
    """Bind language before encoding while preserving sampled text and physical arrays."""
    bound = validate_paired_instruction(instruction)
    sampled = validate_paired_instruction(raw["task_descriptions"])
    result = dict(raw, task_descriptions=list(bound))
    audit = dict(schema="robotwin_paired_instruction_v1", sampled_instruction=list(sampled),
                 bound_instruction=list(bound), sampled_observation_sha256=_observation_hash(raw),
                 effective_observation_sha256=_observation_hash(result))
    return result, audit


def load_benchmark_episode_limits(robotwin_root, tasks):
    """Read the pinned benchmark's control-action horizons without a fallback."""
    import yaml

    if isinstance(tasks, str):
        raise ValueError("tasks must be a nonempty sequence of task names")
    tasks = tuple(tasks)
    if not tasks or any(not isinstance(task, str) for task in tasks):
        raise ValueError("tasks must be a nonempty sequence of task names")
    path = Path(robotwin_root) / "task_config" / "_eval_step_limit.yml"
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Benchmark episode-limit file must contain a task mapping")
    limits = {}
    for task in tasks:
        value = values.get(task)
        if type(value) is not int or value < 1:
            raise ValueError(f"Missing or invalid benchmark episode limit for {task}")
        limits[task] = value
    return limits


def _episode_limit_contract(robotwin_root, task_episode_limits):
    if task_episode_limits is None:
        return {}
    fields = dict(task_episode_limits=dict(sorted(task_episode_limits.items())),
        episode_limit_policy="per-task control-action horizon; true termination takes precedence over timeout")
    if robotwin_root is not None:
        path = Path(robotwin_root) / "task_config" / "_eval_step_limit.yml"
        if path.is_file() and load_benchmark_episode_limits(robotwin_root, task_episode_limits) == task_episode_limits:
            fields["episode_limits_source"] = dict(path=str(path.resolve()), sha256=_sha256(path))
    return fields


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


@contextmanager
def _temporary_instance_method(instance, name, replacement):
    """Restore the exact instance/class lookup even when native calls fail."""
    had_override = name in vars(instance)
    previous = vars(instance).get(name)
    setattr(instance, name, replacement)
    try:
        yield
    finally:
        if had_override:
            setattr(instance, name, previous)
        else:
            delattr(instance, name)


@contextmanager
def _capture_sparse_feedback(task):
    """Observe native feedback without changing its values or adapter logic."""
    native = task.gen_sparse_reward_data
    feedback = {}

    def collect(*args, **kwargs):
        result = native(*args, **kwargs)
        feedback.update(raw_rewards=deepcopy(result[0]),
                        raw_terminations=deepcopy(result[1]),
                        raw_truncations=deepcopy(result[2]),
                        native_info=deepcopy(result[3]))
        return result

    with _temporary_instance_method(task, "gen_sparse_reward_data", collect):
        yield feedback


def _step_with_final_observation(subenv, actions, update_obs):
    """Use the unmodified native full-observation loop for exact accounting.

    Native collect_obs=False omits executed_action_count, including when an
    action is only partly executed at early success. Keep collect_obs=True
    and its count/boundary logic, replacing only intermediate camera reads
    with placeholders. Each replaced get_obs still performs its original
    _update_render, preserving pose synchronization and lighting RNG calls.
    The one deferred real get_obs skips that already-performed update.

    This is restricted to the explicit tasks whose success checks use scene
    state, not intermediate now_obs. Ray-tracer temporal state and pixels are
    not assumed equivalent merely because physical/accounting traces match.
    """
    if subenv.get_instruction() is None:
        subenv.reset(env_seed=None)
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[0] < 1:
        raise ValueError("Require a nonempty RoboTwin action chunk")
    with subenv.lock:
        task = subenv.task
        if getattr(task, "crazy_random_light", False):
            raise RuntimeError("Final-observation capture does not support randomized lights")
        original_get_obs = task.get_obs
        original_update_render = task._update_render

        def deferred_observation():
            original_update_render()
            return None  # Native code uses only list length until returning.

        with _temporary_instance_method(task, "get_obs", deferred_observation):
            reward, termination, truncation, info, placeholders = (
                task.gen_sparse_reward_data(actions, collect_obs=True)
            )
        count = info.get("executed_action_count")
        if (isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer))
                or not 0 <= count <= len(actions)):
            raise RuntimeError("Native RoboTwin loop must report an exact executed_action_count")
        if len(placeholders) != len(actions):
            raise RuntimeError("Native RoboTwin full-observation padding changed")
        # The native loop already called _update_render at every original
        # observation boundary. Do not add a new lighting/RNG transition.
        with _temporary_instance_method(task, "_update_render", lambda: None):
            raw_obs = original_get_obs()
        obs = update_obs(raw_obs)
        obs["instruction"] = task.get_instruction()
    return dict(obs=obs, reward=reward, terminated=termination,
                truncated=truncation, info=info)


def _install_final_observation_capture(venv, update_obs=None):
    """Patch only this single synchronous vector instance; opt-in only."""
    if venv.n_envs != 1 or not isinstance(venv.env_thread_pool, _SynchronousExecutor):
        raise RuntimeError("Final-observation capture requires one synchronous physical environment")
    if update_obs is None:
        from robotwin.envs.vector_env import update_obs
    for subenv in venv.envs:
        if getattr(subenv, "_fb2_final_observation_only", False):
            continue

        def step(actions, _subenv=subenv):
            return _step_with_final_observation(_subenv, actions, update_obs)

        subenv.step = step
        subenv._fb2_final_observation_only = True


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
                       *, task_config_name="demo_clean", final_observation_only=False):
    from omegaconf import OmegaConf

    if task_config_name not in ("demo_clean", "demo_randomized"):
        raise ValueError("task_config_name must be demo_clean or demo_randomized")
    if final_observation_only and task_config_name != "demo_clean":
        raise ValueError("final_observation_only initially supports demo_clean only")
    randomize_appearance = task_config_name == "demo_randomized"
    root = Path(rlinf_root)
    cfg = OmegaConf.load(root / "examples/embodiment/config/env/robotwin_adjust_bottle.yaml")
    return OmegaConf.merge(cfg, dict(
        total_num_envs=1, auto_reset=False, ignore_terminations=False,
        seed=int(seed), group_size=1, use_fixed_reset_state_ids=True,
        seeds_path=None, assets_path=str(robotwin_root), center_crop=False,
        # The optional fast step still invokes the native full-observation
        # loop for exact counts, while returning only its final camera read.
        skip_intermediate_renders=bool(final_observation_only), is_eval=False,
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
                 num_steps=5, task_config_name="demo_clean", output_dir=None,
                 final_observation_only=False, corpus_writer=None,
                 task_episode_limits=None, kv_adapter=None, action_residual=None):
        from fluffy_bassoon.gpu_execution import validate_execution

        if kv_adapter is not None and action_residual is not None:
            raise ValueError("Select only one optional residual interface per backend")
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
        self.task_episode_limits = task_episode_limits
        if task_config_name not in ("demo_clean", "demo_randomized"):
            raise ValueError("task_config_name must be demo_clean or demo_randomized")
        self.task_config_name = task_config_name
        if type(final_observation_only) is not bool:
            raise ValueError("final_observation_only must be a bool")
        if final_observation_only and task_config_name != "demo_clean":
            raise ValueError("final_observation_only initially supports demo_clean only")
        self.final_observation_only = final_observation_only
        self.corpus_writer = corpus_writer
        self._decision_metadata = {}
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
        self.kv_adapter, self.action_residual = kv_adapter, action_residual
        self._public_features = self._last_residual_metadata = None
        if kv_adapter is not None or action_residual is not None:
            from .pi05_residual import ReactiveALOHAActionResidual, ReactiveKVResidual

            hook = kv_adapter if kv_adapter is not None else action_residual
            expected_type = ReactiveKVResidual if kv_adapter is not None else ReactiveALOHAActionResidual
            if not isinstance(hook, expected_type) or hook.training or hook.public_dim != FEATURE_DIM:
                raise ValueError("Require the explicit eval-mode reactive hook with the native public feature dimension")
        self._versions = {name: value._version for name, value in self.model.named_parameters()}
        self.env = None
        self.task_name = None
        self.scene_episode_limit = None
        self.prefix_seconds = self.motor_seconds = 0.0
        self.prefix_calls = self.motor_queries = 0
        self._prepared = self._prefix_mask = self._kv_cache = None
        self._raw_obs = None
        self._paired_instruction = self._instruction_binding = None
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
            observation_capture=("final camera read per chunk; native full-observation action accounting"
                                 if final_observation_only else "native camera read at every action boundary"),
            capture_image_equivalence=("unqualified; requires paired allocated-runtime check"
                                       if final_observation_only else "native reference"),
            appearance_randomization=task_config_name == "demo_randomized",
            step_accounting="RoboTwin full-observation valid_action_mask, not issued length or dense run_steps",
            actual_render_pci=self.render_pci)
        self.contract.update(_episode_limit_contract(robotwin_root, task_episode_limits))
        if kv_adapter is not None or action_residual is not None:
            self.contract["optional_residual"] = dict(configuration=hook.config(), live_qualified=False)
        if corpus_writer is not None:
            corpus_writer.register_runtime(self.contract)

    def _encode(self, raw):
        from rlinf.models.embodiment.openpi.modules.model import preprocess_observation

        # The same binding applies at reset AND every following observation.
        # Keep simulator-sampled text in audit, outside public model inputs.
        instruction = getattr(self, "_paired_instruction", None)
        if instruction is not None:
            raw, self._instruction_binding = bind_paired_instruction(raw, instruction)
        else:
            self._instruction_binding = None
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
        self._public_features = features.detach()
        self._residual_prepared = prepared
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

    def reset(self, task_name, seed, *, paired_instruction=None):
        # RoboTwin seeds NumPy and Torch globally during setup_demo. Keep that
        # scene randomization independent of the steering learner's streams.
        torch = self.torch
        saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 torch.cuda.get_rng_state_all())
        self._paired_instruction = (None if paired_instruction is None
                                    else validate_paired_instruction(paired_instruction))
        self._instruction_binding = None
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
        if self.env is not None and (self.task_name != task_name
                                     or self.scene_episode_limit != self.episode_limit):
            self.env.venv.close()
            self.env = None
        if self.env is None:
            cfg = environment_config(self.rlinf_root, self.robotwin_root, task_name, seed,
                                     self.episode_limit,
                                     task_config_name=self.task_config_name,
                                     final_observation_only=self.final_observation_only)
            self.env = RoboTwinEnv(cfg, 1, seed_offset=0, total_num_processes=1, worker_info=None)
            _install_synchronous_vector_executor(self.env.venv)
            if self.final_observation_only:
                _install_final_observation_capture(self.env.venv)
            self.task_name = task_name
            self.scene_episode_limit = self.episode_limit
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
        raw = self._raw_obs
        reset_info = dict(task=task_name, requested_seed=int(seed), actual_seed=actual_seed,
                              instruction=list(raw["task_descriptions"]),
                              observation_sha256=_observation_hash(raw),
                              features_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
                              actual_render_pci=actual_pci, reset_policy_actions=0,
                              reset_internal_physics_steps="not separately instrumented")
        if self._instruction_binding is not None:
            reset_info["instruction_binding"] = deepcopy(self._instruction_binding)
        if getattr(self, "task_episode_limits", None) is not None:
            reset_info["episode_limit"] = self.episode_limit
        if self.corpus_writer is not None:
            reset_info["corpus_episode_id"] = self.corpus_writer.begin_episode(raw, features, reset_info)
        return features, reset_info

    def decode(self, noise):
        torch = self.torch
        tick = time.monotonic()
        kv_adapter = getattr(self, "kv_adapter", None)
        action_residual = getattr(self, "action_residual", None)
        if kv_adapter is not None and action_residual is not None:
            raise ValueError("Select only one optional residual interface per backend")
        if (kv_adapter is not None or action_residual is not None) and getattr(
                self, "_residual_prepared", None) is not self._prepared:
            raise RuntimeError("Residual public features are not bound to the current prepared observation")
        self._last_residual_metadata = None
        with torch.no_grad():
            tensor = torch.as_tensor(noise, dtype=torch.float32, device=self.model.device)[None]
            cache = self._kv_cache
            if kv_adapter is not None:
                if kv_adapter.training:
                    raise ValueError("Use an eval-mode KV hook for simulator inference")
                cache = kv_adapter(cache, self._prefix_mask, self._public_features)
                self._last_residual_metadata = dict(configuration=kv_adapter.config(),
                                                     public_features=self._public_features.detach().cpu())
            targets = self.model._sample_actions_from_prefix_cache(
                self._prepared, self._prefix_mask, cache, noise=tensor)
            actions = self.model.decode_actions(targets, self._prepared.state)
            base_decoded_actions = actions.detach().clone() if getattr(self, "post_command_transform", None) is not None else None
            if action_residual is not None:
                if action_residual.training:
                    raise ValueError("Use an eval-mode action hook for simulator inference")
                correction = action_residual(actions, self._public_features)
                actions = correction.actions
                self._last_residual_metadata = dict(configuration=action_residual.config(),
                    public_features=self._public_features.detach().cpu(),
                    **{name: getattr(correction, name).detach().cpu() for name in
                       ("base_actions", "requested_delta", "applied_delta", "clipped", "actions")})
            if getattr(self, "record_agent_requested_actions", False):
                self._last_residual_metadata = dict(self._last_residual_metadata or {},
                    public_agent_requested_actions=actions.detach().cpu().clone())
            transform = getattr(self, "post_command_transform", None)
            if transform is not None:
                requested = actions.detach().clone()
                transformed, audit = transform(base_decoded_actions, requested.clone())
                if (not isinstance(transformed, torch.Tensor) or transformed.shape != requested.shape
                        or transformed.device != requested.device or transformed.dtype != requested.dtype
                        or not torch.isfinite(transformed).all() or not isinstance(audit, dict)
                        or not torch.equal(transformed[..., [6, 13]], requested[..., [6, 13]])):
                    raise ValueError("Post-command condition must preserve finite ALOHA layout and exact grippers")
                actions = transformed
                self._last_residual_metadata = dict(self._last_residual_metadata or {},
                    public_agent_requested_actions=requested.cpu(),
                    audit_only_post_command_transform=dict(details=audit, actuator_commands=actions.detach().cpu()))
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
        writer = self.corpus_writer
        metadata, self._decision_metadata = self._decision_metadata, {}
        if getattr(self, "_instruction_binding", None) is not None:
            if "paired_instruction" in metadata:
                raise ValueError("Instruction audit must come from the executed backend binding")
            metadata = dict(metadata, paired_instruction=deepcopy(self._instruction_binding))
        if self._last_residual_metadata is not None:
            if "frozen_pi05_residual" in metadata:
                raise ValueError("Reserved residual metadata must come from the executed backend hook")
            metadata = dict(metadata, frozen_pi05_residual=self._last_residual_metadata)
        if writer is not None:
            writer.begin_decision(noise, actions, metadata)
        started = time.monotonic()
        capture = (_capture_sparse_feedback(self.env.venv.envs[0].task)
                   if writer is not None else nullcontext())
        with capture as native_feedback:
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
        # _encode applies the episode language binding before both prefix and
        # corpus capture. The simulator's original text remains audit-only.
        if getattr(self, "_paired_instruction", None) is not None:
            raw = self._raw_obs
            native_feedback = dict(native_feedback, native_info=dict(
                native_feedback.get("native_info") or {},
                paired_instruction=deepcopy(self._instruction_binding)))
        reward = float(torch.as_tensor(rewards).sum())
        if writer is not None:
            writer.complete_native(next_raw=raw, next_features=features,
                valid_action_mask=mask, reward=reward, terminated=terminal,
                truncated=truncated, executed_steps=actual, success=success,
                duration_seconds=time.monotonic() - started,
                adapter_rewards=rewards, adapter_terminations=terms,
                adapter_truncations=truncs, **native_feedback)
        public_command = {}
        if getattr(self, "record_agent_requested_actions", False):
            requests = self._last_residual_metadata["public_agent_requested_actions"][0, :action_count].numpy().copy()
            valid = torch.as_tensor(mask).detach().cpu().numpy().reshape(-1).astype(bool)
            indices = np.flatnonzero(valid)
            if len(indices) != actual or (len(indices) and indices[-1] >= len(requests)):
                raise RuntimeError("Native executed mask does not index the issued public request prefix")
            public_command = dict(agent_requested_actions=requests, valid_action_mask=valid.copy(),
                last_valid_agent_request=None if not len(indices) else requests[indices[-1]].copy())
        return features, reward, terminal, truncated, dict(
            primitive_steps=actual, issued_primitive_steps=action_count,
            success=success, motor_queries=1, prefix_calls=1,
            observation_sha256=_observation_hash(raw),
            actions_sha256=hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
            prefix_seconds=self.prefix_seconds, motor_seconds=self.motor_seconds, **public_command)

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
                 task_config_name="demo_clean", output_dir=None, *, backend=None,
                 final_observation_only=False, corpus_writer=None,
                 task_episode_limits=None, kv_adapter=None, action_residual=None):
        if kv_adapter is not None and action_residual is not None:
            raise ValueError("Select only one optional residual interface per backend")
        if backend is not None and (kv_adapter is not None or action_residual is not None):
            raise ValueError("An injected backend must own its residual configuration")
        if type(episode_limit) is not int or episode_limit < 1:
            raise ValueError("episode_limit must be a positive integer")
        if type(final_observation_only) is not bool:
            raise ValueError("final_observation_only must be a bool")
        if final_observation_only and task_config_name != "demo_clean":
            raise ValueError("final_observation_only initially supports demo_clean only")
        if task_episode_limits is not None:
            if (not isinstance(task_episode_limits, dict) or not task_episode_limits
                    or any(task not in SOURCE_TASKS + HELDOUT_TASKS
                           or type(limit) is not int or limit < 1
                           for task, limit in task_episode_limits.items())):
                raise ValueError("Require positive integer episode limits for explicit RoboTwin tasks")
            task_episode_limits = dict(task_episode_limits)
        self.task_episode_limits = task_episode_limits
        self.default_episode_limit = episode_limit
        initial_episode_limit = (max(task_episode_limits.values())
                                 if task_episode_limits is not None else episode_limit)
        self.backend = backend if backend is not None else _RoboTwinBackend(
            model_path, norm_stats, robotwin_root, rlinf_root, device, initial_episode_limit,
            action_horizon, num_steps, task_config_name, output_dir,
            final_observation_only=final_observation_only, corpus_writer=corpus_writer,
            task_episode_limits=task_episode_limits, kv_adapter=kv_adapter, action_residual=action_residual)
        self.corpus_writer = corpus_writer
        self.observation_dim = self.backend.observation_dim
        self.noise_shape = tuple(self.backend.noise_shape)
        self.native_horizon = int(self.noise_shape[0])
        self.episode_limit = initial_episode_limit
        self.elapsed_steps = 0
        self.active = False
        self.last_reset_info = None
        self.contract = getattr(self.backend, "contract", {})
        if task_episode_limits is not None:
            self.backend.task_episode_limits = dict(task_episode_limits)
            self.contract = dict(self.contract,
                **_episode_limit_contract(robotwin_root, task_episode_limits))
            self.backend.contract = self.contract
            if corpus_writer is not None:
                corpus_writer.register_runtime(self.contract)

    def episode_limit_for_task(self, task_name):
        if self.task_episode_limits is None:
            return self.default_episode_limit
        if task_name not in self.task_episode_limits:
            raise ValueError(f"No declared episode limit for {task_name}")
        return self.task_episode_limits[task_name]

    def set_recording_context(self, metadata):
        """Set method/phase/trial/policy identity without changing the scene."""
        if self.corpus_writer is not None:
            self.corpus_writer.set_context(metadata)

    def record_decision_metadata(self, metadata):
        """Attach current policy inputs/log-probability/history to next step."""
        if self.corpus_writer is not None:
            self.backend._decision_metadata = deepcopy(metadata)

    def reset(self, task_name, seed, *, paired_instruction=None):
        if task_name not in SOURCE_TASKS + HELDOUT_TASKS:
            raise ValueError("Task is outside the explicit VLRLA twelve-task inventory")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.episode_limit = self.episode_limit_for_task(task_name)
        self.backend.episode_limit = self.episode_limit
        options = ({} if paired_instruction is None else
                   dict(paired_instruction=validate_paired_instruction(paired_instruction)))
        features, info = self.backend.reset(task_name, seed, **options)
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
        truncated = bool(not terminated and
                         (truncated or self.elapsed_steps >= self.episode_limit or budget_truncated))
        if not np.isfinite(reward):
            raise RuntimeError("Nonfinite environment reward")
        self.active = not (terminated or truncated)
        info = dict(info, episode_primitive_steps=self.elapsed_steps,
                    budget_truncated=bool(budget_truncated))
        if self.task_episode_limits is not None:
            info["episode_limit"] = self.episode_limit
        if self.corpus_writer is not None:
            info["corpus_chunk"] = self.corpus_writer.finalize_learning_flags(
                reward=reward, terminated=terminated, truncated=truncated,
                budget_truncated=budget_truncated,
                episode_primitive_steps=self.elapsed_steps)
        return self._features(features), float(reward), bool(terminated), truncated, info

    def check_noise_parity(self, noise):
        return self.backend.check_noise_parity(noise)

    def frozen_fingerprint(self):
        return self.backend.frozen_fingerprint()

    def close(self):
        self.backend.close()
        self.active = False
