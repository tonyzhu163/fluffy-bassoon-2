"""Persistent frozen-native steering collector; one allocated GPU, finite requests.

Physical episodes always finish at their native horizon. PPO records end at
first success; the remaining physical chunks stay in the separate raw history.
The master owns budgets, optimization, request publication and retry decisions.
"""
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from copy import copy, deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from .interface_ppo import InterfaceSpec
from .residual_ppo import ResidualActorCritic, ResidualPPOConfig
from .rlinf_native_ppo import (checked_checkout, cpu_tree, restore_rng, sha,
                               snapshot_rng, tensor_hash)
from .robocasa_interface import FrozenRoboCasaInterface
from .robocasa_pairing import canonical_case, public_snapshot
from .robocasa_residual import (RoboCasaResidualEnvironment, combine_observations,
                                public_observation)


def _save(path, value, *, compressed=False):
    """Publish complete CPU artifacts atomically, including pre-dispatch intents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        if compressed:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, compresslevel=1) as zipped:
                pickle.dump(cpu_tree(value), zipped, protocol=5)
        else:
            torch.save(cpu_tree(value), stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return dict(path=str(path), sha256=sha(path))


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def native_video_writer(path, **options):
    """One encoding thread per camera; preserve every frame/codec/quality."""
    import imageio.v2 as imageio
    return imageio.get_writer(path, **options, ffmpeg_params=["-threads", "1"])


def draw_seed(case, chunk, stream):
    """Stable per-case/chunk streams, independent of lanes, ranks and completion order."""
    identity = dict(version="joint_steering_draw_v1", case=case, chunk=int(chunk), stream=stream)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


def _normal(shape, seed):
    # CPU torch law is identical for every allocated worker/device. Both the
    # actual FP32 draw and its seed/version are saved, so replay needs no guess.
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)


def _raw(environment):
    return cpu_tree(environment.native.current_raw_obs)


class _RecordingEnvironment(RoboCasaResidualEnvironment):
    def __init__(self, *args, canonical_root, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.canonical_root, self.source = Path(canonical_root), source
        self.dispatched_commands, self.completed_commands = [], []

    def reset(self):
        observation = super().reset()
        physical = self.native.env.get_env_attr("_native_pairing")
        snapshot = public_snapshot(observation, self.case, physical)
        directory = self.canonical_root / self.case["task"] / str(self.case["native_seed"])
        marker, report = canonical_case(directory, snapshot, save=lambda p, v: _save(p, v),
            load=lambda p: torch.load(p, map_location="cpu", weights_only=False), source=self.source)
        self.initial_evidence = dict(snapshot=snapshot, canonical=str(marker), pairing=report,
                                     raw=_raw(self), native_counts=self._counts())
        dispatch = self.native.env.step

        def observed_dispatch(actions, *args, **kwargs):
            actual = np.asarray(actions).copy()
            if actual.shape != (1, 12) or not np.isfinite(actual).all():
                raise ValueError("Expected one actual native mapped12 dispatch")
            self.dispatched_commands.append(actual[0])
            result = dispatch(actions, *args, **kwargs)
            self.completed_commands.append(actual[0])
            return result

        self.native.env.step = observed_dispatch
        return observation

    def step_chunk(self, chunk):
        self.dispatched_commands, self.completed_commands = [], []
        result = super().step_chunk(chunk)
        if len(self.completed_commands) != 50:
            raise ValueError("A physical chunk must complete all50 native mapped dispatches")
        return result

    def close(self):
        try:
            super().close()
        finally:
            # An encoder close error must not strand its native simulator.
            if self.native is not None:
                native, self.native = self.native, None
                native.close()


def _qualify_cached(motor, prepared, noise, spec, initial_std):
    """Replay one immutable prefix without additional resets or physical controls."""
    model = motor.model
    original_cache = prepared["prefix"][2]
    all_kv = spec.kind == "kv_all_layers"
    if all_kv and len(original_cache) != spec.kv_layers:
        raise ValueError("Qualification layer count differs from declared all-layer K/V map")
    cache_before = (tuple(tuple(tensor.detach().clone() for tensor in pair) for pair in original_cache)
                    if all_kv else None)
    original = model.build_prefix_cache
    try:
        model.build_prefix_cache = lambda *args, **kwargs: prepared["prefix"]
        baseline, _ = model.predict_action_batch(prepared["observation"], mode="eval",
                                                  noise=torch.as_tensor(noise, device=model.device))
        baseline = np.asarray(cpu_tree(baseline), dtype=np.float32)
    finally:
        model.build_prefix_cache = original
    # The original motor's single-use token remains pending for the real draw.
    # A shallow qualification port consumes only its private token reference;
    # all calls reuse the exact captured prefix and restore native hooks.
    probe = copy(motor)
    probe._prepared = prepared
    zero = probe.execute_batch(prepared, noise=noise,
        steering=np.zeros((len(noise), spec.steering_dim), np.float32), spec=spec)
    if not np.array_equal(zero, baseline):
        raise ValueError("Zero steering differs from the identical cached native draw")
    generator = np.random.default_rng(921117)
    code = generator.normal(0, initial_std, (len(noise), spec.steering_dim)).astype(np.float32)
    probe._prepared = prepared
    consumed = []
    predict = model.predict_action_batch

    def observed_prediction(*args, **kwargs):
        current_prefix = model.build_prefix_cache
        def observed_prefix(*prefix_args, **prefix_kwargs):
            result = current_prefix(*prefix_args, **prefix_kwargs)
            # Snapshot exactly what native denoising receives, before suffix
            # execution can replace/extend its local caches. No new forward.
            consumed.append(tuple(tuple(tensor.detach().clone() for tensor in pair) for pair in result[2]))
            return result
        model.build_prefix_cache = observed_prefix
        try:
            return predict(*args, **kwargs)
        finally:
            model.build_prefix_cache = current_prefix

    try:
        if all_kv:
            model.predict_action_batch = observed_prediction
        changed = probe.execute_batch(prepared, noise=noise, steering=code, spec=spec)
    finally:
        model.predict_action_batch = predict
    cache_targets = []
    if all_kv:
        if len(consumed) != 1 or len(consumed[0]) != spec.kv_layers:
            raise ValueError("Nonzero prediction must consume one complete declared native K/V cache")
        for layer, pair in enumerate(consumed[0]):
            if len(pair) != 2 or len(cache_before[layer]) != 2:
                raise ValueError("Every native cache layer must expose both key and value")
            for target, tensor in enumerate(pair):
                before = cache_before[layer][target]
                unchanged = torch.equal(original_cache[layer][target], before)
                if tensor.shape != before.shape or tensor.dtype != before.dtype or not unchanged:
                    raise ValueError("Qualification changed a native cache format or mutated its original")
                difference = tensor.float() - before.float()
                differing = int(torch.count_nonzero(difference))
                finite = bool(torch.isfinite(tensor).all() and torch.isfinite(difference).all())
                rms = float(difference.square().mean().sqrt())
                if not finite or not np.isfinite(rms) or differing == 0 or not rms > 0:
                    raise ValueError(f"No finite nonzero consumed perturbation at layer{layer} target{target}")
                cache_targets.append(dict(layer=layer, target=("key", "value")[target],
                    shape=list(tensor.shape), dtype=str(tensor.dtype), components=tensor.numel(),
                    differing_components=differing, delta_rms=rms, finite=True,
                    original_cache_unchanged=True))
    delta = changed.astype(np.float64) - baseline
    if not np.isfinite(changed).all() or not np.any(delta):
        raise ValueError("Steering must have finite nonzero influence on decoded native actions")
    return dict(completed=True, interface=spec.record(), batch_size=len(noise),
        zero_native_exact=True, prefix_reused=True, extra_physical_controls=0,
        cache_targets=cache_targets, consumed_cache_layers=spec.kv_layers if all_kv else None,
        original_cache_unchanged=True if all_kv else None,
        consumed_cache_observation="existing nonzero prediction prefix hook; no additional model calls" if all_kv else None,
        probe_seed=921117, probe_coefficient_std=float(initial_std),
        action_delta_rms=float(np.sqrt(np.mean(delta**2))), action_delta_max=float(np.abs(delta).max()),
        baseline_action_rms=float(np.sqrt(np.mean(baseline.astype(np.float64)**2))),
        binary_adapter="unchanged native map_released_control_scores; dispatch observed separately",
        qualification="identity and influence only; scale acceptance belongs to the master")


@torch.no_grad()
def collect_bank(policy, motor, environments, *, request, output, rank, deadline_unix,
                 qualification_path=None):
    """Collect complete finite episodes with deterministic per-case draw scheduling.

    This small port also supports synthetic CPU environments. Records returned
    to PPO are grouped by original case then chronological chunk, never by
    asynchronous lane arrival. Every physical chunk has a durable independent
    artifact, including invalid post-first-success suffixes.
    """
    cases, spec = request["cases"], InterfaceSpec(**request["spec"])
    requested_lanes = request.get("lanes", len(environments))
    barrier = request.get("synchronize_initial_lanes", False)
    if (type(requested_lanes) is not int or requested_lanes < 1 or requested_lanes != len(environments)
            or type(barrier) is not bool or not cases or not environments
            or any(env.cases != cases or env.cursor for env in environments)
            or policy.config.residual_dim != spec.steering_dim):
        raise ValueError("Require fresh lane environments and steering-only policy dimensions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reference = next(policy.parameters())
    deterministic = request.get("deterministic", request["phase"] != "train")
    zero_steering = request.get("zero_steering", request["phase"] == "test_before")
    if type(deterministic) is not bool or type(zero_steering) is not bool:
        raise ValueError("Bind explicit boolean evaluation/zero-steering flags")
    rows, episodes = {}, {}
    upcoming = iter(range(len(cases)))
    resetting, running = {}, {}
    qualification = None
    initial_wave_pending = barrier
    batch_sizes, peak_active_lanes = [], 0

    def start(lane, pool):
        slot = next(upcoming, None)
        if slot is not None:
            if time.time() >= deadline_unix:
                raise TimeoutError("Worker deadline before a new native episode")
            environment = environments[lane]
            environment.seek(slot)
            resetting[pool.submit(environment.reset)] = (lane, slot)

    def accept_reset(future):
        lane, slot = resetting.pop(future)
        observation = future.result()
        directory = output / f"episode-{slot:06d}"
        directory.mkdir(exist_ok=False)
        env = environments[lane]
        initial = dict(case=cases[slot], public=cpu_tree(public_observation(observation)), raw=_raw(env),
                       evidence=cpu_tree(getattr(env, "initial_evidence", None)))
        initial_ref = _save(directory / "initial.pt", initial)
        running[lane] = dict(slot=slot, chunk=0, observation=observation,
            ppo_done=False, directory=directory, initial=initial_ref, chunks=[], valid=[])

    try:
        with ThreadPoolExecutor(max_workers=len(environments)) as pool:
            for lane in range(min(len(environments), len(cases))):
                start(lane, pool)
            while resetting or running:
                if time.time() >= deadline_unix:
                    raise TimeoutError("Worker deadline; preserve completed episodes and partial physical evidence")
                if not running or initial_wave_pending:
                    wait(resetting, timeout=min(1., max(0., deadline_unix - time.time())),
                         return_when=ALL_COMPLETED if initial_wave_pending else FIRST_COMPLETED)
                reset_errors = []
                for future in [item for item in resetting if item.done()]:
                    try:
                        accept_reset(future)
                    except Exception as error:
                        reset_errors.append(error)
                if reset_errors:
                    raise reset_errors[0]  # Every successful sibling reset is already retained.
                if initial_wave_pending:
                    if resetting:
                        continue
                    if len(running) != min(requested_lanes, len(cases)):
                        raise ValueError("Initial synchronized lane wave is incomplete")
                    initial_wave_pending = False
                if not running:
                    continue
                lanes = sorted(running)
                batch_sizes.append(len(lanes))
                peak_active_lanes = max(peak_active_lanes, len(lanes))
                states = [running[lane] for lane in lanes]
                observation = combine_observations([state["observation"] for state in states])
                features, prepared = motor.prepare_batch(observation)
                if features.shape != (len(lanes), policy.config.feature_dim) or not np.isfinite(features).all():
                    raise ValueError("Require finite public pre-action feature rows")
                seeds = [draw_seed(cases[state["slot"]], state["chunk"], "native_gaussian") for state in states]
                native_noise = torch.stack([_normal((50, 32), seed) for seed in seeds])
                if qualification_path is not None and qualification is None:
                    qualification = _qualify_cached(motor, prepared, native_noise, spec, policy.config.initial_std)
                    qualification.update(allocation_id=request.get("allocation_id"),
                        cuda_uuid=request.get("native_cuda_uuid"), completed_unix=time.time())
                    _json(qualification_path, qualification)
                feature_tensor = torch.as_tensor(features, device=reference.device, dtype=reference.dtype)
                distribution, _ = policy._distribution(feature_tensor)
                actor_seeds = [draw_seed(cases[state["slot"]], state["chunk"], "actor_standard_normal")
                               for state in states]
                actor_noise = (None if deterministic or zero_steering else
                    torch.stack([_normal((spec.steering_dim,), seed) for seed in actor_seeds]))
                control, log_prob, value = policy.act(feature_tensor, deterministic=deterministic or zero_steering,
                    **({} if actor_noise is None else dict(action_noise=actor_noise)))
                if zero_steering:
                    control = torch.zeros_like(control)
                    log_prob = distribution.log_prob(control).sum(-1)
                if any(not torch.isfinite(item).all() for item in (
                        distribution.mean, distribution.stddev, control, log_prob, value)):
                    raise FloatingPointError("Nonfinite actor/value output before physical dispatch")
                chunks = motor.execute_batch(prepared, noise=native_noise, steering=control, spec=spec)
                pending = []
                for row, (lane, state) in enumerate(zip(lanes, states)):
                    slot, chunk = state["slot"], state["chunk"]
                    case = cases[slot]
                    record = dict(format="joint_steering_physical_chunk_v1", status="before_dispatch",
                        sequence=request["sequence"], phase=request["phase"], iteration=request["iteration"],
                        rank=rank, case=deepcopy(case), case_slot=slot, chunk_index=chunk,
                        episode_id=f"r{rank:02d}-q{request['sequence']:06d}-e{slot:06d}",
                        source_commit=request["source_commit"], runtime_commit=request["runtime_commit"],
                        policy_iteration=request.get("policy_iteration", request["iteration"] - 1
                            if request["phase"] == "train" else request["iteration"]), interface=spec.record(),
                        observation=cpu_tree(public_observation(state["observation"])),
                        raw_observation=_raw(environments[lane]), features=np.array(features[row], copy=True),
                        native_noise_seed=seeds[row], native_noise=native_noise[row].numpy().copy(),
                        native_noise_law="torch_cpu_float32_standard_normal_v1", torch_version=torch.__version__,
                        actor_noise_seed=actor_seeds[row] if not deterministic and not zero_steering else None,
                        actor_standard_normal=actor_noise[row].numpy().copy() if not deterministic and not zero_steering else None,
                        steering_mean=cpu_tree(distribution.mean[row]), steering_std=cpu_tree(distribution.stddev[row]),
                        control=cpu_tree(control[row]), log_prob=float(log_prob[row]), value=float(value[row]),
                        deterministic=deterministic, zero_steering=zero_steering,
                        requested_commands=np.array(chunks[row], copy=True), valid_ppo=not state["ppo_done"])
                    path = state["directory"] / f"chunk-{chunk:04d}.pkl.gz"
                    _save(path, record, compressed=True)  # Durable before any control dispatch.
                    pending.append((lane, state, record, path))
                futures = [pool.submit(environments[lane].step_chunk, chunks[row])
                           for row, lane in enumerate(lanes)]
                errors, completed_lanes = [], []
                future_items = dict(zip(futures, pending))

                def step_outcomes():
                    # Persist ready siblings immediately; a later failed or
                    # hung lane cannot hide an already completed episode.
                    remaining = max(.001, deadline_unix - time.time())
                    for future in as_completed(future_items, timeout=remaining):
                        lane, state, record, path = future_items[future]
                        try:
                            outcome = future.result()
                        except Exception as error:
                            env = environments[lane]
                            record.update(status="failed_during_dispatch", error=repr(error),
                                dispatched_commands=cpu_tree(getattr(env, "dispatched_commands", [])),
                                executed_commands=cpu_tree(getattr(env, "completed_commands", [])),
                                raw_next_observation=_raw(env))
                            _save(path, record, compressed=True)
                            errors.append(error)
                        else:
                            yield lane, state, record, path, outcome

                for lane, state, record, path, (next_observation, physical_done, info) in step_outcomes():
                    env, slot = environments[lane], state["slot"]
                    if bool(physical_done) != (env.steps == env.horizon):
                        record.update(status="invalid_native_boundary", physical_done=bool(physical_done),
                            observed_steps=env.steps, horizon=env.horizon, raw_next_observation=_raw(env))
                        _save(path, record, compressed=True)
                        errors.append(ValueError("Physical episodes must terminate exactly at the native horizon"))
                        continue
                    first = env.success_steps[0] if env.success_steps else None
                    first_now = first is not None and not state["ppo_done"]
                    valid = not state["ppo_done"]
                    learner_done = valid and (first_now or bool(physical_done))
                    record.update(status="completed", next_observation=cpu_tree(public_observation(next_observation)),
                        raw_next_observation=_raw(env), executed_controls=50,
                        dispatched_commands=cpu_tree(env.dispatched_commands),
                        executed_commands=cpu_tree(env.completed_commands), physical_done=bool(physical_done),
                        first_success_step=first, reward=float(first_now), done=bool(learner_done),
                        episode_start=state["chunk"] == 0, valid_ppo=valid)
                    state["chunks"].append(_save(path, record, compressed=True))
                    if valid:
                        state["valid"].append({key: deepcopy(record[key]) for key in (
                            "features", "control", "log_prob", "value", "reward", "done", "episode_start",
                            "steering_mean", "steering_std",
                            "executed_controls", "case", "case_slot", "chunk_index", "episode_id",
                            "policy_iteration", "first_success_step")})
                    state["ppo_done"] = state["ppo_done"] or first_now or bool(physical_done)
                    if physical_done:
                        final = _save(state["directory"] / "final.pt.gz", dict(public=cpu_tree(next_observation),
                            raw=_raw(env), native_counts=env._counts()), compressed=True)
                        episode = dict(case=deepcopy(cases[slot]), case_slot=slot, episode_id=record["episode_id"],
                            success_once=first is not None, success_at_end=bool(env.success_steps and env.success_steps[-1] == env.steps),
                            first_success_step=first, horizon=env.horizon, completed=True,
                            initial=state["initial"], final=final, chunks=state["chunks"],
                            video_manifest=str(env.output / (env.key + ".json")),
                            monitoring_manifest=str(env.monitoring_root / (env.key + ".json")),
                            valid_ppo_chunks=len(state["valid"]), physical_chunks=state["chunk"] + 1)
                        checkpoint = dict(format="joint_steering_episode_boundary_v1", episode=episode,
                            records=state["valid"], environment=env.state_dict(), worker_rng=snapshot_rng(),
                            request_path=request.get("request_path"), request_sha256=request.get("request_sha256"),
                            source_commit=request["source_commit"], runtime_commit=request["runtime_commit"])
                        episode["checkpoint"] = _save(state["directory"] / "checkpoint.pt", checkpoint)
                        rows[slot], episodes[slot] = state["valid"], episode
                        del running[lane]
                        completed_lanes.append(lane)
                    else:
                        state["chunk"] += 1
                        state["observation"] = next_observation
                _json(output / "progress.json", dict(sequence=request["sequence"],
                    completed_episodes=len(episodes), requested_episodes=len(cases),
                    completed_physical_chunks=sum(len(item["chunks"]) for item in running.values()) +
                        sum(episode["physical_chunks"] for episode in episodes.values()),
                    active_lanes=len(running), resetting_lanes=len(resetting),
                    maximum_native_batch=max(batch_sizes), peak_active_lanes=peak_active_lanes,
                    last_progress_unix=time.time()))
                if errors:
                    raise errors[0]  # Completed sibling evidence was published first.
                for lane in completed_lanes:
                    start(lane, pool)
        if len(episodes) != len(cases):
            raise ValueError("Incomplete bank cannot enter PPO")
        return dict(records=[row for slot in range(len(cases)) for row in rows[slot]],
                    episodes=[episodes[slot] for slot in range(len(cases))], qualification=qualification,
                    execution=dict(requested_lanes=requested_lanes, peak_active_lanes=peak_active_lanes,
                        initial_lane_barrier=barrier, native_batch_sizes=batch_sizes,
                        maximum_native_batch=max(batch_sizes), minimum_native_batch=min(batch_sizes)))
    finally:
        primary_error = sys.exc_info()[0] is not None
        close_errors = []
        for env in environments:
            try:
                env.close()
            except Exception as error:
                close_errors.append(repr(error))
        if close_errors:
            _json(output / "cleanup-errors.json", dict(errors=close_errors, all_lanes_attempted=True))
            if not primary_error:
                raise RuntimeError("Native lane cleanup failed; preserve all episode evidence")



def _calibrate_request(motor, environment, *, request, output):
    """One charged TRAIN reset; public sensitivity only, no policy controls."""
    from .joint_steering_ppo import native_scale_calibration
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        observation = environment.reset()
        initial = _save(output / "initial.pt", dict(case=request["cases"][0],
            public=cpu_tree(public_observation(observation)), raw=_raw(environment),
            evidence=cpu_tree(environment.initial_evidence)))
        receipt = native_scale_calibration(motor, observation, InterfaceSpec(**request["spec"]))
        counts = environment._counts()
        if counts["controls"] != counts["reset_controls"] or environment.steps != 0:
            raise ValueError("Public-observation calibration cannot execute policy controls")
    finally:
        environment.close()
    scope = dict(phase="calibrate", full_physical_episode=False, scoreable_episode=False,
        reset_only=True, policy_controls=0, reset_controls=counts["reset_controls"],
        frame_count=environment.writer.count, videos=environment.writer.refs(), initial=initial,
        case=request["cases"][0], native_counts=counts)
    _json(output / "reset-only-scope.json", scope)
    return dict(calibration=receipt, calibration_scope=scope, episodes=[], records=[])

def _versions(model):
    return tuple((name, tuple(value.shape), str(value.dtype), value._version)
                 for name, value in (*model.named_parameters(), *model.named_buffers()))



def _request_binding(request):
    fixed_spec = {key: value for key, value in request["spec"].items() if key != "fraction"}
    return tensor_hash(dict(spec=fixed_spec, lanes=request["lanes"],
        synchronize_initial_lanes=request.get("synchronize_initial_lanes", False),
        **{key: request[key] for key in (
        "model_directory", "norm_path", "source_commit", "runtime_commit", "policy_config")}))


def _initializer_identity(signature):
    """Content/architecture identity survives reload; GPU IDs and _version do not."""
    return {key: signature[key] for key in ("model_sha256", "model_config_sha256",
                                           "norm_sha256", "parameter_count")}

def _load_native(request, code):
    import multiprocessing as mp
    from hydra import compose, initialize_config_dir
    from rlinf.models.embodiment.openpi import get_model
    from .robocasa_pairing import pairing_native_factory
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each persistent worker requires exactly one allocated visible GPU")
    scheduler_uuid = None
    if os.environ.get("LYCO_BAIGE_GPU_MAP"):
        from fluffy_bassoon.baige_gpu import verify_selected_cuda
        scheduler_uuid = verify_selected_cuda(torch)
    device = torch.cuda.get_device_properties(0)
    checked_checkout(code, request["source_commit"])
    checked_checkout(Path(os.environ["RLINF_PATH"]), request["runtime_commit"])
    mp.set_start_method("spawn", force=True)
    with initialize_config_dir(config_dir=str(code / "configs/native_ppo"), version_base=None):
        cfg = compose(config_name="robocasa365_pi05_eval")
    cfg.rollout.model.model_path = request["model_directory"]
    cfg.rollout.model.openpi_data.norm_stats_path = request["norm_path"]
    cfg.env.eval.episode_horizon_source = "max_episode_steps"
    model = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to("cuda").eval().requires_grad_(False)
    if model.action_horizon != 50 or model.num_steps != 10 or model.action_env_dim != 12:
        raise ValueError("Require original native H50/10-step/12D motor")
    stats = json.loads(Path(request["norm_path"]).read_text())
    stats = stats.get("norm_stats", stats)
    motor = FrozenRoboCasaInterface(SimpleNamespace(model=model, stats=stats))
    signature = dict(model_directory=request["model_directory"],
        model_sha256=sha(Path(request["model_directory"]) / "model.safetensors"),
        model_config_sha256=sha(Path(request["model_directory"]) / "config.json"),
        norm_sha256=sha(request["norm_path"]),
        source_commit=request["source_commit"], runtime_commit=request["runtime_commit"],
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        cuda_uuid=str(device.uuid), scheduler_uuid=scheduler_uuid, device_name=device.name,
        all_parameters_frozen=all(not parameter.requires_grad for parameter in model.parameters()),
        parameter_versions_sha256=tensor_hash(_versions(model)))
    return motor, pairing_native_factory(cfg.env.eval), signature


def run_worker(run: Path, rank: int, code: Path, deadline_unix: float):
    """Serve atomically published requests; never retry a failed native bank."""
    run, code = Path(run), Path(code)
    directory = run / "workers" / f"rank-{rank:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    motor = factory = signature = binding = None
    last_sequence, frozen_fraction, resumed_rng = -1, None, None
    expected_initializer, fresh_qualification = None, True
    allocation_id = f"{os.getpid()}-{time.time_ns()}"
    try:
        if (directory / "workererror.json").exists():
            raise RuntimeError("Existing worker failure requires an explicit diagnosed recovery")
        completed = sorted(directory.glob("result-??????.pt"))
        if completed:
            previous = torch.load(completed[-1], map_location="cpu", weights_only=False)
            if not previous["completed"]:
                raise ValueError("Cannot resume an incomplete worker result")
            last_sequence = int(completed[-1].stem.split("-")[-1])
            request_path = directory / f"request-{last_sequence:06d}.pt"
            if previous["request_sha256"] != sha(request_path):
                raise ValueError("Completed request identity changed")
            resumed_rng = previous["worker_rng"]
            previous_request = torch.load(request_path, map_location="cpu", weights_only=False)
            binding = _request_binding(previous_request)
            if previous.get("request_binding", binding) != binding:
                raise ValueError("Completed worker binding differs from its saved request")
            expected_initializer = _initializer_identity(previous["native_model_frozen_signature"])
            if previous["phase"] != "calibrate":
                frozen_fraction = previous_request["spec"]["fraction"]
        while time.time() < deadline_unix and not (run / "STOP").exists():
            paths = sorted(path for path in directory.glob("request-??????.pt")
                           if int(path.stem.split("-")[-1]) > last_sequence)
            if not paths:
                time.sleep(min(.2, max(0., deadline_unix - time.time())))
                continue
            path = paths[0]
            request = torch.load(path, map_location="cpu", weights_only=False)
            sequence = int(path.stem.split("-")[-1])
            if (request["sequence"] != sequence or last_sequence >= 0 and sequence != last_sequence + 1
                    or request["phase"] not in ("calibrate", "train", "dev", "test_before", "test_after")
                    or type(request["lanes"]) is not int or request["lanes"] < 1):
                raise ValueError("Unexpected worker request sequence/phase/lane contract")
            if request.get("deterministic") is not (request["phase"] != "train") or request.get("zero_steering") is not (request["phase"] == "test_before"):
                raise ValueError("Train is stochastic; evaluation uses the mean; test_before is zero steering")
            if request["phase"] in ("train", "calibrate", "dev"):
                from .joint_steering_ppo import TASKS as train_tasks
                if any(case["task"] not in train_tasks for case in request["cases"]):
                    raise ValueError("Only the five approved ID tasks may enter TRAIN/calibration/DEV")
            spec = InterfaceSpec(**request["spec"])
            if spec.kind not in ("velocity", "kv_all_layers"):
                raise ValueError("Only approved velocity or independent all-layer K/V steering")
            request_binding = _request_binding(request)
            if request["phase"] == "calibrate":
                if sequence != 1 or last_sequence >= 0 or len(request["cases"]) != 1:
                    raise ValueError("Calibration is one initial TRAIN reset per worker before any ordinary bank")
            elif last_sequence < 0:
                raise ValueError("First worker request must be the declared TRAIN-reset calibration")
            elif frozen_fraction is None:
                frozen_fraction = spec.fraction
            elif frozen_fraction != spec.fraction:
                raise ValueError("Freeze the calibrated interface fraction before ordinary collection")
            if binding is not None and request_binding != binding:
                raise ValueError("Persistent motor/interface contract changed")
            if motor is None:
                motor, factory, signature = _load_native(request, code)
                if expected_initializer is not None and _initializer_identity(signature) != expected_initializer:
                    raise ValueError("Reloaded frozen initializer differs from the completed worker boundary")
                binding = request_binding
                if resumed_rng is not None:
                    restore_rng(resumed_rng)
                    resumed_rng = None
            configuration = ResidualPPOConfig(**request["policy_config"])
            if configuration.feature_dim != 2064 or configuration.residual_dim != spec.steering_dim:
                raise ValueError("Steering-only actor requires public2064 features and exact steering width")
            request.update(request_path=str(path), request_sha256=sha(path),
                           allocation_id=allocation_id, native_cuda_uuid=signature["cuda_uuid"])
            from .joint_steering_ppo import JointActionJournal
            import imageio.v2 as imageio
            journal = JointActionJournal(run / "accounting", rank=rank,
                category="train" if request["phase"] in ("train", "calibrate") else "eval",
                scope=f"{request['phase']}-{request['iteration']:03d}")
            output = directory / f"bank-{sequence:06d}"
            environments = [_RecordingEnvironment(factory, request["cases"], horizon=None,
                output=output / "videos" / f"lane-{lane:02d}",
                monitoring_root=output / "monitoring" / f"lane-{lane:02d}",
                journal=journal, writer_factory=native_video_writer, rank=rank,
                episode_id_prefix=f"q{sequence:06d}-lane{lane:02d}-",
                canonical_root=request["canonical_root"], source=dict(rank=rank, sequence=sequence,
                    phase=request["phase"], iteration=request["iteration"],
                    source_commit=request["source_commit"], runtime_commit=request["runtime_commit"]))
                for lane in range(1 if request["phase"] == "calibrate" else request["lanes"])]
            # Environment constructors create only directories; physical reset
            # starts only after a distinct evidence directory exists.
            if request["phase"] == "calibrate":
                result = _calibrate_request(motor, environments[0], request=request, output=output / "calibration")
            else:
                policy = ResidualActorCritic(configuration).to(motor.model.device).eval()
                policy.load_state_dict(request["policy"], strict=True)
                qualification_path = directory / "qualification.json" if fresh_qualification else None
                if qualification_path is not None and qualification_path.exists():
                    qualification_path.replace(directory / f"qualification-prior-{allocation_id}.json")
                result = collect_bank(policy, motor, environments, request=request,
                    output=output / "history", rank=rank, deadline_unix=deadline_unix,
                    qualification_path=qualification_path)
                if fresh_qualification:
                    receipt = json.loads(qualification_path.read_text())
                    if (receipt.get("zero_native_exact") is not True or receipt.get("prefix_reused") is not True
                            or receipt.get("extra_physical_controls") != 0
                            or not receipt.get("action_delta_rms", 0) > 0
                            or receipt.get("allocation_id") != allocation_id):
                        raise ValueError("Fresh allocation needs its own zero-parity/influence receipt")
                    fresh_qualification = False
                    result["qualification_receipt"] = dict(path=str(qualification_path), sha256=sha(qualification_path))
            if (any(parameter.requires_grad for parameter in motor.model.parameters()) or motor.model.training
                    or tensor_hash(_versions(motor.model)) != signature["parameter_versions_sha256"]):
                raise ValueError("Original VLM/action expert frozen-state signature changed")
            result.update(completed=True, request_sha256=request["request_sha256"], rank=rank,
                phase=request["phase"], iteration=request["iteration"], sequence=sequence,
                worker_rng=snapshot_rng(), native_model_frozen_signature=signature,
                charged_controls=journal.charged(), request_binding=binding,
                allocation_id=allocation_id, completed_unix=time.time())
            _save(directory / f"result-{sequence:06d}.pt", result)
            last_sequence = sequence
        _json(directory / "worker-exit.json", dict(last_completed_sequence=last_sequence,
            stop_marker=(run / "STOP").exists(), deadline_reached=time.time() >= deadline_unix,
            ended_unix=time.time()))
    except Exception:
        error_path = directory / "workererror.json"
        if error_path.exists():
            error_path = directory / f"workererror-reentry-{time.time_ns()}.json"
        _json(error_path, dict(rank=rank, last_completed_sequence=last_sequence,
            error=traceback.format_exc(), failed_unix=time.time(), retry=False))
        raise
