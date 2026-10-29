"""Standard LIBERO with a frozen OpenPI motor and the released DSRL SAC learner.

The simulator and model may use separate qualified Python environments. They
communicate only over loopback, inside the same exclusive GPU allocation. Source
learner chronology is retained before any supervised windows are constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import pickle
import random
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
DSRL_COMMIT = "7f48937d4553e95244cd81c79236a3256df80597"
LIBERO_COMMIT = "87edbd1b823e464bc66557d213cc45e9f88aa543"
AUTHKEY = b"fluffy-libero-loopback-v1"
NOISE_DIM, FEATURE_DIM = 32, 58


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value) + "\n")
        stream.flush()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attest_process_gpu(output, *, graphics):
    """Prove the process's actual C/G contexts belong to the allocation."""
    report = dict(pid=os.getpid(), kind="graphics" if graphics else "compute", verified=False)
    try:
        probe = json.loads(Path(os.environ["LYCO_BAIGE_GPU_MAP"]).read_text())
        if probe["status"] != "passed" or len(probe["devices"]) != 1:
            raise ValueError("Require the successful single-GPU allocation probe")
        expected = probe["devices"][0]["uuid"]
        report["expected_uuid"] = expected
        xml = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=25)
        start = next((i for i in (xml.find("<?xml"), xml.find("<nvidia_smi_log")) if i >= 0), -1)
        if start < 0:
            raise ValueError("NVIDIA inventory omitted XML")
        records = []
        for gpu in ET.fromstring(xml[start:]).findall("gpu"):
            for process in gpu.findall("./processes/process_info"):
                if process.findtext("pid", "").strip() == str(os.getpid()):
                    records.append(dict(uuid=gpu.findtext("uuid"), type=process.findtext("type", "")))
        report["processes"] = records
        required = "G" if graphics else "C"
        if (not records or {r["uuid"] for r in records} != {expected}
                or not any(required in r["type"].upper() for r in records)):
            raise RuntimeError("Own process GPU contexts do not attest the allocated device")
        report["verified"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        write_json(output, report)
    return report


def quat_axis_angle(quat):
    quat = np.asarray(quat, np.float64).copy()
    w = float(np.clip(quat[3], -1., 1.))
    den = math.sqrt(max(0., 1. - w * w))
    return np.zeros(3) if math.isclose(den, 0.) else quat[:3] * (2. * math.acos(w) / den)


def public_observation(raw):
    """Only two RGB cameras and robot proprioception; no simulator object state."""
    return dict(image=np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]),
                wrist=np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]),
                state=np.concatenate((raw["robot0_eef_pos"], quat_axis_angle(raw["robot0_eef_quat"]),
                                      raw["robot0_gripper_qpos"])).astype(np.float32))


def resize_rgb(image, size):
    from PIL import Image
    return np.asarray(Image.fromarray(np.asarray(image, np.uint8)).resize((size, size)))


def sac_observation(obs, *, batched=False):
    result = dict(pixels=resize_rgb(obs["image"], 64)[..., None],
                  state=np.asarray(obs["state"], np.float32)[:, None])
    return {key: value[None] for key, value in result.items()} if batched else result


def expand_noise(noise, horizon):
    noise = np.asarray(noise, np.float32)
    if noise.shape != (NOISE_DIM,) or not np.isfinite(noise).all():
        raise ValueError("Require a finite 32-dimensional steering coordinate")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Require a positive native action horizon")
    return np.repeat(noise[None], horizon, axis=0)


def validate_trial_split(args, metadata):
    """Reject scene leakage and oversize source plans before loading the VLA."""
    if (args.attempts < 1 or args.horizon < 1 or args.chunk < 1 or args.history_updates < 0
            or args.max_steps < 1 or args.batch_size < 1 or args.utd < 1 or args.warmup_chunks < 0
            or args.horizon != metadata["horizon"] or args.task_id != metadata["task_id"]
            or args.suite != metadata["suite"] or not 0 < args.train_states <= metadata["initial_states"]
            or not args.eval_states or len(set(args.eval_states)) != len(args.eval_states)
            or any(not args.train_states <= state < metadata["initial_states"] for state in args.eval_states)):
        raise ValueError("Invalid task, interaction budget or disjoint held-out scene bank")
    if args.retry_train_states:
        train, validation, heldout = map(set, (args.retry_train_states,
            args.retry_validation_states, args.eval_states))
        if (args.attempts < 2 or train & validation or (train | validation) & heldout
                or len(train) != len(args.retry_train_states) or len(validation) != len(args.retry_validation_states)
                or any(not 0 <= state < metadata["initial_states"] for state in train | validation)
                or (len(train) + len(validation)) * args.attempts * args.horizon > args.retry_max_steps):
            raise ValueError("Retry teacher scene banks overlap or exceed their explicit interaction budget")


def simulator_server(args):
    import faulthandler
    started = time.monotonic()
    def startup(stage):
        print(json.dumps(dict(simulator_startup=stage,
                              elapsed_seconds=time.monotonic() - started)), flush=True)
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    startup("importing_libero")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from fluffy_bassoon.gpu_execution import validate_execution
    startup("imports_complete")
    libero_root = Path(benchmark.__file__).resolve().parents[3]
    libero_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=libero_root, text=True).strip()
    if libero_commit != LIBERO_COMMIT:
        raise ValueError("Require the standard LIBERO commit pinned by the released DSRL example")
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    startup("suite_constructed")
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    startup("initial_states_loaded")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                            camera_widths=256, render_gpu_device_id=0)
    startup("environment_constructed")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(suite=args.suite, task_id=args.task_id, instruction=task.language,
                    libero_commit=libero_commit,
                    initial_states=len(states), bddl_sha256=sha256(bddl),
                    bddl_file=str(task.bddl_file), problem_folder=str(task.problem_folder),
                    horizon=args.horizon, settle_steps=10, pid=os.getpid(),
                    observation="two RGB cameras and robot proprioception only")
    listener = Listener(("127.0.0.1", args.port), authkey=AUTHKEY)
    write_json(args.output / "simulator-ready.json", metadata)
    startup("listener_ready")
    faulthandler.cancel_dump_traceback_later()
    connection = listener.accept()
    count, success, raw = 0, False, None
    renderer_verified = False
    try:
        while True:
            request = connection.recv()
            command = request["command"]
            if command == "metadata":
                connection.send(metadata)
            elif command == "reset":
                index, seed = int(request["initial_state"]), int(request["seed"])
                if not 0 <= index < len(states):
                    raise ValueError("Invalid fixed initial state")
                np.random.seed(seed)
                random.seed(seed)
                env.seed(seed)
                env.reset()
                raw = env.set_init_state(states[index])
                for _ in range(10):
                    raw, _, _, _ = env.step([0.] * 6 + [-1.])
                if not renderer_verified:
                    attest_process_gpu(args.output / "renderer-binding.json", graphics=True)
                    renderer_verified = True
                count, success = 0, False
                obs = public_observation(raw)
                connection.send(dict(observation=obs, initial_state=index,
                                     initial_observation_sha256=hashlib.sha256(
                                         b"".join(obs[k].tobytes() for k in ("image", "wrist", "state"))).hexdigest()))
            elif command == "step":
                actions = np.asarray(request["actions"], np.float32)
                if raw is None or count >= args.horizon or success:
                    raise RuntimeError("Reset before stepping a completed episode")
                if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                    raise ValueError("Invalid decoded LIBERO action chunk")
                rewards, executed = [], []
                for action in actions[:args.horizon - count]:
                    raw, reward, done, _ = env.step(action.tolist())
                    count += 1
                    rewards.append(float(reward))
                    executed.append(action)
                    success |= bool(done)  # Standard LIBERO done is task success.
                    if done:
                        break
                connection.send(dict(observation=public_observation(raw), rewards=rewards,
                                     executed_actions=np.asarray(executed), primitive_steps=len(executed),
                                     episode_steps=count, success=success, done=success or count >= args.horizon))
            elif command == "close":
                connection.send(dict(closed=True))
                break
            else:
                raise ValueError("Unknown loopback simulator request")
    finally:
        env.close()
        connection.close()
        listener.close()


class SimulatorClient:
    def __init__(self, port):
        self.connection = Client(("127.0.0.1", port), authkey=AUTHKEY)
        self.metadata = self.call(command="metadata")

    def call(self, **request):
        self.connection.send(request)
        return self.connection.recv()

    def reset(self, initial_state, seed):
        return self.call(command="reset", initial_state=int(initial_state), seed=int(seed))

    def step(self, actions):
        return self.call(command="step", actions=actions)

    def close(self):
        self.call(command="close")
        self.connection.close()


class FrozenOpenPI:
    def __init__(self, args):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.openpi_root, text=True).strip()
        if actual != OPENPI_COMMIT:
            raise ValueError("Require the qualified pinned OpenPI source")
        import jax
        from openpi.policies import policy_config
        from openpi.training import config
        if len(jax.devices("gpu")) != 1:
            raise ValueError("Require exactly one allocation-visible JAX GPU")
        ready = json.loads((args.checkpoint / "READY.json").read_text())
        if ready["status"] != "verified" or ready["checkpoint"] != f"gs://openpi-assets/checkpoints/{args.model_config}":
            raise ValueError("Require the verified released checkpoint for this model config")
        conf = config.get_config(args.model_config)
        self.horizon = conf.model.action_horizon
        expected_horizon = 10 if args.model_config == "pi05_libero" else 50
        if self.horizon != expected_horizon or conf.model.action_dim != 32:
            raise ValueError("Native checkpoint action shape changed")
        self.policy = policy_config.create_trained_policy(conf, args.checkpoint,
                                                         sample_kwargs=dict(num_steps=10))
        self.calls = 0
        self.output = args.output
        self.metadata = dict(openpi_commit=actual, checkpoint=ready["checkpoint"],
                             receipt_sha256=sha256(args.checkpoint / "READY.json"),
                             flow_steps=10, predicted_horizon=self.horizon, action_dim=32,
                             weights_frozen=True, jax_devices=list(map(str, jax.devices("gpu"))))
        self.feature_encoder = None
        if args.feature_interface == "vla-prefix":
            from .vla_features import FrozenOpenPIPrefixFeatures
            self.feature_encoder = FrozenOpenPIPrefixFeatures(self.policy,
                checkpoint_receipt_sha256=self.metadata["receipt_sha256"],
                openpi_commit=actual, model_config=args.model_config)

    def infer(self, obs, instruction, noise):
        from openpi_client import image_tools
        element = {"observation/image": image_tools.convert_to_uint8(
                       image_tools.resize_with_pad(obs["image"], 224, 224)),
                   "observation/wrist_image": image_tools.convert_to_uint8(
                       image_tools.resize_with_pad(obs["wrist"], 224, 224)),
                   "observation/state": obs["state"], "prompt": str(instruction)}
        self.calls += 1
        actions = np.asarray(self.policy.infer(element, noise=np.asarray(noise, np.float32))["actions"])
        if self.calls == 1:
            attest_process_gpu(self.output / "motor-binding.json", graphics=False)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Unexpected native motor output {actions.shape}")
        return actions


class OfficialSAC:
    """Unmodified official DSRL actor/critic/update, with explicit sampling/recovery."""
    def __init__(self, root, seed=0):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if actual != DSRL_COMMIT:
            raise ValueError("Require the pinned authors' DSRL source")
        sys.path.insert(0, str(root))
        from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
        observations = dict(pixels=np.zeros((1, 64, 64, 3, 1), np.uint8),
                            state=np.zeros((1, 8, 1), np.float32))
        self.agent = PixelSACLearner(seed, observations, np.zeros((1, 1, 32), np.float32),
            actor_lr=1e-4, critic_lr=3e-4, temp_lr=3e-4,
            hidden_dims=(128, 128, 128), cnn_features=(32, 32, 32, 32),
            cnn_strides=(2, 1, 1, 1), cnn_padding="VALID", latent_dim=50,
            discount=.999, tau=.005, critic_reduction="mean", dropout_rate=0.,
            aug_next=True, use_bottleneck=True, encoder_type="small", encoder_norm="group",
            use_spatial_softmax=True, softmax_temperature=-1, target_entropy="auto",
            num_qs=10, action_magnitude=1., num_cameras=1)
        self.replay, self.updates = [], 0

    def features(self, obs):
        """The final, fixed actor visual bottleneck plus its public robot state."""
        import jax
        import jax.numpy as jnp
        import flax.linen as nn
        from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
        if not hasattr(self, "_features_apply"):
            class FeatureOutput(nn.Module):
                @nn.compact
                def __call__(self, observations, training=False):
                    del training
                    return jnp.concatenate((observations["pixels"],
                        observations["state"].reshape(len(observations["state"]), -1)), axis=-1)
            network = PixelMultiplexer(encoder=Encoder((32, 32, 32, 32), (2, 1, 1, 1), "VALID"),
                                       network=FeatureOutput(), latent_dim=50, use_bottleneck=True)
            self._features_apply = jax.jit(network.apply)
        if "pixels" not in obs:
            obs = sac_observation(obs, batched=True)
        else:
            obs = {k: np.asarray(v)[None] for k, v in obs.items()}
        result = np.asarray(self._features_apply({"params": self.agent._actor.params}, obs))[0]
        if result.shape != (FEATURE_DIM,) or not np.isfinite(result).all():
            raise ValueError("Unexpected frozen teacher encoder feature")
        return result.astype(np.float32)

    def qualify_features(self, obs):
        import flax.linen as nn
        _, capture = self.agent._actor.apply_fn({"params": self.agent._actor.params},
            sac_observation(obs, batched=True),
            capture_intermediates=lambda module, name: name == "__call__" and isinstance(module, nn.LayerNorm),
            mutable=["intermediates"])
        visual = np.tanh(np.asarray(capture["intermediates"]["LayerNorm_0"]["__call__"][0])[0])
        expected = np.concatenate((visual, np.asarray(obs["state"], np.float32)))
        delta = float(np.max(np.abs(expected - self.features(obs))))
        if delta > 1e-5:
            raise RuntimeError("Frozen feature extractor differs from the trained actor bottleneck")
        return dict(passed=True, max_delta=delta, visual_dim=50, state_dim=8,
                    encoder="official small Encoder features32x4 strides2,1,1,1 VALID; Dense50+LayerNorm+tanh")

    def action(self, obs, deterministic=False):
        fn = self.agent.eval_actions if deterministic else self.agent.sample_actions
        return np.asarray(fn(sac_observation(obs, batched=True))).reshape(32).astype(np.float32)

    def update(self, batch_size, sampler):
        from flax.core import freeze
        indices = sampler.integers(0, len(self.replay), batch_size)
        rows = [self.replay[i] for i in indices]
        def stack(items):
            if isinstance(items[0], dict):
                return {k: stack([v[k] for v in items]) for k in items[0]}
            return np.stack(items)
        metrics = self.agent.update(freeze(stack(rows)))
        self.updates += 1
        return {k: float(np.asarray(v)) for k, v in metrics.items() if np.asarray(v).ndim == 0}

    def save(self, path, sampler, contract, **progress):
        import jax
        from flax import serialization
        payload = dict(agent=serialization.to_bytes(self.agent._save_dict),
                       agent_rng=np.asarray(jax.device_get(self.agent._rng)),
                       replay=self.replay, updates=self.updates,
                       sampler=sampler.bit_generator.state, numpy_rng=np.random.get_state(),
                       python_rng=random.getstate(), contract=contract,
                       boundary="complete physical episode", progress=progress)
        temporary = Path(path).with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=5)
        temporary.replace(path)

    def restore(self, path, sampler, contract):
        import jax.numpy as jnp
        from flax import serialization
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if payload["contract"] != contract or payload["boundary"] != "complete physical episode":
            raise ValueError("Source learner recovery contract changed")
        restored = serialization.from_bytes(self.agent._save_dict, payload["agent"])
        self.agent._actor, self.agent._critic = restored["actor"], restored["critic"]
        self.agent._target_critic_params, self.agent._temp = restored["target_critic_params"], restored["temp"]
        self.agent._rng = jnp.asarray(payload["agent_rng"])
        self.replay, self.updates = payload["replay"], payload["updates"]
        sampler.bit_generator.state = payload["sampler"]
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
        return payload["progress"]


def run_episode(sim, motor, choose_noise, *, initial_state, scene_seed, chunk=5,
                native=False, rng=None, feedback=None, bounded=True, record_features=True):
    start = sim.reset(initial_state, scene_seed)
    obs, rows = start["observation"], []
    while True:
        feature_encoder = getattr(motor, "feature_encoder", None)
        public_features = (feature_encoder(obs, sim.metadata["instruction"])
                           if feature_encoder is not None and record_features else None)
        proposal = np.asarray(choose_noise(obs), np.float32)
        actual = np.clip(proposal, -1., 1.) if bounded else proposal
        noise = (rng.standard_normal((motor.horizon, 32)).astype(np.float32) if native
                 else expand_noise(actual, motor.horizon))
        action_chunk = motor.infer(obs, sim.metadata["instruction"], noise)[:chunk]
        outcome = sim.step(action_chunk)
        next_obs = outcome["observation"]
        # Source SAC uses one -1/0 sparse reward per steering decision.
        # The exact executed primitive rewards and count are also retained.
        reward = 0. if outcome["success"] else -1.
        row = dict(sac_observation=sac_observation(obs), sac_next_observation=sac_observation(next_obs),
                   proposed_noise=proposal, executed_noise=actual,
                   native_initial_noise=noise, executed_actions=outcome["executed_actions"],
                   reward=reward, raw_rewards=outcome["rewards"],
                   done=outcome["done"], success=outcome["success"],
                   primitive_steps=outcome["primitive_steps"])
        rows.append(row)
        if public_features is not None:
            row["public_features"] = public_features
        if feedback:
            feedback(actual, reward, outcome["done"])
        obs = next_obs
        if outcome["done"]:
            break
    return rows, dict(initial_state=int(initial_state), scene_seed=int(scene_seed),
                      task_id=sim.metadata.get("task_id"), suite=sim.metadata.get("suite"),
                      semantic_task=Path(sim.metadata.get("bddl_file", "")).stem,
                      success=bool(outcome["success"]), chunks=len(rows),
                      primitive_steps=sum(r["primitive_steps"] for r in rows),
                      raw_return=sum(sum(r["raw_rewards"]) for r in rows),
                      initial_observation_sha256=start["initial_observation_sha256"])


def source_train(args, sim, motor, contract):
    learner = OfficialSAC(args.dsrl_root, args.seed)
    sampler = np.random.default_rng(args.seed)
    primitive_steps, episodes = 0, 0
    if args.resume:
        progress = learner.restore(args.resume, sampler, contract)
        primitive_steps, episodes = progress["primitive_steps"], progress["episodes"]
    while primitive_steps < args.max_steps:
        state = int(sampler.integers(0, args.train_states))
        scene_seed = int(sampler.integers(0, 2**31 - 1))
        warming = len(learner.replay) < args.warmup_chunks
        choose = (lambda _: sampler.standard_normal(32).astype(np.float32)) if warming else learner.action
        started = time.monotonic()
        updates_before = learner.updates
        rows, summary = run_episode(sim, motor, choose, initial_state=state, scene_seed=scene_seed,
                                    chunk=args.chunk, bounded=not warming)
        for row in rows:
            learner.replay.append(dict(observations=row["sac_observation"],
                next_observations=row["sac_next_observation"], actions=row["executed_noise"][None],
                next_actions=row["executed_noise"][None], rewards=np.float32(row["reward"]),
                masks=np.float32(not row["success"]),
                discount=np.float32(.999 ** row["primitive_steps"])))
        primitive_steps += summary["primitive_steps"]
        episodes += 1
        metrics = {}
        if len(learner.replay) > args.warmup_chunks:
            for _ in range(len(rows) * args.utd):
                metrics = learner.update(args.batch_size, sampler)
        trace = args.output / "traces" / f"episode-{episodes:06d}.pkl"
        with trace.open("wb") as stream:
            pickle.dump(dict(rows=rows, summary=summary, source_updates=learner.updates,
                             source_updates_before=updates_before,
                             source_seed=args.seed, episode=episodes), stream, protocol=5)
        summary.update(episode=episodes, primitive_steps_total=primitive_steps,
                       source_updates=learner.updates, warmup=warming,
                       seconds=time.monotonic() - started, metrics=metrics,
                       trace_sha256=sha256(trace))
        append_json(args.output / "progress.jsonl", summary)
        append_json(args.output / "episodes.jsonl", dict(method="dsrl", split="train",
            task_id=args.task_id, scene_seed=scene_seed, trial_seed=args.seed, attempt=1,
            success=summary["success"], primitive_steps=summary["primitive_steps"],
            cumulative_train_steps=primitive_steps, elapsed_seconds=summary["seconds"],
            completed=True, source_updates=learner.updates))
        print(json.dumps(summary), flush=True)
        learner.save(args.output / "source-recovery.pkl", sampler, contract,
                     episodes=episodes, primitive_steps=primitive_steps)
    return learner


def export_histories(args, learner, contract):
    """Relabel only query targets; retain every actual historical action/outcome."""
    from .vla_history import save_dataset
    paths = sorted((args.output / "traces").glob("episode-*.pkl"))
    groups = [paths[i:i + args.attempts] for i in range(0, len(paths), args.attempts)]
    if len(groups) < 2:
        raise ValueError("Need at least two disjoint complete trial groups before student training")
    trials, ids = [], []
    for group in groups:
        rows, starts, source_steps = [], [], []
        for path in group:
            with path.open("rb") as stream:
                episode = pickle.load(stream)
            rows.extend(episode["rows"])
            starts.extend([True] + [False] * (len(episode["rows"]) - 1))
            source_steps.extend([episode["source_updates_before"]] * len(episode["rows"]))
        has_native_features = "public_features" in rows[0]
        observations = np.stack([r["public_features"] if has_native_features
                                 else learner.features(r["sac_observation"]) for r in rows])
        if args.history_objective == "ad" and len(set(source_steps)) < 2:
            # A fixed policy's repetitions are BC, not a source learning history.
            continue
        targets = (np.stack([r["executed_noise"] for r in rows]) if args.history_objective == "ad"
                   else np.stack([np.asarray(learner.agent.eval_actions(
                       {k: v[None] for k, v in r["sac_observation"].items()})).reshape(32) for r in rows]))
        trials.append(dict(observations=observations,
            executed_noise=np.stack([r["executed_noise"] for r in rows]), target_noise=targets,
            rewards=np.asarray([r["reward"] for r in rows], np.float32),
            dones=np.asarray([r["done"] for r in rows], bool), episode_starts=np.asarray(starts, bool),
            teacher_steps=np.asarray(source_steps, np.int64)))
        ids.append(f"source-seed-{args.seed}:{group[0].stem}..{group[-1].stem}")
    if len(trials) < 2:
        raise ValueError("Need two disjoint source learning histories with advancing teacher updates")
    common = dict(motor_sha256=contract["motor"]["receipt_sha256"],
        feature_contract=(contract["feature_contract"] if args.feature_interface == "vla-prefix" else
                          dict(kind="fixed final official SAC actor visual bottleneck50 + robot state8",
                               recovery_sha256=sha256(args.output / "source-recovery.pkl"))),
        noise_contract=contract["noise"], source_steps_cost=sum(
            json.loads(line)["primitive_steps"] for line in (args.output / "progress.jsonl").read_text().splitlines()),
        split="whole disjoint chronological multi-episode groups; one task; smoke data only")
    result = {}
    for name, selected, selected_ids in (("train", trials[:-1], ids[:-1]),
                                          ("validation", trials[-1:], ids[-1:])):
        path = args.output / f"{args.history_objective}-{name}.npz"
        save_dataset(path, selected, objective=args.history_objective,
                     metadata={**common, "trial_ids": selected_ids})
        result[name] = path
    return result


def train_history(args, learner, contract, *, paths=None, artifact_prefix="history", objective=None):
    import torch
    from .vla_history import (HistoryConfig, HistoryNoisePolicy, load_dataset,
                              sample_batch, export_policy)
    from .vla_history_train import supervised_update, validate
    from .warmstart import atomic_save
    paths = export_histories(args, learner, contract) if paths is None else paths
    train, _ = load_dataset(paths["train"], expected_sha256=sha256(paths["train"]))
    validation, _ = load_dataset(paths["validation"], expected_sha256=sha256(paths["validation"]))
    torch.set_num_threads(4)
    candidates = {}
    for mode in ("full", "static"):
        # Matched initialization, data, update count and optimization for static baseline.
        torch.manual_seed(args.seed)
        student = HistoryNoisePolicy(HistoryConfig(train[0]["observations"].shape[-1], 32, width=128, layers=2,
                                      heads=4, context_length=300)).to(args.history_device)
        student.fit_observation_normalization(train)
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.)
        sampler = np.random.default_rng(args.seed)
        for update in range(1, args.history_updates + 1):
            started = time.monotonic()
            batch = sample_batch(train, student, sampler, batch_size=4, max_context=64, mode=mode)
            metrics = supervised_update(student, optimizer, batch, mode=mode)
            row = dict(mode=mode, update=update, seconds=time.monotonic() - started, **metrics)
            if update == 1 or update % 10 == 0 or update == args.history_updates:
                row["validation_nll"] = validate(student, validation, mode=mode, batch_size=2,
                                                batches=1, max_context=64)
                print(json.dumps(dict(history_training=row)), flush=True)
            append_json(args.output / f"{artifact_prefix}-metrics.jsonl", row)
            if update % 10 == 0 or update == args.history_updates:
                atomic_save(args.output / f"{artifact_prefix}-{mode}-recovery.pt",
                    dict(policy=student.state_dict(), optimizer=optimizer.state_dict(),
                    update=update, torch_rng=torch.get_rng_state(), sampler=sampler.bit_generator.state,
                    source_commit=contract["source_commit"], dataset_sha256={k: sha256(v) for k, v in paths.items()},
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        export_policy(args.output / f"{artifact_prefix}-{mode}-candidate.pt", student,
                      manifest=dict(contract=contract, mode=mode, objective=objective or args.history_objective,
                                    updates=args.history_updates,
                                    dataset_sha256={k: sha256(v) for k, v in paths.items()}))
        candidates[mode] = student.eval().requires_grad_(False)
    return candidates


def state_differences(expected, actual, *, path="state", rtol=0., atol=0.):
    """Compare state leaves, independent of MsgPack mapping insertion order."""
    if isinstance(expected, dict) or isinstance(actual, dict):
        if not isinstance(expected, dict) or not isinstance(actual, dict) or set(expected) != set(actual):
            return [dict(path=path, error="mapping keys differ")]
        return [item for key in sorted(expected) for item in
                state_differences(expected[key], actual[key], path=f"{path}/{key}", rtol=rtol, atol=atol)]
    a, b = np.asarray(expected), np.asarray(actual)
    if a.shape != b.shape or a.dtype != b.dtype:
        return [dict(path=path, error="shape or dtype differs", expected_shape=list(a.shape),
                     actual_shape=list(b.shape), expected_dtype=str(a.dtype), actual_dtype=str(b.dtype))]
    equal = (np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
             if a.dtype.kind in "fc" and (rtol or atol) else np.array_equal(a, b))
    if not equal:
        try:
            delta = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        except (TypeError, ValueError):
            delta = None
        return [dict(path=path, error="leaf values differ", maximum_delta=delta,
                     expected_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                     actual_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
                     expected_preview=repr(a.reshape(-1)[:3]), actual_preview=repr(b.reshape(-1)[:3]))]
    return []


def structure_digest(value):
    """Canonical hashes do not depend on mapping insertion order or device type."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            digest.update(b"mapping")
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}".encode())
            for child in item:
                visit(child)
        else:
            array = np.asarray(item)
            digest.update(str(array.dtype).encode() + str(array.shape).encode())
            digest.update(array.tobytes() if array.dtype.kind != "O" else repr(item).encode())
    visit(value)
    return digest.hexdigest()


def qualify_recovery(args, learner, contract, obs):
    """Restore every saved leaf exactly; compare GPU updates numerically.

    GPU convolution gradients need not be bitwise deterministic. An independent
    replay on the original learner measures that floor, without changing the
    fixed float32 comparison tolerance or weakening exact saved-state checks.
    """
    from flax import serialization
    sampler = np.random.default_rng(391)
    path = args.output / "qualification-recovery.pkl"
    learner.save(path, sampler, contract, episodes=0, primitive_steps=0)
    restored = OfficialSAC(args.dsrl_root, args.seed)
    restored_sampler = np.random.default_rng()
    restored.restore(path, restored_sampler, contract)
    def complete_state(source):
        return dict(learning=serialization.to_state_dict(source.agent._save_dict),
                    actor_rng=np.asarray(source.agent._rng), updates=np.int64(source.updates),
                    replay_sha256=np.asarray(structure_digest(source.replay)))
    initial_expected, initial_actual = complete_state(learner), complete_state(restored)
    before = state_differences(initial_expected, initial_actual)
    write_json(args.output / "recovery-initial-comparison.json", dict(differences=before[:20],
        total_differences=len(before), expected_sha256=structure_digest(initial_expected),
        actual_sha256=structure_digest(initial_actual)))
    if before:
        raise RuntimeError("SAC recovery changed an actor/critic/target/optimizer/temperature value")
    expected_action, actual_action = learner.action(obs), restored.action(obs)
    expected_metrics = learner.update(args.batch_size, sampler)
    actual_metrics = restored.update(args.batch_size, restored_sampler)
    updated_expected, updated_actual = complete_state(learner), complete_state(restored)
    after = state_differences(updated_expected, updated_actual)
    tolerance = dict(rtol=1e-5, atol=1e-6)
    numerical = state_differences(updated_expected, updated_actual, **tolerance)
    metrics = state_differences(expected_metrics, actual_metrics, **tolerance)
    # A control reruns the original checkpoint, action and sampled update on the
    # original learner. Its differences diagnose ordinary GPU execution noise.
    learner.restore(path, sampler, contract)
    repeated_action = learner.action(obs)
    repeated_metrics = learner.update(args.batch_size, sampler)
    repeated_state = complete_state(learner)
    control = state_differences(updated_expected, repeated_state)
    write_json(args.output / "recovery-update-comparison.json", dict(differences=after,
        total_differences=len(after), next_action_equal=np.array_equal(expected_action, actual_action),
        next_metrics_equal=expected_metrics == actual_metrics,
        numerical_tolerance=tolerance, numerical_state_differences=numerical,
        numerical_metric_differences=metrics, expected_metrics=expected_metrics, actual_metrics=actual_metrics,
        original_replay_control=dict(action_equal=np.array_equal(expected_action, repeated_action),
            state_differences=control, metric_differences=state_differences(expected_metrics, repeated_metrics),
            numerical_state_differences=state_differences(updated_expected, repeated_state, **tolerance)),
        expected_sha256=structure_digest(updated_expected), actual_sha256=structure_digest(updated_actual)))
    if (not np.array_equal(expected_action, actual_action) or numerical or metrics
            or sampler.bit_generator.state != restored_sampler.bit_generator.state):
        raise RuntimeError("SAC recovery changed the next sampled action or optimizer update")
    learner.restore(path, sampler, contract)
    return dict(passed=True, recovery_sha256=sha256(path), separate_learner=True,
                next_sampled_action_equal=True, next_update_numerically_equal=True,
                next_update_bitwise_equal=not after and expected_metrics == actual_metrics,
                numerical_tolerance=tolerance,
                saved_actor_critic_target_temperature_optimizer_exact=True, replay_sampler_equal=True)


def assert_same_context(first_path, second_path):
    from .vla_history import load_dataset
    first, _ = load_dataset(first_path, expected_sha256=sha256(first_path))
    second, _ = load_dataset(second_path, expected_sha256=sha256(second_path))
    keys = ("observations", "executed_noise", "rewards", "dones", "episode_starts", "teacher_steps")
    if len(first) != len(second) or any(not np.array_equal(a[k], b[k])
        for a, b in zip(first, second, strict=True) for k in keys):
        raise ValueError("AD/DPT comparison changed the actual context instead of only query labels")


def external_candidates(args, sim, motor, contract):
    from .vla_history import load_policy
    if motor.feature_encoder is None:
        raise ValueError("Cross-task candidate evaluation requires the common frozen VLA feature interface")
    candidates, costs = {}, []
    for mode, path, digest in (("full", args.candidate_full, args.candidate_full_sha256),
                               ("static", args.candidate_static, args.candidate_static_sha256)):
        policy, manifest = load_policy(path, expected_sha256=digest, device=args.history_device)
        source = manifest["contract"]
        metadata = source.get("train_metadata", source)
        if (metadata["feature_contract"] != motor.feature_encoder.contract or policy.config.observation_dim != 4104
                or policy.config.noise_dim != 32
                or metadata.get("noise_contract", metadata.get("noise")) != contract["noise"]
                or manifest.get("mode", source.get("mode")) != mode):
            raise ValueError("Candidate feature interface differs from the shared frozen VLA encoder")
        if args.semantic_test:
            tasks = metadata.get("task_manifest", {}).get("tasks", [])
            match = [task for task in tasks if task["suite"] == args.suite and task["task_id"] == args.task_id
                     and task["semantic_task"] == Path(sim.metadata["bddl_file"]).stem]
            if len(match) != 1 or match[0]["split"] != "test":
                raise ValueError("Current semantic task is not in the candidate's declared held-out test bank")
        candidates[mode] = policy
        costs.append(int(metadata.get("total_source_primitive_steps", 0)))
    if len(set(costs)) != 1:
        raise ValueError("Full/static candidates do not share the same source interaction cost")
    return candidates, costs[0]


def evaluate(args, sim, motor, methods, *, learner=None, student=None, train_steps=0,
             contract=None, label_prefix="", rl2_policy=None):
    from .vla_history import FrozenHistoryController
    evaluation = {}
    for method in methods:
        records = []
        rng = np.random.default_rng(args.seed + 10000)
        for state in args.eval_states:
            controller = None
            online, online_sampler, online_steps = None, None, 0
            if method == "dsrl_online":
                online = OfficialSAC(args.dsrl_root, args.seed + state * 1000)
                online_sampler = np.random.default_rng(args.seed + state * 1000)
                online.restore(args.output / "source-recovery.pkl", online_sampler, contract)
                # Each independent deployment starts from the same entire source prior.
                online_sampler = np.random.default_rng(args.seed + state * 1000)
                (args.output / "online-dsrl" / f"state-{state}").mkdir(parents=True, exist_ok=False)
            if method in ("history", "reset", "static"):
                import torch
                torch.manual_seed(args.seed + state * 1000)
                mode = {"history": "full", "reset": "episode", "static": "static"}[method]
                controller = FrozenHistoryController(student["static" if method == "static" else "full"], mode=mode)
            elif method in ("rl2", "rl2_reset", "rl2_trained_reset"):
                import torch
                from .vla_rl2 import FrozenRL2Controller
                torch.manual_seed(args.seed + state * 1000)
                controller = FrozenRL2Controller(rl2_policy,
                    mode="full" if method == "rl2" else "episode")
            attempts = []
            for attempt in range(args.attempts):
                if controller:
                    choose = lambda obs: controller.act(motor.feature_encoder(obs, sim.metadata["instruction"])
                        if motor.feature_encoder is not None else learner.features(obs))
                    feedback = lambda noise, reward, done: controller.observe(
                        executed_noise=noise, reward=reward, done=done)
                else:
                    choose = (online.action if online else learner.action) if method in ("dsrl", "dsrl_online") else lambda _: rng.standard_normal(32).astype(np.float32)
                    feedback = None
                started = time.monotonic()
                updates_before = online.updates if online else None
                rows, summary = run_episode(sim, motor, choose, initial_state=state,
                    scene_seed=args.seed + state * 1000, chunk=args.chunk,
                    native=method == "native", rng=rng, feedback=feedback,
                    bounded=method not in ("native", "tied_gaussian"),
                    record_features=method in ("history", "reset", "static", "rl2", "rl2_reset", "rl2_trained_reset"))
                if online:
                    from .vla_retry_teacher import append_episode_to_replay
                    append_episode_to_replay(online, rows)
                    for _ in range(len(rows) * args.utd):
                        online.update(args.batch_size, online_sampler)
                    online_steps += summary["primitive_steps"]
                    target = args.output / "online-dsrl" / f"state-{state}"
                    online.save(target / "recovery.pkl", online_sampler,
                        dict(parent=contract, heldout_state=state, method=method),
                        episodes=attempt + 1, primitive_steps=online_steps,
                        shared_prior_primitive_steps=train_steps)
                    with (target / f"attempt-{attempt + 1:03d}.pkl").open("xb") as stream:
                        pickle.dump(dict(rows=rows, summary=summary, source_updates_before=updates_before,
                                         source_updates_after=online.updates), stream, protocol=5)
                    summary.update(deployment_primitive_steps=online_steps, source_updates_before=updates_before,
                                   source_updates_after=online.updates, deployment_updates=online.updates - updates_before)
                summary.update(attempt=attempt + 1, method=label_prefix + method, split="eval", task_id=args.task_id,
                               trial_seed=args.seed + state * 1000, cumulative_train_steps=train_steps,
                               elapsed_seconds=time.monotonic() - started, completed=True)
                append_json(args.output / "episodes.jsonl", summary)
                print(json.dumps(summary), flush=True)
                attempts.append(summary)
            records.append(attempts)
        evaluation[label_prefix + method] = dict(success_by_attempt=np.mean(
            [[r["success"] for r in trial] for trial in records], axis=0).tolist(),
            cumulative_success_by_attempt=np.mean(np.maximum.accumulate(
                [[r["success"] for r in trial] for trial in records], axis=1), axis=0).tolist(),
            primitive_steps=sum(r["primitive_steps"] for trial in records for r in trial))
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("env-server", "teacher", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--task-id", type=int, default=57)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--chunk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--dsrl-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-config", choices=("pi05_libero", "pi0_libero"), default="pi05_libero")
    parser.add_argument("--source-commit")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-chunks", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--utd", type=int, default=20)
    parser.add_argument("--train-states", type=int, default=40)
    parser.add_argument("--eval-states", type=int, nargs="+", default=[40, 41, 42, 43])
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--history-updates", type=int, default=0)
    parser.add_argument("--history-device", default="cpu")
    parser.add_argument("--history-objective", choices=("ad", "dpt"), default="ad")
    parser.add_argument("--retry-train-states", type=int, nargs="+")
    parser.add_argument("--retry-validation-states", type=int, nargs="+")
    parser.add_argument("--retry-max-steps", type=int)
    parser.add_argument("--prior-recovery", type=Path)
    parser.add_argument("--prior-contract", type=Path)
    parser.add_argument("--online-dsrl", action="store_true")
    parser.add_argument("--feature-interface", choices=("sac", "vla-prefix"), default="sac")
    parser.add_argument("--compare-supervised", action="store_true")
    parser.add_argument("--rl2-updates", type=int, default=0)
    parser.add_argument("--rl2-max-steps", type=int)
    parser.add_argument("--rl2-max-seconds", type=float)
    parser.add_argument("--rl2-trained-reset", action="store_true")
    parser.add_argument("--candidate-full", type=Path)
    parser.add_argument("--candidate-full-sha256")
    parser.add_argument("--candidate-static", type=Path)
    parser.add_argument("--candidate-static-sha256")
    parser.add_argument("--candidate-label", default="external_")
    parser.add_argument("--semantic-test", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise ValueError("Run models, simulator and tests on allocated Linux")
    if args.mode == "env-server":
        simulator_server(args)
        return
    if (bool(args.retry_train_states) != bool(args.retry_validation_states)
            or (args.retry_train_states and not args.retry_max_steps)):
        raise ValueError("Retry teaching requires explicit disjoint train/validation scenes and a resource budget")
    if bool(args.prior_recovery) != bool(args.prior_contract) or (args.resume and args.prior_recovery):
        raise ValueError("A migrated shared prior needs its original contract and cannot also resume")
    if args.compare_supervised and not args.retry_train_states:
        raise ValueError("A same-data AD/DPT comparison requires explicit same-scene retry teachers")
    if args.rl2_updates and (not args.rl2_max_steps
            or args.rl2_max_steps < args.attempts * args.horizon or not args.retry_train_states):
        raise ValueError("RL2 needs explicit training scenes and a complete-trial interaction budget")
    if args.rl2_trained_reset and not args.rl2_updates:
        raise ValueError("A separately trained episode-reset control requires an explicit RL2 training budget")
    external = (args.candidate_full, args.candidate_full_sha256, args.candidate_static, args.candidate_static_sha256)
    if any(external) and not all(external):
        raise ValueError("External comparison needs hash-bound full and separately trained static candidates")
    if args.semantic_test and not all(external):
        raise ValueError("Semantic test evaluation requires declared external candidates")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    root = Path(__file__).resolve().parents[3]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if actual != args.source_commit or dirty:
        raise ValueError("Require a clean commit-pinned run worktree")
    if args.output.exists() and not (args.resume and args.resume.resolve() == (args.output / "source-recovery.pkl").resolve()):
        raise FileExistsError("Never reuse an output identity")
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    (args.output / "traces").mkdir(exist_ok=bool(args.resume))
    import fcntl
    lock = (args.output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    np.random.seed(args.seed)
    random.seed(args.seed)
    sim = SimulatorClient(args.port)
    validate_trial_split(args, sim.metadata)
    motor = FrozenOpenPI(args)
    if args.chunk > motor.horizon:
        raise ValueError("Executed chunk exceeds the native motor horizon")
    contract = dict(source_commit=actual, seed=args.seed,
                    simulator={k: v for k, v in sim.metadata.items() if k != "pid"},
                    motor=motor.metadata, dsrl_commit=DSRL_COMMIT,
                    noise="32D repeated over native H; initial source/tied baseline Gaussian unbounded; SAC/student clip[-1,1]",
                    feature_dim=FEATURE_DIM if motor.feature_encoder is None else 4104,
                    feature_contract=(dict(kind="fixed final SAC actor encoder") if motor.feature_encoder is None
                                      else motor.feature_encoder.contract),
                    chunk=args.chunk, train_states=args.train_states,
                    horizon=args.horizon, warmup_chunks=args.warmup_chunks,
                    batch_size=args.batch_size, utd=args.utd,
                    scope="native OpenPI LIBERO with official DSRL SAC learner; checkpoint/chunk differ unless explicitly matched")
    if args.resume:
        if json.loads((args.output / "contract.json").read_text()) != contract:
            raise ValueError("Explicit resume changed the original run contract")
    else:
        write_json(args.output / "contract.json", contract)
    append_json(args.output / "runtime.jsonl", dict(hostname=socket.gethostname(), pid=os.getpid(),
                                                   simulator_pid=sim.metadata["pid"], resume=bool(args.resume)))
    try:
        qualification_obs = sim.reset(args.eval_states[0], args.seed + 100000)["observation"]
        if motor.feature_encoder is not None:
            write_json(args.output / "common-feature-parity.json",
                       motor.feature_encoder.qualify(qualification_obs, sim.metadata["instruction"]))
        zeros = expand_noise(np.zeros(32, np.float32), motor.horizon)
        first = motor.infer(qualification_obs, sim.metadata["instruction"], zeros)
        second = motor.infer(qualification_obs, sim.metadata["instruction"], zeros.copy())
        delta = float(np.max(np.abs(first - second)))
        if delta > 1e-5:
            raise RuntimeError("Frozen motor did not reproduce the same native noise input")
        write_json(args.output / "native-noise-parity.json", dict(passed=True, maximum_action_delta=delta,
            native_noise_shape=list(zeros.shape), output_shape=list(first.shape),
            public_state_shape=list(qualification_obs["state"].shape)))
        evaluation = {} if args.resume else evaluate(args, sim, motor, ["native", "tied_gaussian"])
        if args.mode == "teacher":
            if args.prior_recovery:
                learner = OfficialSAC(args.dsrl_root, args.seed)
                prior_contract = json.loads(args.prior_contract.read_text())
                # Loading a saved source learner is an explicit, recorded migration;
                # neither its prior run directory nor checkpoint is changed.
                for key in ("motor", "noise", "dsrl_commit", "chunk", "horizon", "simulator"):
                    before, after = prior_contract[key], contract[key]
                    if key == "simulator":
                        # Older qualified runs already record the BDDL bytes hash.
                        # Added readable names do not alter that scientific contract.
                        after = {k: v for k, v in after.items()
                                 if k in before or k not in ("bddl_file", "problem_folder")}
                    if before != after:
                        raise ValueError(f"Migrated source prior changed its scientific interface: {key}")
                sampler = np.random.default_rng(args.seed)
                progress = learner.restore(args.prior_recovery, sampler, prior_contract)
                train_steps = int(progress["primitive_steps"])
                migration = dict(prior_recovery_sha256=sha256(args.prior_recovery),
                    original_source_commit=prior_contract["source_commit"], current_source_commit=actual,
                    original_contract_sha256=sha256(args.prior_contract), shared_prior_primitive_steps=train_steps,
                    source_weights_unchanged=True, previous_migration=progress.get("migration"))
                progress = dict(progress, migration=migration)
                learner.save(args.output / "source-recovery.pkl", sampler, contract, **progress)
                write_json(args.output / "prior-migration.json", migration)
            else:
                learner = source_train(args, sim, motor, contract)
                train_steps = sum(json.loads(line)["primitive_steps"] for line in
                                  (args.output / "progress.jsonl").read_text().splitlines())
            write_json(args.output / "feature-parity.json", learner.qualify_features(qualification_obs))
            write_json(args.output / "source-recovery-check.json", qualify_recovery(args, learner, contract, qualification_obs))
            evaluation.update(evaluate(args, sim, motor, ["dsrl"], learner=learner, train_steps=train_steps))
            if args.online_dsrl or args.retry_train_states:
                evaluation.update(evaluate(args, sim, motor, ["dsrl_online"], learner=learner,
                                           train_steps=train_steps, contract=contract))
            retry_paths = None
            if args.retry_train_states:
                from .vla_retry_teacher import collect_retry_teachers
                scenes = [dict(trial_id=f"{split}-state-{state}", split=split,
                    initial_state=state, scene_seed=args.seed + state * 1000)
                    for split, bank in (("train", args.retry_train_states),
                                        ("validation", args.retry_validation_states)) for state in bank]
                retry_paths, costs = collect_retry_teachers(source_root=args.dsrl_root,
                    shared_prior_path=args.output / "source-recovery.pkl", prior_contract=contract,
                    fixed_encoder=learner.features, scenes=scenes, attempts=args.attempts,
                    batch_size=args.batch_size, utd=args.utd, chunk=args.chunk,
                    output_dir=args.output / "retry-teachers", seed=args.seed,
                    episode_horizon=args.horizon, max_primitive_steps=args.retry_max_steps,
                    heldout_initial_states=args.eval_states, sim=sim, motor=motor,
                    row_feature_encoder=(None if motor.feature_encoder is None else lambda row: row["public_features"]),
                    fixed_feature_contract=(None if motor.feature_encoder is None else motor.feature_encoder.contract))
                train_steps = costs["total_source_primitive_steps"]
            if args.history_updates:
                objectives = ("ad", "dpt") if args.compare_supervised else (args.history_objective,)
                if args.compare_supervised:
                    for split in ("train", "validation"):
                        assert_same_context(retry_paths["ad"][split], retry_paths["dpt"][split])
                for objective in objectives:
                    prefix = f"{objective}_" if args.compare_supervised else ""
                    student = train_history(args, learner, contract,
                        paths=None if retry_paths is None else retry_paths[objective],
                        artifact_prefix=f"{objective}-history" if args.compare_supervised else "history",
                        objective=objective)
                    evaluation.update(evaluate(args, sim, motor, ["history", "reset", "static"],
                        learner=learner, student=student, train_steps=train_steps, label_prefix=prefix))
            if args.rl2_updates:
                import torch
                from .vla_history import load_dataset
                from .vla_rl2 import RL2Config, make_rl2_policy, train_rl2
                feature_fn = (lambda obs: motor.feature_encoder(obs, sim.metadata["instruction"])) if motor.feature_encoder is not None else learner.features
                policy = make_rl2_policy(RL2Config(observation_dim=contract["feature_dim"]),
                                         device=args.history_device, seed=args.seed)
                # Identical frozen feature statistics for all three meta-training objectives.
                normalization_data = retry_paths["ad"]["train"]
                normalization_trials, _ = load_dataset(normalization_data,
                    expected_sha256=sha256(normalization_data))
                values = torch.as_tensor(np.concatenate([trial["observations"] for trial in normalization_trials]),
                    device=args.history_device, dtype=torch.float32)
                policy.set_feature_normalization(values.mean(0), values.std(0, unbiased=False).clamp(min=1e-4))
                policy, rl2_summary = train_rl2(policy, sim, motor, feature_fn,
                    train_states=args.retry_train_states, heldout_states=args.eval_states + args.retry_validation_states,
                    attempts=args.attempts, chunk=args.chunk, episode_horizon=args.horizon,
                    max_primitive_steps=args.rl2_max_steps, updates=args.rl2_updates,
                    output_dir=args.output / "rl2", seed=args.seed,
                    source_contract=dict(**contract, shared_teacher_primitive_steps=train_steps,
                        feature_normalization_dataset_sha256=sha256(normalization_data)),
                    max_elapsed_seconds=args.rl2_max_seconds)
                evaluation.update(evaluate(args, sim, motor, ["rl2", "rl2_reset"], learner=learner,
                    rl2_policy=policy, train_steps=train_steps + rl2_summary["counters"]["environment_steps"]))
                if args.rl2_trained_reset:
                    # Match initialization, scene sampler, optimizer and nominal
                    # trial/update budget; retain only within-episode state.
                    reset_policy = make_rl2_policy(RL2Config(observation_dim=contract["feature_dim"]),
                                                   device=args.history_device, seed=args.seed)
                    reset_policy.set_feature_normalization(values.mean(0),
                        values.std(0, unbiased=False).clamp(min=1e-4))
                    reset_policy, reset_summary = train_rl2(reset_policy, sim, motor, feature_fn,
                        train_states=args.retry_train_states,
                        heldout_states=args.eval_states + args.retry_validation_states,
                        attempts=args.attempts, chunk=args.chunk, episode_horizon=args.horizon,
                        max_primitive_steps=args.rl2_max_steps, updates=args.rl2_updates,
                        output_dir=args.output / "rl2-trained-reset", seed=args.seed, mode="episode",
                        source_contract=dict(**contract, shared_teacher_primitive_steps=train_steps,
                            feature_normalization_dataset_sha256=sha256(normalization_data)),
                        max_elapsed_seconds=args.rl2_max_seconds)
                    evaluation.update(evaluate(args, sim, motor, ["rl2_trained_reset"], learner=learner,
                        rl2_policy=reset_policy,
                        train_steps=train_steps + reset_summary["counters"]["environment_steps"]))
        else:
            learner = None
        if args.candidate_full:
            candidates, source_cost = external_candidates(args, sim, motor, contract)
            evaluation.update(evaluate(args, sim, motor, ["history", "reset", "static"],
                student=candidates, train_steps=source_cost, label_prefix=args.candidate_label))
        write_json(args.output / "summary.json", dict(completed=True, evaluation=evaluation,
            motor_calls=motor.calls, feature_calls=0 if motor.feature_encoder is None else motor.feature_encoder.calls,
            feature_seconds=0 if motor.feature_encoder is None else motor.feature_encoder.elapsed_seconds))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
