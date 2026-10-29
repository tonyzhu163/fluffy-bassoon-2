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

import numpy as np


OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
DSRL_COMMIT = "7f48937d4553e95244cd81c79236a3256df80597"
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


def simulator_server(args):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                            camera_widths=256, render_gpu_device_id=0)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(suite=args.suite, task_id=args.task_id, instruction=task.language,
                    initial_states=len(states), bddl_sha256=sha256(bddl),
                    horizon=args.horizon, settle_steps=10, pid=os.getpid(),
                    observation="two RGB cameras and robot proprioception only")
    write_json(args.output / "simulator-ready.json", metadata)
    listener = Listener(("127.0.0.1", args.port), authkey=AUTHKEY)
    connection = listener.accept()
    count, success, raw = 0, False, None
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
        self.metadata = dict(openpi_commit=actual, checkpoint=ready["checkpoint"],
                             receipt_sha256=sha256(args.checkpoint / "READY.json"),
                             flow_steps=10, predicted_horizon=self.horizon, action_dim=32,
                             weights_frozen=True, jax_devices=list(map(str, jax.devices("gpu"))))

    def infer(self, obs, instruction, noise):
        from openpi_client import image_tools
        element = {"observation/image": image_tools.convert_to_uint8(
                       image_tools.resize_with_pad(obs["image"], 224, 224)),
                   "observation/wrist_image": image_tools.convert_to_uint8(
                       image_tools.resize_with_pad(obs["wrist"], 224, 224)),
                   "observation/state": obs["state"], "prompt": str(instruction)}
        self.calls += 1
        actions = np.asarray(self.policy.infer(element, noise=np.asarray(noise, np.float32))["actions"])
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
                native=False, rng=None, feedback=None, bounded=True):
    start = sim.reset(initial_state, scene_seed)
    obs, rows = start["observation"], []
    while True:
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
        if feedback:
            feedback(actual, reward, outcome["done"])
        obs = next_obs
        if outcome["done"]:
            break
    return rows, dict(initial_state=int(initial_state), scene_seed=int(scene_seed),
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
        observations = np.stack([learner.features(r["sac_observation"]) for r in rows])
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
        feature_contract=dict(kind="fixed final official SAC actor visual bottleneck50 + robot state8",
                              recovery_sha256=sha256(args.output / "source-recovery.pkl")),
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


def train_history(args, learner, contract):
    import torch
    from .vla_history import (HistoryConfig, HistoryNoisePolicy, load_dataset,
                              sample_batch, export_policy)
    from .vla_history_train import supervised_update, validate
    paths = export_histories(args, learner, contract)
    train, _ = load_dataset(paths["train"], expected_sha256=sha256(paths["train"]))
    validation, _ = load_dataset(paths["validation"], expected_sha256=sha256(paths["validation"]))
    torch.set_num_threads(4)
    candidates = {}
    for mode in ("full", "static"):
        # Matched initialization, data, update count and optimization for static baseline.
        torch.manual_seed(args.seed)
        student = HistoryNoisePolicy(HistoryConfig(58, 32, width=128, layers=2,
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
            append_json(args.output / "history-metrics.jsonl", row)
            if update % 10 == 0 or update == args.history_updates:
                torch.save(dict(policy=student.state_dict(), optimizer=optimizer.state_dict(),
                    update=update, torch_rng=torch.get_rng_state(), sampler=sampler.bit_generator.state,
                    source_commit=contract["source_commit"], dataset_sha256={k: sha256(v) for k, v in paths.items()},
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
                    args.output / f"history-{mode}-recovery.pt")
        export_policy(args.output / f"history-{mode}-candidate.pt", student,
                      manifest=dict(contract=contract, mode=mode, updates=args.history_updates,
                                    dataset_sha256={k: sha256(v) for k, v in paths.items()}))
        candidates[mode] = student.eval().requires_grad_(False)
    return candidates


def evaluate(args, sim, motor, methods, *, learner=None, student=None, train_steps=0):
    from .vla_history import FrozenHistoryController
    evaluation = {}
    for method in methods:
        records = []
        rng = np.random.default_rng(args.seed + 10000)
        for state in args.eval_states:
            controller = None
            if method in ("history", "reset", "static"):
                import torch
                torch.manual_seed(args.seed + state * 1000)
                mode = {"history": "full", "reset": "episode", "static": "static"}[method]
                controller = FrozenHistoryController(student["static" if method == "static" else "full"], mode=mode)
            attempts = []
            for attempt in range(args.attempts):
                if controller:
                    choose = lambda obs: controller.act(learner.features(obs))
                    feedback = lambda noise, reward, done: controller.observe(
                        executed_noise=noise, reward=reward, done=done)
                else:
                    choose = learner.action if method == "dsrl" else lambda _: rng.standard_normal(32).astype(np.float32)
                    feedback = None
                started = time.monotonic()
                _, summary = run_episode(sim, motor, choose, initial_state=state,
                    scene_seed=args.seed + state * 1000, chunk=args.chunk,
                    native=method == "native", rng=rng, feedback=feedback,
                    bounded=method not in ("native", "tied_gaussian"))
                summary.update(attempt=attempt + 1, method=method, split="eval", task_id=args.task_id,
                               trial_seed=args.seed + state * 1000, cumulative_train_steps=train_steps,
                               elapsed_seconds=time.monotonic() - started, completed=True)
                append_json(args.output / "episodes.jsonl", summary)
                print(json.dumps(summary), flush=True)
                attempts.append(summary)
            records.append(attempts)
        evaluation[method] = dict(success_by_attempt=np.mean(
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
    args = parser.parse_args()
    if sys.platform != "linux":
        raise ValueError("Run models, simulator and tests on allocated Linux")
    if args.mode == "env-server":
        simulator_server(args)
        return
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
    sim, motor = SimulatorClient(args.port), FrozenOpenPI(args)
    if (not 1 <= args.chunk <= motor.horizon or args.horizon != sim.metadata["horizon"]
            or args.task_id != sim.metadata["task_id"] or args.suite != sim.metadata["suite"]
            or not 0 < args.train_states <= sim.metadata["initial_states"]
            or any(not args.train_states <= state < sim.metadata["initial_states"] for state in args.eval_states)):
        raise ValueError("Motor chunk, simulator task, horizon or held-out scene split differs")
    contract = dict(source_commit=actual, seed=args.seed,
                    simulator={k: v for k, v in sim.metadata.items() if k != "pid"},
                    motor=motor.metadata, dsrl_commit=DSRL_COMMIT,
                    noise="32D repeated over native H; initial source/tied baseline Gaussian unbounded; SAC/student clip[-1,1]",
                    feature_dim=FEATURE_DIM, chunk=args.chunk, train_states=args.train_states,
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
            learner = source_train(args, sim, motor, contract)
            write_json(args.output / "feature-parity.json", learner.qualify_features(qualification_obs))
            train_steps = sum(json.loads(line)["primitive_steps"] for line in
                              (args.output / "progress.jsonl").read_text().splitlines())
            evaluation.update(evaluate(args, sim, motor, ["dsrl"], learner=learner, train_steps=train_steps))
            if args.history_updates:
                student = train_history(args, learner, contract)
                evaluation.update(evaluate(args, sim, motor, ["history", "reset", "static"],
                                           learner=learner, student=student, train_steps=train_steps))
        else:
            learner = None
        write_json(args.output / "summary.json", dict(completed=True, evaluation=evaluation,
                                                      motor_calls=motor.calls))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
