"""Outcome-blind native ReGround batch054 reset and observation diagnostics."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import multiprocessing as mp
import os
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import reference as ref
from opening import identical, leaves, observations
from reground.eval import rollout
from reground.policies.vla import VLA, chunk_seed
import retry
import symmeta_adapter as stream

SCENE_SEED = 0
CASES = [("spatial_0", start, 0) for start in range(21, 25)]


def leaf_receipt(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=sha256(value.tobytes()).hexdigest())


def compare(left, right):
    changes = {key: "missing" for key in sorted(left.keys() ^ right.keys())}
    for key in sorted(left.keys() & right.keys()):
        a, b = np.asarray(left[key]), np.asarray(right[key])
        if identical(a, b):
            continue
        info = dict(left=leaf_receipt(a), right=leaf_receipt(b))
        if a.shape == b.shape and a.dtype == b.dtype == np.uint8 and a.ndim == 3 and a.shape[-1] == 3:
            delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
            changed = delta != 0
            locations = np.argwhere(changed.any(axis=-1))
            info.update(pixels=len(locations), per_channel=changed.sum(axis=(0, 1)).tolist(),
                        max_per_channel=delta.max(axis=(0, 1)).tolist(), first_yx=locations[:12].tolist(),
                        bbox_yx=[locations.min(axis=0).tolist(), locations.max(axis=0).tolist()])
        changes[key] = info
    return changes


def render_state(env):
    from OpenGL import GL
    import mujoco

    sim = env.rs.sim
    names = {name: getattr(sim.model, name) for name in (
        "body_pos", "body_quat", "cam_pos", "cam_quat", "cam_fovy", "light_pos", "light_dir",
        "light_active", "light_ambient", "light_diffuse", "light_specular", "geom_rgba")}
    for name in ("cam_xpos", "cam_xmat", "light_xpos", "light_xdir", "qacc_warmstart", "ctrl"):
        names[name] = getattr(sim.data, name)
    return dict(arrays=dict(leaves("render", names)), meta=dict(
        pid=os.getpid(), egl=os.environ.get("MUJOCO_EGL_DEVICE_ID"), mujoco=mujoco.__version__,
        xml_sha256=sha256(sim.model.get_xml().encode()).hexdigest(),
        **{name: GL.glGetString(getattr(GL, "GL_" + name.upper())).decode()
           for name in ("vendor", "renderer", "version")}))


class DiagnosticConn:
    """Keep the released worker loop and reset. Handle read-only diagnostics between RPCs."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.env = None
        self.opening = None

    def recv(self):
        command, arg = self.pipe.recv()
        while command == "diagnostic":
            try:
                initial = dict(observations([self.opening]))
                render = render_state(self.env)
                repeated = [dict(observations([self.env.render_obs()])) for _ in range(2)]
                self.pipe.send(dict(original=initial, repeated=repeated,
                                    original_after=dict(observations([self.opening])), render=render))
            except Exception:
                from traceback import format_exc
                self.pipe.send({"__error__": format_exc()})
            command, arg = self.pipe.recv()
        self.command = command
        return command, arg

    def send(self, value):
        if self.command == "reset" and "obs" in value:
            self.opening = value["obs"]
        self.pipe.send(value)


def diagnostic_worker(pipe, env_kwargs, constructor_seed):
    from reground.envs import libero_env

    original = libero_env.TaskEnv
    conn = DiagnosticConn(pipe)

    def construct(*args, **kwargs):
        if constructor_seed is not None:
            np.random.seed(constructor_seed)
            random.seed(constructor_seed)
        conn.env = original(*args, **kwargs)
        return conn.env

    with patch.object(libero_env, "TaskEnv", construct):
        rollout._env_worker(conn, env_kwargs)


class DiagnosticPool(rollout.EnvPool):
    def __init__(self, transport, constructor_seed):
        self.transport = transport
        self.constructor_seed = constructor_seed
        super().__init__(len(CASES))

    def _start(self, index):
        self.ctx = mp.get_context(self.transport)
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=diagnostic_worker,
                                args=(child, self.env_kwargs, self.constructor_seed), daemon=True)
        proc.start()
        child.close()
        self.conns[index], self.procs[index], self.tasks[index] = parent, proc, None


@contextmanager
def pool(transport, seed):
    workers = DiagnosticPool(transport, seed)
    try:
        yield workers
    finally:
        workers.close()


def native_reset(workers, keys):
    idx = list(range(len(keys)))
    expected = [(task, retry.CELL, start, SCENE_SEED, True) for task, start, _ in keys]
    pids = [proc.pid for proc in workers.procs]
    rows = workers.call(idx, "reset", expected)
    states = workers.call(idx, "full_state", [None] * len(keys))
    ref.require([proc.pid for proc in workers.procs] == pids, "Reset workers respawned")
    ref.require([state["task"] for state in states] == [key[0] for key in keys], "Wrong reset task")
    return rows, states


def input_arrays(raw, mapped, rows, states):
    arrays = dict(raw_tasks=np.array([row["description"] for row in rows]), starts=np.array(CASES),
                  scene_seeds=np.full(len(CASES), SCENE_SEED, np.int64), X0=np.stack([row["x"] for row in rows]))
    for i, state in enumerate(states):
        arrays.update(leaves(f"sim/{i}", state["state"]))
    arrays.update({"raw/" + key: value for key, value in observations(raw).items()})
    arrays.update({"mapped/" + key: value for key, value in observations(mapped).items()})
    seeds = [rollout.Episode(*key).noise_seed for key in CASES]
    arrays.update(tasks=arrays["raw_tasks"], noise_seeds=np.array(seeds),
                  chunk_seeds=np.array([chunk_seed(seed, 0) for seed in seeds]))
    return arrays


class ModelBoundary(Exception):
    pass


class ObservationPath:
    """Run the released policy wrapper up to the model boundary, with no weights."""

    def __init__(self):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.policies.factory import make_pre_post_processors
        from reground.policies.vla import RENAME, VLA_REPOS

        repo = VLA_REPOS["pi05"]
        config = PreTrainedConfig.from_pretrained(repo, revision=ref.POLICY)
        self.wrapper = object.__new__(VLA)
        self.wrapper.device = "cpu"
        self.wrapper.chunk_size, self.wrapper.noise_dim = config.chunk_size, config.max_action_dim
        self.wrapper.env_pre, _ = LiberoEnv().get_env_processors()
        self.wrapper.pre, _ = make_pre_post_processors(
            config, repo, pretrained_revision=ref.POLICY,
            preprocessor_overrides={"device_processor": {"device": "cpu"},
                                    "rename_observations_processor": {"rename_map": RENAME["pi05"]}})
        self.wrapper.policy = SimpleNamespace(predict_action_chunk=self.capture)

    def capture(self, batch, noise):
        self.inputs = {}
        for key, value in batch.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            self.inputs.update(leaves(key, value))
        self.inputs["noise"] = noise.detach().cpu().numpy().copy()
        raise ModelBoundary

    def inspect(self, raw, yaws, tasks, seeds):
        original = observations(raw)
        mapped = [stream.policy_observation(obs, stream.public(obs)["robot0_eef_pos"].copy(), yaw)
                  for obs, yaw in zip(raw, yaws, strict=True)]
        mapped_before = observations(mapped)
        after_mapping = compare(original, observations(raw))
        try:
            self.wrapper.predict_chunk(mapped, tasks, seeds, 0)
        except ModelBoundary:
            return mapped, self.inputs, dict(mapping=after_mapping, raw_after_policy=compare(original, observations(raw)),
                                             mapped_after_policy=compare(mapped_before, observations(mapped)))
        raise RuntimeError("Missing released policy-input boundary")


def saved_observations(tape):
    by_lane = [{} for _ in CASES]
    for key, value in tape.items():
        if not key.startswith("raw/"):
            continue
        _, lane, obs, *path = key.split("/")
        ref.require(obs == "obs" and path, key)
        branch = by_lane[int(lane)]
        for node in path[:-1]:
            branch = branch.setdefault(node, {})
        branch[path[-1]] = value.copy()
    return by_lane


def saved_openings(attempts, output, processor):
    archive = {}
    pins = dict(policy=ref.POLICY, code=os.environ["RETRY_PIN"], upstream=ref.PIN,
                cell=retry.CELL, label=0, keys=list(map(list, CASES)))
    supports = list(map(list, map(retry.support_key, CASES)))
    for attempt in ("retry02", "retry04"):
        path = attempts / attempt / "batch054"
        contract = ref.read(path / "contract.json")
        ref.require(all(contract[key] == value for key, value in pins.items()), "Archived enrollment or pin differs")
        rows = ref.read(path / "support.json")
        fields = ("support", "query", "initial_sim_sha256", "estimate", "scene_seed", "noise_seed")
        enrollment = [{key: row[key] for key in fields} for row in rows]
        ref.require([row["support"] for row in enrollment] == supports and
                    [row["query"] for row in enrollment] == pins["keys"], "Archived support assignment differs")
        with np.load(path / "opening.npz", allow_pickle=False) as tape:
            opening = {key: tape[key] for key in tape.files if key != "proposal"}
        ref.require(identical(opening["starts"], np.array(CASES)), "Opening tape enrollment differs")
        yaws = [row["estimate"][0] for row in enrollment]
        mapped, processed, mutations = processor.inspect(saved_observations(opening), yaws,
                                                         opening["tasks"].tolist(), opening["noise_seeds"].tolist())
        expected = {"mapped/" + key: val for key, val in observations(mapped).items()}
        archive[attempt] = (opening, enrollment, processed)
        ref.write(output / f"{attempt}-archive.json", dict(
            original_leaves={k: leaf_receipt(v) for k, v in opening.items()}, support=enrollment,
            mutations=mutations, mapped_vs_saved=compare(expected, {k: opening[k] for k in expected}),
            noise_vs_saved=compare({"noise": processed["noise"]}, {"noise": opening["noise"]})))
    left, right = archive.values()
    ref.write(output / "archive-compare.json", dict(opening=compare(left[0], right[0]),
                                                    support_equal=left[1] == right[1],
                                                    processed=compare(left[2], right[2])))
    ref.require(left[1] == right[1], "Archived support metadata differs")
    return {name: record[0] for name, record in archive.items()}, right[1]


def capture(workers, processor, yaws, archive, path, reset=None):
    rows, states = native_reset(workers, CASES) if reset is None else reset
    raw = [row["obs"] for row in rows]
    first = observations(raw)
    mapped, processed, mutations = processor.inspect(raw, yaws, [row["description"] for row in rows],
                                                      [rollout.Episode(*key).noise_seed for key in CASES])
    arrays = input_arrays(raw, mapped, rows, states)
    for key, value in first.items():
        arrays["raw/" + key] = value
    arrays["noise"] = processed["noise"]
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    details = []
    for i, record in enumerate(workers.call(list(range(len(CASES))), "diagnostic", [None] * len(CASES))):
        original = {key.replace(f"{i}/obs/", "0/obs/", 1): value for key, value in first.items()
                    if key.startswith(f"{i}/obs/")}
        render = record["render"]
        ref.require("NVIDIA" in render["meta"]["vendor"].upper(), "Wrong native EGL vendor")
        np.savez_compressed(path.parent / f"{path.name}-lane{i}-rerender.npz", **{
            **render["arrays"], **{f"{trial}/{key}": value for trial, obs in enumerate(record["repeated"])
                                  for key, value in obs.items()}})
        details.append(dict(renderer=render["meta"],
                            parent_vs_child=compare(original, record["original"]),
                            child_retained_after_render=compare(record["original"], record["original_after"]),
                            repeated=[compare(record["original"], obs) for obs in record["repeated"]],
                            render_leaves={k: leaf_receipt(v) for k, v in render["arrays"].items()}))
    ref.write(path.with_suffix(".json"), dict(leaves={k: leaf_receipt(v) for k, v in arrays.items()},
              mutations=mutations, archive={name: compare(tape, arrays) for name, tape in archive.items()}, renderer=details))
    return arrays, processed, mutations, details


def qualify(output, processor, archive, enrollment, forkserver_only=False):
    import socket
    import torch
    from fluffy_bassoon.baige_gpu import cuda_pci, pci

    gpu = ref.read(output / "gpu.json")
    device, = gpu["devices"]
    ref.require(gpu["hostname"] == socket.gethostname() and torch.cuda.device_count() == 1 and
                pci(device["pci_bus_id"]) == cuda_pci(0) and
                torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090", "Wrong allocated GPU")
    torch.cuda.set_device(0)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(ref.egl_device(0))
    mp.set_forkserver_preload(["preload_env"])
    modes = (("forkserver", None), ("spawn", None), ("spawn", SCENE_SEED))
    if forkserver_only:
        modes = modes[:1]
    results = []
    for transport, seed in modes:
        mode = f"{transport}-{'native' if seed is None else 'seed0'}"
        with pool(transport, seed) as workers:
            _, states = native_reset(workers, list(map(retry.support_key, CASES)))
        support = [stream.digest(state["state"]["sim"]) for state in states]
        matches_support = support == [row["initial_sim_sha256"] for row in enrollment]
        trials = []
        for repetition in range(2):
            # Keep the original sequential lifecycle in round 0. Round 1 tests two parked
            # pools, as needed for comparison before either live arm queries the model.
            args = (processor, [row["estimate"][0] for row in enrollment], archive)
            with pool(transport, seed) as workers:
                if repetition:
                    first = native_reset(workers, CASES)
                    with pool(transport, seed) as other:
                        # Reproduce the scoring preflight ordering. Extra EGL calls
                        # in either process occur after both original reset returns.
                        second = native_reset(other, CASES)
                        left = capture(workers, *args, output / f"{mode}-{repetition}-A", reset=first)
                        right = capture(other, *args, output / f"{mode}-{repetition}-B", reset=second)
                else:
                    left = capture(workers, *args, output / f"{mode}-{repetition}-A")
            if not repetition:
                with pool(transport, seed) as other:
                    right = capture(other, *args, output / f"{mode}-{repetition}-B")
            trials.append(dict(lifecycle=("parked" if repetition else "sequential"), opening=compare(left[0], right[0]),
                               processed=compare(left[1], right[1]), archive={name: [compare(tape, row[0]) for row in (left, right)] for name, tape in archive.items()},
                               mutation_detected=any(any(row[2].values()) or any(
                                   d["parent_vs_child"] or d["child_retained_after_render"] for d in row[3])
                                   for row in (left, right)),
                               repeat_render_drift=any(any(any(d["repeated"]) for d in row[3]) for row in (left, right))))
        results.append(dict(mode=mode, support_match=matches_support, support_sim_sha256=support, trials=trials,
                            exact=matches_support and all(not (t["opening"] or t["processed"] or t["mutation_detected"]
                                                               or all(any(gaps) for gaps in t["archive"].values()))
                                                          for t in trials)))
        ref.write(output / "paired-report.json", dict(code=os.environ["RENDER_PIN"], cases=CASES,
                                         paired_capture="both-before-rerender", trials=results))
    return any(row["exact"] for row in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("attempts", type=Path)
    parser.add_argument("mode", choices=("archive", "cuda", "cuda-forkserver"))
    args = parser.parse_args()
    ref.configure()
    ref.require(retry.schedule()[54] == CASES, "Released batch054 differs")
    processor = ObservationPath()
    opening, support = saved_openings(args.attempts, args.output, processor)
    if args.mode != "archive" and not qualify(args.output, processor, opening, support,
                                               forkserver_only=args.mode == "cuda-forkserver"):
        raise SystemExit("No exact native opening reproduced. See paired-report.json.")


if __name__ == "__main__":
    main()
