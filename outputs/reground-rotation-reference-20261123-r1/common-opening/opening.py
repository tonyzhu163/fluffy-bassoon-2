"""Capture one native query zero per registered batch, then verify and replay only that query."""
from contextlib import ExitStack
from hashlib import sha256
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

import reference as ref
from reground.eval import rollout
from reground.method.controller import Base
from reground.policies.vla import chunk_seed


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def leaves(name, value):
    """Preserve every raw observation and full_state leaf, including dtype and shape."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(f"{name}/{key}", item)
    elif value is None:
        yield name + "/None", np.empty(0, np.uint8)
    else:
        array = np.asarray(value)
        ref.require(not array.dtype.hasobject, f"Unsupported opening field {name}")
        yield name, array.copy()


def observations(items):
    return dict(pair for i, item in enumerate(items) for pair in leaves(f"{i}/obs", item))


def identical(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class Captured(Exception):
    """End the unscored capture at the first policy return, before any action."""


class Opening:
    def __init__(self, ctx, job, original, work, tape=None):
        self.ctx, self.job, self.original = ctx, job, original
        self.work, self.tape = work, tape
        self.episodes = list(map(tuple, job["episodes"]))
        self.batch = -1
        self.receipts = {}
        self.pending = None
        self.call, self.predict = ctx.pool.call, ctx.vla.predict_chunk
        if tape is not None:
            self.manifest = ref.read(tape / "opening.json")
            ref.require(self.manifest["identity"] == self.identity(), "Tape identity differs")

    def identity(self):
        return dict(episodes=[list(e) for e in self.episodes], seed=self.job["seed"],
                    shard=self.job["shard"], cell=self.job["cell"], policy=self.ctx.vla.revision,
                    upstream=ref.PIN, code=os.environ["SYMMETA_CODE_REV"])

    def check(self, name, got, expected):
        if not identical(got, expected):
            path = self.work / f"mismatch-{self.batch}.npz"
            np.savez_compressed(path, field=name, expected=expected, actual=got)
            raise ValueError(f"Opening differs at batch {self.batch}, {name}. Diagnose {path}")

    def reset(self, idx, cmd, args):
        outs = self.call(idx, cmd, args)
        if cmd != "reset":
            return outs
        self.batch += 1
        size = ref.BATCH
        self.keys = self.episodes[self.batch * size:(self.batch + 1) * size]
        ref.require(self.pending is None and list(idx) == list(range(size)) and len(self.keys) == size
                    and list(args) == [(t, self.job["cell"], i, 0, True) for t, i, _ in self.keys],
                    "Different registered batch, ordering, scene seed or render request")
        states = self.call(idx, "full_state", [None] * size)
        self.pending = dict(episodes=np.array([f"{t}/{i}/{r}" for t, i, r in self.keys]),
                            scene_seed=np.zeros(size, np.int64), tasks=np.array([o["description"] for o in outs]))
        for lane, (key, out, state) in enumerate(zip(self.keys, outs, states, strict=True)):
            self.check(f"{lane}/sim_task", state["task"], key[0])
            obs = out["obs"]
            evidence = self.original[key]
            self.check(f"{lane}/original/X0", out["x"], evidence["x"])
            self.check(f"{lane}/original/eef", obs["robot_state"]["eef"]["pos"], evidence["eef"])
            self.check(f"{lane}/original/jaw", obs["robot_state"]["gripper"]["qpos"], evidence["jaw"])
            camera = obs["pixels"]["image"]
            digest = sha256(camera.tobytes()).hexdigest()
            if digest != evidence["camera"]:
                np.save(self.work / f"historical-camera-mismatch-{self.batch}-{lane}.npy", camera)
                self.check(f"{lane}/original/agentview_sha256", digest, evidence["camera"])
            self.pending.update(leaves(f"{lane}/sim", state["state"]))
            self.pending[f"{lane}/X0"] = out["x"].copy()
        self.reset_obs = observations([o["obs"] for o in outs])
        self.pending.update(self.reset_obs)
        return outs

    def query(self, items, tasks, seeds, query):
        if query != 0:
            ref.require(self.pending is None, "Missing first query")
            return self.predict(items, tasks, seeds, query)
        inputs = self.pending
        ref.require(inputs is not None, "Unexpected second opening")
        policy_obs = observations(items)
        ref.require(policy_obs.keys() == self.reset_obs.keys(), "Opening observation schema changed")
        for key, value in policy_obs.items():
            self.check("policy/" + key, value, self.reset_obs[key])
        self.check("tasks", tasks, inputs["tasks"])
        self.check("original/noise_seed", seeds, [self.original[k]["nonce"] for k in self.keys])
        inputs["noise_seed"] = np.asarray(seeds)
        inputs["chunk_seed"] = np.array([chunk_seed(s, query) for s in seeds])
        name = f"batch{self.batch}.npz"
        if self.tape is None:
            native_noise = self.ctx.vla._noise

            def observe_noise(actual_seeds, index):
                ref.require("noise" not in inputs and index == 0 and list(actual_seeds) == list(seeds),
                            "Unexpected native noise call")
                tensor = native_noise(actual_seeds, index)
                inputs["noise"] = tensor.detach().cpu().numpy().copy()
                return tensor

            with patch.object(self.ctx.vla, "_noise", observe_noise):
                proposal = self.predict(items, tasks, seeds, query)
            ref.require("noise" in inputs and proposal.shape == (ref.BATCH, 50, 7)
                        and proposal.dtype == np.float32 and np.isfinite(proposal).all(), "Invalid native opening")
            np.savez_compressed(self.work / name, **inputs, proposal=proposal)
            digest = file_hash(self.work / name)
        else:
            path = self.tape / name
            digest = file_hash(path)
            self.check("tape_sha256", digest, self.manifest["batches"][name])
            inputs["noise"] = self.ctx.vla._noise(seeds, query).detach().cpu().numpy()
            with np.load(path, allow_pickle=False) as z:
                ref.require(set(z.files) == set(inputs) | {"proposal"}, "Different reset/input schema")
                for key, value in inputs.items():
                    self.check(key, value, z[key])
                proposal = z["proposal"]
        self.receipts[name] = digest
        self.pending = None
        if self.tape is None:
            raise Captured
        return proposal

    def hooks(self):
        stack = ExitStack()
        stack.enter_context(patch.object(self.ctx.pool, "call", self.reset))
        stack.enter_context(patch.object(self.ctx.vla, "predict_chunk", self.query))
        return stack

    def capture(self):
        for start in range(0, len(self.episodes), ref.BATCH):
            episodes = [rollout.Episode(*e) for e in self.episodes[start:start + ref.BATCH]]
            try:
                rollout.run_batch(self.ctx.vla, self.ctx.pool, episodes, Base(), self.job["cell"])
            except Captured:
                continue
            raise ValueError("Expected one native opening query")

    def finish(self):
        names = {f"batch{b}.npz" for b in range(len(self.episodes) // ref.BATCH)}
        ref.require(self.receipts.keys() == names and self.pending is None, "Missing batch openings")
        ref.write(self.work / "opening.json", dict(identity=self.identity(), batches=self.receipts))
