"""Separate uncorrected support, then matched native retries under a frozen estimate."""
from contextlib import contextmanager
import json
import multiprocessing as mp
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

import reference as ref
import symmeta_adapter as stream
from opening import file_hash, identical, leaves, observations
from fluffy_bassoon.xemb.embodiment import retarget_action_from_motor, rot2
from reground.envs.libero_env import N_SETTLE, max_steps
from reground.eval import jobs, rollout
from reground.method.controller import Base
from reground.policies.vla import N_EXEC, chunk_seed

CELL = "rot_z/90"
ARMS = ("translation", "translation_wrist")
TASKS = ("10_1", "10_3", "goal_1", "goal_2", "object_1", "object_3", "spatial_0", "spatial_2")
require = ref.require


def schedule():
    full = [(task, i, 0) for task in TASKS for i in range(15, 50)]
    for label in ref.SEEDS:
        units = ref.units("base", [label])
        actual = [tuple(key) for j in units for key in j["episodes"]]
        require(actual == full and all(j["cell"] == CELL for j in units), "Released grid differs")
    require(N_SETTLE == N_EXEC == 10 and ref.BATCH == 4, "Release cadence differs")
    return [full[i:i+4] for i in range(0, len(full), 4)]


def support_key(key):
    task, start, rep = key
    require(15 <= start < 50 and rep == 0, key)
    return task, (start - 15) % 15, rep


def issue(proposal, estimate, arm):
    require(arm in ARMS, arm)
    yaw, gain, grip = estimate
    require(np.isfinite(estimate).all() and gain > 0 and grip in (-1, 1), estimate)
    bounded = np.clip(proposal, stream.LO, stream.HI)
    if tuple(estimate) == stream.IDENTITY:
        commands = bounded.copy()
        commands[..., 6] = proposal[..., 6]  # stock Base passes the raw gripper
    else:
        commands = retarget_action_from_motor(bounded, yaw, gain, grip, stream.LO, stream.HI)
    if arm == "translation_wrist":
        angular_xy = bounded[..., 3:5] @ rot2(-yaw).T
        commands[..., 3:5] = np.clip(angular_xy, -1, 1)
    return commands


class Meter:
    """Journal native calls before dispatch and after return for interrupted-attempt accounting."""
    def __init__(self, path, stage):
        self.log = path.open("x", buffering=1)
        self.stage = stage
        self.counts = {}

    def event(self, op, n, fn):
        record = [self.stage, op, n]
        self.log.write(json.dumps(["begin", *record]) + "\n")
        value = fn()
        self.log.write(json.dumps(["end", *record]) + "\n")
        self.counts[op] = self.counts.get(op, 0) + n
        return value


class Pool:
    def __init__(self, native, meter):
        self.native, self.meter = native, meter

    def call(self, idx, op, args):
        if op not in ("reset", "step"):
            return self.native.call(idx, op, args)
        pids = [self.native.procs[i].pid for i in idx] if op == "reset" else []
        result = self.meter.event(op, len(idx), lambda: self.native.call(idx, op, args))
        if op == "reset":
            retried = sum(self.native.procs[i].pid != pid for i, pid in zip(idx, pids, strict=True))
            if retried:
                self.meter.event("internal_reset_retry", retried, lambda: None)
        return result


class ImportedEnvPool(rollout.EnvPool):
    """Keep the released child target and RPCs. Each child gets a fresh simulator and GL context."""
    def _start(self, i):
        self.ctx = mp.get_context("forkserver")
        super()._start(i)


@contextmanager
def fresh(work, stage, n):
    require(not (work / (stage + ".jsonl")).exists(), "Attempt must be fresh")
    meter = Meter(work / (stage + ".jsonl"), stage)
    pool = None
    try:
        mp.set_forkserver_preload(['preload_env'])
        pool = ImportedEnvPool(n)
        yield Pool(pool, meter), meter
    finally:
        try:
            if pool is not None:
                pool.close()
        finally:
            meter.log.close()


def proposal_ok(a, n):
    require(a.dtype == np.float32 and a.shape == (n, 50, 7) and np.isfinite(a).all(), "Invalid proposal")


def calibrate(vla, pool, pipe, keys, work):
    eps = [rollout.Episode(*support_key(k)) for k in keys]
    n = len(eps)
    idx = list(range(n))
    outs = pool.call(idx, "reset", [(e.task, CELL, e.init_id, e.scene_seed, True) for e in eps])
    obs = [o["obs"] for o in outs]
    tasks = [o["description"] for o in outs]
    seeds = [e.noise_seed for e in eps]
    before = list(map(stream.public, obs))
    snapshots = [before]
    # Full-state observations are read only by the separate layout/opening audit.
    layout = [stream.digest(o["state"]["sim"]) for o in pool.call(idx, "full_state", [None]*n)]
    ctrl = Base()
    x = np.stack([o["x"] for o in outs])
    ctrl.reset(x)
    pipe.send(("begin", n))
    pipe.recv()
    commands, proposals, rows = [], [], []
    for i, e in enumerate(eps):
        rows.append(dict(support=[e.task, e.init_id, e.rep], query=keys[i], noise_seed=e.noise_seed,
                         scene_seed=e.scene_seed, initial_sim_sha256=layout[i], updates=[], queries=[]))
    for t in range(15):
        if t % N_EXEC == 0:
            query = t // N_EXEC
            chunk = pool.meter.event("vla_batch", 1, lambda: vla.predict_chunk(obs, tasks, seeds, query))
            proposal_ok(chunk, n)
            proposals.append(chunk.copy())
            for i, row in enumerate(rows):
                row["queries"].append(dict(control=t, chunk_seed=chunk_seed(seeds[i], query),
                                           proposal_sha256=stream.digest(chunk[i])))
            ctrl.on_chunk(x, chunk)
        issued = ctrl.act(x, t % N_EXEC)
        commands.append(issued.copy())
        endpoint = (t+1) % stream.CHUNK == 0
        outs = pool.call(idx, "step", [(a, endpoint) for a in issued])
        require(t == 14 or not any(o["success"] for o in outs), "Support terminated before 15 controls")
        x = np.stack([o["x"] for o in outs])
        if endpoint:
            obs = [o["obs"] for o in outs]
            after = list(map(stream.public, obs))
            recent = np.stack(commands[-stream.CHUNK:])
            for i, row in enumerate(rows):
                pipe.send(("observe", (i, before[i], after[i], recent[:, i])))
                update = pipe.recv()
                require(update["chunk"] == (t+1)//5, "Reader state leaked")
                np.testing.assert_array_equal(update["issued_mean"], recent[:, i].mean(axis=0))
                update.update(end_control=t+1, before_sha256=stream.digest(before[i]["agentview_image"]),
                              after_sha256=stream.digest(after[i]["agentview_image"]))
                row["updates"].append(update)
            snapshots.append(after)
            before = after
    summaries = pool.call(idx, "summary", [None]*n)
    speeds = np.stack([s.pop("speeds") for s in summaries])
    limit = jobs.speed_limit("pi05")
    for i, row in enumerate(rows):
        row["estimate"] = row["updates"][-1]["next_adapter"]
        summary = summaries[i]
        violations = int((speeds[i] > limit).sum())
        row["support_safety"] = dict(**summary, steps=15, success=int(outs[i]["success"]),
            speed_viol=violations, unsafe=int(any(summary[k] for k in ("displaced","fallen","contact_steps"))
                                              or violations > 0))
    public_tape = {k:np.stack([[lane[k] for lane in snap] for snap in snapshots]) for k in before[0]}
    np.savez_compressed(work / "support.npz", proposals=np.stack(proposals), issued=np.stack(commands),
                        speeds=speeds, **public_tape)
    ref.write(work / "support.json", rows)
    return rows


class Retry:
    needs_sim = False

    def __init__(self, vla, pool, keys, support, arm, work):
        self.vla, self.pool, self.keys, self.arm, self.work = vla, pool, keys, arm, work
        self.estimates = [tuple(s["estimate"]) for s in support]
        self.layouts = [s["initial_sim_sha256"] for s in support]
        self.queries = []

    def call(self, idx, op, args):
        outs = self.pool.call(idx, op, args)
        if op != "reset":
            return outs
        expected = [(t, CELL, i, 0, True) for t, i, _ in self.keys]
        require(list(args) == expected and list(idx) == list(range(len(self.keys))), "Query reset differs")
        self.anchor = [stream.public(o["obs"])["robot0_eef_pos"].copy() for o in outs]
        self.opening = dict(raw_tasks=np.array([o["description"] for o in outs]), starts=np.array(self.keys),
                            scene_seeds=np.zeros(len(outs), np.int64), X0=np.stack([o["x"] for o in outs]))
        states = self.pool.call(idx, "full_state", [None]*len(outs))
        for i, state in enumerate(states):
            require(state["task"] == self.keys[i][0] and stream.digest(state["state"]["sim"]) != self.layouts[i],
                    "Support and query reset to the same state or wrong task")
            self.opening.update(leaves(f"sim/{i}", state["state"]))
        self.opening.update({"raw/" + k: v for k, v in observations([o["obs"] for o in outs]).items()})
        return outs

    def predict_chunk(self, obs, tasks, seeds, query):
        require(query == len(self.queries) and seeds == [rollout.Episode(*k).noise_seed for k in self.keys],
                "Policy query cadence or seeds differ")
        mapped = [stream.policy_observation(o, anchor, est[0])
                  for o, anchor, est in zip(obs, self.anchor, self.estimates, strict=True)]
        self.queries.append(dict(control=query*N_EXEC, chunk_seeds=[chunk_seed(s, query) for s in seeds]))
        def predict():
            return self.pool.meter.event("vla_batch", 1, lambda: self.vla.predict_chunk(mapped, tasks, seeds, query))
        if query:
            return predict()
        inputs = self.opening
        inputs.update({"mapped/" + k: v for k, v in observations(mapped).items()})
        inputs.update(tasks=np.array(tasks), noise_seeds=np.array(seeds),
                      chunk_seeds=np.array(self.queries[-1]["chunk_seeds"]))
        path = self.work / "opening.npz"
        if self.arm == ARMS[0]:
            noise = self.vla._noise
            def record(actual, q):
                require(list(actual) == list(seeds) and q == 0 and "noise" not in inputs, "Wrong native noise")
                tensor = noise(actual, q)
                inputs["noise"] = tensor.detach().cpu().numpy().copy()
                return tensor
            with patch.object(self.vla, "_noise", record):
                proposal = predict()
            proposal_ok(proposal, len(obs))
            require("noise" in inputs, "Native prediction skipped noise")
            np.savez_compressed(path, **inputs, proposal=proposal)
        else:
            inputs["noise"] = self.vla._noise(seeds, query).detach().cpu().numpy()
            require(file_hash(path) == ref.read(self.work / (ARMS[0] + ".json"))["opening_sha256"],
                    "Stored opening digest changed")
            with np.load(path, allow_pickle=False) as tape:
                require(set(tape.files) == set(inputs) | {"proposal"}, "Opening fields differ")
                for key, value in inputs.items():
                    require(identical(tape[key], value), f"Paired opening differs at {key}")
                proposal = tape["proposal"]
            self.pool.meter.event("replayed_opening_batch", 1, lambda: None)
        self.opening = None
        return proposal

    def reset(self, x):
        self.deviations = [[] for _ in x]

    def on_chunk(self, x, chunk):
        proposal_ok(chunk, len(x))
        self.chunk = chunk
        self.commands = np.stack([issue(a[:N_EXEC], est, self.arm)
                                  for a, est in zip(chunk, self.estimates, strict=True)])

    def act(self, x, k, sim_states=None):
        require(sim_states is None, "Unexpected simulator input to controller")
        cmd = self.commands[:, k]
        for d, dev in zip(self.deviations, np.linalg.norm(cmd[:, :6] - self.chunk[:, k, :6], axis=1)):
            d.append(float(dev))
        return cmd

    def u_dev_mean(self, i):
        return float(np.mean(self.deviations[i]))


def run_pair(vla, pipe, reader, keys, work, pool_factory=fresh):
    """The reference native episode loop owns all test actions, replans, stopping and safety."""
    with pool_factory(work, "support", len(keys)) as (pool, meter):
        support = calibrate(vla, pool, pipe, keys, work)
        support_budget = meter.counts.copy()
    ref.write(work / "contract.json", dict(keys=keys, label=0, code=os.environ["RETRY_PIN"], reader=reader,
              upstream=ref.PIN, policy=vla.revision, cell=CELL, batch=len(keys), support_budget=support_budget))
    lim = jobs.speed_limit("pi05")
    require(lim and lim > 0, "Missing reference unsafe speed envelope")
    for arm in ARMS:
        with pool_factory(work, arm, len(keys)) as (pool, meter):
            bridge = Retry(vla, pool, keys, support, arm, work)
            eps = [rollout.Episode(*key) for key in keys]
            result = rollout.run_batch(bridge, bridge, eps, bridge, CELL, speed_limit=lim)
            jobs._save_traces(work / arm / "traces/retry.npz", result.traces)
            ref.write(work / (arm + ".json"), dict(rows=result.rows, arm=arm, budget=meter.counts,
                      queries=bridge.queries, opening_sha256=file_hash(work / "opening.npz"), speed_limit=lim))


def validate(work, keys, policy=ref.POLICY, native_horizon=True):
    contract = ref.read(work / "contract.json")
    require(contract["keys"] == list(map(list, keys)) and contract["code"] == os.environ["RETRY_PIN"] and
            contract["upstream"] == ref.PIN and contract["reader"]["weights_sha256"] == stream.WEIGHTS and
            contract["policy"] == policy and contract["label"] == 0 and contract["cell"] == CELL, "Wrong batch")
    n = len(keys)
    support = ref.read(work / "support.json")
    require(len(support) == n and contract["support_budget"]["step"] == 15*n and
            contract["support_budget"]["reset"] == n and contract["support_budget"]["vla_batch"] == 2,
            "Incomplete support")
    with np.load(work / "support.npz") as tape:
        proposals = np.concatenate((tape["proposals"][0,:,:10], tape["proposals"][1,:,:5]), axis=1)
        proposals[..., :6] = np.clip(proposals[..., :6], -1, 1)
        issued = tape["issued"].transpose(1, 0, 2)
        np.testing.assert_array_equal(issued, proposals)
        snapshots = {k:tape[k] for k in ("agentview_image","robot0_eef_pos","robot0_gripper_qpos")}
        speeds = tape["speeds"]
        require(speeds.shape==(n,15) and np.isfinite(speeds).all(), "Support speed trace differs")
        for i, (s, k) in enumerate(zip(support, keys, strict=True)):
            expect = rollout.Episode(*support_key(k))
            require(s["query"] == list(k) and s["support"] == list(support_key(k)) and
                    s["noise_seed"] == expect.noise_seed and s["scene_seed"] == 0 and
                    [q["control"] for q in s["queries"]] == [0,10] and
                    [q["chunk_seed"] for q in s["queries"]] == [chunk_seed(expect.noise_seed,q) for q in (0,1)] and
                    [u["end_control"] for u in s["updates"]] == [5,10,15] and
                    s["estimate"] == s["updates"][-1]["next_adapter"], "Support pairing or reader differs")
            for j, u in enumerate(s["updates"]):
                np.testing.assert_array_equal(u["issued_mean"], issued[i, j*5:j*5+5].mean(axis=0))
                for tag, ix in (("before",j),("after",j+1)):
                    require(u[tag+"_sha256"]==stream.digest(snapshots["agentview_image"][ix,i]), "Support image")
                    np.testing.assert_array_equal(u[tag+"_eef"],snapshots["robot0_eef_pos"][ix,i])
                    np.testing.assert_array_equal(u[tag+"_jaw_qpos"],snapshots["robot0_gripper_qpos"][ix,i])
            safety = s["support_safety"]
            require(safety["speed_viol"]==int((speeds[i]>jobs.speed_limit("pi05")).sum()) and
                    safety["unsafe"]==int(any(safety[k] for k in ("speed_viol","displaced","fallen","contact_steps"))),
                    "Invalid support safety summary")
    with np.load(work / "opening.npz") as tape:
        proposal, x0 = tape["proposal"], tape["X0"]
        proposal_ok(proposal, n)
        require(tape["starts"].tolist() == np.array(keys).tolist(), "Opening keys differ")
    opening_sha = file_hash(work / "opening.npz")
    for arm in ARMS:
        report = ref.read(work / (arm + ".json"))
        traces = jobs.load_traces(work / arm)
        rows = report["rows"]
        counts = report["budget"]
        require(report["arm"] == arm and report["opening_sha256"] == opening_sha and
                report["speed_limit"] == jobs.speed_limit("pi05") and set(traces) == set(keys) and
                counts["reset"] == n and counts["step"] == sum(r["steps"] for r in rows), "Invalid query budget")
        qcount = (max(r["steps"] for r in rows)+9)//10
        require(len(report["queries"]) == qcount and counts.get("vla_batch",0) == qcount-int(arm==ARMS[1])
                and counts.get("replayed_opening_batch",0) == int(arm==ARMS[1]), "Missing native query")
        for q, row in enumerate(report["queries"]):
            require(row == dict(control=10*q, chunk_seeds=[chunk_seed(rollout.Episode(*k).noise_seed,q)
                                                           for k in keys]), "Wrong query sequence")
        for i, (row, key, s) in enumerate(zip(rows, keys, support, strict=True)):
            tr, steps = traces[key], row["steps"]
            require((row["task"], row["init_id"], row["rep"]) == key and
                    row["noise_seed"] == rollout.Episode(*key).noise_seed and row["success"] in (0,1)
                    and 0 < steps <= max_steps(key[0]) and (not native_horizon or row["success"]
                    or steps == max_steps(key[0])), "Invalid native stopping or case")
            for name, shape in (("A", (steps,7)), ("A_vla", (steps,7)), ("X", (steps+1,25)), ("speed", (steps,))):
                require(tr[name].shape == shape and np.isfinite(tr[name]).all(), name)
            np.testing.assert_array_equal(tr["X"][0], x0[i])
            np.testing.assert_array_equal(tr["A_vla"][:10], proposal[i,:min(10,steps)])
            np.testing.assert_array_equal(tr["A"], issue(tr["A_vla"], s["estimate"], arm))
            require(row["speed_viol"] == int((tr["speed"] > report["speed_limit"]).sum()) and
                    row["unsafe"] == int(any(row[k] for k in ("speed_viol","fallen","displaced","contact_steps"))),
                    "Native unsafe metric differs")
