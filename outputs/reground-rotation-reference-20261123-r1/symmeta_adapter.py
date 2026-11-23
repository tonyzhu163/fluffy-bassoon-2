"""Evaluate frozen SymMeta through the released ReGround job writer and episode loop."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import gzip
from hashlib import sha256
import json
from multiprocessing import Pipe
import os
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np

import reference
from reference import require, write
from symmeta_reader import WEIGHTS
from fluffy_bassoon.xemb.embodiment import retarget_action_from_motor, retarget_observation_for_motor


HERE = Path(__file__).resolve().parent
STORE = reference.ROOT / "symmeta-native"
IDENTITY = (0., 1., 1)
CHUNK = 5
CFG = {"reader_seed": 41, "reader_chunk": CHUNK, "max_updates": 24}
METHOD = "symmeta"
LO = np.full(7, -1, np.float32)
HI = -LO


def digest(value):
    a = np.ascontiguousarray(value)
    return sha256(a.tobytes()).hexdigest()


def public(obs):
    state = obs["robot_state"]
    p = dict(agentview_image=obs["pixels"]["image"], robot0_eef_pos=state["eef"]["pos"],
             robot0_gripper_qpos=state["gripper"]["qpos"])
    image = p["agentview_image"]
    require(image.shape == (256, 256, 3) and image.dtype == np.uint8
            and p["robot0_eef_pos"].shape == (3,) and p["robot0_gripper_qpos"].shape == (2,),
            "Unexpected public LeRobot observation")
    return p


def policy_observation(obs, anchor, yaw):
    if yaw == 0:
        return obs
    state = obs["robot_state"]
    pos = retarget_observation_for_motor({"robot0_eef_pos": state["eef"]["pos"]}, anchor, yaw)
    return {**obs, "robot_state": {**state, "eef": {**state["eef"], "pos": pos["robot0_eef_pos"]}}}


@contextmanager
def reader_process():
    archive = Path(os.environ["SYMMETA_SOURCE"])
    # Keep the historical JAX/Flax/OpenCV environment out of the native policy process.
    paths = (HERE.parent / "symmeta-original-pro-20261123-r1", archive / "src",
             archive / "outputs/xemb-icml-campaign-20261121-r1",
             archive / "outputs/libero-xemb-ur5e-sysid-20261120-r1",
             archive / "outputs/libero-crossarm-20261120-r1", Path(os.environ["SYMMETA_CV2"]))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, paths)), CUDA_VISIBLE_DEVICES="",
               JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false", REGROUND_PINNED_OFFLINE="0")
    pipe, child = Pipe()
    with pipe, child:
        proc = subprocess.Popen([os.environ["SYMMETA_PYTHON"], "-Bu", str(HERE / "symmeta_reader.py"),
                                 str(child.fileno())], pass_fds=(child.fileno(),), env=env)
        child.close()
        try:
            info = pipe.recv()
            require(info["update_limit"] == CFG["max_updates"], "Unexpected reader update limit")
            yield pipe, info
        finally:
            if proc.poll() is None:
                try:
                    pipe.send(("close", None))
                except (BrokenPipeError, ConnectionResetError):
                    pass
            if proc.wait() and sys.exc_info()[0] is None:
                raise RuntimeError("CPU reader exited with an error")


class Lane:
    def __init__(self, obs):
        self.before = public(obs)
        self.anchor = self.before["robot0_eef_pos"].copy()
        self.estimate = IDENTITY
        self.issued = []
        self.steps = 0
        self.updates = []
        self.queries = []
        self.ended = False
        self.first_changed = None
        self.midpoint_renders = 0


class MatchedBatch:
    """Only this harness sees rollout metadata. The estimator receives public() slices."""

    needs_sim = False

    def __init__(self, vla, pool, pipe):
        self.vla, self.pool, self.pipe = vla, pool, pipe

    def call(self, idx, cmd, args):
        if cmd == "reset":
            outs = self.pool.call(idx, cmd, args)
            self.lanes = [Lane(o["obs"]) for o in outs]
            self.horizons = [o["max_steps"] for o in outs]
            self.pipe.send(("begin", len(idx)))
            self.pipe.recv()
            return outs
        if cmd != "step":
            require(cmd in ("frame", "summary"), "Unexpected simulator access")
            return self.pool.call(idx, cmd, args)

        rendered = []
        for i, (action, native_render) in zip(idx, args):
            lane = self.lanes[i]
            endpoint = (lane.steps + 1) % CHUNK == 0 and len(lane.updates) < CFG["max_updates"]
            rendered.append((action, native_render or endpoint))
            lane.midpoint_renders += int(endpoint and not native_render)
        outs = self.pool.call(idx, cmd, rendered)
        for i, (action, native_render), out in zip(idx, args, outs):
            lane = self.lanes[i]
            lane.steps += 1
            lane.ended = out["success"] or lane.steps == self.horizons[i]
            if len(lane.updates) < CFG["max_updates"]:
                lane.issued.append(action)
                if len(lane.issued) == CHUNK and not lane.ended:
                    after = public(out["obs"])
                    issued = np.stack(lane.issued)
                    start = time.perf_counter()
                    self.pipe.send(("observe", (i, lane.before, after, issued)))
                    update = self.pipe.recv()
                    update.update(ipc_and_compute_s=time.perf_counter() - start, end_control=lane.steps,
                                  before_sha256=digest(lane.before["agentview_image"]),
                                  after_sha256=digest(after["agentview_image"]))
                    require(update["chunk"] == len(lane.updates) + 1, "Unexpected reader state")
                    lane.estimate = tuple(update["next_adapter"])
                    lane.before = after
                    lane.issued.clear()
                    lane.updates.append(update)
            # Keep the upstream policy's last-observation semantics even for ended lanes.
            if not native_render:
                out.pop("obs", None)
        return outs

    def predict_chunk(self, observations, tasks, seeds, query):
        from reground.policies.vla import chunk_seed
        items = [policy_observation(o, s.anchor, s.estimate[0]) for o, s in zip(observations, self.lanes)]
        result = self.vla.predict_chunk(items, tasks, seeds, query)
        for i, (lane, obs, item, seed) in enumerate(zip(self.lanes, observations, items, seeds)):
            if not lane.ended:
                lane.queries.append(dict(start_control=lane.steps, chunk_seed=chunk_seed(seed, query),
                                         estimate=lane.estimate, proposal_sha256=digest(result[i]),
                                         public_eef=obs["robot_state"]["eef"]["pos"].tolist(),
                                         policy_eef=item["robot_state"]["eef"]["pos"].tolist()))
        return result

    def reset(self, _x):
        self.deviations = [[] for _ in self.lanes]

    def on_chunk(self, _x, chunk):
        require(chunk.shape == (len(self.lanes), 50, 7) and np.isfinite(chunk).all(),
                "Unexpected pi05 chunk")
        self.chunk = chunk

    def act(self, _x, k, sim_states=None):
        require(sim_states is None, "Unexpected simulator state")
        if k % CHUNK == 0:
            proposals = self.chunk[:, k:k + CHUNK]
            self.commands = proposals.copy()
            # Use the stock Base clipping in the identity prefix, including its raw gripper.
            self.commands[:, :, :6] = np.clip(proposals[:, :, :6], -1., 1.)
            for i, lane in enumerate(self.lanes):
                if lane.estimate != IDENTITY:
                    self.commands[i] = retarget_action_from_motor(
                        np.clip(proposals[i], LO, HI), *lane.estimate, LO, HI)
        issued = self.commands[:, k % CHUNK]
        for i, lane in enumerate(self.lanes):
            proposal = self.chunk[i, k]
            base = proposal.copy()
            base[:6] = np.clip(base[:6], -1., 1.)
            if not lane.ended and lane.first_changed is None and np.any(base != issued[i]):
                lane.first_changed = lane.steps
            self.deviations[i].append(float(np.linalg.norm(issued[i, :6] - proposal[:6])))
        return issued

    def u_dev_mean(self, i):
        return float(np.mean(self.deviations[i]))


def run_job(job, ctx, pipe, info, dest, *, fixture=False):
    from reground.eval import jobs, rollout
    from reground.method import build
    from reground.envs.libero_env import N_SETTLE
    from reground.policies.vla import N_EXEC

    require(N_EXEC == 10 and N_SETTLE == 10, "Unexpected native control cadence or settle")
    upstream_batch = rollout.run_batch
    costs = []

    def batch(vla, pool, eps, _, cell, **kw):
        bridge = MatchedBatch(vla, pool, pipe)
        result = upstream_batch(bridge, bridge, eps, bridge, cell, **kw)
        for row, lane in zip(result.rows, bridge.lanes):
            require(lane.steps == row["steps"], "Extraneous steps")
            costs.append(dict(task=row["task"], init_id=row["init_id"], rep=row["rep"],
                              adapt_seed=job["seed"], noise_seed=row["noise_seed"], scene_seed=0,
                              native_resets=1, settle_controls=N_SETTLE, extra_resets=0, probe_controls=0,
                              offline_target_transitions=0, scored_controls=lane.steps,
                              online_target_transitions=CHUNK * len(lane.updates),
                              extra_render_requests=lane.midpoint_renders,
                              first_changed_control=lane.first_changed, updates=lane.updates,
                              queries=lane.queries))
        return result

    # job selection, prompts, scene seeds, success/safety/termination, CSV, NPZ and manifests
    # are the upstream evaluator's. Only its controller + public sensor routing change.
    with patch.object(jobs, "job_dir", lambda _: dest), patch.object(build, "make_controller", lambda *a, **kw: None), \
            patch.object(rollout, "run_batch", batch):
        jobs.run_eval_job(job, ctx)
    write(dest / "calibration.json", costs)
    manifest = reference.read(dest / "manifest.json")
    manifest.update(reader=info, code_revision=os.environ["SYMMETA_CODE_REV"],
                    adapter_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    reader_bridge_sha256=sha256((HERE/"symmeta_reader.py").read_bytes()).hexdigest(),
                    assignment_sha256=sha256((HERE / "evidence/symmeta-jobs.json.gz").read_bytes()).hexdigest(),
                    reference_assignment_sha256=sha256((HERE/"evidence/rotation-jobs.json.gz").read_bytes()).hexdigest(),
                    scene_seed=0, reader_chunk=CHUNK, policy_n_exec=N_EXEC, native_settle=N_SETTLE,
                    cpu_fixture=fixture, device=ctx.device, runtime_root=str(reference.ROOT),
                    accelerator=os.environ.get("SYMMETA_ACCELERATOR", "cpu"),
                    checkpoint_sha256=WEIGHTS, separate_target_fit_transitions=0)
    write(dest / "manifest.json", manifest)
    validate(dest, job, fixture)


def validate(dest, job, fixture=False):
    from reground.eval import jobs
    from reground.envs.libero_env import max_steps
    from reground.utils import cfg_id, noise_seed
    from reground.policies.vla import chunk_seed
    with (dest / "results.csv").open() as f:
        rows = csv.DictReader(f)
        require(rows.fieldnames == jobs.FIELDS, "Results schema differs")
        rows = list(rows)
    traces = jobs.load_traces(dest)
    costs = reference.read(dest / "calibration.json")
    manifest = reference.read(dest / "manifest.json")
    require(manifest["git_sha"] == reference.PIN and manifest["cfg"] == CFG and manifest["cpu_fixture"] == fixture
            and manifest["n_grid"] == len(job["full_episodes"]) and manifest["checkpoint_sha256"] == WEIGHTS
            and manifest["method"] == METHOD and manifest["adapt_seed"] == job["seed"]
            and manifest["adapter_sha256"] == sha256(Path(__file__).read_bytes()).hexdigest()
            and manifest["reader_bridge_sha256"] == sha256((HERE/"symmeta_reader.py").read_bytes()).hexdigest()
            and (fixture or manifest["vla_revision"] == reference.POLICY), "Invalid manifest")
    keys = [(r["task"], int(r["init_id"]), int(r["rep"])) for r in rows]
    require(keys == list(map(tuple, job["episodes"])) and set(traces) == set(keys)
            and len(costs) == len(rows), "Incomplete assigned cases")
    for key, row, cost in zip(keys, rows, costs):
        task, init, rep = key
        t = int(row["steps"])
        tr = traces[key]
        require(0 < t <= max_steps(task) and (row["success"] == "1" or t == max_steps(task))
                and row["success"] in ("0", "1") and int(row["noise_seed"]) == noise_seed(init, rep)
                and row["method"] == METHOD and row["cfg_id"] == cfg_id(CFG) and row["policy"] == "pi05"
                and row["shift"] == "rot_z" and float(row["level"]) == 90 and row["split"] == "test"
                and int(row["adapt_seed"]) == job["seed"] and (cost["task"], cost["init_id"], cost["rep"]) == key
                and cost["noise_seed"] == int(row["noise_seed"]) and cost["scored_controls"] == t, row)
        for field, shape in (("X", (t+1, 25)), ("A", (t, 7)), ("A_vla", (t, 7)), ("speed", (t,))):
            require(tr[field].shape == shape and np.isfinite(tr[field]).all(), field)
        require(tr["ref"].shape == (t, 3) and np.isnan(tr["ref"]).all(), "Unexpected reference trajectory")
        require(int(row["speed_viol"]) == int((tr["speed"] > jobs.speed_limit("pi05")).sum()) and
                int(row["unsafe"]) == int(any(int(row[f]) for f in ("displaced","fallen","contact_steps","speed_viol"))),
                "Unsafe or speed count differs")
        ends = list(range(CHUNK, min(CHUNK * CFG["max_updates"] + 1, t), CHUNK))
        require([u["end_control"] for u in cost["updates"]] == ends, "Future or missing reader update")
        for u, end in zip(cost["updates"], ends):
            np.testing.assert_array_equal(u["issued_mean"], tr["A"][end-CHUNK:end].mean(axis=0))
        require([q["start_control"] for q in cost["queries"]] == list(range(0,t,10))
                and [q["chunk_seed"] for q in cost["queries"]] ==
                [chunk_seed(noise_seed(init, rep), k) for k in range((t+9)//10)], "Query cadence/noise differs")
        base = tr["A_vla"].copy()
        base[:, :6] = np.clip(base[:, :6], -1., 1.)
        np.testing.assert_array_equal(base[:10], tr["A"][:10])
        np.testing.assert_array_equal(base[:, 3:6], tr["A"][:, 3:6])
        changed = np.flatnonzero(np.any(base != tr["A"], axis=1))
        require(cost["first_changed_control"] == (int(changed[0]) if len(changed) else None),
                "Calibration receipt differs from commands")


def assignments(seeds):
    with gzip.open(HERE / "evidence/symmeta-jobs.json.gz", "rt") as f:
        jobs = json.load(f)
    expected = [{**j, "method": METHOD, "cfg": CFG} for j in reference.grid()]
    require(jobs == expected, "Matched assignment manifest differs from the release schedule")
    return [j for j in jobs if j["seed"] in seeds]


def units(seeds):
    from reground.eval.jobs import shard_jobs
    return shard_jobs(assignments(seeds), 4 * reference.BATCH)


def lane(args):
    from reground.eval import jobs
    import torch
    from fluffy_bassoon.baige_gpu import cuda_pci, pci

    probe = reference.read(args.attempt / "gpu.json")
    require(probe["hostname"] == socket.gethostname() and probe["status"] == "passed"
            and probe["allocation_type"] == "baige_container" and torch.cuda.is_available()
            and torch.cuda.device_count() == len(probe["devices"]), "Mismatched allocation probe")
    device = probe["devices"][args.rank]
    require(args.rank >= 0 and device["allocation_ordinal"] == args.rank
            and cuda_pci(args.rank) == pci(device["pci_bus_id"])
            and device["cuda_name"] == torch.cuda.get_device_name(args.rank), "GPU identity differs")
    allowed = device["cuda_name"] == "NVIDIA RPBZZZ6" or any(
        name in device["cuda_name"].upper() for name in ("RTX 4090", "RTX PRO 6000"))
    require(allowed, f"Unexpected device {device['cuda_name']}")
    torch.cuda.set_device(args.rank)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(reference.egl_device(args.rank))
    os.environ["SYMMETA_ACCELERATOR"] = torch.cuda.get_device_name(args.rank)
    ctx = jobs.WorkerCtx("pi05", f"cuda:{args.rank}", reference.BATCH, lambda m: print(m, flush=True))
    try:
        with reader_process() as (pipe, info):
            for job in units(args.seeds):
                if args.shards is not None and job["shard"] not in args.shards:
                    continue
                key = reference.label(job)
                dest = STORE / "shards" / key
                # Each publication is a whole native shard. Failed attempts never enter the result store.
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.with_suffix(".lock").open("a") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if dest.exists():
                        validate(dest, job)
                        continue
                    if ctx._vla is None:
                        reference.strict_policy(ctx)
                    stage = Path(tempfile.mkdtemp(prefix=f"seed{job['seed']}-shard{job['shard']}-", dir=args.attempt))
                    run_job(job, ctx, pipe, info, stage)
                    stage.rename(dest)
                    print(f"published {dest}", flush=True)
    finally:
        ctx.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("lane", "cpu", "manifest", "pair"))
    p.add_argument("--seeds", nargs="+", type=int, choices=reference.SEEDS, default=reference.SEEDS)
    p.add_argument("--shards", nargs="+", type=int, choices=range(18))
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--attempt", type=Path)
    args = p.parse_args()
    reference.configure()
    if args.mode != "manifest":
        require(args.attempt is not None, "--attempt is required")
    if args.mode == "lane":
        lane(args)
    elif args.mode == "cpu":
        from symmeta_cpu import qualify
        qualify(args.attempt)
    elif args.mode == "manifest":
        print(json.dumps(assignments(args.seeds), indent=1))
    else:
        pair(args)


def pair(args):
    """Read-only post-run match against the release runner's base and ReGround shards."""
    from reground.eval.jobs import load_traces
    rows = []
    for job in units(args.seeds):
        if args.shards is not None and job["shard"] not in args.shards:
            continue
        dest = STORE / "shards" / reference.label(job)
        validate(dest, job)
        inputs = {}
        for method in ("base", "reground"):
            other = next(j for j in reference.units(method, [job["seed"]]) if j["shard"] == job["shard"])
            path = reference.output(other)
            reference.check_eval(path, other)
            inputs[method] = path
        traces, base, fitted = (load_traces(p) for p in (dest, inputs["base"], inputs["reground"]))
        tables = [list(csv.DictReader((p/"results.csv").read_text().splitlines())) for p in (dest, *inputs.values())]
        for ours, native, reground in zip(*tables):
            keys = ("policy", "task", "suite", "shift", "level", "adapt_seed", "split", "init_id", "rep", "noise_seed")
            require(all(ours[k] == native[k] == reground[k] for k in keys), "Different assignments")
            k = (ours["task"], int(ours["init_id"]), int(ours["rep"]))
            prefix = min(10, int(ours["steps"]), int(native["steps"]))
            if prefix < 10:
                require(ours["steps"] == native["steps"] and ours["success"] == native["success"],
                        "Different stopping in the common opening")
            np.testing.assert_array_equal(traces[k]["X"][0], fitted[k]["X"][0], err_msg=f"{k} fitted reset")
            opening = min(prefix, int(reground["steps"]))
            np.testing.assert_array_equal(traces[k]["A_vla"][:opening], fitted[k]["A_vla"][:opening],
                                          err_msg=f"{k} fitted first policy query")
            for field, n in (("X", prefix+1), ("A", prefix), ("A_vla", prefix), ("speed", prefix)):
                np.testing.assert_array_equal(traces[k][field][:n], base[k][field][:n], err_msg=f"{k} {field}")
            rows.append({**{k:ours[k] for k in keys}, "matched_prefix":prefix,
                         **{f"{m}_{v}":int(r[v]) for m,r in zip((METHOD,"base","reground"), (ours,native,reground))
                            for v in ("steps","success","unsafe")}})
    write(args.attempt / "paired-cases.json", rows)


if __name__ == "__main__":
    main()
