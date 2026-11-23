"""Decompose streaming SymMeta's result on ReGround's released 90-degree rotation test.

Each arm reruns the 280 distinct task-start pairs (seed label 0) through the same released
episode loop as the streaming run, changing only where the correction comes from, when it
starts, whether wrist commands are rotated and whether the policy's EEF input is rotated.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reground-rotation-reference-20261123-r1"))

import reference  # noqa: E402
from reference import require, write  # noqa: E402
import symmeta_adapter as base  # noqa: E402
from fluffy_bassoon.xemb.embodiment import retarget_action_from_motor  # noqa: E402

CHUNK = base.CHUNK
EXACT = (90.0, 1.0, 1)
SUPPORT_CONTROLS = 15
SEED = 0
STORE = reference.ROOT / "decompose-native"
# source: where the correction comes from. start: first control it can change.
# wrist: rotate the wrist commands too. obs: rotate the policy's EEF input, as the streaming run did.
ARMS = {
    "exact_asrun_0": dict(source="exact", start=0, wrist=False, obs=True),
    "exact_asrun_10": dict(source="exact", start=10, wrist=False, obs=True),
    "exact_world_0": dict(source="exact", start=0, wrist=False, obs=False),
    "exact_full_0": dict(source="exact", start=0, wrist=True, obs=False),
    "exact_full_10": dict(source="exact", start=10, wrist=True, obs=False),
    "stream_full": dict(source="stream", start=10, wrist=True, obs=False),
    "retry_full": dict(source="support", start=0, wrist=True, obs=False),
}


def support_start(init_id: int) -> int:
    """A calibration start (0-14) of the same task; the scored starts are 15-49."""
    return (init_id - 15) % 15


def corrected(proposals, estimate, wrist: bool):
    """Commands issued for one five-control block of policy proposals."""
    out = np.array(proposals, dtype=np.float32)
    out[..., :6] = np.clip(out[..., :6], -1.0, 1.0)
    if tuple(estimate) != base.IDENTITY:
        out = retarget_action_from_motor(np.clip(proposals, base.LO, base.HI), *estimate, base.LO, base.HI, rotate_wrist=wrist)
    return out


class ArmBatch(base.MatchedBatch):
    def __init__(self, vla, pool, pipe, arm, given=None, support=False):
        super().__init__(vla, pool, pipe)
        self.arm, self.given, self.support = arm, given, support
        self.reads = support or arm["source"] == "stream"

    def estimate(self, i, lane):
        if self.support:
            return base.IDENTITY
        if self.arm["source"] == "stream":
            return tuple(lane.estimate)
        fixed = EXACT if self.arm["source"] == "exact" else self.given[i]
        return fixed if lane.steps >= self.arm["start"] else base.IDENTITY

    def call(self, idx, cmd, args):
        if cmd == "step" and not self.reads:
            outs = self.pool.call(idx, cmd, args)
            for i, out in zip(idx, outs):
                lane = self.lanes[i]
                lane.steps += 1
                lane.ended = out["success"] or lane.steps == self.horizons[i]
            return outs
        outs = super().call(idx, cmd, args)
        if cmd == "reset":
            if self.support:
                # One control past the support so the third chunk is read before the episode ends.
                for out in outs:
                    out["max_steps"] = SUPPORT_CONTROLS + 1
                self.horizons = [SUPPORT_CONTROLS + 1] * len(outs)
            for lane in self.lanes:
                lane.applied = []
        return outs

    def predict_chunk(self, observations, tasks, seeds, query):
        estimates = [self.estimate(i, lane) for i, lane in enumerate(self.lanes)]
        items = [base.policy_observation(o, lane.anchor, est[0]) if self.arm["obs"] else o
                 for o, lane, est in zip(observations, self.lanes, estimates)]
        result = self.vla.predict_chunk(items, tasks, seeds, query)
        for lane, est in zip(self.lanes, estimates):
            if not lane.ended:
                lane.queries.append(dict(start_control=lane.steps, estimate=list(est)))
        return result

    def act(self, _x, k, sim_states=None):
        require(sim_states is None, "Unexpected simulator state")
        if k % CHUNK == 0:
            proposals = self.chunk[:, k:k + CHUNK]
            self.commands = np.array(proposals, dtype=np.float32)
            for i, lane in enumerate(self.lanes):
                est = self.estimate(i, lane)
                if not lane.ended:
                    lane.applied.append(list(est))
                self.commands[i] = corrected(proposals[i], est, self.arm["wrist"])
        issued = self.commands[:, k % CHUNK]
        for i in range(len(self.lanes)):
            self.deviations[i].append(float(np.linalg.norm(issued[i, :6] - self.chunk[i, k, :6])))
        return issued


def run_job(name, job, ctx, pipe, info, dest):
    from reground.eval import jobs, rollout
    from reground.method import build

    arm = ARMS[name]
    upstream = rollout.run_batch
    receipts = []

    def batch(vla, pool, eps, _, cell, **kw):
        given, support = None, [None] * len(eps)
        if arm["source"] == "support":
            first = [replace(e, init_id=support_start(e.init_id), video=False) for e in eps]
            probe = ArmBatch(vla, pool, pipe, arm, support=True)
            upstream(probe, probe, first, probe, cell, **kw)
            # A support that ended early keeps the nominal interface.
            given = [tuple(lane.updates[-1]["next_adapter"]) if len(lane.updates) == SUPPORT_CONTROLS // CHUNK else base.IDENTITY
                     for lane in probe.lanes]
            support = [dict(init_id=e.init_id, noise_seed=e.noise_seed, controls=lane.steps, updates=lane.updates,
                            issued_identity=all(tuple(a) == base.IDENTITY for a in lane.applied))
                       for e, lane in zip(first, probe.lanes)]
        bridge = ArmBatch(vla, pool, pipe, arm, given)
        result = upstream(bridge, bridge, eps, bridge, cell, **kw)
        for row, lane, sup in zip(result.rows, bridge.lanes, support):
            require(lane.steps == row["steps"], "Extraneous steps")
            receipts.append(dict(task=row["task"], init_id=row["init_id"], rep=row["rep"], noise_seed=row["noise_seed"],
                                 scored_controls=lane.steps, applied=lane.applied, queries=lane.queries,
                                 updates=lane.updates, support=sup))
        return result

    with patch.object(jobs, "job_dir", lambda _: dest), patch.object(build, "make_controller", lambda *a, **kw: None), \
            patch.object(rollout, "run_batch", batch):
        jobs.run_eval_job(job, ctx)
    write(dest / "calibration.json", receipts)
    manifest = reference.read(dest / "manifest.json")
    manifest.update(arm=name, arm_config=arm, reader=info, code_revision=os.environ["SYMMETA_CODE_REV"],
                    arms_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), device=ctx.device,
                    accelerator=os.environ.get("SYMMETA_ACCELERATOR", "cpu"), support_controls=SUPPORT_CONTROLS)
    write(dest / "manifest.json", manifest)
    validate(name, dest, job)


def validate(name, dest, job):
    """The issued commands are the recorded proposals under the recorded corrections, and each arm kept its rule."""
    from reground.eval import jobs
    from reground.envs.libero_env import max_steps

    arm = ARMS[name]
    with (dest / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    traces = jobs.load_traces(dest)
    receipts = reference.read(dest / "calibration.json")
    manifest = reference.read(dest / "manifest.json")
    keys = [(r["task"], int(r["init_id"]), int(r["rep"])) for r in rows]
    require(manifest["arm"] == name and manifest["arm_config"] == arm and manifest["git_sha"] == reference.PIN
            and manifest["arms_sha256"] == sha256(Path(__file__).read_bytes()).hexdigest()
            and keys == list(map(tuple, job["episodes"])) and len(receipts) == len(rows), "Invalid or incomplete shard")
    for key, row, rec in zip(keys, rows, receipts):
        t = int(row["steps"])
        tr = traces[key]
        require(0 < t <= max_steps(key[0]) and (row["success"] == "1" or t == max_steps(key[0])) and row["method"] == name
                and row["shift"] == "rot_z" and float(row["level"]) == 90 and int(row["adapt_seed"]) == SEED
                and (rec["task"], rec["init_id"], rec["rep"]) == key and rec["scored_controls"] == t, row)
        applied = [tuple(a) for a in rec["applied"]]
        require(len(applied) == -(-t // CHUNK), "Missing correction record")
        for b, est in enumerate(applied):
            block = slice(b * CHUNK, min((b + 1) * CHUNK, t))
            np.testing.assert_allclose(tr["A"][block], corrected(tr["A_vla"][block], est, arm["wrist"]), atol=1e-6)
        head = arm["start"] // CHUNK
        if arm["source"] == "exact":
            require(applied == [base.IDENTITY] * min(head, len(applied)) + [EXACT] * max(0, len(applied) - head), "Exact arm rule")
        elif arm["source"] == "support":
            sup = rec["support"]
            last = tuple(sup["updates"][-1]["next_adapter"]) if len(sup["updates"]) == SUPPORT_CONTROLS // CHUNK else base.IDENTITY
            require(sup["init_id"] == support_start(key[1]) and sup["issued_identity"] and sup["controls"] <= SUPPORT_CONTROLS + 1
                    and set(applied) == {last}, "Retry arm rule")
        else:
            require(applied[:2] == [base.IDENTITY] * min(2, len(applied)), "Streaming arm changed its opening")


def units(name):
    from reground.eval.jobs import shard_jobs

    jobs = [{**j, "method": name, "cfg": dict(ARMS[name])} for j in reference.grid() if j["seed"] == SEED]
    return shard_jobs(jobs, 4 * reference.BATCH)


def lane(args):
    for name in args.arms:
        # The proven lane (GPU checks, shard locks, staging) with this arm's jobs, store, runner and checks.
        with patch.multiple(base, STORE=STORE / name, units=lambda seeds, name=name: units(name),
                            run_job=lambda job, ctx, pipe, info, dest, name=name: run_job(name, job, ctx, pipe, info, dest),
                            validate=lambda dest, job, fixture=False, name=name: validate(name, dest, job)):
            base.lane(args)


def summarize(args):
    out = {}
    for name in args.arms:
        rows = []
        for job in units(name):
            dest = STORE / name / "shards" / reference.label(job)
            if not dest.exists():
                continue
            validate(name, dest, job)
            rows += list(csv.DictReader((dest / "results.csv").read_text().splitlines()))
        by_task = {}
        for r in rows:
            by_task.setdefault(r["task"], []).append(int(r["success"]))
        out[name] = dict(config=ARMS[name], episodes=len(rows), successes=sum(int(r["success"]) for r in rows),
                         unsafe=sum(int(r["unsafe"]) for r in rows),
                         by_task={task: [sum(v), len(v)] for task, v in sorted(by_task.items())})
    write(args.attempt / "decompose-summary.json", out)
    print(json.dumps({k: [v["successes"], v["episodes"]] for k, v in out.items()}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("lane", "summary"))
    p.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    p.add_argument("--shards", nargs="+", type=int, choices=range(18))
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--attempt", type=Path, required=True)
    args = p.parse_args()
    args.seeds = [SEED]
    reference.configure()
    (lane if args.mode == "lane" else summarize)(args)


if __name__ == "__main__":
    main()
