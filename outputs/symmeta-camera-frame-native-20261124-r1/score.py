"""Reconcile every issued and physical command and score matched native outcomes."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import math
from pathlib import Path
from statistics import mean

import numpy as np

from protocol import MANIFEST_SHA, RELEASE_PIN, atomic, digest, encode, parse, require


def fp(a):
    arr = np.ascontiguousarray(a)
    return digest(repr((arr.shape, arr.dtype.str)).encode() + arr.tobytes())


def xy_rotation(yaw):
    theta = math.radians(yaw)
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s], [s, c]], np.float32)


def commands(proposal, correction, truth):
    """The motor inverts a chunk. The simulator applies its forward map to one control at a time."""
    issued = np.clip(proposal, -1, 1).astype(np.float32)
    inv = np.float32(1 / np.clip(correction["xy_scale"], 0.25, 4))
    issued[:, :2] = (issued[:, :2] @ xy_rotation(-correction["yaw_deg"]).T) * inv
    issued[:, 2] *= inv
    issued[:, 6] *= correction["gripper_sign"]
    issued = np.clip(issued, -1, 1)
    actual = issued.copy()
    rotation = xy_rotation(truth["yaw_deg"]).T
    for i, command in enumerate(issued):
        # A batched matrix product rounds differently from the native per-control dot at oblique angles.
        actual[i, :2] = (command[:2] * np.float32(truth["xy_scale"])) @ rotation
    actual[:, 2] *= truth["xy_scale"]
    actual[:, 6] *= truth["gripper_sign"]
    return issued, np.clip(actual, -1, 1)


def inspect(row, arm, obj, cpu):
    c = row["corrections"][arm]
    q, ref = row["query"], row["reference"]
    rec, env, body = obj["native"], obj["env"], obj["body"]
    reset, n = rec["reset"], rec["controls"]
    require(env["source_pin"] == RELEASE_PIN and env["manifest_sha256"] == MANIFEST_SHA
            and all(env[k] == ref[k] for k in ("released_row_fingerprint", "bddl_sha256"))
            and all(env[k] == q[k] for k in ("suite", "task_id", "init_state_id", "seed"))
            and env["perturbation_plan"]["case_id"] == "BASE"
            and reset["physics"] == env["initial_state_fingerprint"] != row["support_physics"]
            and reset["physics"] != row.get("calibration_physics")
            and env["scene_fingerprint"] == reset["scene_fingerprint"]
            and env["perturbation_plan_fingerprint"] == reset["perturbation_plan"], "release/reset")
    require(body["arm"] == reset["arm"] == "UR5e" and body["joint_count"] == reset["joint_count"] == 6
            and body["controller"] == "OSC_POSE" and body["gripper"] == "PandaGripper"
            and body["model_class"] == reset["model_class"] == row["model_class"]
            and reset["worker_step_index"] == 0 and reset["action_schema"] == row["action_schema"]
            and rec["inferred"] == c and reset["horizon"] == ref["horizon"]
            and rec["released_prompt"] == reset["prompt"] == row["prompt"], "motor and correction")
    require("llvmpipe" in reset["renderer"].lower() if cpu else reset["renderer_vendor"] == "NVIDIA Corporation",
            "renderer")
    horizon = reset["horizon"]
    seed = q["seed"] + 817117
    noise = np.random.default_rng(seed).standard_normal(((horizon+4)//5, 10, 32)).astype("f4")
    require(reset["noise_seed"] == seed and reset["noise_sha256"] == fp(noise)
            and len(obj["chunks"]) == (n+4)//5 and 0 < n <= horizon, "noise or episode length")
    rewards = []
    for i, block in enumerate(obj["chunks"]):
        p = np.asarray(block["raw"], np.float32)
        require(p.shape == (10, 7) and np.isfinite(p).all() and block["noise"] == fp(noise[i]), "policy output")
        if not i:
            require(fp(p) == rec["opening_proposal_sha256"] and block["motor_inputs"] == reset["motor_frame0"],
                    "opening policy call")
        issued, physical = commands(p[:5], c, row["true_map"])
        steps = block["steps"]
        count = min(5, n-5*i)
        require(len(steps) == count and np.array_equal(issued[:count], np.asarray(block["issued"], np.float32))
                and np.array_equal(physical[:count], np.asarray([s["action"] for s in steps], np.float32)),
                f"wrong issued or physical command at chunk {i}")
        for s in steps:
            rewards.append(s["reward"])
            require(s["index"] == len(rewards) and type(s["success"]) is bool and
                    s["reward"] == int(s["success"]) and s["terminated"] is s["success"]
                    and s["truncated"] is (len(rewards) == horizon and not s["success"])
                    and (len(rewards) == n or not (s["terminated"] or s["truncated"])), "native control flags")
    require(all(steps[-1][k] == rec[k] for k in ("success", "terminated", "truncated"))
            and rewards == rec["native_rewards"] and rec["native_return"] == sum(rewards) == int(rec["success"]),
            "native endpoint")
    if cpu:
        require(n == 10, "CPU fixture length")
    else:
        require(rec["terminated"] != rec["truncated"], "incomplete native retry")
    return int(rec["success"])


def task_interval(values):
    """Exact discrete percentile interval for equal-task draws, allowing unequal cases per task."""
    hist = Counter({Fraction(0): 1})
    for _ in values:
        acc = Counter()
        for x, frequency in hist.items():
            for y in values:
                acc[x+y] += frequency
        hist = acc
    endpoints = []
    total = sum(hist.values())
    for num in (1, 39):
        upto = 0
        for x, frequency in sorted(hist.items()):
            upto += frequency
            if 40*upto >= num*total:
                endpoints.append(100*float(x)/len(values))
                break
    return endpoints


def covariance(x, y):
    return None if len(x) < 2 else sum((a-mean(x))*(b-mean(y)) for a,b in zip(x,y))/(len(x)-1)


def describe(rows):
    def stats(items):
        a = [r["success"]["A"] for r in items]
        b = [r["success"]["B"] for r in items]
        return dict(starts=len(items), successes={arm:sum(r["success"][arm] for r in items)
                    for arm in items[0]["success"]}, a_only=sum(x > y for x,y in zip(a,b)),
                    b_only=sum(x < y for x,y in zip(a,b)), paired_success_pp=100*(mean(a)-mean(b)),
                    mean_abs_yaw_deg={arm:mean(r["yaw_error_deg"][arm] for r in items) for arm in ("A","B")},
                    success_covariance=covariance(a,b))

    tasks = defaultdict(list)
    for r in rows:
        tasks[(r["suite"], r["task_id"])].append(r)
    by_task = {f"{s}/{i}":stats(items) for (s,i),items in tasks.items()}
    effects = [Fraction(v["a_only"]-v["b_only"], v["starts"]) for v in by_task.values()]
    reductions = [v["mean_abs_yaw_deg"]["B"]-v["mean_abs_yaw_deg"]["A"] for v in by_task.values()]
    gains = [100*float(x) for x in effects]
    out = dict(**stats(rows), tasks=by_task, task_count=len(tasks), task_weighted_success_gain_pp=mean(gains),
               task_cluster_ci95_pp=task_interval(effects), task_effect_range_pp=[min(gains),max(gains)],
               task_yaw_error_reduction_success_gain_cov_deg_pp=covariance(reductions,gains),
               task_mean_ab_success_covariance=covariance(
                   [v["successes"]["A"]/v["starts"] for v in by_task.values()],
                   [v["successes"]["B"]/v["starts"] for v in by_task.values()]))
    for dim in ("angle_deg", "regime"):
        out[dim] = {str(k):stats([r for r in rows if r[dim] == k]) for k in sorted({r[dim] for r in rows})}
    return out


def records(out):
    manifest_raw = (out / "panel.json").read_bytes()
    panel = parse(manifest_raw)
    cpu = panel["cpu_fixture"]
    sites = panel["rows"]
    expected = {out/"pairs"/f"{r['index']:02d}.json.gz" for r in sites}
    require(expected == set((out/"pairs").glob("*")) and len(sites) == len(expected), "incomplete or extra cases")
    results = defaultdict(list)
    for r in sites:
        data = parse(gzip.decompress((out/"pairs"/f"{r['index']:02d}.json.gz").read_bytes()))
        require(data["panel_sha256"] == digest(manifest_raw) and data["index"] == r["index"]
                and set(data["arms"]) == set(r["corrections"]), "pair identity")
        arms = data["arms"]
        first = next(iter(arms.values()))
        for item in arms.values():
            require(item["native"]["reset"] == first["native"]["reset"] and item["body"] == first["body"]
                    and item["env"] == first["env"] and item["chunks"][0]["raw"] == first["chunks"][0]["raw"],
                    "unpaired native opening")
        successes = {k:inspect(r,k,v,cpu) for k,v in arms.items()}
        yaw_err = {k:abs((c["yaw_deg"]-r["true_map"]["yaw_deg"]+180)%360-180)
                   for k,c in r["corrections"].items()}
        results[r["study"]].append(dict(**{k:r[k] for k in ("index","suite","task_id","regime","angle_deg")},
                                        success=successes, yaw_error_deg=yaw_err))
    if cpu:
        return dict(cpu_fixture=True, scored_native_retries=0, pairs=len(sites),
                    checked_controls=sum(10*len(r["corrections"]) for r in sites), pin=panel["pin"])
    return dict(cpu_fixture=False, pin=panel["pin"], release_sha256=panel["release_sha256"],
                studies={k:describe(v) for k,v in results.items()}, paired_rows=dict(results))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    out = p.parse_args().out
    s = records(out)
    atomic(out/("cpu-readback.json" if s["cpu_fixture"] else "native-results.json"), encode(s))
