"""Independent CPU replay of every selected overlap, sent command and paired native endpoint."""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from contract import ARMS, IDENTITY, MANIFEST_SHA, MODES, RELEASE_SHA, SOURCE_SHA, TASKS, WEIGHT_SHA
from contract import cases, digest, require, save, source_law


def fp(array):
    a = np.ascontiguousarray(array)
    return digest(repr((a.shape, a.dtype.str)).encode() + a.tobytes())


def rotation(deg):
    angle = math.radians(deg)
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], np.float32)


def native_commands(plan, c):
    cmds = np.clip(plan, -1., 1.).copy()
    inverse = np.float32(1.0 / float(np.clip(c["xy_scale"], .25, 4.)))
    cmds[:, :2] = (cmds[:, :2] @ rotation(-c["yaw_deg"]).T) * inverse
    cmds[:, 2] *= inverse
    cmds[:, 6] *= c["gripper_sign"]
    return np.clip(cmds, -1., 1.)


def source_xyz(cmds, c):
    xyz = np.asarray(cmds, np.float32)[:, :3].copy() * np.float32(c["xy_scale"])
    xyz[:, :2] = xyz[:, :2] @ rotation(c["yaw_deg"]).T
    return np.clip(xyz, -1., 1.).astype(float)


def reference_index(tape, k, c, law):
    prev = tape[k-1]
    delta = np.subtract(prev["tcp"][-1], prev["before"])
    measured = delta.copy()
    measured[:2] = delta[:2] @ rotation(c["yaw_deg"]).T
    previous_cmd = np.asarray(prev["original"], np.float32)[:5]
    sent = native_commands(previous_cmd, c)
    xyz = source_xyz(sent, c)
    carry = source_xyz(tape[k-2]["sent"], c).mean(axis=0) if k >= 2 else np.zeros(3)
    points = [np.zeros(3)]
    for step in range(5):
        points.append((xyz[:step+1].sum(axis=0) @ law[:3] + (step+1)*(carry @ law[3:])) / 5)
    errors = np.linalg.norm(points-measured, axis=1)
    j = min(range(6), key=lambda i: (errors[i], -i))
    info = tape[k]["alignment"]
    for actual, expected in [(info["delta_local_m"], delta), (info["delta_source_m"], measured),
                             (info["donor_issued_xyz"], sent[:, :3]), (info["prefix_m"], points),
                             (info["distances_m"], errors)]:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    return j


def mixed(raw, prev, j):
    result = raw.copy()
    donor = np.clip(np.asarray(prev["original"], dtype=np.float32)[j:j+5, :3], -1, 1)
    result[:5, :3] = .5 * np.clip(raw[:5, :3], -1, 1) + .5 * donor
    return result


def verify(obj, row, mode, *, cpu=False):
    q, support = row["query"], row["support"]
    rec, reset, full = obj["native"], obj["native"]["reset"], obj["environment"]
    model = obj["model"]
    require(obj["case"] == row and obj["mode"] == mode and obj["fixture"] is cpu and
            obj["source_sha256"] == SOURCE_SHA and obj["weights_sha256"] == (None if cpu else WEIGHT_SHA),
            "case or source identity")
    # Both hashed panel archives identify the object arena as the floor-mounted variant.
    expected_class = ("OnTheGround" if q["suite"] == "libero_object" else "Mounted") + q["arm"]
    require(model["arm"] == reset["arm"] == q["arm"] and model["model_class"] == expected_class and
            model["model_class"] == reset["model_class"] and model["joint_count"] == len(model["joint_names"]) == 7
            and model["gripper"] == "PandaGripper" and model["controller"] == "OSC_POSE" and model["control_freq"] == 20
            and model["tcp_site"] == "gripper0_grip_site", "compiled robot or control contract")
    require(all(full[k] == q[k] for k in ("suite", "task_id", "seed", "init_state_id")) and
            full["perturbation_plan"]["case_id"] == q["variant"] and full["source_pin"] == RELEASE_SHA and
            full["manifest_sha256"] == MANIFEST_SHA and reset["physics"] == full["initial_state_fingerprint"] and
            reset["physics"] != support["physics"] and reset["scene_fingerprint"] == full["scene_fingerprint"] and
            reset["worker_step_index"] == 0 and reset["prompt"] == rec["released_prompt"] and
            reset["action_schema"]["control_mode"] == "relative" and reset["action_schema"]["shape"] == [7],
            "native reset or hidden shift")
    require(("llvmpipe" in reset["renderer"].lower()) if cpu else reset["renderer_vendor"] == "NVIDIA Corporation",
            "unqualified renderer")
    h, width = obj["noise_shape"]
    n = rec["controls"]
    noise_seed = q["seed"] + 817117
    require(h >= 10 and width > 0 and reset["noise_seed"] == noise_seed and 0 < n <= reset["horizon"],
            "noise schedule or horizon")
    noise = np.random.default_rng(noise_seed).standard_normal(((reset["horizon"]+4)//5, h, width)).astype("f4")
    require(fp(noise) == reset["noise_sha256"], "incorrect full noise bank")
    rewards = rec["native_rewards"]
    require(len(rewards) == n and all(r in (0., 1.) for r in rewards) and rec["native_return"] == sum(rewards) and
            type(rec["success"]) is bool and rec["success"] == bool(max(rewards)), "native predicate / reward")
    if not cpu:
        require(rec["terminated"] != rec["truncated"] and (not rec["truncated"] or n == reset["horizon"]) and
                (not rec["success"] or rec["terminated"]), "incomplete native episode")
    c = IDENTITY if mode == "raw" else support["tuned"]
    require(rec["inferred"] == c, "wrong correction")
    tape = obj["chunks"]
    require(len(tape) == (n+4)//5, "inference cadence")
    law = source_law()
    counts = Counter()
    histogram = Counter()
    for k, chunk in enumerate(tape):
        raw = np.asarray(chunk["original"], np.float32)
        actual = np.asarray(chunk["output"], np.float32)
        count = min(5, n-5*k)
        require(raw.shape == actual.shape == (h, 7) and np.isfinite(raw).all() and
                chunk["noise_sha256"] == fp(noise[k]) and chunk["prompt"] == reset["prompt"],
                "original proposal, prompt or per-query noise changed")
        require(np.shape(chunk["before"]) == (3,) and np.shape(chunk["tcp"]) == (count, 3) and
                np.isfinite(chunk["tcp"]).all() and np.isfinite(chunk["before"]).all() and
                (not k or chunk["before"] == tape[k-1]["tcp"][-1]), "broken causal TCP trace")
        if not k:
            require(rec["opening_proposal_sha256"] == fp(raw), "modified opening")
        j = None
        if k and mode == "progress":
            j = reference_index(tape, k, c, law)
        else:
            require(chunk["alignment"] is None, "unexpected observation-conditioned phase")
            if k and mode == "fixed":
                j = 5
        require(chunk["j"] == j, "wrong donor phase")
        expected = raw if j is None else mixed(raw, tape[k-1], j)
        require(np.array_equal(actual, expected), "full plan differs including wrist, gripper or suffix")
        sent = np.asarray(chunk["sent"], np.float32)
        require(np.array_equal(sent, native_commands(expected[:count], c)), "issued command mismatch")
        if j is not None:
            histogram[j] += 1
            counts["eligible_boundaries"] += 1
            counts["phase_departures"] += int(j != 5)
            baseline = native_commands(raw[:count], c)
            counts["changed_sent_vs_tuned_controls"] += int(np.any(sent != baseline, axis=1).sum())
            counts["changed_canonical_vs_tuned_controls"] += int(np.any(
                np.clip(actual[:count, :3], -1, 1) != np.clip(raw[:count, :3], -1, 1), axis=1).sum())
            fixed = native_commands(mixed(raw, tape[k-1], 5)[:count], c)
            counts["changed_sent_vs_same_history_fixed_controls"] += int(np.any(sent != fixed, axis=1).sum())
    return dict(counts=dict(counts), phase_histogram=dict(histogram))


def pairing(rows):
    for a, b in combinations(rows, 2):
        x, y = rows[a], rows[b]
        require(x["environment"] == y["environment"] and x["model"] == y["model"] and
                x["native"]["reset"] == y["native"]["reset"] and x["case"] == y["case"] and
                x["noise_shape"] == y["noise_shape"], "unpaired physical reset, weights, public observation or noise bank")
        left, right = x["chunks"][0], y["chunks"][0]
        same = left["policy_input"] == right["policy_input"] and left["prompt"] == right["prompt"]
        if a != "raw" and b != "raw":
            require(same and left["sent"] == right["sent"], "tuned opening conditions differ")
        if same:
            require(left["original"] == right["original"], "unequal original proposal at identical inputs and noise")


def interval(task_differences, denominator):
    dist = Counter({0: 1})
    for _ in task_differences:
        new = Counter()
        for v, freq in dist.items():
            for d in task_differences:
                new[v+d] += freq
        dist = new
    cuts = []
    count = 0
    for v, f in sorted(dist.items()):
        count += f
        while len(cuts) < 2 and 40*count >= (1, 39)[len(cuts)]*sum(dist.values()):
            cuts.append(100*v/denominator)
    return cuts


def endpoints(groups):
    denom = len(groups)*len(TASKS)
    out = {"cases": denom, "successes": {m: sum(r[m] for g in groups for r in g) for m in MODES},
           "contrasts": {}}
    for a, b in (("progress","raw"), ("progress","tuned"), ("progress","fixed"), ("fixed","tuned")):
        diffs = [int(r[a])-int(r[b]) for g in groups for r in g]
        by_task = [sum(int(g[i][a])-int(g[i][b]) for g in groups) for i in range(8)]
        out["contrasts"][a+"-minus-"+b] = dict(net_pp=100*sum(diffs)/denom, wins=diffs.count(1),
                losses=diffs.count(-1), ties=diffs.count(0), rescue_denominator=denom-out["successes"][b],
                harm_denominator=out["successes"][b], task_bootstrap_ci95_pp=interval(by_task, denom))
    return out


def analyze(folder, arm, cpu=False):
    key = json.loads((folder/"origin.json").read_bytes())
    require(key["arm"] == arm and key["fixture"] is cpu, "invalid shard origin")
    expected = set()
    results, details = [], []
    for row in cases(arm):
        q = row["query"]
        quartet, diagnostics = {}, {}
        for mode in MODES:
            name = f"{q['suite']}-{q['task_id']}-{mode}.json"
            expected.add(name)
            obj = json.loads((folder/"episodes"/name).read_bytes())
            require(obj["origin"] == key, "mixed runtime")
            diagnostics[mode] = verify(obj, row, mode, cpu=cpu)
            quartet[mode] = obj
        pairing(quartet)
        results.append({m: x["native"]["success"] for m, x in quartet.items()})
        raw = quartet["raw"]
        shared = {k: raw["environment"][k] for k in ("released_row_fingerprint", "bddl_sha256")}
        shared.update({k: raw["native"]["reset"][k] for k in ("noise_seed", "noise_sha256", "horizon", "prompt")})
        details.append(dict(query=q, matched_start=shared, diagnostics=diagnostics, success=results[-1]))
    require({p.name for p in (folder/"episodes").glob("*.json")} == expected, "incomplete or extra endpoints")
    return results, details


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iiwa", type=Path)
    p.add_argument("--kinova3", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    shards = {a: getattr(args, a.lower()) for a in ARMS if getattr(args, a.lower())}
    require(shards, "supply at least one shard")
    answer, groups, origins = {}, [], []
    for arm, folder in shards.items():
        group, details = analyze(folder, arm)
        groups.append(group)
        origins.append(json.loads((folder/"origin.json").read_bytes()))
        totals = {}
        for mode in MODES:
            counts, histogram = Counter(), Counter()
            for task in details:
                counts.update(task["diagnostics"][mode]["counts"])
                histogram.update(task["diagnostics"][mode]["phase_histogram"])
            totals[mode] = dict(counts=dict(counts), phase_histogram=dict(histogram))
        answer[arm] = dict(native=endpoints([group]), intervention_totals=totals, tasks=details)
    if len(groups) == 2:
        require(origins[0]["pin"] == origins[1]["pin"],
                "cross-arm pooling requires the same pinned evaluator")
        require([r["matched_start"] for r in answer["IIWA"]["tasks"]] ==
                [r["matched_start"] for r in answer["Kinova3"]["tasks"]],
                "across-arm released row, native task or common noise mismatch")
        answer["pooled_equal_arm"] = endpoints(groups)
    save(args.out, answer)
