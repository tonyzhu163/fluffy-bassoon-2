"""Paired fresh retries at query start 14, with fixed camera-pose information on support only."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import re
import runpy
import socket
import sys
from types import SimpleNamespace

import numpy as np
from orbit import ARMS, OUTPUTS, checked, corrections, nuisance, panel, require, write_json

QUERY_START = 14
OFFSET = 817117


def query(row):
    tasks = json.loads((OUTPUTS / "libero-meta-learning-20261117-r1/SPLITS.json").read_bytes())["test"]
    suite = row["suite"]
    require(row["task_id"] in tasks[suite], "Unknown released test task")
    seed = 41800000 + 10000 * sorted(tasks).index(suite) + 100 * row["task_id"] + QUERY_START
    return dict(arm="UR5e", suite=suite, task_id=row["task_id"], variant="BASE", init_state_id=QUERY_START, seed=seed)


def shared_scale_grip(row):
    receipt = row["support"]
    with np.load(io.BytesIO(checked(Path(receipt["path"]), receipt["sha256"])), allow_pickle=False) as z:
        np.testing.assert_array_equal(z["issued_mean"], row["issued"])
        np.testing.assert_array_equal(z["delta_eef"], row["delta_eef"])
        np.testing.assert_array_equal(z["chunk_steps"], [5, 5, 5])
        np.testing.assert_array_equal(z["valid"], [True, True, True])
        return nuisance(z["issued_mean"], z["delta_eef"], z["delta_jaw"])


def paired_cell(episode, sim, policy, row, heads, reset_refs, *, fixture=False):
    case = query(row)
    key = (case["suite"], case["task_id"], case["init_state_id"])
    require(tuple(heads) == ARMS, "Missing or unexpected intervention")

    def check(reset):
        require(reset["physics"] != row["physics"] and reset["worker_step_index"] == 0 and
                reset["arm"] == "UR5e" and reset["joint_count"] == 6 and
                reset["noise_seed"] == case["seed"] + OFFSET and bool(reset["prompt"]) and
                (fixture or "nvidia" in reset["renderer_vendor"].lower()), "Invalid initial state")
        require(reset_refs.setdefault(key, reset) == reset, "Unmatched reset, observation, prompt, or noise")

    arms = {}
    opening = None
    for name, (yaw, scale, grip) in heads.items():
        rec = episode(sim, policy, case, SimpleNamespace(name=row["embodiment"]),
                      check_reset=check, yaw_hat=yaw, scale_hat=scale, grip_hat=grip)
        require(rec["reset"] == reset_refs[key] and rec["released_prompt"] == rec["reset"]["prompt"] and
                rec["inferred"] == dict(yaw_deg=yaw, xy_scale=scale, gripper_sign=grip),
                "Wrong query inputs or incomplete reset receipt")
        proposal = rec["opening_proposal_sha256"]
        require(isinstance(proposal, str) and re.fullmatch("[0-9a-f]{64}", proposal)
                and (opening is None or proposal == opening), "Opening proposal mismatch")
        opening = proposal
        n = rec["controls"]
        horizon = rec["reset"]["horizon"]
        require(type(rec["success"]) is bool and type(rec["terminated"]) is bool and type(rec["truncated"]) is bool
                and rec["terminated"] != rec["truncated"] and (not rec["success"] or rec["terminated"])
                and type(n) is int and 0 < n <= horizon and (not rec["truncated"] or n == horizon)
                and len(rec["native_rewards"]) == n and all(map(math.isfinite, rec["native_rewards"]))
                and sum(rec["native_rewards"]) == rec["native_return"], "Incomplete native outcome")
        arms[name] = rec
    return dict(index=row["index"], angle=row["angle"], support_sha256=row["support"]["sha256"],
                support_physics=row["physics"], embodiment=row["embodiment"], query=case, arms=arms)


def summarize(rows):
    expected = {(r["index"], r["angle"]) for r in panel() if r["angle"]}
    require(len(rows) == len(expected) and {(r["index"], r["angle"]) for r in rows} == expected,
            "Incomplete comparison")
    comparisons = {}
    for a, b in (("symmeta_camera", "symmeta"), ("hs_camera", "hs"), ("symmeta_camera", "hs_camera"),
                 ("symmeta_camera", "unadapted"), ("oracle", "symmeta_camera")):
        by_task = {}
        for row in rows:
            key = f"{row['query']['suite']}/{row['query']['task_id']}"
            by_task.setdefault(key, []).append(int(row["arms"][a]["success"]) - int(row["arms"][b]["success"]))
        diffs = np.concatenate(list(by_task.values()))
        effects = {k: float(np.mean(v)) for k, v in by_task.items()}
        draws = [(x + y) / 2 for x in effects.values() for y in effects.values()]
        comparisons[a + "_minus_" + b] = dict(wins=int(np.sum(diffs > 0)), losses=int(np.sum(diffs < 0)),
                                               ties=int(np.sum(diffs == 0)), task_effects=effects,
                                               difference=float(np.mean(diffs)), task_bootstrap_95=[
                                                   float(np.quantile(draws, q, method="inverted_cdf"))
                                                   for q in (.025, .975)])
    return dict(cells=len(rows), episodes=len(rows) * len(ARMS), support_tasks=2,
                successes={k: sum(r["arms"][k]["success"] for r in rows) for k in ARMS}, comparisons=comparisons)


def run(out):
    # Import the verified runtime at its deployment path, not an ambient preinstalled package.
    sys.path.insert(0, str(OUTPUTS / "symmeta-perceiver-retry-20261123-r1"))
    import paired_inputs
    native = paired_inputs.native()
    flow = paired_inputs.module("matched_flow", OUTPUTS / "symmeta-matched-flow-20261123-r1/live.py")
    runtime = flow.qualify_runtime(flow.load_fk())
    probe = json.loads((out / "gpu.json").read_bytes())
    require(probe["status"] == "passed" and probe["hostname"] == socket.gethostname()
            and probe["allocation_type"] == "baige_container" and probe["purpose"] == "evaluation"
            and len(probe["devices"]) == 1, "Wrong or missing one-card distributed evaluation receipt")
    card = probe["devices"][0]["cuda_name"].upper()
    require(card in {"NVIDIA RPBZZZ6", "NVIDIA GEFORCE RTX 4090"} or "RTX PRO 6000" in card, "Wrong GPU")
    egl_device = runpy.run_path(str(OUTPUTS / "reground-rotation-reference-20261123-r1/reference.py"))["egl_device"]
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device(0))
    rows = [r for r in panel() if r["angle"]]
    # Resolve every support before loading weights or consuming an episode.
    heads = [corrections(r, *shared_scale_grip(r)) for r in rows]
    write_json(out / "runtime.json", dict(**runtime, egl_device=os.environ["MUJOCO_EGL_DEVICE_ID"],
                                          cells=len(rows), commit=os.environ["ORBIT_DEPLOYMENT"]))
    policy = native.load_frozen_policy("pi05")
    refs, results = {}, []
    with native.UR5eSimulator() as sim:
        for row, predictions in zip(rows, heads, strict=True):
            # Warm the renderer on first create for this released task before sealing a common reset.
            if (row["suite"], row["task_id"], QUERY_START) not in refs:
                sim.call("create", dict(query(row), embodiment=row["embodiment"]))
            rec = paired_cell(native.run_episode, sim, policy, row, predictions, refs)
            native.write_json(out / f"cell-{row['index']:02d}-{row['angle']:+d}.json", rec)
            results.append(rec)
            print(f"Verified {len(results)}/{len(rows)} six-arm cells", flush=True)
    native.write_json(out / "summary.json", summarize(results))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
