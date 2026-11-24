"""Collect, fit both prefix arms through frozen pi0.5 and score the held-out replay. One GPU job step."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "frozen-vla-proposal-shift-20261124-r1")]

import harness  # noqa: F401  puts the motor directory on the path
import ur5e_motor
import collect
import fit as fitting
import prefix as px


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--pin", required=True)
    p.add_argument("--updates", type=int, default=fitting.UPDATES)
    args = p.parse_args()
    from ur5e_sim_worker import UR5eSimulator
    import jax
    import openpi_engine

    policy = ur5e_motor.load_frozen_pi05()
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        sim = UR5eSimulator()
    with sim:
        sets, receipts = collect.collect(sim, args.out)
    for split in sets.values():
        for row in split:
            row["item"] = ur5e_motor.to_openpi_observation(row.pop("public"), harness.TASK)
    if not sets["fit"]:
        raise RuntimeError("No donor replay earned native reward on this host; nothing to fit")
    engine = openpi_engine.Engine(policy, fitting.SEED)
    probe = sets["fit"][0]
    noise = np.random.default_rng(fitting.SEED).standard_normal(
        (policy._model.action_horizon, policy._model.action_dim)).astype(np.float32)
    before = np.asarray(policy.infer(probe["item"], noise=noise)["actions"])
    report = dict(pin=args.pin, collection=receipts, fit_rows=len(sets["fit"]),
                  heldout_rows=len(sets["heldout"]), arms={})
    fitted = {}
    for arm in px.ARMS:
        tic = time.monotonic()
        params, receipt = fitting.fit(engine, sets["fit"], arm, updates=args.updates,
                                      log=lambda row, arm=arm: print(arm, json.dumps(row), flush=True))
        receipt.update(pin=args.pin, seconds=time.monotonic() - tic,
                       fit_episodes=sorted({row["episode"] for row in sets["fit"]}),
                       device_peak_bytes=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use"))
        report["arms"][arm] = px.save(args.out / f"prefix-{arm}.npz", params, receipt)
        fitted[f"prefix-{arm}"] = params
    fitted["prefix-random"] = px.random_like(fitted["prefix-shift"], fitting.SEED)
    if sets["heldout"]:
        report["heldout"] = fitting.heldout(engine, sets["heldout"], fitted)
    # Through the installed wrapper: an untagged call, then a tagged call with an all-zero shift.
    import adapter as lane
    lane.REGISTRY["zero-shift"] = {"shift": np.zeros_like(fitted["prefix-shift"]["shift"])}
    lane.install()
    after = np.asarray(policy.infer(probe["item"], noise=noise)["actions"])
    zero = np.asarray(policy.infer(dict(probe["item"], prompt=harness.TASK + lane.tag("zero-shift")),
                                   noise=noise)["actions"])
    report["untreated_probe_identical_after_fit"] = bool(np.array_equal(before, after))
    report["zero_shift_max_abs_difference_from_stock"] = float(np.max(np.abs(zero - before)))
    ur5e_motor.write_json(args.out / "fit.json", report)
    if not report["untreated_probe_identical_after_fit"]:
        raise RuntimeError("The untreated policy output changed during fitting")


if __name__ == "__main__":
    main()
