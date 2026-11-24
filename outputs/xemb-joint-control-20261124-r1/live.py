"""Closed-loop frozen pi0.5 evaluation for Track 1: Kinova3 & UR5e under native JOINT_POSITION."""
import argparse
import os
import subprocess
from multiprocessing import Pipe
from pathlib import Path

import numpy as np

from ur5e_motor import (
    NOISE_SEED_OFFSET,
    array_digest,
    load_frozen_pi05,
    public_digest,
    to_openpi_observation,
    write_json,
)
from ur5e_sim_worker import SIM_PYTHON

HERE = Path(__file__).resolve().parent
ARMS = ("Kinova3", "UR5e")
METHODS = ("OSC_POSE", "ik_naive", "ik_pose_track")
SEEDS = {"libero_spatial": 171117, "libero_goal": 171118}
LAYOUTS = (0, 2, 3)


def main(out: Path, fixture: bool):
    out.mkdir(parents=True, exist_ok=True)
    policy = None if fixture else load_frozen_pi05()
    pipe, peer = Pipe()
    args = [str(SIM_PYTHON), "-Bu", str(HERE / "sim_server.py"), str(peer.fileno())]
    if fixture:
        args.append("--fixture")
    worker = subprocess.Popen(
        args,
        pass_fds=(peer.fileno(),),
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    )
    peer.close()

    def call(op, value=None):
        pipe.send((op, value))
        ok, result = pipe.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    results = []
    try:
        arms = ("Kinova3",) if fixture else ARMS
        layouts = (0,) if fixture else LAYOUTS
        for arm in arms:
            for layout in layouts:
                for suite, base_seed in SEEDS.items():
                    if fixture and suite != "libero_goal":
                        continue
                    seed = base_seed + 100 * layout
                    case = dict(arm=arm, suite=suite, layout=layout, seed=seed)
                    meta = call("case", case)
                    noise = np.random.default_rng(seed + NOISE_SEED_OFFSET).standard_normal(
                        ((meta["horizon"] + 4) // 5, 10, 32)
                    ).astype(np.float32)
                    tape = None
                    if fixture:
                        data = Path("/mowenjie/fluffy-bassoon-2/outputs/libero-meta-learning-20261117-r1/devref01/reference")
                        tape = np.load(data / (suite + "-actions.npy"))
                    for method in METHODS:
                        first = call("start", method)
                        raw = first.pop("obs")
                        start_dig = public_digest(raw)
                        record = dict(
                            case=case,
                            method=method,
                            meta=meta,
                            public=start_dig,
                            noise_sha256=array_digest(noise),
                            rewards=[],
                            canonical=[],
                            native=[],
                            fixture=fixture,
                        )
                        n = 0
                        while n < meta["horizon"]:
                            item = to_openpi_observation(raw, meta["prompt"], canonicalize_quat=False)
                            proposal = (
                                np.asarray(policy.infer(item, noise=noise[n // 5])["actions"], np.float32)
                                if policy is not None
                                else tape[n : n + 10].astype(np.float32)
                            )
                            commands = np.clip(proposal[:5], -1.0, 1.0)
                            step = call("chunk", commands)
                            raw = step.pop("obs")
                            k = len(step["rewards"])
                            record["rewards"].extend(step["rewards"])
                            record["native"].extend(step["native"])
                            record["canonical"].extend(commands[:k].tolist())
                            n += k
                            if step["success"] or step["truncated"] or (fixture and n >= 10):
                                break
                        sats = 0
                        if method != "OSC_POSE" and record["native"]:
                            arr = np.asarray(record["native"])[:, :-1]
                            sats = int(np.sum(np.any(np.abs(arr) >= 0.999, axis=1)))
                        record.update(
                            success=bool(step["success"]),
                            truncated=bool(step["truncated"]),
                            controls=n,
                            sat_frac=sats / max(1, n),
                        )
                        write_json(out / f"{arm}-{suite}-{layout}-{method}.json", record)
                        results.append(
                            dict(
                                **case,
                                method=method,
                                success=record["success"],
                                controls=n,
                                sat_frac=record["sat_frac"],
                            )
                        )
                        print(
                            f"{arm:7s} {suite}/{layout} {method:13s} succ={int(record['success'])} "
                            f"ctrl={n:3d} sat={record['sat_frac']:.2f}",
                            flush=True,
                        )
        call("close")
        if worker.wait(timeout=30):
            raise RuntimeError("Simulator worker failed")
    finally:
        pipe.close()
        if worker.poll() is None:
            worker.terminate()
            worker.wait()

    summary = {}
    for arm in arms:
        summary[arm] = {}
        for m in METHODS:
            sub = [r for r in results if r["arm"] == arm and r["method"] == m]
            summary[arm][m] = dict(
                successes=sum(r["success"] for r in sub),
                total=len(sub),
                mean_controls=float(np.mean([r["controls"] for r in sub])) if sub else 0.0,
                mean_sat=float(np.mean([r["sat_frac"] for r in sub])) if sub else 0.0,
            )
    write_json(out / "live-summary.json", dict(fixture=fixture, rows=results, summary=summary))
    print("\n=== TRACK 1 LIVE SUMMARY ===", flush=True)
    for arm in arms:
        print(arm, summary[arm], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--fixture", action="store_true")
    a = p.parse_args()
    main(a.out, a.fixture)
