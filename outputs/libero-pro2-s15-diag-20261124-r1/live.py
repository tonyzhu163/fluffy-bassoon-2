"""Closed-loop frozen pi0.5 evaluation for Track 2: S15 causal ablation & S16 rectification."""
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
    to_openpi_observation,
    write_json,
)
from ur5e_sim_worker import SIM_PYTHON

HERE = Path(__file__).resolve().parent
TASKS = [
    ("libero_spatial", 0),
    ("libero_object", 1),
    ("libero_goal", 3),
    ("libero_10", 5),
]
LAYOUTS = (13, 14)
CONDITIONS = (
    "BASE",
    "S15_full",
    "S15_ext_only",
    "S15_wrist_only",
    "S15_wrist_zero",
    "S16_raw",
    "S16_oracle_rect",
    "S16_selfcal_rect",
)


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
        tasks = TASKS[:1] if fixture else TASKS
        layouts = LAYOUTS[:1] if fixture else LAYOUTS
        for idx, (suite, tid) in enumerate(tasks):
            for layout in layouts:
                seed = 202611 + 100 * idx + layout
                case = dict(suite=suite, task_id=tid, layout=layout, seed=seed)
                meta = call("case", case)
                noise = np.random.default_rng(seed + NOISE_SEED_OFFSET).standard_normal(
                    ((meta["horizon"] + 4) // 5, 10, 32)
                ).astype(np.float32)
                for cond in CONDITIONS:
                    first = call("start", cond)
                    raw = first.pop("obs")
                    selfcal = first.get("selfcal")
                    n = 0
                    rewards = []
                    while n < meta["horizon"]:
                        item = to_openpi_observation(raw, meta["prompt"], canonicalize_quat=False)
                        if policy is not None:
                            proposal = np.asarray(policy.infer(item, noise=noise[n // 5])["actions"], np.float32)
                        else:
                            proposal = np.zeros((10, 7), dtype=np.float32)
                        commands = np.clip(proposal[:5], -1.0, 1.0)
                        step = call("chunk", commands)
                        raw = step.pop("obs")
                        k = len(step["rewards"])
                        rewards.extend(step["rewards"])
                        n += k
                        if step["success"] or step["truncated"] or (fixture and n >= 10):
                            break
                    row = dict(
                        **case,
                        condition=cond,
                        success=bool(step["success"]),
                        controls=n,
                        selfcal=selfcal,
                        noise_sha256=array_digest(noise),
                    )
                    results.append(row)
                    print(
                        f"{suite}/{tid} L{layout} {cond:16s} succ={int(row['success'])} ctrl={n:3d} selfcal={selfcal}",
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
    for cond in CONDITIONS:
        sub = [r for r in results if r["condition"] == cond]
        summary[cond] = dict(
            successes=sum(r["success"] for r in sub),
            total=len(sub),
            mean_controls=float(np.mean([r["controls"] for r in sub])) if sub else 0.0,
        )
    write_json(out / "live-summary.json", dict(fixture=fixture, rows=results, summary=summary))
    print("\n=== TRACK 2 LIVE SUMMARY ===", flush=True)
    for cond in CONDITIONS:
        s = summary[cond]
        print(f"{cond:18s}: {s['successes']}/{s['total']} (mean_ctrl={s['mean_controls']:.1f})", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--fixture", action="store_true")
    a = p.parse_args()
    main(a.out, a.fixture)
