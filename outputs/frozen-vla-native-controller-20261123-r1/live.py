"""Run the same frozen JAX policy in closed loop on native OSC and joint controllers."""
import argparse
import json
import os
import subprocess
from multiprocessing import Pipe
from pathlib import Path
import sys

import numpy as np

from ur5e_motor import NOISE_SEED_OFFSET, array_digest, load_frozen_pi05, public_digest, to_openpi_observation, write_json
from ur5e_sim_worker import SIM_PYTHON

HERE = Path(__file__).resolve().parent
METHODS = ("OSC_POSE", "hold", "ik", "calibrated")
SEEDS = {"libero_spatial": 171117, "libero_goal": 171118}


def main(out, fixture):
    policy = None if fixture else load_frozen_pi05()
    pipe, peer = Pipe()
    args = [str(SIM_PYTHON), "-Bu", str(HERE/"sim_server.py"), str(peer.fileno())]
    if fixture:
        args.append("--fixture")
    worker = subprocess.Popen(args, pass_fds=(peer.fileno(),), env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    peer.close()

    def call(op, value=None):
        pipe.send((op, value))
        ok, result = pipe.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    results = []
    try:
        for layout in ((0,) if fixture else (0, 2, 3)):
            for suite, base_seed in SEEDS.items():
                if fixture and suite != "libero_goal":
                    continue
                seed = base_seed + 100 * layout
                case = dict(suite=suite, layout=layout, seed=seed)
                meta = call("case", case)
                noise = np.random.default_rng(seed + NOISE_SEED_OFFSET).standard_normal(
                    ((meta["horizon"]+4)//5, 10, 32)).astype(np.float32)
                tape = None
                if fixture:
                    data = Path("/mowenjie/fluffy-bassoon-2/outputs/libero-meta-learning-20261117-r1/devref01/reference")
                    tape = np.load(data/(suite+"-actions.npy"))
                opening = None
                for method in METHODS:
                    first = call("start", method)
                    raw = first.pop("obs")
                    start = public_digest(raw)
                    record = dict(case=case, method=method, meta=meta, schema={k:np.asarray(v).tolist()
                                  for k,v in first.items()}, public=start, noise_sha256=array_digest(noise),
                                  rewards=[], canonical=[], native=[], fixture=fixture)
                    n = 0
                    while n < meta["horizon"]:
                        item = to_openpi_observation(raw, meta["prompt"], canonicalize_quat=False)
                        assert item["observation/state"].shape == (8,)
                        assert item["observation/image"].shape == item["observation/wrist_image"].shape == (224,224,3)
                        proposal = (np.asarray(policy.infer(item, noise=noise[n//5])["actions"], np.float32)
                                    if policy is not None else tape[n:n+10].astype(np.float32))
                        assert proposal.shape == (10,7) and np.isfinite(proposal).all()
                        if n == 0:
                            key = dict(public=start, opening=array_digest(proposal), noise=record["noise_sha256"],
                                       policy_inputs={k:array_digest(v) for k,v in item.items() if k != "prompt"})
                            if opening is not None and key != opening:
                                raise ValueError("Unmatched policy opening across native controllers")
                            opening = key
                            record.update(opening=key["opening"], policy_inputs=key["policy_inputs"])
                        commands = np.clip(proposal[:5], -1, 1)
                        step = call("chunk", commands)
                        raw = step.pop("obs")
                        k = len(step["rewards"])
                        record["rewards"].extend(step["rewards"])
                        record["native"].extend(np.asarray(step["native"]).tolist())
                        record["canonical"].extend(commands[:k].tolist())
                        n += k
                        if step["success"] or step["truncated"] or (fixture and n >= 10):
                            break
                    record.update(success=step["success"], truncated=bool(step["truncated"]), controls=n)
                    assert bool(sum(record["rewards"])) == record["success"]
                    assert fixture or record["success"] or record["truncated"]
                    if method == METHODS[0]:
                        source_success = record["success"]
                    record["matched_source_success"] = source_success
                    write_json(out/f"{suite}-{layout}-{method}.json", record)
                    results.append(dict(**case, method=method, success=record["success"],
                                        source_success=source_success, controls=n))
                    print(suite, layout, method, n, record["success"], flush=True)
        call("close")
        if worker.wait(timeout=30):
            raise RuntimeError("Simulator worker failed")
    finally:
        pipe.close()
        if worker.poll() is None:
            worker.terminate()
            worker.wait()
    summary = {m: dict(all_success=sum(r["success"] for r in results if r["method"]==m),
                       cases=sum(r["method"]==m for r in results),
                       eligible_success=sum(r["success"] for r in results if r["method"]==m and r["source_success"]),
                       eligible=sum(r["source_success"] for r in results if r["method"]==m))
               for m in METHODS}
    write_json(out/"live-summary.json", dict(fixture=fixture, rows=results, counts=summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    main(args.out, args.fixture)
