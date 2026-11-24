"""Independently check cross-archive physical trace overlap and single-pair DIS inputs."""
import argparse
from hashlib import sha256
import itertools
from pathlib import Path

import cv2
import numpy as np

from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps
from source import read_json, require, sha, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    seal = read_json(args.data / "source-seal.json")
    groups = {}
    receipts = {}
    cv2.setNumThreads(1)
    for group, name in (("tuned_fit", "train-tuned"), ("geom_fit", "train-geom"), ("judge_fit", "fit"),
                        ("cal", "reader-cal"), ("eval", "reader-eval")):
        path = args.data / f"{name}.npz"
        require(sha(path) == seal["sha256"][path.name], path)
        with np.load(path, allow_pickle=False) as f:
            before, after, valid = (f[k] for k in ("img_before", "img_after", "valid"))
            commands = f["cmd"] if group == "judge_fit" else f["issued_mean"][:, :, :2]
            require(np.array_equal(before[:, 1:], after[:, :-1]) and valid.all(), group)
            # Hash observed camera endpoints and issued XY in a common representation across archives.
            digests = {sha256(before[i].tobytes() + after[i].tobytes() +
                              np.ascontiguousarray(commands[i]).tobytes()).hexdigest() for i in range(len(valid))}
            groups[group] = digests
            receipts[group] = dict(rows=len(valid), distinct_physical_prefixes=len(digests))
            if group not in ("cal", "eval"):
                continue
            maps = f["dis_flow"]
            count, max_residual = 0, 0.
            # Instantiate a fresh OpenCV estimator for every original held-out pair. It sees neither
            # neighbor, either reader, nor any command. This checks state reuse in archival batch DIS.
            for i, j in itertools.product(range(len(valid)), range(3)):
                independent = compute_dis_flow_maps(before[i, j:j+1], after[i, j:j+1], valid[i, j:j+1])[0]
                max_residual = max(max_residual, float(np.max(np.abs(independent - maps[i, j]))))
                count += 1
            receipts[group].update(single_pair_flow_checks=count, max_flow_difference=max_residual)
            require(max_residual == 0, f"Per-pair and archived DIS differ for {group}")
    overlap = {f"{a}__{b}": len(groups[a] & groups[b]) for a,b in itertools.combinations(groups, 2)}
    require(not any(overlap.values()), "Physical prefix reused across roles")
    write_json(args.output, dict(data_pin=seal["revision"], source_seal_sha256=sha(args.data/"source-seal.json"),
                                 roles=receipts, cross_population_overlaps=overlap))


if __name__ == "__main__":
    main()
