"""Merge score shards, verify common source/native anchors, and pair by task."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("no_op", "public_ik", "displaced", "oracle_q", "oracle_q_visibility", "matched_source")
CONTRASTS = (("no_op", "public_ik"), ("displaced", "public_ik"), ("no_op", "matched_source"),
             ("public_ik", "oracle_q"), ("oracle_q", "oracle_q_visibility"),
             ("oracle_q_visibility", "matched_source"), ("public_ik", "matched_source"))


def interval(x, draws):
    low, high = np.quantile(x[draws].mean(axis=1), [0.025, 0.975])
    return dict(mean=float(x.mean()), lo=float(low), hi=float(high), positive=int((x > 0).sum()))


def summarize(work, out):
    root = work / "followup"
    with (Path(__file__).resolve().parent.parent / "appearance/motor.json").open() as f:
        prior = json.load(f)["tasks"]
    rows = []
    for shard in ("scores-0-5.csv", "scores-5-10.csv"):
        with (root / shard).open() as f:
            rows.extend(csv.DictReader(f))
    tasks = [r["task"] for r in prior]
    table = {(r["task"], r["condition"]): r for r in rows}
    expected = {(t, c) for t in tasks for c in ("source_reference", *CONDITIONS)}
    assert len(rows) == len(table) == 70 and expected == table.keys()
    direct = {c: [] for c in CONTRASTS[-4:]}
    for i, t in enumerate(tasks):
        with np.load(root / "actions" / f"{i:02d}.npz") as actions:
            assert hashlib.sha256(actions["source_reference"].tobytes()).hexdigest() == prior[i]["source_chunk_sha256"]
            for a, b in direct:
                diff = (actions[a][:, :3] - actions[b][:, :3]).astype(float)
                direct[a, b].append(np.sqrt(np.mean(diff ** 2)))
    with (Path(__file__).resolve().parent.parent / "appearance/scores.csv").open() as f:
        for r in csv.DictReader(f):
            if r["contrast"] == "uncropped_native":
                for k in ("first_xyz_rms", "xyz_rms", "rotation_rms", "gripper_rms", "action_rms", "max_abs"):
                    assert r[k] == table[r["task"], "no_op"][k]
    with (out / "scores.csv").open("w") as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    draws = np.random.default_rng(20261123).integers(10, size=(20000, 10))
    summary = {}
    for m in ("xyz_rms", "first_xyz_rms", "rotation_rms", "gripper_rms", "action_rms"):
        values = {c: np.array([float(table[t, c][m]) for t in tasks]) for c in CONDITIONS}
        summary[m] = {c: interval(v, draws) for c, v in values.items()}
        for a, b in CONTRASTS:
            summary[m][f"{a}_minus_{b}"] = interval(values[a] - values[b], draws)
    summary["direct_xyz_rms"] = {f"{a}_vs_{b}": interval(np.asarray(v), draws) for (a, b), v in direct.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.work, args.output)
