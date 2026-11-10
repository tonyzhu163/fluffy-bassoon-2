"""Audit both physically executed actions and score the predeclared balanced hidden goals."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def label_chunks(data, horizon=50):
    """Return the actual controls following each query boundary and a delivered-control mask."""
    commands, offsets = data["commands"], data["offsets"]
    assert commands.ndim == 2 and commands.shape[1] == 12 and np.isfinite(commands).all()
    assert int(offsets[-1]) == len(commands) and np.array_equal(offsets[:-1], np.arange(0,len(commands),5))
    targets = np.zeros((len(offsets)-1, horizon, 12), np.float32)
    mask = np.zeros(targets.shape[:2], bool)
    for i, offset in enumerate(offsets[:-1]):
        executed = commands[offset:offset+horizon]
        targets[i,:len(executed)] = executed
        mask[i,:len(executed)] = True
    return targets, mask


def audit(directory, admission):
    seeds = read(admission)["admitted"]
    rows = []
    samples = 0
    for seed in seeds:
        scene = directory/str(seed)
        pickup = read(scene/"pickup.json")
        views, results = [], []
        with np.load(scene/"pickup.npz") as tape:
            assert tape["commands"].shape == (pickup["controls"],12)
            assert tape["post"].shape[0] == pickup["controls"]
        for side in ("left","right"):
            record = read(scene/f"{side}.json")
            assert record["seed"] == seed and record["side"] == side
            assert record["decision"] == pickup["decision"] and record["prefix"] == pickup["controls"]
            assert record["branch_controls"] + record["prefix"] <= 500
            assert len(record["placed"]) == 2 and sum(record["placed"]) <= 1
            with np.load(scene/f"{side}.npz") as data:
                actions, mask = label_chunks(data)
                frames, states = data["frames"], data["states"]
                assert frames.dtype == np.uint8 and frames.shape[1:] == (3,256,256,3)
                assert len(frames) == len(states) == len(actions)+1
                assert frames.shape[0] == len(data["public"]) == len(data["offsets"])
                assert len(data["commands"]) == record["branch_controls"]
                views.append((frames[0].copy(), states[0].copy()))
                chosen = ("left","right").index(side)
                if record["placed"][chosen]:
                    assert len(mask) and mask.any(1).all()
                    samples += len(actions)
            results.append(record)
        for a,b in zip(*views):
            np.testing.assert_array_equal(a,b)
        rows.append({"seed":seed, "pickup":pickup["controls"], "grasped":pickup["fruit"] is not None,
                     "outcomes":[r["placed"] for r in results], "branch_controls":[r["branch_controls"] for r in results],
                     "phases":[r["phases"] for r in results]})
    outcomes = np.array([r["outcomes"] for r in rows],dtype=bool)
    target_hits = outcomes[:,[0,1],[0,1]]
    both = int(target_hits.all(1).sum())
    oracle = int(outcomes.any(1).sum())
    fixed = outcomes.sum((0,2)).tolist()
    return {"scenes":rows, "native_grasps":sum(r["grasped"] for r in rows), "two_sided":both,
            "target_specific_labels":target_hits.sum(0).tolist(), "verified_action_chunks":samples,
            "balanced_goal_cases":2*len(rows), "per_goal_oracle":oracle, "fixed_left_right":fixed,
            "qualifies":both >= 5 and oracle > max(fixed)}


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("directory",type=Path)
    ap.add_argument("admission",type=Path)
    args=ap.parse_args()
    print(json.dumps(audit(args.directory,args.admission),indent=2))
