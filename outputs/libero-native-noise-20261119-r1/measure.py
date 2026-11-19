"""Measure same-reset native query trajectories and within-query opening proposals."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


TASKS = (0, 2, 3, 7, 10, 11, 13, 15, 17, 22, 23, 27, 29, 33, 34)
REPEATS = range(20)
CONTROLS = (1, 5, 25, 50, 100)
HORIZON = 400


@dataclass
class Query:
    initial: np.ndarray
    state: np.ndarray
    embedding: np.ndarray
    proposals: np.ndarray
    positions: np.ndarray
    success: bool


def read_query(folder, published, seed, stream):
    record = json.loads((folder / "result.json").read_text())
    if record != published or record["arm"] != folder.name or record["policy_stream"] != [seed, stream]:
        raise ValueError(f"{folder}: episode record disagrees with published trial")
    with np.load(folder / "public.npz", allow_pickle=False) as tape:
        initial, positions, issued = tape["initial_position"], tape["positions"], tape["issued"]
        rewards, n, terminal = tape["rewards"], tape["controls"].item(), tape["terminal"].item()
    if not (type(n) is int and 1 <= n <= HORIZON and n == record["controls"]
            and positions.shape == (n, 3) and issued.shape == (n, 7) and rewards.shape == (n,)
            and initial.shape == (3,) and all(np.isfinite(a).all() for a in (initial, positions, rewards))):
        raise ValueError(f"{folder}: invalid control tape")
    success = record["success"]
    if not (type(success) is bool and terminal is success and record["truncated"] is (not success)
            and (success or n == HORIZON) and record["score"] == rewards.sum()
            and np.flatnonzero(rewards > 0).tolist() == ([n - 1] if success else [])):
        raise ValueError(f"{folder}: native outcome disagrees with reward or termination tape")

    with np.load(folder / "decisions.npz", allow_pickle=False) as decisions:
        state, embedding, proposals = (decisions[k][0] for k in ("state", "embedding", "proposals"))
        slot, offset, length = (decisions[k][0].item() for k in ("index", "offset", "length"))
        choices = decisions["index"]
    coin = np.random.default_rng(np.random.SeedSequence([seed, stream, 2]))
    if not np.array_equal(choices, coin.integers(2, size=record["replans"])):
        raise ValueError(f"{folder}: archived choices disagree with the collector's coin stream")
    if not (state.shape == (8,) and np.array_equal(state[:3], initial.astype(np.float32))
            and np.isfinite(state).all() and embedding.ndim == 1 and embedding.size
            and np.isfinite(embedding).all()):
        raise ValueError(f"{folder}: invalid opening readout")
    if not (proposals.shape == (2, 10, 7) and np.isfinite(proposals).all()
            and type(slot) is int and slot in (0, 1) and offset == 0 and type(length) is int
            and length == min(5, n) and np.array_equal(proposals[slot, :length], issued[:length])):
        raise ValueError(f"{folder}: selected opening prefix disagrees with issued controls")
    return Query(initial, state, embedding, proposals, positions, success)


def summary(values):
    median, p90 = np.percentile(values, [50, 90]).tolist() if values else (None, None)
    return {"n": len(values), "median": median, "p90": p90}


def measure(root):
    wrist_cm = {control: [] for control in CONTROLS}
    per_task = {}
    pooled = np.zeros((2, 2), dtype=int)
    openings = []
    sources = set()
    for task in TASKS:
        counts = np.zeros_like(pooled)
        for repeat in REPEATS:
            folder = root / f"task{task:02d}-r{repeat:02d}"
            trial = json.loads((folder / "trial.json").read_text())
            sources.add((trial["source_pin"], trial["design_sha256"], trial["env_seed"]))
            row = trial["manifest_row"]
            if (row["split"], row["task_id"], row["repeat"]) != ("fitting", task, repeat):
                raise ValueError(f"{folder}: unexpected manifest row")
            a, b = (read_query(folder / arm, trial["episodes"][arm], row["seed"], stream)
                    for stream, arm in enumerate(("query0", "query1"), start=1))
            if not all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("initial", "state", "embedding")):
                raise ValueError(f"{folder}: opening wrist, state or embedding differs between queries")

            counts[int(a.success), int(b.success)] += 1
            for control in CONTROLS:
                if control <= min(len(a.positions), len(b.positions)):
                    # Each tape starts with the wrist position after physical control 1.
                    delta = a.positions[control - 1].astype(np.float64) - b.positions[control - 1]
                    wrist_cm[control].append(100 * np.linalg.norm(delta))
            for arm, query in (("query0", a), ("query1", b)):
                x, y = query.proposals
                delta = x[0, :3].astype(np.float64) - y[0, :3]
                openings.append(dict(task=task, repeat=repeat, arm=arm,
                                     first_xyz_l2_raw=float(np.linalg.norm(delta)),
                                     gripper_diff_first_five_raw=bool(np.any(x[:5, 6] != y[:5, 6]))))
        per_task[task] = counts.tolist()
        pooled += counts

    gripper_diffs = sum(row["gripper_diff_first_five_raw"] for row in openings)
    if len(sources) != 1:
        raise ValueError(f"Mixed collector, design or simulator seeds: {sources}")
    collector, design, env_seed = sources.pop()
    return {
        "collector_commit": collector, "design_sha256": design, "env_seed": env_seed,
        "task_ids": TASKS, "repeats": list(REPEATS), "n_reset_pairs": int(pooled.sum()),
        "wrist_distance_cm_after_control": {k: summary(v) for k, v in wrist_cm.items()},
        "native_outcomes": {
            "rows": ["query0 failure", "query0 success"], "columns": ["query1 failure", "query1 success"],
            "pooled": pooled.tolist(), "by_task": per_task,
        },
        "opening_proposals": {
            "comparison": "Two slots from one query input. Only the selected prefix was issued.",
            "first_xyz_l2_raw": summary([row["first_xyz_l2_raw"] for row in openings]),
            "gripper_diff_first_five_raw": {"different": gripper_diffs, "n": len(openings)},
            "by_query": openings,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fitting_root", type=Path, help="Published fitting directory containing taskNN-rNN")
    parser.add_argument("destination", type=Path, help="Output JSON path")
    args = parser.parse_args()
    report = measure(args.fitting_root)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
