"""Compare fixed circular readouts on archived cross-arm motion vectors."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "symmeta-matched-flow-20261123-r1/evidence/target-details.json.gz"


def wrap(degrees):
    return (degrees + 180) % 360 - 180


def readouts(commands, motion, valid):
    commands, motion = commands[valid], motion[valid]
    if not len(commands):
        raise ValueError("Support has no valid votes")
    magnitude = np.linalg.norm(motion, axis=1)
    if np.any(magnitude == 0):
        raise ValueError("Zero length visual vector in a valid chunk")
    unit = motion / magnitude[:, None]
    dot = (commands * unit).sum(axis=1)
    cross = commands[:, 0] * unit[:, 1] - commands[:, 1] * unit[:, 0]
    votes = dot + 1j * cross
    angles = np.angle(votes, deg=True)
    distance = np.abs(wrap(angles[:, None] - angles[None, :]))
    return {
        "uniform": np.angle(votes.sum(), deg=True),
        "magnitude": np.angle((magnitude * votes).sum(), deg=True),
        "medoid": angles[distance.sum(axis=1).argmin()],
        "last": angles[-1],
        "first": angles[0],
        "nominal": 0.0,
    }


def evaluate(rows, reader):
    records = []
    for row in rows:
        estimates = readouts(np.asarray(row["command_unit"]), np.asarray(row[f"{reader}_xy"]),
                             np.asarray(row["weights"]) > 0)
        np.testing.assert_allclose(wrap(estimates["uniform"] - row[f"{reader}_yaw_deg"]), 0, atol=1e-10)
        errors = {k: abs(wrap(v - row["true_yaw_deg"])) for k, v in estimates.items()}
        records.append({"task": f'{row["suite"]}/{row["task_id"]}', "regime": row["regime"],
                        "index": row["index"], "errors": errors})
    return records


def summarize(rows):
    tasks = sorted({r["task"] for r in rows})
    methods = list(rows[0]["errors"])
    by_task = []
    for task in tasks:
        group = [r for r in rows if r["task"] == task]
        by_task.append([[r["errors"][k] for k in methods] for r in group])
    # Each task contributes the same number of cases within each reported stratum.
    means = np.array(by_task).mean(axis=1)
    changes = means - means[:, [methods.index("uniform")]]
    rng = np.random.default_rng(20261123)
    draws = rng.integers(len(tasks), size=(20000, len(tasks)))
    ci = np.quantile(changes[draws].mean(axis=1), [0.025, 0.975], axis=0)
    return {"n": len(rows), "tasks": tasks, "heads": {
        k: {"mae_deg": means[:, i].mean(), "paired_delta_deg": changes[:, i].mean(),
            "ci95_delta_deg": ci[:, i].tolist()} for i, k in enumerate(methods)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reader", choices=["learned", "linear"])
    reader = parser.parse_args().reader
    encoded = DATA.read_bytes()
    rows = json.loads(gzip.decompress(encoded))
    if len(rows) != 96 or len({r["index"] for r in rows}) != 96:
        raise ValueError("Incomplete archived panel")
    scored = evaluate(rows, reader)
    held_out = [r for r in scored if r["regime"] != "familiar"]
    if len(held_out) != 64:
        raise ValueError("Held-out split differs from the archived panel")
    groups = {"all": scored, "ood": held_out}
    for regime in sorted({r["regime"] for r in scored}):
        groups[regime] = [r for r in scored if r["regime"] == regime]
    report = {"reader": reader, "source_sha256": hashlib.sha256(encoded).hexdigest(),
              "bootstrap_seed": 20261123, "bootstrap_draws": 20000,
              "groups": {name: summarize(group) for name, group in groups.items()}}
    (HERE / f"{reader}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["groups"]["ood"]["heads"], indent=2))


if __name__ == "__main__":
    main()
