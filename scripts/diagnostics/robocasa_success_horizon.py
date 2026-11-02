#!/usr/bin/env python
"""Per-task first-success quantiles from RoboCasa monitoring records.

Reads simulator-only monitoring JSON (never Flora or learner inputs) and
reports, per task, how many episodes ever succeeded and the p95/p99 policy
control of first success, plus p99 rounded up to a whole H50 chunk.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Monitoring directories, searched recursively")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    by_task = {}
    for root in args.roots:
        for path in sorted(Path(root).rglob("*.json")):
            row = json.loads(path.read_text())
            counts = row.get("native_counts", {})
            if "first_success_control" not in counts:
                continue  # Recorded before first-success logging existed.
            by_task.setdefault(row["case"]["task"], []).append(
                (counts["first_success_control"], bool(row["simulator_final_success"])))
    result = {}
    for task, rows in sorted(by_task.items()):
        steps = np.asarray([step for step, _ in rows if step is not None], dtype=float)
        result[task] = dict(episodes=len(rows), ever_succeeded=len(steps),
            final_success=sum(final for _, final in rows),
            succeeded_then_failed_at_end=sum(step is not None and not final for step, final in rows))
        if len(steps):
            p95, p99 = (float(np.percentile(steps, q, method="higher")) for q in (95, 99))
            result[task].update(p95=p95, p99=p99, max=float(steps.max()),
                                p99_horizon=int(math.ceil(p99 / 50) * 50))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
