"""Plot measured curves without extending short runs or mixing protocols."""

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    groups = {}
    for path in sorted(args.root.glob("*/config.json")):
        config = json.loads(path.read_text())
        contract = config["contract"]
        metrics = path.with_name("metrics.jsonl")
        if not metrics.exists():
            continue
        points = [json.loads(line) for line in metrics.read_text().splitlines()
                  if line.strip()]
        points = [x for x in points if "evaluation" in x]
        if points:
            groups.setdefault(contract["benchmark"], []).append((contract, points))
    if not groups:
        raise ValueError("No measured evaluation points to plot")
    for benchmark, runs in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for contract, points in runs:
            label = f'{contract["core"]} seed {contract["seed"]}'
            if any(x["gpu_hours"] is None for x in points):
                raise ValueError("GPU-hour curves require GPU runs")
            hours = [x["gpu_hours"] for x in points]
            for axis, split in zip(axes[:2], ("train_eval", "test")):
                axis.plot(hours, [x["evaluation"][split]["normal"]["success"] for x in points],
                          marker=".", label=label)
            last = points[-1]["evaluation"]["test"]
            axes[2].plot(np.arange(1, len(last["normal"]["success_by_episode"]) + 1),
                         last["normal"]["success_by_episode"], marker=".", label=label)
        for axis in axes:
            axis.set_ylim(0, 1)
            axis.set_ylabel("Episode success")
            axis.grid(alpha=.2)
        axes[0].set_title("Training families, fresh instances")
        axes[1].set_title("Held-out tasks")
        axes[2].set_title("Final held-out adaptation")
        for axis in axes[:2]:
            axis.set_xlabel("Measured allocated GPU-hours")
        axes[2].set_xlabel("Adaptation episode")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{benchmark}: bounded pilot, {len(runs)} runs; no seed uncertainty estimate")
        fig.savefig(args.root / f"{benchmark}-gpu-hours.png", dpi=160)
        fig.savefig(args.root / f"{benchmark}-gpu-hours.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
