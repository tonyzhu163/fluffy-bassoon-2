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
    for path in sorted(args.root.rglob("config.json")):
        config = json.loads(path.read_text())
        contract = config.get("contract")
        if not contract or contract.get("throughput_qualification_only") or contract.get("qualification_only") or contract.get("mode") == "one_update_throughput":
            continue
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
            method = contract.get("method", {})
            name = (method.get("display_name", method.get("name")) if isinstance(method, dict) else method)
            name = name or contract.get("core", "unknown method")
            if contract.get("core") == "persistent_ctm":
                length = contract["optimization"].get("replay_chunk_length", 0)
                name = f'CTM T{contract["core_config"]["ticks"]}' + (f' cached {length}' if length else ' full trial')
            label = f'{name}, seed {contract["seed"]}'
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
        for metric, xkey, xlabel in (("gpu-hours", "gpu_hours", "GPU Hours"),
                                    ("training-gpu-hours", "training_gpu_hours", "Training GPU Hours"),
                                    ("environment-steps", "environment_steps", "Training Environment Steps")):
            figure, panels = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            for contract, points in runs:
                method = contract.get("method", {})
                name = (method.get("display_name", method.get("name")) if isinstance(method, dict) else method)
                name = name or contract.get("core", "unknown method")
                if contract.get("core") == "persistent_ctm":
                    name = f'CTM T{contract["core_config"]["ticks"]}'
                    if contract["optimization"].get("replay_chunk_length"):
                        name += f' cached {contract["optimization"]["replay_chunk_length"]}'
                for panel, split in zip(panels, ("train_eval", "test")):
                    panel.plot([p[xkey] for p in points],
                               [p["evaluation"][split]["normal"]["success"] for p in points],
                               label=f'{name}, seed {contract["seed"]}')
            for panel, title in zip(panels, ("ML10 Train", "ML10 Held-out Families")):
                panel.set(title=title if benchmark == "ml10" else title.replace("ML10", benchmark),
                          xlabel=xlabel, ylabel="Success", ylim=(0, 1))
                panel.grid(alpha=.25)
            panels[0].legend(fontsize=7)
            figure.suptitle("Measured single-seed curves; no uncertainty bands")
            figure.savefig(args.root / f"{benchmark}-comparison-{metric}.png", dpi=180)
            figure.savefig(args.root / f"{benchmark}-comparison-{metric}.pdf")
            plt.close(figure)


if __name__ == "__main__":
    main()
