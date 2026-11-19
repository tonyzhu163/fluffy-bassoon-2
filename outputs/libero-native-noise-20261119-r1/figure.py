"""Draw the trajectories on the collector's original published example."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fitting", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    case = args.fitting / "task00-r00"
    trial = json.loads((case / "trial.json").read_text())
    fig, (top, height) = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    for arm, color, style in (("query0", "#176b85", "-"), ("query1", "#ba572d", "--")):
        with np.load(case / arm / "public.npz", allow_pickle=False) as tape:
            xyz = np.vstack([tape["initial_position"], tape["positions"]])
            xyz = (xyz - xyz[0]) * 100
            success = bool(tape["terminal"])
        assert success == trial["episodes"][arm]["success"]
        label = ("Retry A" if arm == "query0" else "Retry B") + (" · success" if success else " · timeout")
        for ax, x, y in ((top, xyz[:, 0], xyz[:, 1]), (height, np.arange(len(xyz)), xyz[:, 2])):
            ax.plot(x, y, color=color, ls=style, lw=1.7, alpha=0.9, label=label)
            ax.scatter(x[-1], y[-1], color=color, marker="s", s=23, zorder=3)
    for ax in (top, height):
        ax.scatter(0, 0, marker="o", s=38, color="#252627", zorder=4, label="Shared start")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    top.set(title="Wrist path from above", xlabel="X from start (cm)", ylabel="Y from start (cm)", aspect="equal")
    height.set(title="Wrist height over time", xlabel="Controls executed", ylabel="Height from start (cm)")
    height.legend(frameon=False, fontsize=9)
    fig.suptitle("Close the top drawer · two π0.5 retries from the same pose", size=14)
    fig.savefig(args.destination, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
