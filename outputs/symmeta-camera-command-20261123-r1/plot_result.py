"""Plot mean absolute command-yaw error from the three completed camera interventions."""
import gzip
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVIDENCE = Path(__file__).resolve().parent / "evidence"


def read(stem):
    with gzip.open(EVIDENCE / (stem + ".json.gz"), "rt") as f:
        return json.load(f)


def main():
    world_error = read("world")["summary"]["ood_combined"]["yaw_mae_deg"]
    orbital_world = mean(row["yaw_error_deg"] for row in read("replay")["rows"])
    physical = {a: read(f"azimuth{a:+d}")["summary"]["all"]["0"] for a in (-30, 0, 30)}
    panels = [
        ("Full image roll", "64 held-out traces", read("roll")["summary"]["ood"], world_error),
        ("Cropped image roll", "Same 64 traces · 1.4× zoom", read("moderate")["summary"]["ood"], world_error),
        ("Camera orbit in MuJoCo", "8 fresh replay trajectories", physical, orbital_world),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9), layout="constrained")
    for ax, (title, sample, data, ref) in zip(axes, panels):
        values = {int(k): v for k, v in data.items()}
        angles = sorted(values)
        for method, label, color, marker in [
            ("model_native", "SymMeta", "#174f8a", "o"),
            ("hs_native", "Classical flow", "#b45424", "s"),
        ]:
            ax.plot(angles, [values[a][method]["mae_deg"] for a in angles],
                    label=label, color=color, marker=marker, markerfacecolor="white",
                    markersize=5.8, linewidth=1.8, markeredgewidth=1.6)
        ax.axhline(ref, linestyle="--", color="#48544b", label="Trusted world-frame motion", linewidth=1.5)
        ax.set_title(f"{title}\n{sample}", fontsize=11, loc="left", pad=8)
        ax.set_xticks(angles)
        ax.set_xlabel("Camera angle (°)")
        ax.set_ylim(0, 110 if title == "Full image roll" else 50)
        ax.grid(axis="y", color="#dcdfe1", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean absolute command-yaw error (°)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(EVIDENCE / "camera-yaw-error.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
