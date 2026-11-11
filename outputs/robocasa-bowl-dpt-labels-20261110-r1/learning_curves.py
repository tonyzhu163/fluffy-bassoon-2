"""Plot the recorded checkpoint-selection panels without re-running a model."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    folder = Path(__file__).resolve().parent
    primary = json.loads((folder / "development.json").read_text())["selection"]
    oracle = json.loads((folder / "oracle-development.json").read_text())["selection"]
    steps = [128, 512, 1024]
    series = [
        ("Memoryless LoRA", "#31599d", primary, "memoryless"),
        ("DPT + LoRA", "#be5d28", primary, "own"),
        ("True-side LoRA", "#27735a", oracle, "oracle"),
    ]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#263238", "xtick.color": "#45505a",
        "ytick.color": "#45505a", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.3))
    for label, color, values, key in series:
        rows = [values["development"][str(step)][key] for step in steps]
        axes[0].plot(steps, [r["masked_error"] for r in rows],
                     marker="o", color=color, lw=2, label=label, ms=6)
        axes[1].plot(steps, [r["native"] for r in rows],
                     marker="o", color=color, lw=2, label=label, ms=6)
    axes[0].set_ylabel("Masked flow-prediction error")
    axes[0].set_title("Shared-observation imitation error", loc="left", fontsize=12, pad=13)
    axes[0].set_ylim(.08, .17)
    axes[1].set_ylabel("Correct goal cases / 16")
    axes[1].set_title("Physical placements on 8 development scenes", loc="left", fontsize=12, pad=13)
    axes[1].set_ylim(-.3, 6)
    axes[1].set_yticks(range(7))
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(steps, labels=["128", "512", "1,024"])
        ax.set_xlabel("Training updates", labelpad=9)
        ax.grid(axis="y", color="#e5e8eb", lw=.7)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(.52, .02))
    fig.subplots_adjust(left=.09, right=.98, bottom=.24, top=.88, wspace=.30)
    fig.savefig(folder / "learning-curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
