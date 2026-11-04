"""Render a frozen partial DEV snapshot; no model computation or selection."""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

source = Path(sys.argv[1])
data = json.loads(source.read_text())
labels = {
    "OpenDrawer": "Open drawer",
    "CloseToasterOvenDoor": "Close toaster door",
    "PickPlaceDrawerToCounter": "Drawer → counter",
    "CoffeeSetupMug": "Coffee: set up mug",
    "TurnOffStove": "Turn off stove",
}
colors = ["#2563eb", "#d97706", "#7c3aed", "#059669", "#dc2626"]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=True, sharey=True)
fig.patch.set_facecolor("#ffffff")
for ax, row, color in zip(axes.flat, data["rows"], colors):
    scores = sorted(row["scores"], key=lambda s: s["step"])
    x = [s["step"] for s in scores]
    y = [100 * s["successes"] / s["cases"] for s in scores]
    lo, hi = [], []
    for s in scores:
        n = s["cases"]
        p = s["successes"] / n
        z = 1.959963984540054
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        lo.append(100*(center-half))
        hi.append(100*(center+half))
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5)
    for xx, yy, s in zip(x, y, scores):
        ax.annotate(f'{s["successes"]}/64', (xx, yy), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color=color)
    ax.set_title(labels[row["task"]], loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(87, 133)
    ax.set_ylim(0, 100)
    ax.set_xticks([90, 100, 110, 120, 130])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelbottom=True, labelleft=True, labelsize=9)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    ax.set_xlabel("PPO update (training order)", fontsize=9)
    ax.set_ylabel("Native success (%)", fontsize=9)
axes.flat[5].axis("off")
axes.flat[5].text(0, 0.93, "PARTIAL DEV RESULTS", fontsize=12, fontweight="bold", va="top", transform=axes.flat[5].transAxes)
axes.flat[5].text(0, 0.78,
    "64 fixed fresh cases per point\nSame case banks and sampler RNG\nShading: pointwise 95% Wilson intervals\n\n"
    "Updates 0–80: still pending at snapshot\nEvaluation runs 130 → 0\nPlot reads left → right in training order\n\n"
    "No winner locked; independent TEST follows.\nConnecting lines are only visual guides.",
    fontsize=10, va="top", linespacing=1.5, color="#374151", transform=axes.flat[5].transAxes)
fig.suptitle("PPO checkpoint curves so far — late training window", x=0.055, ha="left", fontsize=17, fontweight="bold")
fig.text(0.055, 0.923, "Snapshot: 4 Nov 2026, 13:22 Shanghai · First success within 600 controls · Earlier training checkpoints pending", fontsize=10, color="#4b5563")
fig.subplots_adjust(left=0.065, right=0.975, top=0.84, bottom=0.09, wspace=0.30, hspace=0.48)
target = source.with_suffix(".png")
fig.savefig(target, dpi=180, facecolor=fig.get_facecolor())
print(target.resolve())
