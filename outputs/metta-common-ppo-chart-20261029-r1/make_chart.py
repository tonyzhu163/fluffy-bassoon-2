"""Data-only chart; imports no learner, simulator or torch model."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--reader", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=True, parents=True)
spec = importlib.util.spec_from_file_location("saved_array_reader", a.reader)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
paths = {
    "GRU": "metta-ml10-common-gru-baige-20261029-r1/train",
    "Transformer-XL": "metta-ml10-common-xl-baige-20261029-r1/train",
    "CTM": "metaworld-ml10-matched-baige-20261028-r1/ctm/train",
    "CMM": "cmm-ml10-matched-baige-20261029-r1/train",
}
colors = dict(zip(paths, ["#27856b", "#3e73b2", "#d47738", "#8b5bb1"]))
rows, cells, contracts = [], [], []
for name, relative in paths.items():
    directory = a.root / relative
    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    cell = reader.audit_cell(directory / "config.json", verify_hashes=True)
    assert cell["completed"] and cell["checkpoint_hash_verified"]
    assert cell["environment_steps"] == 10_000_000 and cell["optimizer_steps"] == 4000
    assert 212500 <= config["parameters"] <= 250000
    contracts.append(config["contract"])
    cells.append(cell)
    counter = summary["counters"]
    row = dict(name=name, parameters=config["parameters"], seed=config["contract"]["seed"],
        training_steps=counter["environment_steps"], training_gpu_hours=counter["training_seconds"] / 3600,
        evaluation_gpu_hours=counter["evaluation_seconds"] / 3600,
        driver_gpu_hours=counter["cumulative_elapsed_seconds"] / 3600,
        evaluation_steps=counter["evaluation_steps"], source=config["contract"]["source_commit"],
        checkpoint_sha256=summary["checkpoint_sha256"])
    for split in ("train_eval", "test"):
        for condition in ("normal", "memory_reset", "reward_blind"):
            row[f"{split}_{condition}_percent"] = summary["last_evaluation"][split][condition]["success"] * 100
    rows.append(row)
for key in ("protocol", "benchmark", "benchmark_seed", "seed", "pool_hashes", "goal_visible",
            "resample_instance", "reward", "observation", "normalization", "actor_critic",
            "parameter_budget", "attempts", "horizon", "num_envs", "optimization", "optimizer", "evaluation"):
    assert all(c[key] == contracts[0][key] for c in contracts), key
(a.output / "audit.json").write_text(json.dumps(dict(cells=cells, rows=rows,
    reader_sha256=hashlib.sha256(a.reader.read_bytes()).hexdigest(),
    chart_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    interpretation="One training seed; deterministic K25, trained K5; original clock. No seed uncertainty estimate."), indent=2) + "\n")
with (a.output / "endpoints.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "semibold"})

def style(ax):
    ax.set_facecolor("#f7f9fc")
    ax.grid(color="#dce2e9", linewidth=.7)
    ax.set_axisbelow(True)

fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.3), sharex=True, sharey=True)
fig.subplots_adjust(top=.75, bottom=.22, left=.075, right=.98, wspace=.10)
fig.suptitle("ML10 common-PPO controls", fontsize=21, weight="semibold", y=.98)
fig.text(.5, .918, "Matched 10M training interactions · common ~250k parameter cap · seed 1000",
         ha="center", color="#485567", fontsize=11)
handles = [Line2D([], [], marker="o", linestyle="", color=colors[r["name"]],
    markersize=8, label=f'{r["name"]}  ({r["parameters"]:,})') for r in rows]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .875), ncol=4,
           frameon=False, fontsize=10, columnspacing=1.6, handletextpad=.3)
for ax, split, title in zip(axes, ("train_eval", "test"),
        ("Fresh instances of training families", "Unseen task families")):
    style(ax)
    for row in rows:
        x, y = row["training_gpu_hours"], row[f"{split}_normal_percent"]
        ax.scatter(x, y, s=110, c=colors[row["name"]], edgecolors="white", linewidths=1.2, zorder=3)
        name = "XL" if row["name"] == "Transformer-XL" else row["name"]
        ax.annotate(f"{name}  {y:.1f}%", (x, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", color=colors[row["name"]], weight="semibold", fontsize=11)
    ax.set(title=title, xlim=(0, 1.5), ylim=(-2, 30),
           xlabel="Training GPU-hours (collection + PPO)")
    ax.set_xticks(np.arange(0, 1.51, .3))
axes[0].set_ylabel("Episode success (%)")
fig.text(.5, .105, "Each point is a completed 10M-step run, not an interpolated learning curve.", ha="center", fontsize=10)
fig.text(.5, .06, "Evaluation and setup time excluded: monitoring frequency differed. Trained K5; evaluated K25 with original clock.",
         ha="center", fontsize=9.5, color="#566273")
for ext in ("png", "pdf"):
    fig.savefig(a.output / f"common-ppo-ml10-endpoints.{ext}", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10.5, 5.1), constrained_layout=True)
style(ax)
x = np.arange(4)
for offset, condition, label, color in [(-.25, "normal", "Normal", "#466c9b"),
        (0., "memory_reset", "Frozen reset between attempts", "#83a5c1"),
        (.25, "reward_blind", "Scalar reward hidden", "#c6d3df")]:
    bars = ax.bar(x + offset, [r[f"test_{condition}_percent"] for r in rows], .24, label=label, color=color)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
ax.set(xticks=x, xticklabels=list(paths), ylim=(0, 35), ylabel="Held-out success (%)",
       title="ML10 frozen memory controls · one trained checkpoint per core")
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.supxlabel("A frozen reset lesion is not an independently trained reset control.", fontsize=10)
for ext in ("png", "pdf"):
    fig.savefig(a.output / f"common-ppo-ml10-memory-controls.{ext}", dpi=180)
plt.close(fig)
assert "torch" not in __import__("sys").modules
print(json.dumps({"audited_runs": len(rows), "output": str(a.output), "rows": rows}))
