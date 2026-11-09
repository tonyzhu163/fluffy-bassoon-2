"""Plot preserved original-SFT and frozen-expert AD steering readouts."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
NATIVE = ROOT / "outputs/robocasa-eight-hour-babysit-20261104-r1"
STEERING = ROOT / "outputs/robocasa-three-task-kv-velocity-meta-20261105-r1"
TASKS = [
    ("OpenDrawer", "Drawer", "ID"),
    ("CloseToasterOvenDoor", "Toaster", "ID"),
    ("CoffeeSetupMug", "Coffee", "ID"),
    ("CloseFridge", "Fridge", "OOD"),
    ("TurnOnMicrowave", "Microwave", "OOD"),
]
SERIES = [
    ("sft", "Original SFT", "#64748b", "o"),
    ("kv", "K/V steering · AD · full history", "#2563eb", "s"),
    ("velocity", "Velocity steering · AD · full history", "#e56b24", "^"),
]
sources = []
curves = {key: {} for key, *_ in SERIES}


def read_source(path):
    data = path.read_bytes()
    sources.append({"path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(data).hexdigest()})
    value = json.loads(data)
    assert value["completed"] is True, path
    return value


for split in ("ID", "OOD"):
    native = read_source(NATIVE / f"NATIVE-ANCHOR-{split}-RESULT.json")
    for task, _, task_split in TASKS:
        if task_split != split:
            continue
        rows = native["curves"][task]
        assert all(rows[str(j)]["denominator"] == 50 for j in range(6))
        curves["sft"][task] = [rows[str(j)]["counts"]["native_initial"] for j in range(6)]
    for key, interface in (("kv", "kv_bridge"), ("velocity", "velocity_bridge")):
        data = read_source(STEERING / "readouts" / interface / "ad" / split / "SUMMARY.json")
        assert data["interface"] == interface and data["objective"] == "ad"
        assert data["native_parameters_frozen"] is True
        assert data["deployment_optimizer_updates"] == 0
        assert data["cases_per_task"] == 50
        for task, _, task_split in TASKS:
            if task_split != split:
                continue
            row = data["curves"][task]["full"]
            assert row["denominator"] == 50
            assert len(row["first_success"]) == 6
            curves[key][task] = row["first_success"]

assert all(0 <= n <= 50 for tasks in curves.values() for row in tasks.values() for n in row)
means = {key: np.mean([curves[key][task] for task, *_ in TASKS], axis=0) * 2
         for key, *_ in SERIES}
endpoint = [{"task": title, "split": split,
             **{key: curves[key][task][5] * 2 for key, *_ in SERIES}}
            for task, title, split in TASKS]
endpoint.append({"task": "Equal-task mean", "split": "ID + OOD",
                 **{key: float(means[key][5]) for key, *_ in SERIES}})
with (OUT / "query-success.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["task", "split", "method", "prior_episodes", "successes", "cases", "success_percent"])
    for task, _, split in TASKS:
        for key, label, *_ in SERIES:
            for j, count in enumerate(curves[key][task]):
                writer.writerow([task, split, label, j, count, 50, count * 2])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), sharex=True, sharey=True)
fig.subplots_adjust(left=.065, right=.985, top=.82, bottom=.19, hspace=.32, wspace=.16)
x = np.arange(6)
for i, ax in enumerate(axes.flat):
    if i < 5:
        task, title, split = TASKS[i]
        ax.set_title(f"{title}  ·  {split}", loc="left", pad=10)
    else:
        ax.set_title("Equal-task mean  ·  all five tasks", loc="left", pad=10)
        ax.set_facecolor("#f6f8fb")
    for key, label, color, marker in SERIES:
        y = np.asarray(curves[key][task]) * 2 if i < 5 else means[key]
        ax.plot(x, y, color=color, marker=marker, markersize=5.5,
                linewidth=2.1, label=label, zorder=3)
    ax.set_xlim(-.12, 5.12)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#dce1e8", linewidth=.8, zorder=0)
    ax.tick_params(axis="both", length=0, pad=7)
    if i % 3 == 0:
        ax.set_ylabel("Query success (%)")
    if i >= 3:
        ax.set_xlabel("Completed prior episodes")
fig.suptitle("Frozen-expert AD steering vs original SFT", fontsize=21,
             fontweight="bold", x=.065, ha="left", y=.965)
fig.text(.065, .916, "Success in each query episode · 50 cases per task · 0 is the first query",
         fontsize=12, color="#475569")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.055, .89),
           ncol=3, frameon=False, columnspacing=2.0)
fig.text(.065, .035,
         "K/V and velocity retain full history; SFT has no learned history conditioning.\n"
         "One fitted AD model per interface · zero evaluation updates · descriptive results; no significance claim.",
         fontsize=10.5, color="#475569", linespacing=1.6)
for suffix in ("png", "svg"):
    fig.savefig(OUT / f"sft-kv-velocity-query-curves.{suffix}", dpi=180, facecolor="white")
plt.close(fig)

manifest = {"metric": "First success within each individual query episode; not cumulative ever-success",
            "cases_per_task": 50, "prior_episodes": list(range(6)),
            "steering_objective": "AD", "memory": "full",
            "sources": sources, "counts": curves,
            "equal_task_mean_percent": {key: value.tolist() for key, value in means.items()},
            "after_five_preceding_episodes_percent": endpoint,
            "limitations": ["One fitted model per interface", "No significance claim",
                            "Stage 1 fits precede the later stopping-rule repair",
                            "DPT task evaluations are not included because they were not completed"]}
(OUT / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(endpoint, indent=2))
print(OUT / "sft-kv-velocity-query-curves.png")
