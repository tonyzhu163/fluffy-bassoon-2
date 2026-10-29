"""Report paired robot outcomes, retaining the scene as the sampling unit.

Input is the integration driver's episodes.jsonl. Supervised losses and retry
success-once are not substituted for per-attempt robotic success.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random


def load_episodes(path):
    rows = []
    identities = set()
    required = ("method", "split", "task_id", "scene_seed", "trial_seed", "attempt",
                "success", "primitive_steps", "completed")
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if any(key not in row for key in required):
            raise ValueError("Episode is missing its method/scene/outcome/cost identity")
        if (type(row["success"]) is not bool or type(row["completed"]) is not bool
                or type(row["attempt"]) is not int or row["attempt"] < 1
                or type(row["primitive_steps"]) is not int or row["primitive_steps"] < 1):
            raise ValueError("Episode outcomes, attempt numbers or physical costs are invalid")
        identity = tuple(row[key] for key in ("method", "split", "task_id", "scene_seed", "trial_seed", "attempt"))
        # Training may evaluate multiple checkpoints on the same scene bank.
        identity += (row.get("cumulative_train_steps", 0),)
        if identity in identities:
            raise ValueError("Duplicate episode identity; never silently count it twice")
        identities.add(identity)
        rows.append(row)
    if not rows:
        raise ValueError("No episode records")
    return rows


def mean(values):
    return sum(values) / len(values)


def trial_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["completed"]:
            key = tuple(row[key] for key in ("task_id", "scene_seed", "trial_seed"))
            groups[key].append(row)
    for key, trial in groups.items():
        trial.sort(key=lambda row: row["attempt"])
        if [row["attempt"] for row in trial] != list(range(1, len(trial) + 1)):
            raise ValueError(f"Incomplete/nonconsecutive attempts in trial {key}")
    return dict(groups)


def summarize(rows):
    groups = trial_groups(rows)
    if not groups:
        return dict(completed_trials=0)
    counts = {len(trial) for trial in groups.values()}
    if len(counts) != 1:
        raise ValueError("Do not pool unequal completed retry budgets")
    attempts = counts.pop()
    by_attempt = []
    for index in range(attempts):
        by_attempt.append(dict(attempt=index + 1, success_rate=mean(
            [float(trial[index]["success"]) for trial in groups.values()]),
            mean_cumulative_primitive_steps=mean([sum(row["primitive_steps"] for row in trial[:index + 1])
                                                 for trial in groups.values()])))
    return dict(completed_trials=len(groups), independent_scenes=len({(key[0], key[1]) for key in groups}),
        tasks=len({key[0] for key in groups}), attempts=attempts,
        first_attempt_success=by_attempt[0]["success_rate"],
        final_attempt_success=by_attempt[-1]["success_rate"],
        mean_trial_success=mean([mean([float(row["success"]) for row in trial]) for trial in groups.values()]),
        success_at_least_once=mean([float(any(row["success"] for row in trial)) for trial in groups.values()]),
        primitive_steps=sum(row["primitive_steps"] for trial in groups.values() for row in trial),
        by_attempt=by_attempt)


def paired_comparison(rows_a, rows_b, *, repetitions=2000, seed=734001):
    """Bootstrap scenes, preserving all stochastic trials and correlated attempts."""
    a, b = trial_groups(rows_a), trial_groups(rows_b)
    if a.keys() != b.keys() or not a:
        raise ValueError("Paired methods must complete the identical task/scene/trial bank")
    scenes = defaultdict(list)
    for key in a:
        if len(a[key]) != len(b[key]):
            raise ValueError("Paired methods must use the identical attempt budget")
        difference = mean([float(row["success"]) for row in a[key]]) - mean(
            [float(row["success"]) for row in b[key]])
        scenes[key[:2]].append(difference)
    differences = [mean(values) for values in scenes.values()]
    result = dict(independent_scenes=len(differences),
                  mean_trial_success_difference=mean(differences), sampling_unit="task/scene")
    # A one-scene interval would report fake certainty, even with many attempts.
    if len(differences) < 2:
        result["bootstrap_95_percent_interval"] = None
        return result
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(repetitions))
    result["bootstrap_95_percent_interval"] = [samples[int(.025 * (repetitions - 1))],
                                                samples[int(.975 * (repetitions - 1))]]
    return result


def report(rows):
    cells = defaultdict(list)
    for row in rows:
        key = (row["method"], row["split"], row.get("cumulative_train_steps", 0))
        cells[key].append(row)
    summary = {}
    for key, values in sorted(cells.items()):
        label = f"{key[0]}/{key[1]}/train_steps={key[2]}"
        if key[1] == "train":
            # Success across successive source-learning episodes is a training
            # curve, not independent retry trials. Preserve its raw ordering.
            summary[label] = dict(completed_episodes=sum(row["completed"] for row in values),
                success_rate=mean([float(row["success"]) for row in values if row["completed"]])
                if any(row["completed"] for row in values) else None,
                primitive_steps=sum(row["primitive_steps"] for row in values))
        else:
            summary[label] = summarize(values)
    comparisons = {}
    for (method, split, checkpoint), values in cells.items():
        if split == "train" or method in ("native", "tied_gaussian"):
            continue
        for reference in ("native", "static", "reset"):
            candidates = [(key, reference_rows) for key, reference_rows in cells.items()
                          if key[0] == reference and key[1] == split
                          and key[2] in (checkpoint, 0)]
            if len(candidates) != 1:
                continue
            try:
                comparisons[f"{method}-{reference}/{split}/train_steps={checkpoint}"] = paired_comparison(
                    values, candidates[0][1])
            except ValueError as error:
                comparisons[f"{method}-{reference}/{split}/train_steps={checkpoint}"] = dict(unpaired=str(error))
    return dict(cells=summary, paired=comparisons,
        interpretation="Episode success uses paired scenes. Retry success-once alone does not establish learning.")


def plot_curves(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = defaultdict(list)
    for row in rows:
        if row["split"] != "train":
            cells[(row["method"], row.get("cumulative_train_steps", 0))].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for (method, checkpoint), values in sorted(cells.items()):
        result = summarize(values)
        if not result["completed_trials"]:
            continue
        curve = result["by_attempt"]
        label = f"{method} ({checkpoint:,} train steps)" if checkpoint else method
        axes[0].plot([row["attempt"] for row in curve], [100 * row["success_rate"] for row in curve],
                     marker="o", label=label)
        axes[1].plot([row["mean_cumulative_primitive_steps"] for row in curve],
                     [100 * row["success_rate"] for row in curve], marker="o", label=label)
    axes[0].set_xlabel("Deployment attempt")
    axes[1].set_xlabel("Mean cumulative deployment environment steps")
    for axis in axes:
        axis.set_ylim(0, 100)
        axis.set_ylabel("Success on this attempt (%)")
        axis.grid(alpha=.2)
    if cells:
        axes[1].legend(fontsize=7)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args(argv)
    rows = load_episodes(args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report(rows), indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot_curves(rows, args.plot)


if __name__ == "__main__":
    main()
