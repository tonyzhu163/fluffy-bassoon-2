"""Paired official-panel estimates, conditioning on the three registered reader seeds."""

import numpy as np

from fluffy_bassoon.xemb.embodiment import EVAL_REGIMES, circular_yaw_error_deg
from fluffy_bassoon.xemb.wider_family import operator_error, rot2_batch

ARMS = ("original", "candidate")
METRICS = ("yaw_mae_deg", "operator_error", "scale_rel_err", "grip_acc",
           "all_correct", "yaw_correct", "scale_correct")
BOOT_SEED = 20261123
DRAWS = 20000


def trace_metrics(meta, predictions):
    yaw, gain, grip = predictions
    true_op = np.array([r["xy_scale"] for r in meta])[:, None, None] * rot2_batch([r["yaw_deg"] for r in meta])
    errors = operator_error(true_op, gain[:, None, None] * rot2_batch(yaw))
    out = []
    for r, y, s, g, err in zip(meta, yaw, gain, grip, errors, strict=True):
        yaw_error = circular_yaw_error_deg(float(y), r["yaw_deg"])
        scale_error = abs(float(s) - r["xy_scale"]) / r["xy_scale"]
        grip_ok = int(g) == r["gripper_sign"]
        yaw_ok, scale_ok = yaw_error <= 22.5, scale_error <= 0.25
        out.append(dict(yaw_deg=float(y), xy_scale=float(s), gripper_sign=int(g),
                        yaw_mae_deg=yaw_error, operator_error=float(err), scale_rel_err=scale_error,
                        grip_acc=int(grip_ok), all_correct=int(yaw_ok and scale_ok and grip_ok),
                        yaw_correct=int(yaw_ok), scale_correct=int(scale_ok)))
    return out


def interval(point, draws):
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return dict(original=float(point[0]), candidate=float(point[1]),
                candidate_minus_original=float(point[1] - point[0]), paired_task_ci95=[float(lo), float(hi)])


def summarize(rows, seeds):
    # Canonical case order on both arms and every seed, with each task kept intact.
    by_key = {(r["reader_seed"], r["index"]): r for r in rows}
    if len(by_key) != len(rows) or set(by_key) != {(s, i) for s in seeds for i in range(96)}:
        raise ValueError("Expected one paired row for each of 96 supports per seed")
    first = [by_key[seeds[0], i] for i in range(96)]
    # Pack all methods and seeds on the same trace axis before selecting or resampling tasks.
    values = np.empty((96, len(seeds), len(ARMS), len(METRICS)), dtype=np.float64)
    for j, seed in enumerate(seeds):
        for i, anchor in enumerate(first):
            row = by_key[seed, i]
            if row["support_sha256"] != anchor["support_sha256"]:
                raise ValueError(f"Different support for trace {i}, seed {seed}")
            for a, arm in enumerate(ARMS):
                values[i, j, a] = [row[arm][metric] for metric in METRICS]
    tasks = sorted({(r["suite"], r["task_id"]) for r in first})
    if len(tasks) != 8:
        raise ValueError("Expected eight official tasks")
    picks = np.random.default_rng(BOOT_SEED).integers(len(tasks), size=(DRAWS, len(tasks)))
    weights = np.stack([(picks == i).mean(axis=1) for i in range(len(tasks))], axis=1)
    results = {}
    for group in (*EVAL_REGIMES, "ood_combined", "overall"):
        task_indices = []
        for suite, tid in tasks:
            indices = [r["index"] for r in first if (r["suite"], r["task_id"]) == (suite, tid) and
                       (group == "overall" or group == r["regime"] or
                        (group == "ood_combined" and r["regime"] in ("compositional_ood", "continuous_angle_ood")))]
            expected = {"overall": 12, "ood_combined": 8}.get(group, 4)
            if len(indices) != expected:
                raise ValueError(f"Unexpected {group} membership for {suite}/{tid}")
            task_indices.append(indices)
        # [task, trace within task, seed, arm, metric].
        data = values[task_indices]
        task_means = data.mean(axis=1)
        points = task_means.mean(axis=0)
        resampled = (weights @ task_means.reshape(len(tasks), -1)).reshape(DRAWS, *points.shape)
        diffs = resampled[:, :, 1] - resampled[:, :, 0]
        per_seed = {s: {m: interval(points[j, :, k], diffs[:, j, k]) for k, m in enumerate(METRICS)}
                    for j, s in enumerate(seeds)}
        mean_seed = {m: interval(points.mean(axis=0)[:, k], diffs[:, :, k].mean(axis=1))
                     for k, m in enumerate(METRICS)}

        # Median yaw is computed on each seed's traces, including repeat tasks in each draw.
        yaw = data[..., 0]
        medians = np.median(yaw, axis=(0, 1))
        boot_medians = np.concatenate([np.median(yaw[draw], axis=(1, 2)) for draw in np.array_split(picks, 100)])
        median_diff = boot_medians[:, :, 1] - boot_medians[:, :, 0]
        for j, s in enumerate(seeds):
            per_seed[s]["yaw_median_deg"] = interval(medians[j], median_diff[:, j])
        mean_seed["yaw_median_deg"] = interval(medians.mean(axis=0), median_diff.mean(axis=1))
        results[group] = dict(n_traces=data.shape[0] * data.shape[1], n_tasks=len(tasks),
                              by_seed=per_seed, mean_over_seeds=mean_seed)
    return results


def render(result):
    lines = ["# Seven-shift schedule control", "",
             "Three paired Perceiver seeds on recorded UR5e supports. Differences are candidate minus original. "
             "Negative differences favor the candidate for errors. Positive differences favor it for accuracies. "
             "Intervals resample eight whole tasks and keep seeds paired.", "",
             "| Panel | Seed | Original yaw MAE (°) | Candidate yaw MAE (°) | Paired difference [95%] |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for g in ("ood_combined", "overall"):
        for s, metrics in result[g]["by_seed"].items():
            v = metrics["yaw_mae_deg"]
            lo, hi = v["paired_task_ci95"]
            lines.append(f"| {g} | {s} | {v['original']:.2f} | {v['candidate']:.2f} | "
                         f"{v['candidate_minus_original']:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    for name, group in result.items():
        lines.extend(["", f"## {name} ({group['n_traces']} traces)", "",
                      "| Seed-averaged metric | Original | Candidate | Paired difference [95%] |",
                      "| --- | ---: | ---: | ---: |"])
        for m, v in group["mean_over_seeds"].items():
            lo, hi = v["paired_task_ci95"]
            lines.append(f"| {m} | {v['original']:.4f} | {v['candidate']:.4f} | "
                         f"{v['candidate_minus_original']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    return "\n".join(lines) + "\n"
