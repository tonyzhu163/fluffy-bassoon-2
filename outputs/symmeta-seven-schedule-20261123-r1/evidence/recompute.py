"""Recompute the recorded-panel estimates. Pass the gpu01 folder and PR7 summary path."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


SEEDS = (41, 42, 43)
ARMS = ("original", "candidate")
METRICS = ("yaw_mae_deg", "operator_error", "scale_rel_err", "grip_acc",
           "all_correct", "yaw_correct", "scale_correct")
REGIMES = ("familiar", "compositional_ood", "continuous_angle_ood")
BOOTSTRAP_SEED = 20261123
DRAWS = 20_000


def read_json(path):
    if path.exists():
        return json.loads(path.read_bytes())
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return json.load(stream)


def independent_errors(row, prediction):
    angle = math.remainder(row["yaw_deg"] - prediction["yaw_deg"], 360.0)
    gain_ratio = row["xy_scale"] / prediction["xy_scale"]
    operator = abs(gain_ratio * np.exp(1j * np.deg2rad(angle)) - 1)
    scale_error = abs(prediction["xy_scale"] - row["xy_scale"]) / row["xy_scale"]
    grip_ok = row["gripper_sign"] == prediction["gripper_sign"]
    yaw_ok = abs(angle) <= 22.5
    scale_ok = scale_error <= 0.25
    return (abs(angle), operator, scale_error, grip_ok, yaw_ok and scale_ok and grip_ok,
            yaw_ok, scale_ok)


def describe(seed_means):
    return dict(by_seed=list(map(float, seed_means)), mean=float(np.mean(seed_means)),
                population_sd=float(np.std(seed_means, ddof=0)))


def analyze(gpu_dir, prior_path):
    res = gpu_dir / "results"
    summary = read_json(res / "summary.json")
    records = [read_json(res / f"paired-cases-{s}.json") for s in SEEDS]
    rows = records[0]
    tasks = sorted({(r["suite"], r["task_id"]) for r in rows})
    assert len(tasks) == 8 and len({r["support_sha256"] for r in rows}) == 96
    assert len({(r["suite"], r["task_id"], r["embodiment"]) for r in rows}) == 96
    identity = tuple(k for k in rows[0] if k not in (*ARMS, "reader_seed"))
    values = np.empty((3, 96, 2, len(METRICS)))
    raw_max = 0.0
    for j, seed in enumerate(SEEDS):
        assert len(records[j]) == 96
        for i, row in enumerate(records[j]):
            assert (row["index"], row["reader_seed"]) == (i, seed)
            assert all(row[k] == rows[i][k] for k in identity)
            for a, arm in enumerate(ARMS):
                vals = independent_errors(row, row[arm])
                saved = [row[arm][k] for k in METRICS]
                np.testing.assert_allclose(vals, saved, atol=1e-12, rtol=0)
                raw_max = max(raw_max, float(np.max(np.abs(np.subtract(vals, saved)))))
                values[j, i, a] = vals

    picks = np.random.default_rng(BOOTSTRAP_SEED).choice(8, size=(DRAWS, 8), replace=True)
    checked_cells, max_cell_diff = 0, 0.0
    primary, task_detail, panels, all96, by_protocol = {}, [], {}, {}, {}
    for group in (*REGIMES, "ood_combined", "overall"):
        groups = REGIMES if group == "overall" else REGIMES[1:] if group == "ood_combined" else (group,)
        indices = np.array([[i for i, r in enumerate(rows)
                             if (r["suite"], r["task_id"]) == task and r["regime"] in groups]
                            for task in tasks])
        assert indices.shape == (8, 4 * len(groups))
        ref = summary["panels"][group]
        assert ref["n_tasks"] == 8 and ref["n_traces"] == indices.size
        # Axes are task, supports within task, training seed, arm, metric.
        data = values.transpose(1, 0, 2, 3)[indices]
        for k, metric in enumerate((*METRICS, "yaw_median_deg")):
            median = metric == "yaw_median_deg"
            measured = data[..., 0 if median else k]
            reduce = np.median if median else np.mean
            point = reduce(measured, axis=(0, 1))
            # Draw task indices directly and keep every selected seed/arm/shift together.
            # Repeated tasks contribute repeated supports to each seed's median.
            if median:
                boot = np.concatenate([np.median(measured[b], axis=(1, 2))
                                       for b in np.array_split(picks, 100)])
            else:
                boot = measured.mean(axis=1)[picks].mean(axis=1)
            diffs = np.diff(boot, axis=-1)[..., 0]
            info = {}
            for j, seed in enumerate((*SEEDS, "mean")):
                if seed == "mean":
                    p, d, saved = point.mean(axis=0), diffs.mean(axis=1), ref["mean_over_seeds"][metric]
                else:
                    p, d, saved = point[j], diffs[:, j], ref["by_seed"][str(seed)][metric]
                lo, hi = np.quantile(d, [.025, .975])
                actual = np.array([*p, p[1]-p[0], lo, hi])
                reported = np.array([saved["original"], saved["candidate"], saved["candidate_minus_original"],
                                     *saved["paired_task_ci95"]])
                np.testing.assert_allclose(actual, reported, atol=1e-10, rtol=0)
                max_cell_diff = max(max_cell_diff, float(np.max(np.abs(actual-reported))))
                checked_cells += 1
                info[str(seed)] = dict(historic=p[0], schedule_only=p[1], difference=p[1]-p[0],
                                       paired_task_95=[lo, hi])
            if metric == "yaw_mae_deg":
                panels[group] = info["mean"]
                if group == "ood_combined":
                    primary = info
                    for t, (suite, task_id) in enumerate(tasks):
                        p = measured[t].mean(axis=(0, 1))
                        task_detail.append(dict(suite=suite, task_id=task_id, historic=p[0],
                                                schedule_only=p[1], difference=p[1]-p[0]))
                    for a, name in enumerate(("seven_historic", "seven_schedule_only")):
                        by_protocol[name] = {"ood_yaw_mae_deg": describe(point[:, a])}
            if group == "overall" and metric == "operator_error":
                all96 = info
                for a, name in enumerate(("seven_historic", "seven_schedule_only")):
                    by_protocol[name]["all96_operator"] = describe(point[:, a])

    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == "02f10b464a91abc616ac600c2be025550d1b8f85350228c8645155a19fc7db18"
    prior = read_json(prior_path)
    assert prior["synthetic"] is False and prior["step_scale"] == 1.0
    for method in ("perceiver_paper", "perceiver_cont_warm", "perceiver_cont"):
        selection = sorted((r for r in prior["rows"] if r["model"] == method and r["condition"] == "recorded"
                            and r["chunks"] == 3), key=lambda row: row["seed"])
        assert [r["seed"] for r in selection] == list(SEEDS)
        by_protocol["pr7_" + method] = dict(
            ood_yaw_mae_deg=describe([r["yaw_mae_ood_deg"] for r in selection]),
            all96_operator=describe([r["mean"] for r in selection]))
    contrasts = {}
    for left, right in (("seven_schedule_only", "seven_historic"),
                        ("pr7_perceiver_cont_warm", "seven_schedule_only"),
                        ("pr7_perceiver_cont_warm", "seven_historic"),
                        ("pr7_perceiver_cont_warm", "pr7_perceiver_cont"),
                        ("seven_historic", "pr7_perceiver_paper")):
        contrasts[left + "_minus_" + right] = {
            m: describe(np.subtract(by_protocol[left][m]["by_seed"], by_protocol[right][m]["by_seed"]))
            for m in by_protocol[left]}
    return dict(seeds=SEEDS, distinct_supports=96, ood_supports=64, whole_tasks=8, trace_seed_pairs=288,
                predictions_checked=int(values.size/len(METRICS)), regime_counts=Counter(r["regime"] for r in rows),
                bootstrap_seed=BOOTSTRAP_SEED, task_draws=DRAWS, raw_metric_max_abs_discrepancy=raw_max,
                production_metric_records_checked=checked_cells, summary_max_abs_discrepancy=max_cell_diff,
                ood_yaw_mae_deg=primary, yaw_mae_deg_by_panel=panels, ood_yaw_mae_deg_by_task=task_detail,
                operator_all96=all96, protocol_descriptives=by_protocol, seed_paired_contrasts=contrasts)


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1]), Path(sys.argv[2])), indent=2, allow_nan=False))
