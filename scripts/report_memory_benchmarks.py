#!/usr/bin/env python3
"""Audit saved benchmark arrays and plot measured learning/adaptation curves.

This reader imports no learner or simulator and never loads model tensors.
It can inspect active snapshots; only terminal cells receive checkpoint audits.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def mean(values):
    return statistics.fmean(values)


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError(f"{label}: stored {actual} != raw {expected}")


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def read_rows(path, terminal):
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    trailing = bool(lines and not lines[-1].endswith(b"\n"))
    if trailing:
        if terminal:
            raise ValueError(f"Terminal log has an incomplete line: {path}")
        lines.pop()
    return [json.loads(line) for line in lines if line.strip()], trailing


def audit_evaluation(record, contract, label):
    outcomes, returns = record["outcomes"], record["native_returns"]
    families = record["family_by_lane"]
    trials, attempts = record["trials"], contract["evaluation"]["attempts"]
    if not (len(outcomes) == len(returns) == len(families) == trials and trials > 0):
        raise ValueError(f"Evaluation lane dimensions differ: {label}")
    if any(len(row) != attempts for row in outcomes + returns):
        raise ValueError(f"Evaluation episode dimensions differ: {label}")
    if any(type(v) is not bool for row in outcomes for v in row):
        raise ValueError(f"Non-binary success array: {label}")
    if not finite(returns):
        raise ValueError(f"Nonfinite raw returns: {label}")
    names = sorted(set(families))
    family_curves = {name: [mean([outcomes[i][e] for i, family in enumerate(families) if family == name])
                            for e in range(attempts)] for name in names}
    curve = [mean([family_curves[name][e] for name in names]) for e in range(attempts)]
    if len(record["success_by_episode"]) != attempts or set(record["per_family"]) != set(names):
        raise ValueError(f"Stored curve dimensions differ: {label}")
    if any(len(record["per_family"][name]) != attempts for name in names):
        raise ValueError(f"Stored family curve dimensions differ: {label}")
    close(record["success"], mean(curve), label + "/success")
    for actual, expected in zip(record["success_by_episode"], curve):
        close(actual, expected, label + "/success curve")
    for name in names:
        for actual, expected in zip(record["per_family"][name], family_curves[name]):
            close(actual, expected, label + "/family curve")
    return_curve = [mean([row[e] for row in returns]) for e in range(attempts)]
    native_return = mean(return_curve)
    if "mean_return" in record:
        if len(record["return_by_episode"]) != attempts:
            raise ValueError(f"Stored return curve dimensions differ: {label}")
        close(record["mean_return"], native_return, label + "/return")
        for actual, expected in zip(record["return_by_episode"], return_curve):
            close(actual, expected, label + "/return curve")
    result = dict(success=mean(curve), success_by_episode=curve,
                  mean_return=native_return, return_by_episode=return_curve,
                  last_minus_first_return=return_curve[-1] - return_curve[0],
                  last_minus_first_success=curve[-1] - curve[0], trials=trials,
                  environment_steps=record["environment_steps"])
    details = record.get("episode_metrics")
    if details and all(details):
        if len(details) != trials or any(len(row) != attempts for row in details):
            raise ValueError(f"Episode diagnostic dimensions differ: {label}")
        steps = [[item["steps"] for item in row] for row in details]
        if any(type(x) is not int or not 1 <= x <= contract["horizon"] for row in steps for x in row):
            raise ValueError(f"Invalid real episode length: {label}")
        close(record["environment_steps"], sum(map(sum, steps)), label + "/real steps")
        result["steps_by_episode"] = [mean([row[e] for row in steps]) for e in range(attempts)]
        if contract["benchmark"] == "halfcheetah-vel":
            result["absolute_velocity_error_by_episode"] = [mean([
                row[e]["absolute_velocity_error"] for row in details]) for e in range(attempts)]
        if contract["benchmark"] == "pointmaze":
            for row in details:
                if any(not 0 <= item["coverage"] <= 1 for item in row):
                    raise ValueError(f"Invalid coverage: {label}")
            result["coverage_by_episode"] = [mean([row[e]["coverage"] for row in details])
                                             for e in range(attempts)]
            # The original artifact writes 0 for Jaccard(empty, empty). Mark
            # those pairs undefined here: an idle agent has not explored anew.
            jaccard = []
            for e in range(attempts):
                valid = []
                for row in details:
                    value = row[e].get("episode_jaccard")
                    empty_pair = e > 0 and row[e]["coverage"] <= 1 / 17 and row[e - 1]["coverage"] <= 1 / 17
                    if value is not None and not empty_pair:
                        if not 0 <= value <= 1:
                            raise ValueError(f"Invalid Jaccard: {label}")
                        valid.append(value)
                jaccard.append(mean(valid) if valid else None)
            result["episode_jaccard_by_episode"] = jaccard
            result["jaccard_note"] = "empty/empty visited sets are undefined, not diverse exploration"
    return result


def audit_cell(path, *, verify_hashes=True):
    config = json.loads(path.read_text())
    contract = config.get("contract", {})
    if contract.get("throughput_qualification_only") or contract.get("benchmark") not in (
            "pointmaze", "halfcheetah-vel", "ml1-push"):
        return None
    directory = path.parent
    summary_path = directory / "summary.json"
    terminal = summary_path.exists()
    rows, trailing = read_rows(directory / "metrics.jsonl", terminal)
    if not rows:
        return None
    if not finite(rows):
        raise ValueError(f"Nonfinite metrics: {directory}")
    previous_steps, previous_time, previous_iteration = -1, -1., -1
    optimizer_steps = 0
    points = {}
    last_evaluation, last_raw_evaluation = {}, None
    for row in rows:
        steps, elapsed, iteration = row["environment_steps"], row["elapsed_seconds"], row["iteration"]
        if steps < previous_steps or elapsed < previous_time or iteration < previous_iteration:
            raise ValueError(f"Nonmonotonic progress: {directory}")
        previous_steps, previous_time, previous_iteration = steps, elapsed, iteration
        if "training" in row:
            training = row["training"]
            optimizer_steps += training["optimizer_steps"]
            if training["active_action_samples"] > training["forward_samples"]:
                raise ValueError(f"Real samples exceed executed samples: {directory}")
        if "evaluation" in row:
            last_raw_evaluation = row["evaluation"]
            verified = {split: {condition: audit_evaluation(value, contract,
                        f"{directory.name}/{iteration}/{split}/{condition}")
                        for condition, value in conditions.items()}
                        for split, conditions in row["evaluation"].items()}
            points[steps] = dict(environment_steps=steps, elapsed_seconds=elapsed,
                gpu_hours=row["gpu_hours"], training_gpu_hours=row["training_gpu_hours"],
                iteration=iteration, evaluation=verified)
            last_evaluation = verified
    summary, hash_verified = None, False
    if terminal:
        summary = json.loads(summary_path.read_text())
        if summary["source_commit"] != contract["source_commit"] or summary["iteration"] != previous_iteration:
            raise ValueError(f"Summary source or iteration differs: {directory}")
        counters = summary["counters"]
        if config.get("resume"):
            raise ValueError(f"Resumed counters require the parent run's evidence: {directory}")
        if counters["environment_steps"] != previous_steps or counters["optimizer_steps"] != optimizer_steps:
            raise ValueError(f"Summary interaction/optimizer counters differ: {directory}")
        if verify_hashes:
            digest = hashlib.sha256()
            with (directory / "latest.pt").open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != summary["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash differs: {directory}")
            hash_verified = True
        if summary["last_evaluation"] != last_raw_evaluation:
            raise ValueError(f"Summary final evaluation differs: {directory}")
    memory_effects = {}
    for split, conditions in (last_raw_evaluation or {}).items():
        if "normal" in conditions and "memory_reset" in conditions:
            for normal, reset in zip(conditions["normal"]["native_returns"], conditions["memory_reset"]["native_returns"]):
                close(normal[0], reset[0], f"{directory.name}/{split}/paired first episode")
    for split, conditions in last_evaluation.items():
        normal = conditions.get("normal")
        if normal is None:
            continue
        effects = {}
        for condition in ("memory_reset", "reward_blind"):
            if condition not in conditions:
                continue
            other = conditions[condition]
            effects[condition] = dict(normal_minus_condition_return=normal["mean_return"] - other["mean_return"],
                normal_minus_condition_success=normal["success"] - other["success"])
        memory_effects[split] = effects
    # Independent loss/return checks above do not promote single-seed evidence.
    return dict(path=str(directory), benchmark=contract["benchmark"], core=contract["core"],
        seed=contract["seed"], memory_mode=contract.get("memory_mode", "meta"),
        core_config=contract["core_config"], parameters=config["parameters"],
        protocol=contract["protocol"], pool_hashes=contract["pool_hashes"],
        source_commit=contract["source_commit"], terminal=terminal,
        completed=bool(summary and summary["completed"]), checkpoint_hash_verified=hash_verified,
        trailing_partial_line_ignored=trailing, iteration=previous_iteration,
        environment_steps=previous_steps, optimizer_steps=optimizer_steps,
        elapsed_seconds=summary["counters"]["cumulative_elapsed_seconds"] if summary else previous_time,
        points=list(points.values()), final_evaluation=last_evaluation, memory_effects=memory_effects)


def label(cell):
    name = {"persistent_ctm": "CTM", "transformer_xl": "Transformer-XL", "cmm": "CMM", "gru": "GRU"}[cell["core"]]
    if cell["core"] == "cmm" and cell["core_config"].get("ltm_tokens") == 0:
        name += " STM"
    return f"{name}, seed {cell['seed']}" + (" (active)" if not cell["terminal"] else
           " (capped)" if not cell["completed"] else "")


def write_plots(cells, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    colors = {"gru": "#43856a", "transformer_xl": "#5279b8", "persistent_ctm": "#ce693a", "cmm": "#8d62af"}
    for benchmark in sorted({cell["benchmark"] for cell in cells}):
        group = [cell for cell in cells if cell["benchmark"] == benchmark]
        primary = "mean_return" if benchmark == "halfcheetah-vel" else "success"
        ylabel = "Mean native episode return" if primary == "mean_return" else "Episode success"
        splits = [s for s in ("train_eval", "test", "ood") if any(s in cell["final_evaluation"] for cell in group)]
        figure, axes = plt.subplots(2, len(splits), figsize=(5 * len(splits), 7), squeeze=False,
                                    constrained_layout=True)
        for column, split in enumerate(splits):
            for cell in group:
                points = [p for p in cell["points"] if split in p["evaluation"]]
                if not points:
                    continue
                color = colors[cell["core"]]
                style = "-" if cell["terminal"] else "--"
                axes[0, column].plot([p["environment_steps"] for p in points],
                    [p["evaluation"][split]["normal"][primary] for p in points],
                    color=color, linestyle=style, marker=".", label=label(cell))
                last = cell["final_evaluation"][split]
                key = "return_by_episode" if primary == "mean_return" else "success_by_episode"
                for condition, condition_style in (("normal", style), ("memory_reset", ":")):
                    if condition in last:
                        values = last[condition][key]
                        axes[1, column].plot(range(1, len(values) + 1), values, color=color,
                            linestyle=condition_style, label=label(cell) + (" reset lesion" if condition != "normal" else ""))
            axes[0, column].set(title=split.replace("_", " "), xlabel="Real training interactions", ylabel=ylabel)
            axes[1, column].set(xlabel="Adaptation episode", ylabel=ylabel)
            for axis in axes[:, column]:
                axis.grid(alpha=.2)
                if primary == "success":
                    axis.set_ylim(0, 1)
        axes[0, 0].legend(fontsize=7)
        axes[1, 0].legend(fontsize=6)
        figure.suptitle(benchmark + ": measured curves; active/capped cells marked; no seed uncertainty estimate")
        figure.savefig(output / (benchmark + "-learning.png"), dpi=150)
        figure.savefig(output / (benchmark + "-learning.pdf"))
        plt.close(figure)
        figure, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 3.8), squeeze=False,
                                    constrained_layout=True)
        for column, split in enumerate(splits):
            for cell in group:
                points = [p for p in cell["points"] if split in p["evaluation"]]
                if points:
                    axes[0, column].plot([p["gpu_hours"] for p in points],
                        [p["evaluation"][split]["normal"][primary] for p in points],
                        color=colors[cell["core"]], linestyle="-" if cell["terminal"] else "--",
                        marker=".", label=label(cell))
            axes[0, column].set(title=split.replace("_", " "), xlabel="Measured driver GPU-hours", ylabel=ylabel)
            axes[0, column].grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        figure.suptitle(benchmark + ": driver time includes collection/evaluation; launcher overhead reported separately")
        figure.savefig(output / (benchmark + "-gpu-hours.png"), dpi=150)
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-hashes", action="store_true", help="Provisional reader only; never marks hashes verified")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)
    cells = []
    for root in args.roots:
        for config in sorted(root.rglob("config.json")):
            if not config.with_name("metrics.jsonl").exists():
                continue
            cell = audit_cell(config, verify_hashes=not args.skip_hashes)
            if cell is not None:
                cells.append(cell)
    if not cells:
        raise ValueError("No learning runs with saved evaluations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(cells=cells, interpretation="Single-seed screens are not confirmed architecture wins; trained reset controls and calibration are separate gates")
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Saved-array benchmark audit", "", "Native return is primary for HalfCheetah. PointMaze/ML1 success accompanies adaptation and memory controls.", "",
             "| Task | Core | Seed | State | Real steps | Held-out return | Held-out success | Normal − reset return |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for cell in cells:
        final = cell["final_evaluation"].get("test", {}).get("normal", {})
        effect = cell["memory_effects"].get("test", {}).get("memory_reset", {})
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            cell["benchmark"], cell["core"], cell["seed"], "complete" if cell["completed"] else "capped" if cell["terminal"] else "active",
            cell["environment_steps"], round(final.get("mean_return", 0.), 4),
            round(final.get("success", 0.), 4), round(effect.get("normal_minus_condition_return", 0.), 4)))
    lines.extend(["", "Active/capped results retain achieved budgets; no curve is extrapolated. A frozen reset lesion is not an independently trained reset control.",
                  "PointMaze empty/empty visited-cell Jaccard pairs are treated as undefined in derived summaries; the original artifacts are preserved.", ""])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))
    if args.plots:
        write_plots(cells, args.output_dir)
    print(json.dumps({"cells": len(cells), "terminal": sum(c["terminal"] for c in cells), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
