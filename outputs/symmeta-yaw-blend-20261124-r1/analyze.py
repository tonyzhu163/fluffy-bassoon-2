"""Score frozen corrections on CPU. From the repo root, run python -B path/to/analyze.py."""

import csv
import gzip
import hashlib
import json
import math
import random
import runpy
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from statistics import fmean, quantiles

ARCHIVE = Path("outputs/symmeta-perceiver-retry-20261123-r1/evidence")
INPUTS = (ARCHIVE / "train02pro/predictions.json", ARCHIVE / "native02g4090/pairs.csv.gz",
          Path("src/fluffy_bassoon/xemb/embodiment.py"))
HASHES = ("c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b",
          "be37297b1950f4ec3463a55ce9e60617eba2318c42c363504d42844eb3d8cf1d",
          "e0ed86b5fc85bbde7b4dd95b0ec2b520b792325d0e6f135465132cc2dc99c41c")
PANELS = ("gpu09test", "gpu09iiwa", "gpu06kinova3", "gpu08pi0fix")
SEEDS = (41, 42, 43)
DRAWS, BOOT_SEED = 20_000, 1796


def pair_key(row):
    return row["panel"], int(row["index"]), int(row["seed"])


def operator_error(yaw_error, gain_ratio):
    angle = math.radians(yaw_error)
    return math.hypot(gain_ratio * math.cos(angle) - 1, gain_ratio * math.sin(angle))


def summarize(task_rows, draws):
    # The design is balanced. Each task contributes the same number of OOD rows.
    assert len({len(rows) for rows in task_rows}) == 1
    task_means = [tuple(map(fmean, zip(*rows))) for rows in task_rows]
    result = {"pairs": sum(map(len, task_rows))}
    for metric, offset in (("yaw_mae_deg", 0), ("mean_operator_error", 2)):
        tuned, blend = zip(*(means[offset:offset + 2] for means in task_means))
        diffs = [b - t for t, b in zip(tuned, blend)]
        estimates = [fmean(diffs[i] for i in indices) for indices in draws]
        cuts = quantiles(estimates, n=40, method="inclusive")
        result[metric] = dict(tuned=fmean(tuned), blend=fmean(blend), blend_minus_tuned=fmean(diffs),
                              task_ci95=[cuts[0], cuts[-1]])
    return result


def analyze(predictions, outcomes, label_source):
    receipt = {}
    for path, canonical, expected in zip((predictions, outcomes, label_source), INPUTS, HASHES):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: {digest}")
        receipt[str(canonical)] = dict(bytes=len(data), sha256=digest)
    module = runpy.run_path(str(label_source))
    labels = {s.name: s for s in module["EVAL_EMBODIMENTS"]}
    yaw_error = module["circular_yaw_error_deg"]
    with gzip.open(outcomes, "rt") as stream:
        native_rows = list(csv.DictReader(stream))
    native = {pair_key(r): r for r in native_rows}
    assert len(native) == len(native_rows) == 1152
    cells, task_ids = set(), set()
    scores = defaultdict(list)
    sign_counts = {panel: Counter() for panel in (*PANELS, "pooled")}
    subset = Counter()
    closer = Counter()

    for row in json.loads(predictions.read_bytes())["rows"]:
        key = pair_key(row)
        panel, _, seed = key
        pair = native.pop(key)
        for field in ("suite", "task_id", "embodiment", "regime", "tape_sha256"):
            assert str(row[field]) == pair[field], (key, field)
        task = (row["suite"], row["task_id"])
        spec = labels[row["embodiment"]]
        assert spec.regime == row["regime"]
        cells.add((panel, task, seed, spec.name))
        task_ids.add(task)
        t, s = row["tuned"], row["symmeta"]
        for group in (panel, "pooled"):
            counts = sign_counts[group]
            counts["all_pairs"] += 1
            counts["all_agree"] += t["gripper_sign"] == s["gripper_sign"]
            if spec.regime != "familiar":
                counts["ood_pairs"] += 1
                counts["ood_tuned_correct"] += t["gripper_sign"] == spec.gripper_sign
                counts["ood_symmeta_correct"] += s["gripper_sign"] == spec.gripper_sign
        if spec.regime == "familiar":
            continue
        arc = (s["yaw_deg"] - t["yaw_deg"] + 180) % 360 - 180
        assert abs(arc) < 180, "Antipodal yaws have no unique circular midpoint"
        baseline = yaw_error(t["yaw_deg"], spec.yaw_deg)
        blend = yaw_error(t["yaw_deg"] + arc / 2, spec.yaw_deg)
        ratio = spec.xy_scale / t["xy_scale"]
        scores[panel, task].append((baseline, blend, operator_error(baseline, ratio), operator_error(blend, ratio)))

        if panel == "gpu09test" and spec.regime == "compositional_ood":
            success = (int(pair["tuned"]), int(pair["symmeta"]))
            outcome = {(1, 0): "tuned_only", (0, 1): "symmeta_only", (1, 1): "both", (0, 0): "neither"}[success]
            subset[outcome] += 1
            if outcome == "symmeta_only":
                closer["yaw"] += yaw_error(s["yaw_deg"], spec.yaw_deg) < baseline
                closer["gain"] += abs(s["xy_scale"] - spec.xy_scale) < abs(t["xy_scale"] - spec.xy_scale)

    tasks = sorted(task_ids)
    assert not native and len(tasks) == 8 and cells == set(product(PANELS, tasks, SEEDS, labels))
    assert sum(subset.values()) == 96 and sign_counts["pooled"]["ood_pairs"] == 768
    rng = random.Random(BOOT_SEED)
    draws = [rng.choices(range(len(tasks)), k=len(tasks)) for _ in range(DRAWS)]
    metrics = {p: summarize([scores[p, t] for t in tasks], draws) for p in PANELS}
    pooled = [[row for p in PANELS for row in scores[p, t]] for t in tasks]
    metrics["pooled"] = summarize(pooled, draws)
    return dict(inputs=receipt, reader_seeds=SEEDS, task_clusters=tasks,
                bootstrap=dict(draws=DRAWS, seed=BOOT_SEED, ci="paired percentile, linear interpolation"),
                offline_ood=metrics, gripper_sign=sign_counts,
                archived_ur5e_pi05_compositional=dict(pairs=sum(subset.values()), outcomes=subset,
                                                      symmeta_closer_on_symmeta_only=closer))


if __name__ == "__main__":
    paths = list(map(Path, sys.argv[1:])) or INPUTS
    print(json.dumps(analyze(*paths), indent=2, allow_nan=False))
