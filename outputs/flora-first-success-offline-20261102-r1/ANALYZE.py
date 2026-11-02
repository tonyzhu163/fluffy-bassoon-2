"""Offline check of binary Flora success rules against simulator truth.

Reads only the original-prompt labels saved in the rubric replay (no scorer,
no simulator). Thresholds are chosen on CALIBRATION/OpenDrawer and applied
unchanged to TRAIN/OpenDrawer (the Flora-PPO episodes).
"""
import collections
import itertools
import json
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
episodes = json.loads((SRC / "PAIRED-EPISODES.json").read_text())


def features(label):
    p = {int(k): v for k, v in label["probabilities"].items()}
    return {
        "argmax5": float(label["score"] == 5),
        "P5": p[5],
        "P45": p[4] + p[5],
        "expected": label["expected_score"],
        "progress": label["progress"],
    }


def auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def confusion(rows, key, tau):
    tp = sum(r["y"] and r[key] >= tau for r in rows)
    fp = sum((not r["y"]) and r[key] >= tau for r in rows)
    pos = sum(r["y"] for r in rows)
    neg = len(rows) - pos
    return {
        "tau": tau,
        "tp": tp,
        "fp": fp,
        "positives": pos,
        "negatives": neg,
        "recall": tp / pos if pos else None,
        "false_positive_rate": fp / neg if neg else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def case_of(e):
    return e["native_id"].split("-rank-")[0]


rows = collections.defaultdict(list)
for e in episodes:
    for which in ("saved_original", "rerun_original"):
        r = features(e[which])
        r.update(y=e["truth"]["success"], id=e["id"], case=case_of(e),
                 instruction=e["native_instruction"], label=which)
        rows[(e["dataset"], e["task_key"], which)].append(r)

KEYS = ["argmax5", "P5", "P45", "expected", "progress"]
result = {"source": str(SRC.name), "banks": {}, "rules": {}, "train_clusters": {}}

for (ds, task, which), rs in sorted(rows.items()):
    pos = lambda k: [r[k] for r in rs if r["y"]]
    neg = lambda k: [r[k] for r in rs if not r["y"]]
    result["banks"][f"{ds}/{task}/{which}"] = {
        "n": len(rs),
        "successes": sum(r["y"] for r in rs),
        "auroc": {k: auroc(pos(k), neg(k)) for k in KEYS[1:]},
        "argmax5": confusion(rs, "argmax5", 1.0),
    }

# Thresholds fitted on CALIBRATION/OpenDrawer (saved labels): the smallest
# threshold whose false-positive rate is <= target, applied to TRAIN.
cal = rows[("CALIBRATION", "OpenDrawer", "saved_original")]
train = rows[("TRAIN", "OpenDrawer", "saved_original")]
train_rerun = rows[("TRAIN", "OpenDrawer", "rerun_original")]
for key in KEYS[1:]:
    grid = sorted({r[key] for r in cal + train})
    for target in (0.10, 0.05):
        ok = [t for t in grid if confusion(cal, key, t)["false_positive_rate"] <= target]
        tau = min(ok)
        result["rules"][f"{key}@calFPR<={target}"] = {
            "calibration": confusion(cal, key, tau),
            "train_saved": confusion(train, key, tau),
            "train_rerun": confusion(train_rerun, key, tau),
        }
    # Oracle on TRAIN for reference only: best achievable FPR at recall >= 0.8.
    feasible = [confusion(train, key, t) for t in grid]
    feasible = [c for c in feasible if c["recall"] >= 0.8]
    result["rules"][f"{key}@TRAIN-oracle-recall>=0.8"] = min(
        feasible, key=lambda c: c["false_positive_rate"])

# Within-case ranking on TRAIN: PPO advantages compare episodes from similar
# states, so pairwise accuracy inside a case matters more than pooled AUROC.
by_case = collections.defaultdict(list)
for r in train:
    by_case[r["case"]].append(r)
for key in KEYS[1:]:
    wins = total = 0
    for rs in by_case.values():
        for a, b in itertools.product(rs, rs):
            if a["y"] and not b["y"]:
                total += 1
                wins += (a[key] > b[key]) + 0.5 * (a[key] == b[key])
    result.setdefault("within_case_pairwise", {})[key] = {
        "pairs": total, "accuracy": wins / total if total else None}

# Do the argmax-5 false positives concentrate in particular cases/targets?
fp = [r for r in train if not r["y"] and r["argmax5"]]
fails = [r for r in train if not r["y"]]
case_fail = collections.Counter(r["case"] for r in fails)
case_fp = collections.Counter(r["case"] for r in fp)
case_succ = collections.Counter(r["case"] for r in train if r["y"])
result["train_clusters"] = {
    "false_positives": len(fp),
    "failures": len(fails),
    "cases": len(by_case),
    "cases_with_false_positive": len(case_fp),
    "top_cases": [
        {"case": c, "false_pos": n, "failures": case_fail[c],
         "successes": case_succ[c]}
        for c, n in case_fp.most_common(10)
    ],
    "by_instruction": {
        ins: {"false_pos": sum(r["instruction"] == ins for r in fp),
              "failures": sum(r["instruction"] == ins for r in fails)}
        for ins in sorted({r["instruction"] for r in train})
    },
    "false_positive_P5_quartiles": sorted(r["P5"] for r in fp)[::max(1, len(fp) // 4)],
    "true_positive_P5_quartiles": sorted(
        r["P5"] for r in train if r["y"] and r["argmax5"])[::40],
    "saved_vs_rerun_argmax5_flips": sum(
        a["argmax5"] != b["argmax5"] for a, b in zip(train, train_rerun)),
}

# Mean reward on successes minus failures, for the graded reward used by
# Flora-PPO and for binary rules.
P5_TAU = result["rules"]["P5@calFPR<=0.05"]["calibration"]["tau"]
mean = lambda xs: sum(xs) / len(xs)
rewards = {
    "graded_progress": lambda r: r["progress"],
    "argmax5": lambda r: r["argmax5"],
    f"P5>={P5_TAU}": lambda r: float(r["P5"] >= P5_TAU),
}
result["train_reward_gap"] = {
    name: {"success_mean": mean([fn(r) for r in train if r["y"]]),
           "failure_mean": mean([fn(r) for r in train if not r["y"]])}
    for name, fn in rewards.items()
}
for v in result["train_reward_gap"].values():
    v["gap"] = v["success_mean"] - v["failure_mean"]

# Failures that still pass the calibrated P5 rule: candidates for drawer
# ground truth (factorial plan Stage A).
result["confident_false_positives"] = sorted(
    ({"id": r["id"], "P5": r["P5"], "instruction": r["instruction"]}
     for r in train if not r["y"] and r["P5"] >= P5_TAU),
    key=lambda r: -r["P5"])

OUT.write_text(json.dumps(result, indent=1) + "\n")
