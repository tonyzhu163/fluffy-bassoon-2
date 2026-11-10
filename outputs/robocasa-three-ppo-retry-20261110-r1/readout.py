"""Audit matched native attempts, and summarize the independent offline learning curves."""
import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ("OpenDrawer", "CloseToasterOvenDoor", "TurnOnSinkFaucet")
ARMS = ("support", "frozen", "memoryless", "own", "donor")


def read(path):
    return json.loads(path.read_text())


def collect(folder):
    rows = []
    seen = set()
    for task in TASKS:
        for bank in range(4):
            shard = folder / f"{task}-{bank}"
            out = shard / f"bank-{bank:02d}"
            complete = read(out / "complete.json")
            assert len(complete["seeds"]) == 8
            panels = {k:read(out / f"{k}.json") for k in ARMS}
            support = panels["support"]
            assert len(set(support["scene_xml"])) == 8 and seen.isdisjoint(support["scene_xml"])
            seen.update(support["scene_xml"])
            assert all(k == "support" or v["noise_sha256"] == panels["own"]["noise_sha256"] for k,v in panels.items())
            assert panels["own"]["noise_sha256"] != support["noise_sha256"]
            donors = panels["donor"]["donor_lanes"]
            assert sorted(donors) == list(range(8)) and all(i != d for i,d in enumerate(donors))
            for k,p in panels.items():
                assert len(p["chunks"]) == 12 and p["reset"]["physical_state_exact"] and p["reset"]["public_state_exact"]
                assert p["scene_xml"] == support["scene_xml"]
                assert [x["seed"] for x in p["episodes"]] == complete["seeds"]
            for i, seed in enumerate(complete["seeds"]):
                row = {"task":task, "bank":bank, "scene":seed, "donor_scene":complete["seeds"][donors[i]]}
                for arm, p in panels.items():
                    episode = p["episodes"][i]
                    audit = episode["native_counts"]
                    assert audit["controls"] - audit["reset_controls"] == 600
                    assert episode["success_once"] == (episode["first_success_control"] is not None)
                    row[arm] = int(episode["success_once"])
                rows.append(row)
    assert len(seen) == len(rows) == 96
    return rows


def counts(rows):
    return {"scenes":len(rows), **{arm:sum(r[arm] for r in rows) for arm in ARMS}}


def contrasts(rows):
    ans = {}
    for a,b in (("own","donor"), ("own","memoryless"), ("memoryless","frozen"), ("own","frozen")):
        vals = [r[a]-r[b] for r in rows]
        wins, losses = vals.count(1), vals.count(-1)
        ans[f"{a}_minus_{b}"] = {"wins":wins, "losses":losses, "ties":len(vals)-wins-losses,
                                  "delta_pp":100*np.mean(vals) if vals else None}
    return ans


def native_readout(folder):
    rows = collect(folder)
    summary = {}
    for task in (*TASKS,"pooled"):
        subset = [r for r in rows if r["task"] == task or task == "pooled"]
        failed = [r for r in subset if not r["support"]]
        summary[task] = {"all":counts(subset), "failed_support":counts(failed),
                         "paired":contrasts(subset), "paired_failed_support":contrasts(failed)}
    rng = np.random.default_rng(151)
    diff = np.array([[[r["own"]-r["donor"] for r in rows if r["task"] == t and r["bank"] == i]
                      for i in range(4)] for t in TASKS])
    boot = np.stack([task.mean(1)[rng.integers(4, size=(20000,4))].mean(1) for task in diff])
    summary["pooled"]["own_minus_donor_bank_bootstrap95_pp"] = (100*np.quantile(boot.mean(0),[.025,.975])).tolist()
    return {"per_scene":rows,"summary":summary,"episodes":5*len(rows),"controls":600*5*len(rows)}


def curve_readout(folder):
    seeds = [991, 1481, 2029]
    phases = [144, 288, 576, 1152, 2304]
    arms = ["frozen", "memoryless", "own", "same_task_donor"]
    records, keys = [], None
    for seed in seeds:
        for step in phases:
            raw = read(folder / f"seed-{seed}/step-{step:04d}/held-out.json")
            ids = [(r["task"],r["rank"],r["lane"],r["chunk"]) for r in raw]
            assert len(ids) == len(set(ids)) == 576
            if keys is None:
                keys = ids
            assert ids == keys
            row = {"seed":seed,"epoch":step/288}
            for arm in arms:
                values = [r[arm] for r in raw]
                assert np.isfinite(values).all()
                row[arm] = np.mean(values)
            records.append(row)
    means = [{"epoch":step/288, **{a:np.mean([r[a] for r in records if r["epoch"]==step/288]) for a in arms}}
             for step in phases]
    minima = {a:min(means, key=lambda row:row[a])["epoch"] for a in ("memoryless","own")}
    return {"records":records,"mean_by_epoch":means,"lowest_validation_error_epoch":minima}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind",choices=["native","curves"])
    ap.add_argument("directory",type=Path)
    args = ap.parse_args()
    print(json.dumps((native_readout if args.kind=="native" else curve_readout)(args.directory),indent=2))
