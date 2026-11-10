"""Join the preselected confirmation panels and retain compact summaries in worker logs."""
import argparse
import json
from pathlib import Path

import numpy as np

from readout import TASKS, counts, contrasts, curve_readout, native_readout


def report(study, fixed, curves):
    lock = json.loads((study / "selected.json").read_text())
    rows = []
    fixed_scores = native_readout(fixed)
    for t in TASKS:
        for worker in range(3):
            ref = json.loads((study / f"{t}-{worker}/complete.json").read_text())
            assert ref["selection"] == lock and ref["bank"] == worker+1
            panels = {k:json.loads(Path(v).read_text()) for k,v in ref["sources"].items()}
            support = panels["support"]
            assert len({p["noise_sha256"] for k,p in panels.items() if k != "support"}) == 1
            for arm,p in panels.items():
                assert p["scene_xml"] == support["scene_xml"] and p["reset"]["physical_state_exact"] and p["reset"]["public_state_exact"]
                assert p["donor_lanes"] == support["donor_lanes"]
            for i, episode in enumerate(support["episodes"]):
                row = {"task":t,"bank":worker+1,"scene":episode["seed"]}
                for arm,p in panels.items():
                    e = p["episodes"][i]
                    assert e["seed"] == episode["seed"]
                    assert e["native_counts"]["controls"] - e["native_counts"]["reset_controls"] == 600
                    row[arm] = int(e["success_once"])
                rows.append(row)
    assert len({(r["task"],r["scene"]) for r in rows}) == len(rows) == 72
    original = {(r["task"],r["scene"]):r for r in fixed_scores["per_scene"]}
    selected = {}
    for t in (*TASKS,"pooled"):
        subset = [r for r in rows if t=="pooled" or r["task"]==t]
        failures = [r for r in subset if not r["support"]]
        selected[t] = {"all":counts(subset), "failed_support":counts(failures), "paired":contrasts(subset),
                       "paired_failed_support":contrasts(failures)}
        for a in ("own","memoryless"):
            delta = [r[a]-original[r["task"],r["scene"]][a] for r in subset]
            selected[t][f"{a}_selected_vs_two_epochs"] = dict(wins=delta.count(1), losses=delta.count(-1),
                                                               delta_pp=100*np.mean(delta))
    result = {"fixed":fixed_scores, "curves":curve_readout(curves), "selection":lock,
              "confirmed":{"summary":selected,"per_scene":rows}}
    (study / "all-results.json").write_text(json.dumps(result,indent=2))
    for t in (*TASKS,"pooled"):
        f = fixed_scores["summary"][t]
        print("FIXED",t,json.dumps({"all":f["all"],"failed":f["failed_support"],"memory":f["paired"]["own_minus_donor"]}),flush=True)
        s = selected[t]
        print("CONFIRMED",t,json.dumps({"all":s["all"],"failed":s["failed_support"],"memory":s["paired"]["own_minus_donor"]}),flush=True)
    print("FIXED_INTERVAL",fixed_scores["summary"]["pooled"]["own_minus_donor_bank_bootstrap95_pp"],flush=True)
    print("LOCK",json.dumps(lock),flush=True)
    for epoch in result["curves"]["mean_by_epoch"]:
        print("CURVE",json.dumps(epoch),flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    for key in ("study","fixed","curves"):
        ap.add_argument(key,type=Path)
    a=ap.parse_args()
    report(a.study,a.fixed,a.curves)
