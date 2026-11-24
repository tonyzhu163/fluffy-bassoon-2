"""Freeze public predictions to disk before joining them to offline native labels."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from common import save, sha
from reader import load_public, predict

METHODS = ("visual", "clock", "constant", "instant")


def freeze(model_path, folder, cases, output):
    model = json.loads(model_path.read_text())
    raw = {}
    for case in cases:
        name = folder/f"public-{case}.npz"
        t,x = load_public(name)
        predictions = predict(t,x,model)
        for stop in range(1,len(t)):
            replay = predict(t[:stop],x[:stop],model)
            assert all(np.allclose(replay[k],v[:stop],rtol=0,atol=1e-14) for k,v in predictions.items())
        raw[case] = dict(t=t.tolist(),public_sha256=sha(name),**{k:v.tolist() for k,v in predictions.items()})
    save(output,dict(model_sha256=sha(model_path),cases=raw))


def first(t, event):
    return int(t[np.flatnonzero(event)[0]]) if np.any(event) else None


def summarize(pred_path, folder, output):
    pred = json.loads(pred_path.read_text())["cases"]
    rows, cases = [], {}
    for case, record in pred.items():
        t = np.array(record["t"])
        truth = json.loads((folder/f"truth-{case}.json").read_text())
        if "y" in truth:
            y = np.array(truth["y"],bool)
            physical = truth["first"]
            q = truth["q"]
        else:
            drawer = np.array(truth["q"])
            ever = np.maximum.accumulate(drawer < -0.14)
            physical = first(np.arange(len(drawer)),ever)
            y,q = ever[t],drawer[t]
        assert np.array_equal(y,t >= physical) if physical is not None else not y.any()
        c = dict(sample_count=len(t), positives=int(y.sum()), first_open=physical,methods={})
        if "joint_alignment_rmse_rad" in truth:
            c.update(state_offset=truth["state_offset"],joint_alignment_rmse_rad=truth["joint_alignment_rmse_rad"])
        for m in METHODS:
            p = np.asarray(record[m])
            decision = p >= 0.5
            c["methods"][m] = dict(brier=float(np.mean((p-y)**2)), fp=int(np.sum(decision & ~y)),
                                   fn=int(np.sum(~decision & y)), first=first(t,decision))
        cases[case] = c
        for j,k in enumerate(t):
            rows.append(dict(case=case,t=int(k),drawer_q=float(q[j]),ever_open=int(y[j]),
                             **{m:record[m][j] for m in METHODS}))
    clusters = sorted({key.split("-")[0] for key in cases})
    cluster_scores = np.array([[np.mean([v["methods"][m]["brier"] for key,v in cases.items() if key.split("-")[0]==cl])
                                for m in METHODS] for cl in clusters])
    differences = cluster_scores[:,0] - cluster_scores[:,1]
    ix = np.random.default_rng(20261123).integers(len(clusters),size=(20000,len(clusters)))
    ci = np.quantile(differences[ix].mean(axis=1),[0.025,0.975]).tolist()
    out = dict(prediction_file_sha256=sha(pred_path),cases=cases,cluster_ids=clusters,
               mean_brier=dict(zip(METHODS,np.mean(cluster_scores,axis=0).tolist())),
               visual_minus_clock_by_cluster=differences.tolist(),paired_start_bootstrap_95=ci)
    # Compare all opposite-label pairs at identical elapsed time, within the same new layout for crossed commands.
    cells = {}
    for row in rows:
        group = row["case"].split("-")[0] if len(clusters) < len(cases) else "all"
        cells.setdefault((group,row["t"]),[]).append(row)
    rankings = []
    for members in cells.values():
        for a in members:
            for b in members:
                if a["ever_open"] and not b["ever_open"]:
                    rankings.append([float(a[m]>b[m])+0.5*float(a[m]==b[m]) for m in METHODS])
    out["time_matched_opposite_label_pairs"] = len(rankings)
    if rankings:
        out["time_matched_ranking"] = dict(zip(METHODS,np.mean(rankings,axis=0).tolist()))
    save(output.with_suffix(".json"),out)
    with gzip.open(output.with_suffix(".csv.gz"),"wt") as handle:
        writer = csv.DictWriter(handle,fieldnames=rows[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode",choices=["predict","score"])
    p.add_argument("input",type=Path)
    p.add_argument("directory",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("cases",nargs="*")
    a = p.parse_args()
    if a.mode == "predict":
        assert a.cases and not a.output.exists()
        freeze(a.input,a.directory,a.cases,a.output)
    else:
        summarize(a.input,a.directory,a.output)
