"""Evaluate an independently released bank with the primary's sealed models, no fitting."""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from bank import CASES_SHA, HERE, SOURCE_PIN, cases, sha
from camera import FULL, view_inputs
from data import SENSOR, read, write
from evaluate import decode, digest, get_embodiment_spec, infer, load_models
from replay import prepared
from fluffy_bassoon.xemb.equiv_v2 import predict_embodiment_v2
from model import CFG
from summarize import estimate

LEARNED = ("A", "B", "frozen_warp", "shipped")
HEADS = ("yaw_deg", "xy_scale", "gripper_sign")


def load_bank(path, expected_sha, size):
    assert sha(path) == expected_sha and path.name == f"bank{size}.json"
    b = read(path)
    expected = {c["index"]:c for c in cases() if size == 64 or (size == 1 and c["index"] == 8) or
                (size == 32 and c["slot"] >= 8)}
    assert b["cases_sha256"] == CASES_SHA and len(b["rows"]) == len(expected) == size
    assert set(expected) == {r["index"] for r in b["rows"]}
    for r in b["rows"]:
        c = expected[r["index"]]
        assert all(r[k] == c[k] for k in ("shard","suite","task_id","query_seed","query_layout","noise_seed","regime"))
        assert r["arm"] == "Kinova3" and r["role"] == "heldout" and r["source_collector_pin"] == SOURCE_PIN
        assert r["archive_sha256"] == c["support_sha256"] == sha(r["archive_npz"])
        assert r["input_physics"] == c["physics"] and r["seed"] == c["support_seed"] and r["init_state_id"] == 11
        assert r["native_control_id"] == c["embodiment"] and r["assigned_view"] == ("orbit+" if c["angle_deg"]>0 else "orbit-")
        diag = path.parent/r["id"]/"diagnostics.json"
        assert b["diagnostic_sha256"][r["id"]] == sha(diag)
        assert read(path.parent/r["id"]/"row.json") == r
        spec = get_embodiment_spec(c["embodiment"])
        assert (spec.yaw_deg,spec.xy_scale,spec.gripper_sign) == (r["control_yaw_deg"],r["xy_scale"],r["gripper_sign"])
        for key, angle in (("nominal",0),("orbit+",36),("orbit-",-36)):
            v = r["views"][key]
            assert v["angle_deg"] == angle and sha(v["passive_npz"]) == v["sha256"]
    return sorted(b["rows"],key=lambda r:r["index"])


def observation(r):
    # Open only the assigned shifted camera NPZ. Opposite calibration supplies a matrix only.
    name = r["assigned_view"]
    view = r["views"][name]
    with np.load(view["passive_npz"],allow_pickle=False) as z:
        rgb = np.concatenate([z["frame0_img"][None],z["img_after"]])
        np.testing.assert_array_equal(z["img_before"],rgb[:-1])
        controls = {k:z[k] for k in SENSOR}
    assert controls["valid"].all() and rgb.dtype == np.uint8
    h = np.asarray(view["H_ref_to_view"],dtype=np.float64)
    opp = r["views"]["orbit+" if name=="orbit-" else "orbit-"]
    return rgb, controls, h, np.asarray(opp["H_ref_to_view"],dtype=np.float64)


def batches(rows, model, calibration):
    feature_rows, receipts = [], []
    for r in rows:
        rgb, sensor, correct, opposite = observation(r)
        h = correct if calibration == "true" else opposite
        if model in ("A","B"):
            vis = view_inputs(rgb,FULL,h,model)
            data = dict(**vis, **sensor)
        else:
            frame = "A" if model == "frozen_warp" else "B"
            data, _, _ = prepared(rgb,FULL,h,sensor,frame)
        provenance = dict(index=r["index"], task=f"{r['suite']}/{r['task_id']}", suite=r["suite"], task_id=r["task_id"],
                          embodiment=r["native_control_id"], regime=r["regime"], view=r["assigned_view"],
                          angle_deg=r["views"][r["assigned_view"]]["angle_deg"], observed_sha256=digest(rgb),
                          source_sha256=r["archive_sha256"], native_view_sha256=r["views"][r["assigned_view"]]["sha256"],
                          calibration_sha256=digest(h), sensor_sha256={k:digest(v) for k,v in sensor.items()},
                          feature_sha256={k:digest(v) for k,v in data.items()}, model=model, calibration=calibration)
        feature_rows.append(data)
        receipts.append(provenance)
    return {k:np.stack([r[k] for r in feature_rows]) for k in feature_rows[0]}, receipts


def fixture(rows, out):
    models = {m:batches(rows,m,"true") for m in LEARNED}
    for x,y in (("A","frozen_warp"),("B","shipped")):
        for key in (*SENSOR,"img_before","img_after"):
            np.testing.assert_array_equal(models[x][0][key],models[y][0][key])
    entries = [models[m][1][0] for m in LEARNED]
    for e in entries:
        for key in ("observed_sha256","sensor_sha256","calibration_sha256","source_sha256","native_view_sha256"):
            assert e[key]==entries[0][key]
    write(out/"fixture-inputs.json",dict(scored=False,rows=entries))


def summarize(rows, truths):
    # Pair before averaging. Every task draw retains the full matched arms and all seeds.
    by = {(r["index"],r["seed"],r["model"],r["calibration"]):r for r in rows}
    assert len(by) == len(rows)
    groups = defaultdict(lambda:defaultdict(list))
    for r in rows:
        key, m, cal = (r["task"],r["seed"]), r["model"], r["calibration"]
        i, s = r["index"], r["seed"]
        truth = truths[i]
        def metrics(row):
            return dict(yaw_mae_deg=abs((row["yaw_deg"]-truth["yaw_deg"]+180)%360-180),
                        gain_mae=abs(row["xy_scale"]-truth["xy_scale"]),
                        gripper_accuracy=float(row["gripper_sign"]==truth["gripper_sign"]))
        comps = {f"{m}/{cal}":metrics(r)}
        rivals = []
        if cal=="true" and m in ("A","B"):
            rivals = [(n, "true") for n in LEARNED if n!=m]
        if cal=="opposite":
            rivals = [(m,"true")]
        for other, other_cal in rivals:
            ref = metrics(by[i,s,other,other_cal])
            comps[f"{m}/{cal} minus {other}/{other_cal}"] = {k:v-ref[k] for k,v in metrics(r).items()}
        for scope in ("all",r["regime"],str(r["angle_deg"])):
            for contrast, result in comps.items():
                for metric,value in result.items():
                    groups[scope,contrast,metric][key].append(value)
    output = {}
    for (scope,contrast,metric),v in groups.items():
        output.setdefault(scope,{}).setdefault(contrast,{})[metric] = estimate(v)
    return output


def main(args):
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    rows = load_bank(args.bank,args.bank_sha,args.size)
    if args.mode == "fixture":
        assert args.size == 1 and not args.fit
        fixture(rows,args.out)
        return
    assert args.size in (32,64) and sha(args.fit/"sealed.json") == args.seal_sha
    seal = read(args.fit/"sealed.json")
    assert seal["steps"] == int(min(seal["source_B_mean"],key=seal["source_B_mean"].get))
    assert set(seal["training_exposures"]) == {"41","42","43"}
    models = load_models(args.fit)
    rows_out = []
    truths = {r["index"]:dict(yaw_deg=r["control_yaw_deg"],xy_scale=r["xy_scale"],gripper_sign=r["gripper_sign"]) for r in rows}
    for m in LEARNED:
        for cal in (("true",) if m=="shipped" else ("true","opposite")):
            for start in range(0,len(rows),12):
                chunk = rows[start:start+12]
                batch, meta = batches(chunk,m,cal)
                # PR7's archived inference batch is 12. Pad final batch and discard padding.
                batch = {k:np.concatenate([v,np.repeat(v[-1:],12-len(v),axis=0)]) for k,v in batch.items()}
                for s in (41,42,43):
                    p, path, hs = models[m if m in ("A","B") else "shipped",s]
                    vals = decode(infer(p,batch)) if m in ("A","B") else predict_embodiment_v2(p,CFG,batch)
                    for j, entry in enumerate(meta):
                        y,g,t = (v[j].item() for v in vals)
                        assert np.isfinite(y) and np.isfinite(g) and g>0 and t in (-1,1)
                        rows_out.append(dict(entry,seed=s,checkpoint_path=path,checkpoint_sha256=hs,
                                             yaw_deg=y,xy_scale=g,gripper_sign=t))
    assert len(rows_out) == args.size*7*3
    records = {(r["index"],r["seed"],r["model"]):r for r in rows_out if r["calibration"]=="true"}
    for r in rows:
        for s in (41,42,43):
            arm_rows = [records[r["index"],s,m] for m in LEARNED]
            for field in ("observed_sha256","calibration_sha256","sensor_sha256","native_view_sha256","source_sha256"):
                assert all(a[field]==arm_rows[0][field] for a in arm_rows)
    pred = args.out/"predictions.json.gz"
    with gzip.open(pred,"wt") as file:
        json.dump(rows_out,file,separators=(",",":"),allow_nan=False)
    write(args.out/"offline.json",summarize(rows_out,truths))
    # The native runner accepts only assigned-view, seed-41, true-H triples plus the separate privileged map.
    sites = []
    for r in rows:
        corrections = {m:{k:records[r["index"],41,m][k] for k in HEADS} for m in LEARNED}
        corrections["true_map"] = truths[r["index"]]
        c = next(c for c in cases() if c["index"] == r["index"])
        sites.append(dict(c,corrections=corrections,calibration_physics=r["physics"], source_sha256=r["archive_sha256"],
                          observed_sha256=records[r["index"],41,"A"]["observed_sha256"],
                          calibration_sha256=records[r["index"],41,"A"]["calibration_sha256"]))
    checkpoint = {m:records[rows[0]["index"],41,m]["checkpoint_sha256"] for m in LEARNED}
    write(args.out/"full-head-release.json",dict(cases_sha256=CASES_SHA,bank=str(args.bank),bank_sha256=args.bank_sha,
          predictions=str(pred),predictions_sha256=sha(pred),fit=str(args.fit),seal_sha256=args.seal_sha,
          selected_step=seal["steps"],seed=41,checkpoint_sha256=checkpoint,rows=sites,source_fit_revision=seal["revision"],
          execution_revision=subprocess.check_output(["git","-C",HERE,"rev-parse","HEAD"],text=True).strip()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("fixture","release"))
    parser.add_argument("bank",type=Path)
    parser.add_argument("bank_sha")
    parser.add_argument("size",type=int,choices=(1,32,64))
    parser.add_argument("out",type=Path)
    parser.add_argument("--fit",type=Path)
    parser.add_argument("--seal-sha")
    main(parser.parse_args())
