"""Independently reconstruct endpoints, interpolation, calibration and pooled intervals from saved rows."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

import inputs as io

METHODS = ("single", "tt_mid", "tg_mid", "tt_cal", "tg_cal")


def load_rows(path):
    with gzip.open(path, "rt") as f:
        return list(csv.DictReader(f))


def vec(rows, k):
    return np.array([float(r[k]) for r in rows])


def angle(z):
    return np.rad2deg(np.angle(z))


def unit(y):
    return np.exp(1j*np.deg2rad(y))


def verify_rows(rows, endpoints, weights):
    seen = set()
    src = {(r["case"],r["seed"]):r for r in endpoints}
    assert len(src) == len(endpoints)
    for r in rows:
        i,j = int(r["i"]),int(r["j"])
        key = r["case"],i,j
        assert i != j and key not in seen
        seen.add(key)
        a,b = src[r["case"],i],src[r["case"],j]
        assert all(a[k]==b[k]==r[k] for k in ("task","panel","trace","shift","regime"))
        assert all(a[k]==float(r[k]) for k in ("start","t","gain","sign","y","g","s"))
        assert (b["t"],b["h"]) == (float(r["tt"]),float(r["tg"]))
        assert (str(a["tv"]),str(b["tv"]),str(b["hv"])) == tuple(r[k] for k in ("tv","ttv","tgv"))
    assert len(seen) == len(src)*2
    base = unit(vec(rows,"t"))
    arcs = []
    for k in ("tt","tg"):
        delta = angle(unit(vec(rows,k)) / base)
        diff = vec(rows,k)-vec(rows,"t")
        ok = np.array([r["tv"]=="True" and r[k+"v"]=="True" for r in rows]) & (diff % 360 != 180)
        arcs.append(np.where(ok,delta,0))
    out = np.stack([base, base*unit(arcs[0]/2), base*unit(arcs[1]/2),
                    base*unit(arcs[0]*weights[0]), base*unit(arcs[1]*weights[1])], axis=1)
    logged = np.stack([vec(rows,"yaw_"+m) for m in METHODS],axis=1)
    dev = float(abs(angle(out/unit(logged))).max())
    np.testing.assert_allclose(dev,0,atol=2e-12)
    return dev, arcs


def cal_test(rows, seal, arcs):
    # Direct evaluation of every breakpoint, independent of the rational slope sweep.
    result = {}
    for k,d in zip(("tt","tg"),arcs,strict=True):
        residual = angle(unit(vec(rows,"t")-vec(rows,"y")))
        nz = d != 0
        breaks = np.concatenate([((180*j-residual[nz])/d[nz]) for j in range(-2,3)])
        saved = [float(Fraction(r["weights"][k])) for r in seal["all_calibration_refits"]]
        cand = np.unique(np.r_[0.,.5,1.,breaks[(breaks>0)&(breaks<1)],saved])
        err = abs((residual[:,None]+d[:,None]*cand+180) % 360 - 180)
        group = np.array([r["task"] for r in rows])
        avg = np.array([err[group==t].mean(axis=0) for t in seal["task_order"]])
        gap = 0.
        for rec,w in zip(seal["all_calibration_refits"],saved,strict=True):
            losses = np.average(avg,axis=0,weights=rec["multiplicity"])
            gap = max(gap,float(losses[cand==w][0]-min(losses)))
        np.testing.assert_allclose(gap,0,atol=2e-12)
        result[k] = dict(distinct_breakpoint_candidates=len(cand),max_loss_above_optimum=gap)
    return result


def pooled(rows, arcs, fit, counts, task_ids, reported):
    selected = np.array([r["regime"]!="familiar" for r in rows])
    t,y,ratio = (vec(rows,"t")[selected],vec(rows,"y")[selected],(vec(rows,"g")/vec(rows,"gain"))[selected])
    task = np.array([r["task"] for r in rows])[selected]
    d,e = (x[selected] for x in arcs)
    tasks = sorted(set(task))
    groups = [task==name for name in tasks]
    order = [tuple(c) for c in counts]
    cal = {tuple(r["multiplicity"]):tuple(float(Fraction(r["weights"][k])) for k in ("tt","tg"))
           for r in fit["all_calibration_refits"]}
    # Evaluate methods directly as unit complex rotations, then average cases within sampled tasks.
    draws = {}
    for c,(w,v) in cal.items():
        z = unit(t-y)[:,None] * unit(np.stack((0*d,d/2,e/2,d*w,e*v),axis=1))
        err = np.stack([abs(angle(z)),abs(ratio[:,None]*z-1)],axis=2)
        delta = np.stack((err[:,2]-err[:,1],err[:,4]-err[:,3]),axis=1)
        draws[c] = np.stack([delta[g].mean(axis=0) for g in groups])
    frozen = draws[(1,1,1,1)]
    conditional = frozen[task_ids].mean(axis=1)
    joint = np.stack([draws[c][task_ids[n]].mean(axis=0) for n,c in enumerate(order)])
    max_diff = 0.
    for i,m in enumerate(("tg_mid_minus_tt_mid","tg_cal_minus_tt_cal")):
        for k,name in enumerate(("yaw","operator")):
            r = reported[name][m]
            observed = [frozen.mean(axis=0)[i,k],*np.percentile(conditional[:,i,k],[2.5,97.5]),
                        *np.percentile(joint[:,i,k],[2.5,97.5])]
            expected = [r["mean"],*r["conditional"],*r["joint"]]
            max_diff = max(max_diff,max(abs(np.subtract(observed,expected))))
    np.testing.assert_allclose(max_diff,0,atol=2e-12)
    return dict(draws=len(task_ids),max_mean_or_interval_difference=float(max_diff))


def previous(endpoints,role):
    def summary(rows):
        a,b,y = (np.asarray([r[k] for r in rows]) for k in ("t","h","y"))
        ratio = np.array([r["g"]/r["gain"] for r in rows])
        base = unit(a-y)
        mid = base*unit(angle(unit(b-a))/2)
        return [abs(angle(base)).mean(),abs(angle(mid)).mean(),abs(ratio*base-1).mean(),abs(ratio*mid-1).mean()]
    if role=="eval":
        hist = io.read_json(io.checked(io.SRC/"judge-evidence/score01/metrics.json",
                        "bb906c94f62e2173794ba57fe30498df2c3e1d5be60becb0466815f84d208796"))["test_only"]
        ref = [hist[k]["mean"] for k in ("mae_tuned","mae_midpoint","operator_tuned","operator_midpoint")]
        check = summary(endpoints)
        np.testing.assert_allclose(ref,check,atol=2e-12,rtol=0)
        return dict(tuned_yaw=check[0],old_same_seed_midpoint_yaw=check[1],values_checked=4)
    hist = io.read_json(io.OLD)["offline_ood"]
    sanity = io.read_json(io.checked(io.RETRY/"train02pro/training-sanity.json",
                          "9b5232526159a9395ff65cbbee39572a6c62578d642abd008452f4c8dc5368ff"))
    errors = []
    for panel,ref in hist.items():
        rows = [r for r in endpoints if (panel=="pooled" or r["panel"]==panel) and r["regime"]!="familiar"]
        vals = [ref[k][m] for k in ("yaw_mae_deg","mean_operator_error") for m in ("tuned","blend")]
        errors.extend(abs(np.array(summary(rows))-vals))
    for r in sanity["seeds"]:
        rows = [e for e in endpoints if e["panel"]=="gpu09test" and e["seed"]==r["seed"]]
        for arch,col in (("tuned","t"),("symmeta","h")):
            for subset,key in ((rows,"yaw_mae_deg"),([e for e in rows if e["regime"]!="familiar"],"yaw_mae_ood_deg")):
                mae = np.mean([abs(angle(unit(e[col]-e["y"]))) for e in subset])
                errors.append(abs(mae-r[arch]["exported"][key]))
    np.testing.assert_allclose(errors,0,atol=2e-12)
    return dict(values_checked=len(errors),max_difference=max(errors))


def raw_target():
    path = io.RETRY/"train02pro/predictions.json"
    export = io.read_json(path)["rows"]
    rawroot = io.RETRY/"native02g4090/pairs"
    for r in export:
        p = rawroot/r["panel"]/f"s{r['seed']}"/f"{r['index']:02d}.json"
        doc = io.read_json(p)
        pair = doc["pair"]
        canonical = (json.dumps(pair, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode()
        assert hashlib.sha256(canonical).hexdigest() == doc["sha256"]
        record = pair["row"]
        for k in r:
            assert record[k] == r[k], (p,k)
        for k in ("symmeta","tuned"):
            assert pair["arms"][k]["inferred"] == r[k]
    return dict(individual_pair_files=len(export),identical_factor_triples=2*len(export))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    args = p.parse_args()
    assert os.sched_getaffinity(0)=={26,27} and os.environ.get("CUDA_VISIBLE_DEVICES")==""
    start = time.monotonic()
    out = args.out
    m = io.read_json(out/"metrics.json")
    assert m["weights_sha256"] == io.digest(out/"weights.json")
    seal = io.read_json(out/"weights.json")
    w = tuple(seal["weights"][a]["weight"] for a in ("tt","tg"))
    rng = np.random.default_rng(20261124)
    ids = rng.integers(4,size=(20000,4))
    counts = [np.bincount(row,minlength=4) for row in ids]
    result = {}
    for role,get in (("cal",lambda:io.source("cal")),("eval",lambda:io.source("eval")),("target",io.target)):
        endpoints = get()
        rows = load_rows(out/(role+"-pairs.csv.gz"))
        dev,arcs = verify_rows(rows,endpoints,w)
        result[role] = dict(raw_rows=len(rows), independent_emitted_yaws=len(rows)*5,max_yaw_diff_deg=dev)
        if role == "cal":
            result[role]["calibration_loss_readback"] = cal_test(rows,seal,arcs)
        else:
            result[role]["previous_same_seed_summaries"] = previous(endpoints,role)
            result[role]["pooled_primary_intervals"] = pooled(rows,arcs,seal,counts,rng.integers(8,size=(20000,8)),
                                                              m[role]["groups"]["pooled"])
    result.update(native_export_readback=raw_target(), pin=m["pin"],inputs=io.CHECKED,
                  wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time())
    result["artifacts"] = {p.name:io.digest(p) for p in sorted(out.iterdir()) if p.suffix in (".gz",".json") and p.name!="readback.json"}
    io.write_json(out/"readback.json",result)
    print("Raw verification complete",io.digest(out/"readback.json"))


if __name__ == "__main__":
    main()
