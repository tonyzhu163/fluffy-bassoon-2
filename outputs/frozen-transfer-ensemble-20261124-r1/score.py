"""Score the pre-registered frozen ensemble controls on the pinned Baige CPU deployment."""
import argparse
import csv
import gzip
import itertools
import os
import platform
import socket
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import inputs as io
from calibrate import knots, optimum, wrap

METHODS = ("single", "tt_mid", "tg_mid", "tt_cal", "tg_cal")
DIFFS = ((2,1), (4,3), (1,0), (2,0), (3,0), (4,0))
NAMES = METHODS + tuple(f"{METHODS[a]}_minus_{METHODS[b]}" for a,b in DIFFS)
B = 20000
SEED = 20261124


def pairs(endpoints):
    groups = defaultdict(dict)
    for r in endpoints:
        assert r["seed"] not in groups[r["case"]]
        groups[r["case"]][r["seed"]] = r
    assert len({next(iter(g.values()))["trace"] for g in groups.values()}) == len(groups)
    rows = []
    for case in sorted(groups):
        g = groups[case]
        assert set(g) == set(io.SEEDS)
        for i,j in itertools.permutations(io.SEEDS, 2):
            a,b = g[i],g[j]
            assert all(a[k] == b[k] for k in ("task","panel","trace","shift","regime","start","y","g","s"))
            assert np.isfinite(a["t"]) and np.isfinite(a["gain"]) and a["gain"] > 0 and a["sign"] in (-1,1)
            assert (not b["hv"] or np.isfinite(b["h"]))
            r = {k:v for k,v in a.items() if k not in ("seed","h","hv")}
            r.update(i=i, j=j, tt=b["t"], ttv=b["tv"], tg=b["h"], tgv=b["hv"])
            rows.append(r)
    return rows


def columns(rows):
    d = {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    for k in ("tt","tg"):
        arc = wrap(d[k] - d["t"])
        valid = d["tv"] & d[k+"v"] & np.isfinite(arc) & (abs(arc) != 180)
        d[k+"_arc"] = np.where(valid, arc, 0.)
        d[k+"_ok"] = valid
    return d


def predict(d, weights):
    base = d["t"]
    return np.stack([wrap(base), *(wrap(base + w*d[a+"_arc"]) for a,w in
                      zip(("tt","tg","tt","tg"), (.5,.5,*weights), strict=True))], axis=-1)


def outcomes(d, yaw):
    err = abs(wrap(yaw - d["y"][:, None]))
    ratio = (d["g"] / d["gain"])[:, None]
    op = np.hypot(ratio*np.cos(np.deg2rad(err))-1., ratio*np.sin(np.deg2rad(err)))
    return np.stack((err,op), axis=-1)


def write_rows(out, role, d, yaw):
    counts = len(yaw)
    fields = list(d) + ["yaw_"+s for s in METHODS]
    with gzip.open(out / (role+"-pairs.csv.gz"), "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(zip(*(d[k] for k in d), *yaw.T, strict=True))
    return dict(physical_cases=len(set(d["case"])), ordered_rows=counts,
                tasks=sorted(set(d["task"])), grip_accuracy=float(np.mean(d["s"]==d["sign"])),
                degeneracies={a:dict(fallback=int((~d[a+"_ok"]).sum()), equal_valid=int(
                    ((d[a+"_arc"] == 0) & d[a+"_ok"]).sum())) for a in ("tt","tg")})


def weights_fit(cal_rows):
    counts = sorted({tuple(c.count(i) for i in range(4)) for c in itertools.product(range(4), repeat=4)})
    result = {}
    for arch in ("tt","tg"):
        tasks, table = knots(cal_rows, arch)
        assert tuple(tasks) == io.SRC_TASKS["cal"]
        result[arch] = {c:optimum(table, c) for c in counts}
    return counts, result


def with_contrasts(scores):
    return np.concatenate((scores, np.stack([scores[...,a,:] - scores[...,b,:] for a,b in DIFFS], axis=-2)), axis=-2)


def task_draws(rng, n):
    ids = rng.integers(n, size=(B,n))
    return np.stack([(ids == i).sum(axis=1) for i in range(n)], axis=1)


def summarize(d, by_weight, masks, cal_index, test_counts, original):
    tasks = sorted(set(d["task"]))
    result = {}
    for label, mask in masks:
        idx = [np.flatnonzero(mask & (d["task"]==t)) for t in tasks]
        assert len({len(ids) for ids in idx}) == 1 and len(idx[0]) > 0
        table = with_contrasts(np.stack([by_weight[:,ids,:,:].mean(axis=1) for ids in idx], axis=1))
        point = table[original].mean(axis=0)
        conditional = np.einsum("bt,tmk->bmk", test_counts, table[original]) / len(tasks)
        joint = np.empty_like(conditional)
        for n in range(len(table)):
            take = cal_index == n
            joint[take] = np.einsum("bt,tmk->bmk", test_counts[take], table[n]) / len(tasks)
        bounds = [np.quantile(draw, [.025,.975],axis=0) for draw in (conditional,joint)]
        result[label] = dict(cases=len(set(d["case"][mask])), ordered_rows=int(mask.sum()),
            grip_accuracy=float(np.mean(d["s"][mask] == d["sign"][mask])),
            **{metric:{name:dict(mean=point[i,k], conditional=bounds[0][:,i,k].tolist(),
                                             joint=bounds[1][:,i,k].tolist()) for i,name in enumerate(NAMES)}
                         for k,metric in enumerate(("yaw","operator"))})
    return result


def screen(d, role, frozen, counts, ws, order, draws, out):
    original = counts.index((1,1,1,1))
    yaw = predict(d, frozen)
    coverage = write_rows(out, role, d, yaw)
    all_rows = np.ones(len(yaw), bool)
    selected = d["regime"] != "familiar" if role == "target" else all_rows
    masks = [("pooled",selected)]
    if role == "target":
        masks += [("familiar",~selected),("all_12_shifts",all_rows)]
        for v in sorted(set(d["regime"][selected])):
            masks.append(("regime/"+v, d["regime"] == v))
    for field in ("panel","i","j","shift"):
        for val in sorted(set(d[field][selected])):
            masks.append((f"{field}/{val}", selected & (d[field]==val)))
    for i,j in itertools.permutations(io.SEEDS,2):
        mask = selected & (d["i"]==i) & (d["j"]==j)
        masks.append((f"pair/{i}-{j}",mask))
    # Joint calibration uncertainty uses the exact optimizer for each of the 35 possible task counts.
    by_weight = np.stack([outcomes(d,predict(d,w)) for w in ws])
    scored = summarize(d, by_weight, masks, order, draws, original)
    cell_means = {}
    for task in sorted(set(d["task"])):
        mask = selected & (d["task"] == task)
        means = with_contrasts(by_weight[original,mask].mean(axis=0))
        cell_means[task] = {name:dict(yaw=means[i,0],operator=means[i,1]) for i,name in enumerate(NAMES)}
    panel_seed = {}
    for panel in sorted(set(d["panel"])):
        for i,j in itertools.permutations(io.SEEDS,2):
            keep = selected & (d["panel"]==panel) & (d["i"]==i) & (d["j"]==j)
            means = with_contrasts(by_weight[original,keep].mean(axis=0))
            panel_seed[f"{panel}/{i}-{j}"] = {n:dict(yaw=means[k,0],operator=means[k,1]) for k,n in enumerate(NAMES)}
    return dict(coverage=coverage, groups=scored, per_task=cell_means, per_panel_seed_pair=panel_seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pin", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    assert os.sched_getaffinity(0) == {26,27} and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=io.REPO,text=True).strip()==args.pin
    assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no","--",str(io.HERE)],cwd=io.REPO)
    args.out.mkdir(parents=True,exist_ok=False)
    cal = pairs(io.source("cal"))
    counts, fits = weights_fit(cal)
    ws = [tuple(float(fits[a][c][0]) for a in ("tt","tg")) for c in counts]
    unit = (1,1,1,1)
    frozen = ws[counts.index(unit)]
    weights = {a:dict(weight=float(fits[a][unit][0]), exact=str(fits[a][unit][0]), loss=float(fits[a][unit][1]))
               for a in ("tt","tg")}
    seal = dict(pin=args.pin, registered_plan_sha256=io.digest(io.HERE/"PLAN.md"), weights=weights,
                inputs=dict(io.CHECKED), task_order=io.SRC_TASKS["cal"],
                all_calibration_refits=[dict(multiplicity=c,weights={a:str(fits[a][c][0]) for a in ("tt","tg")}) for c in counts],
                frozen_utc=datetime.now(timezone.utc).isoformat())
    io.write_json(args.out/"weights.json",seal)
    print("Source-only weights frozen", io.digest(args.out/"weights.json"), weights, flush=True)
    cal_d = columns(cal)
    cal_cov = write_rows(args.out,"cal",cal_d,predict(cal_d,frozen))
    rng = np.random.default_rng(SEED)
    cal_counts = task_draws(rng,4)
    lookup = {c:k for k,c in enumerate(counts)}
    order = np.array([lookup[tuple(c)] for c in cal_counts])
    result = dict(pin=args.pin, host=socket.gethostname(), affinity=sorted(os.sched_getaffinity(0)),
                  python=platform.python_version(),numpy=np.__version__, weights=weights, cal_coverage=cal_cov,
                  weights_sha256=io.digest(args.out/"weights.json"),bootstrap=dict(draws=B,seed=SEED,
                  order=["cal","eval","target"], calibrated="joint cal/test by task, conditional test only"))
    for role,get in (("eval",lambda:io.source("eval")),("target",io.target)):
        d = columns(pairs(get()))
        result[role] = screen(d,role,frozen,counts,ws,order,task_draws(rng,8),args.out)
    result.update(inputs=io.CHECKED,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time())
    io.write_json(args.out/"metrics.json",result)


if __name__ == "__main__":
    main()
