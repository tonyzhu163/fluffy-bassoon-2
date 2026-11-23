"""Read-only panel accounting and post-rollout pairing against both archived controls."""
import argparse
import csv
import json

from common import BUDGETS, SOURCE_STORE, WORK, configure, ref
from runner import Inputs, panel_jobs


def read_rows(d):
    with (d / "results.csv").open() as f:
        return {(r["task"],int(r["init_id"]),int(r["rep"])):r for r in csv.DictReader(f)}


def collect(seed, panel, budget):
    import numpy as np
    from reground.envs.libero_env import max_steps
    from reground.eval.jobs import load_traces
    inputs = Inputs(seed, budget or 15)
    low = None
    if budget is not None:
        root = WORK / "production" / f"seed{seed}" / f"b{budget}" / "eval"
        matches = [p for p in root.glob("*/" + panel) if p.is_dir()]
        ref.require(len(matches) == 1, f"Expected one evaluation version for {root}/{panel}")
        low, = matches
    rows, seconds, horizon_seconds = [], 0, 0
    for job in panel_jobs(inputs, panel):
        shard = f"shard{job['shard']:03d}"
        root = SOURCE_STORE / "shards" / f"seed{seed}" / "rot_z/90"
        dirs = {m:root / m / shard for m in ("base","reground")}
        if low:
            dirs["low"] = low / shard
            ref.check_eval(dirs["low"],job)
        tables = {m:read_rows(d) for m,d in dirs.items()}
        traces = {m:load_traces(d) for m,d in dirs.items()} if low else {}
        keys = [tuple(e) for e in job["episodes"]]
        for key in keys:
            control = tables["reground"][key]
            seconds += float(control["wall_s"])
            row = dict(seed=seed,task=key[0],start=key[1],rep=key[2])
            for method,table in tables.items():
                rr = table[key]
                for col in ("policy","task","suite","shift","level","adapt_seed","split","init_id","rep","noise_seed"):
                    ref.require(control[col] == rr[col], (key, col))
                for metric in ("success","unsafe","steps"):
                    row[method + "_" + metric] = int(rr[metric])
                if low and method!="low":
                    a,b=traces["low"][key],traces[method][key]
                    n=min(10,len(a["A_vla"]),len(b["A_vla"]))
                    row[method+"_opening_exact"] = bool(np.array_equal(a["X"][0],b["X"][0])
                        and np.array_equal(a["A_vla"][:n],b["A_vla"][:n]))
            rows.append(row)
        for i in range(0,len(keys),4):
            batch = keys[i:i+4]
            # Measured simulator/policy/controller batch time scaled to maximum horizon
            # at the same rate. This is a scenario estimate, not a runtime bound.
            sec = sum(float(tables["reground"][key]["wall_s"]) for key in batch)
            horizon_seconds += sec*max(max_steps(key[0]) for key in batch)/max(int(tables["reground"][key]["steps"]) for key in batch)
    result = dict(seed=seed,panel=panel,budget=budget,episodes=len(rows),reference_rollout_gpu_s=seconds,
                  horizon_scaled_rollout_gpu_s=horizon_seconds,
                  counts={k:sum(r[k] for r in rows) for k in rows[0] if k.endswith(("_success","_unsafe","_steps"))})
    if low:
        for m in ("base","reground"):
            result[m+"_opening_mismatches"] = sum(not r[m+"_opening_exact"] for r in rows)
    return dict(summary=result, rows=rows)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,choices=ref.SEEDS,required=True)
    p.add_argument("--panel",choices=["panel32","full"],default="panel32")
    p.add_argument("--budget",choices=BUDGETS,type=int,help="omit for archived controls alone")
    args=p.parse_args()
    configure()
    print(json.dumps(collect(args.seed,args.panel,args.budget),allow_nan=False))
