"""Reconcile the two independent worker jobs and compute the registered task-paired endpoints once."""
import argparse
import gzip
import itertools
import zlib
from pathlib import Path

import numpy as np

from audit import inspect_episode, opening
from cohort import (ALONG, ESSENTIAL, ROBOTS, TASKS, assignments, atomic, checked, digest, encode, order, parse, require)

METHODS = ESSENTIAL+ALONG
PRIMARY = ("structured_lateral","direct_lateral")
CONTRASTS = [PRIMARY,("true","raw")]
for reader in ("structured","direct"):
    for reference in ("passive","along"):
        CONTRASTS.append((reader+"_lateral",reader+"_"+reference))
    CONTRASTS.extend([(reader+"_lateral","raw"),("true",reader+"_lateral")])
CONTRASTS.extend([("structured_passive","direct_passive"),("structured_along","direct_along")])


def interval(effects):
    if len(effects)!=4:
        return None
    x = np.array(effects)
    boot = np.array(list(itertools.product(range(4),repeat=4)))
    return np.quantile(x[boot].mean(axis=1),[.025,.975]).tolist()


def paired(rows, methods, coeffs):
    pairs = [r for r in rows if all(m in r["methods"] for m in methods)]
    def diff(r):
        return sum(c*int(r["methods"][m]["success"]) for c,m in zip(coeffs,methods))
    by_task = {}
    for suite,task in TASKS:
        group = [r for r in pairs if (r["suite"],r["task"])==(suite,task)]
        by_task[f"{suite}/{task}"] = dict(n=len(group),pp=100*np.mean([diff(r) for r in group]) if group else None)
    x = [v["pp"] for v in by_task.values() if v["n"]]
    out = dict(n=len(pairs),by_task=by_task,task_mean_pp=float(np.mean(x)) if x else None,
               descriptive_ci95_pp=interval(x),complete=len(pairs)==32)
    if len(methods)==2:
        wins = sum(diff(r)>0 for r in pairs)
        losses = sum(diff(r)<0 for r in pairs)
        ref_wins = sum(r["methods"][methods[1]]["success"] for r in pairs)
        out.update(rescues=wins,losses=losses,both_success=ref_wins-losses,both_fail=len(pairs)-ref_wins-wins,
                   comparator_failures=len(pairs)-ref_wins,comparator_successes=ref_wins)
    return out


def rates(rows):
    result = {}
    for m in METHODS:
        v = [r["methods"][m] for r in rows if m in r["methods"]]
        n = sum(b["controls"] for b in v)
        bounds = [sum(b["xy_bound_counts"][k] for b in v) for k in range(2)]
        result[m] = dict(successes=sum(int(b["success"]) for b in v),attempts=len(v),controls=n,
                         xy_bound_counts=bounds,xy_saturated_controls=sum(b["xy_saturated_controls"] for b in v),
                         mean_desired_to_received_xy_mm_equivalent=50*sum(b["total_desired_to_received_xy"] for b in v)/n if n else None)
    return result


def group_readback(arm, path, panel, sha, subset=None):
    rows = {r["index"]:r for r in panel["cases"] if r["arm"]==arm}
    result = {}
    for i,r in rows.items():
        if subset is not None and i!=subset:
            continue
        record = dict(next(c for c in assignments() if c["index"]==i),methods={})
        result[i] = record
        anchor = None
        record["integrity_errors"] = []
        read_errors = (OSError,EOFError,zlib.error,ValueError,TypeError,KeyError,IndexError,AssertionError)
        for group in (ESSENTIAL,ALONG):
            f = path/"groups"/f"{i:02d}-{group[0]}.json"
            if not f.exists():
                continue
            try:
                g = parse(f.read_bytes())
                require(g["index"]==i and g["panel_sha256"]==sha and
                        g["methods"]==[m for m in order(r,group) if m in r["estimates"]],"group identity")
                require(anchor is None or anchor==g["opening"],"different official group openings")
                anchor = g["opening"]
                for name in g["methods"]:
                    require((path/"episodes"/f"{i:02d}-{name}.json.gz").is_file(),"missing closed-group receipt "+name)
            except read_errors as e:
                record["integrity_errors"].append(dict(group=group[0],reason=str(e)))
        for name in METHODS:
            f = path/"episodes"/f"{i:02d}-{name}.json.gz"
            if not f.exists():
                continue
            try:
                raw = f.read_bytes()
                data = parse(gzip.decompress(raw))
                require(name in r["estimates"] and data["index"]==i and data["method"]==name and
                        data["panel_sha256"]==sha and data["cpu_fixture"] is False, "episode-panel join")
                ep = data["episode"]
                value = inspect_episode(r,name,ep)
                observed = opening(ep)
                require(anchor is None or anchor==observed,"unpaired physical and motor opening")
                anchor = observed
                record["methods"][name] = dict(value,sha256=digest(raw))
            except read_errors as e:
                record["integrity_errors"].append(dict(method=name,reason=str(e)))
        if anchor:
            record["opening_sha256"] = digest(encode(anchor))
            record["physics"] = anchor["reset"]["physics"]
    return result


def analyze(a):
    p = parse(checked(a.panel,a.sha))
    require(p["assignments"]==assignments(), "registration")
    if a.first is not None:
        arm = next(r["arm"] for r in assignments() if r["index"]==a.first)
        path = a.ur5e if arm=="UR5e" else a.iiwa
        r = group_readback(arm,path,p,a.sha,subset=a.first)[a.first]
        require(not r["integrity_errors"] and all(m in r["methods"] for m in ESSENTIAL), "first essential group not complete")
        atomic(a.out/"first-group.json",encode(dict(panel_sha256=a.sha,index=a.first,arm=arm,
            verified_control_count=sum(v["controls"] for v in r["methods"].values()),
            opening_sha256=r["opening_sha256"],tapes={k:v["sha256"] for k,v in r["methods"].items()})))
        return
    combined = {r["index"]:dict(r,methods={}) for r in assignments()}
    jobs = {}
    for arm,path in zip(ROBOTS,(a.ur5e,a.iiwa)):
        require((path/"exit.txt").exists() or a.budget_exhausted, "one final native look after both jobs exit")
        gpu = parse((path/"gpu.json").read_bytes())
        jobs[arm] = dict(host=gpu["hostname"],path=str(path),gpu_sha256=digest((path/"gpu.json").read_bytes()),
                         exit=(path/"exit.txt").read_text().strip() if (path/"exit.txt").exists() else None)
        combined.update(group_readback(arm,path,p,a.sha))
    require(jobs["UR5e"]["host"]!=jobs["IIWA"]["host"], "two independent official job IDs required")
    rows = list(combined.values())
    contrasts = {f"{m} minus {b}":paired(rows,(m,b),(1,-1)) for m,b in CONTRASTS}
    for base in ("passive","along"):
        names = ("structured_lateral","structured_"+base,"direct_lateral","direct_"+base)
        contrasts["reader interaction lateral minus "+base] = paired(rows,names,(1,-1,-1,1))
    cross = {}
    for key in ("arm","family","sign","layout","task_index"):
        cross[key] = {}
        for v in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key]==v]
            cross[key][str(v)] = dict(rates=rates(sub),primary=paired(sub,PRIMARY,(1,-1)))
    primary = contrasts[" minus ".join(PRIMARY)]
    native_errors = {r["index"]:r["integrity_errors"] for r in rows if r.get("integrity_errors")}
    qualified = (p["source_qualified"] and not p["quarantine"] and not native_errors and primary["complete"])
    summary = dict(panel_sha256=a.sha,jobs=jobs,cases=rows,methods=rates(rows),paired=contrasts,
                   source_qualified=p["source_qualified"],source_qualification_reason=p["unresolved_reason"],
                   architecture_comparison_qualified=qualified,native_integrity_errors=native_errors,quarantine=p["quarantine"],strata=cross)
    atomic(a.out/"native-results.json",encode(summary))
    atomic(a.out/"paired-cases.json",encode(rows))


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("panel",type=Path)
    cli.add_argument("sha")
    cli.add_argument("ur5e",type=Path)
    cli.add_argument("iiwa",type=Path)
    cli.add_argument("out",type=Path)
    cli.add_argument("--first",type=int,choices=range(32))
    cli.add_argument("--budget-exhausted",action="store_true")
    analyze(cli.parse_args())
