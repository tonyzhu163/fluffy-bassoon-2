"""Independent CPU check of the frozen assignments and eight-task paired uncertainty."""
import argparse
import ast
from collections import Counter
import cmath
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def require(ok, label):
    if not ok:
        raise ValueError(label)


def hashfile(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def exact_bootstrap(diffs):
    # Eight draws from eight tasks, compressed into 6,435 compositions.
    n = len(diffs)
    combos = itertools.combinations(range(2*n-1), n-1)
    weights, values = [], []
    for breaks in combos:
        counts = np.diff((-1, *breaks, 2*n-1)) - 1
        weights.append(math.factorial(n) // math.prod(math.factorial(int(c)) for c in counts))
        values.append(np.dot(counts, diffs) / n)
    order = np.argsort(values)
    cdf = np.cumsum(np.asarray(weights)[order])
    require(cdf[-1] == n**n, "Bootstrap mass differs")
    return [float(np.asarray(values)[order[np.searchsorted(cdf,p*n**n)]]) for p in (.025,.975)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--checkout",type=Path,required=True)
    p.add_argument("--export",type=Path,required=True)
    args=p.parse_args()
    run=args.run
    src=load(run/"source-rules.json")
    case=load(run/"target-cases.json")
    outcome=load(run/"soft-inner-summary.json")
    target_sha="c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b"
    require(hashfile(args.export)==outcome["predictions_sha256"]==target_sha,"Target export mismatch")
    require(hashfile(run/"source-rules.json")==outcome["source_rules_sha256"],"Source rules mismatch")
    n=0
    for path,sha in src["input_hashes"].items():
        f=Path(path)
        require(hashfile(f)==sha,path)
        if f.suffix==".npz":
            with np.load(f,allow_pickle=False) as d:
                require(d["valid"][:,:3].shape[1]==3,f.name)
                n+=len(d["valid"])
    require(n==src["source_trace_count"]==1466,"Wrong source membership")
    choices={}
    for r in src["rules"]:
        tab=r["source_scores"]
        require([q["threshold"] for q in tab]==[0,.05,.1,.15,.2,.3,.4,.5,"always"],"Wrong gate grid")
        for q in tab:
            np.testing.assert_allclose(q["mean"],(q["nominal"]+q["shifted"])/2,atol=1e-14,rtol=0)
        require(r["threshold"]==min(tab,key=lambda q:q["mean"])["threshold"],"Source argmin differs")
        choices[(r["reader"],r["seed"])]=r["threshold"]
    require(set(choices)==set(itertools.product(("symmeta","tuned"),(41,42,43))),"Reader identities")
    module=ast.parse((args.checkout/"src/fluffy_bassoon/xemb/embodiment.py").read_text())
    specs={}
    for node in module.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="EVAL_EMBODIMENTS" for t in node.targets):
            for call in node.value.elts:
                name,yaw,gain,grip,_=map(ast.literal_eval,call.args)
                specs[name]=(gain*cmath.exp(1j*math.radians(yaw)),grip)
    require(len(specs)==12,"Wrong native transforms")
    export={(r["seed"],r["index"]):r for r in load(args.export)["rows"] if r["panel"]=="gpu09test"}
    require(len(export)==288 and len(case)==576,"Wrong target count")
    keys=set()
    groups={}
    for r in case:
        k=(r["reader"],r["seed"],r["index"])
        require(k not in keys, "Duplicate case")
        keys.add(k)
        raw=export[k[1:]]
        for item in ("suite","task_id","embodiment","regime","tape_sha256"):
            require(r[item]==raw[item],item)
        truth,grip=specs[raw["embodiment"]]
        pred=raw[k[0]]
        estimate=pred["xy_scale"]*cmath.exp(1j*math.radians(pred["yaw_deg"]))
        threshold=choices[k[:2]]
        leave=threshold=="always" or abs(estimate-1)<threshold
        require(r["leave_alone"]==leave and r["grip_true"]==grip and r["grip_raw"]==pred["gripper_sign"] and
                r["grip_gated"]==(1 if leave else pred["gripper_sign"]),"Wrong decisions")
        errors=[abs(truth/estimate-1),abs(truth/(1 if leave else estimate)-1),abs(truth-1)]
        np.testing.assert_allclose(errors,[r[v] for v in ("raw_error","gated_error","identity_error")],
                                   atol=1e-11,rtol=1e-11)
        condition="nominal" if r["embodiment"]=="fam_nom" else "shifted"
        require(r["nominal"]==(condition=="nominal"),"Wrong stratum")
        groups.setdefault((k[0],condition),[]).append((r,errors))
    result={}
    for (reader,condition),rows in groups.items():
        reported=outcome["results"][reader][condition]
        arr=np.array([e for r,e in rows])
        required={k:np.mean(x) for k,x in zip(("raw_error","gated_error","identity_error"),arr.T)}
        required.update(assignments=len(rows),unique_tapes=len({r["tape_sha256"] for r,e in rows}),
            applications=sum(not r["leave_alone"] for r,e in rows),abstentions=sum(r["leave_alone"] for r,e in rows),
            raw_wrong_gripper=sum(r["grip_raw"]!=r["grip_true"] for r,e in rows),
            gated_wrong_gripper=sum(r["grip_gated"]!=r["grip_true"] for r,e in rows))
        for key,value in required.items():
            np.testing.assert_allclose(value,reported[key],atol=1e-11,rtol=1e-11)
        tasks=sorted({(r["suite"],r["task_id"]) for r,e in rows})
        require(len(tasks)==8,"Wrong number of tasks")
        diffs=np.array([np.mean([e[1]-e[0] for r,e in rows if (r["suite"],r["task_id"])==t]) for t in tasks])
        rng=np.random.default_rng(20261123)
        sampling=rng.integers(0,8,size=(10000,8))
        ci=np.percentile(diffs[sampling].mean(axis=1),[2.5,97.5])
        np.testing.assert_allclose(ci,reported["gated_minus_raw_ci95"],atol=1e-11,rtol=1e-11)
        exact=exact_bootstrap(diffs)
        by_shift={}
        for name in sorted({r["embodiment"] for r,e in rows}):
            mr=[r for r,e in rows if r["embodiment"]==name]
            by_shift[name]=dict(abstentions=sum(r["leave_alone"] for r in mr),
                raw_wrong_grip=sum(r["grip_raw"]!=r["grip_true"] for r in mr),
                gated_wrong_grip=sum(r["grip_gated"]!=r["grip_true"] for r in mr))
        result.setdefault(reader,{})[condition]=dict(gated_minus_raw_mean=float(np.mean(diffs)),
            task_bootstrap_exact_95=exact,by_shift=by_shift)
    print(json.dumps(dict(sources_rehashed=len(src["input_hashes"]),source_trace_count=n,
        source_argmins_checked=len(choices),case_count=len(keys),conditions_verified=4,
        support_count_per_reader=96,task_count=8,bootstrap_ordered_samples=8**8,results=result),indent=2))


if __name__=="__main__":
    main()
