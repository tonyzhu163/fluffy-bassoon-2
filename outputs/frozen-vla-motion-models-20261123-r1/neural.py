"""Test a conditional 4-unit neural residual at the same 15-control calibration budget."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import experiment as ex
from motion import fit_source, fit_target, ridge


def train(rows, seed):
    prior, _ = fit_source(rows)
    x, y = ex.query_panel(rows)
    x, y = x.reshape(-1, 6), y.reshape(-1, 3)
    scale = np.sqrt((x*x).mean(axis=0))
    z = x / scale
    centered = y - x @ prior
    yscale = np.sqrt((centered*centered).mean())
    target = centered / yscale
    rng = np.random.default_rng(seed)
    width = 4
    count = 6*width + width + width*3
    def unpack(p):
        return p[:24].reshape(6,width), p[24:28], p[28:].reshape(width,3)
    def objective(p):
        w, b, v = unpack(p)
        a, offset = np.tanh(z@w+b), np.tanh(b)
        h = a-offset
        err = h @ v - target
        dy = 2 / err.size * err
        dh = dy @ v.T
        da = dh * (1 - a*a)
        grad = np.concatenate(((z.T @ da).ravel(), (da-dh*(1-offset*offset)).sum(axis=0),
                               (h.T @ dy).ravel()))
        penalty = 1e-3
        return np.mean(err*err) + penalty*np.mean(p*p), grad+2*penalty/count*p
    init = rng.normal(0, .15, size=count)
    out = minimize(objective, init, jac=True, method="L-BFGS-B", options=dict(maxiter=5000, ftol=1e-8))
    ex.check(np.isfinite(out.fun) and out.fun < objective(init)[0], "Neural optimization failed")
    w,b,v = unpack(out.x)
    hidden = np.tanh(z@w+b)-np.tanh(b)
    projection = np.linalg.lstsq(x, hidden, rcond=None)[0]
    unexplained = hidden - x@projection
    gain = np.linalg.norm(x[:,:3]) / np.linalg.norm(unexplained)
    return dict(prior=prior, scale=scale, yscale=yscale, w=w, b=b, v=v, projection=projection,
                gain=gain, optimizer=dict(iterations=int(out.nit),objective=float(out.fun),
                                         converged=bool(out.success),message=str(out.message)))


def encode(x, model):
    h = np.tanh((x/model["scale"]) @ model["w"] + model["b"]) - np.tanh(model["b"])
    pred = x @ model["prior"] + model["yscale"] * (h @ model["v"])
    extra = model["gain"] * (h - x @ model["projection"])
    return pred, np.concatenate((x[...,:3], extra), axis=-1)


def evaluate(model, traces, strength):
    # Each unique task pair is weighted equally, across all source calibration/query restarts.
    tasks = sorted({r["task"] for r in traces})
    ids = np.array([tasks.index(r["task"]) for r in traces])
    x, y = ex.query_panel(traces)
    pred, features = encode(x, model)
    grids = {m:np.full((len(tasks),len(tasks)),np.nan) for m in ("linear","neural_source","neural_linear_head","neural_full")}
    values = {m:np.empty((len(traces),len(traces))) for m in grids}
    for i,support in enumerate(traces):
        current = fit_target("jacobian",model["prior"],support,strength)
        values["linear"][i] = ex.errors(current, x,y)
        values["neural_source"][i] = 1000*np.linalg.norm(pred-y,axis=-1).mean(axis=-1)
        for name, dims in (("neural_linear_head",3),("neural_full",features.shape[-1])):
            f = features[...,:dims]
            delta = ridge(f[i], y[i]-pred[i],strength)
            values[name][i] = 1000*np.linalg.norm(pred+f@delta-y,axis=-1).mean(axis=-1)
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            if i != j:
                for m,g in grids.items():
                    g[i,j] = values[m][np.ix_(ids==i,ids==j)].mean()
    return grids


def intervals(grid):
    from scipy.stats import t
    n = len(grid)
    omitted = np.array([np.nanmean(grid[np.ix_(np.arange(n)!=i,np.arange(n)!=i)]) for i in range(n)])
    se = np.sqrt((n-1)/n*np.sum((omitted-omitted.mean())**2))
    center = float(np.nanmean(grid))
    delta = float(t.ppf(.975,n-1)*se)
    return dict(mean_mm=center, interval_mm=[center-delta,center+delta],task_count=n)


def to_list(value):
    if isinstance(value,dict):
        return {k:to_list(v) for k,v in value.items()}
    if isinstance(value,np.ndarray):
        return value.tolist()
    return value


def support_spectrum(traces):
    s = np.linalg.svd(np.stack([r["inputs"][:,:3] for r in traces]),compute_uv=False)
    return dict(median_stable_rank=float(np.median((s*s).sum(axis=1)/s[:,0]**2)),
                median_singular_values=np.median(s,axis=0).tolist(),
                median_weakest_over_strongest=float(np.median(s[:,-1]/s[:,0])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--splits",type=Path,required=True)
    p.add_argument("--linear",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    fit_raw=args.linear.read_bytes()
    ex.check(sha256(fit_raw).hexdigest()=="acf6150ecd7937733d77516cabfaca4a7648531009e3d7c3264c025170279f73","Unexpected original source fit")
    strength=json.loads(fit_raw)["choice"]["jacobian"]
    splits = json.loads(args.splits.read_bytes())
    source = ex.source_traces(args.data_root,splits)
    groups = {arm:[r for r in source if r["arm"]==arm] for arm in ("Panda","Sawyer")}
    seeds = [41,42,43]
    trained, source_grids = {}, defaultdict(list)
    per_seed = defaultdict(dict)
    def record(label,seed,grids):
        source_grids[label].append(grids)
        per_seed[label][seed] = {m:float(np.nanmean(g)) for m,g in grids.items()}
    for seed in seeds:
        trained[seed] = {}
        for arm in ("Panda","Sawyer"):
            name = "Sawyer" if arm=="Panda" else "Panda"
            model = train(groups[arm],seed)
            trained[seed][arm]=model
            record(f"source_{name}",seed,evaluate(model,groups[name],strength))
        trained[seed]["pooled"] = train(source,seed)
    ex.dump(args.out/"models.json",to_list(trained))
    targets=ex.target_traces(args.data_root,splits)
    for arm, rows in targets.items():
        for seed in seeds:
            record(arm,seed,evaluate(trained[seed]["pooled"],rows,strength))
    results={}
    for name, entries in source_grids.items():
        grids={m:np.mean([g[m] for g in entries],axis=0) for m in entries[0]}
        results[name]=dict(errors={m:intervals(g) for m,g in grids.items()},
            neural_full_minus={m:intervals(grids["neural_full"]-g) for m,g in grids.items() if m!="neural_full"},
            seed_errors_mm=per_seed[name],spectrum=support_spectrum(targets[name] if name in targets else groups[name.removeprefix("source_")]))
        print(name, {m:round(np.nanmean(g),3) for m,g in grids.items()}, flush=True)
    ex.dump(args.out/"neural-summary.json",results)


if __name__=="__main__":
    main()
