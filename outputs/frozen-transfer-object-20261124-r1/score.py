"""Fit source-only contact readers and score whole-task paired effects on unseen arms."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from public import METHODS

PAIRS = ["task", "arm", "donor", "control"]
SCORES = list(METHODS)


def load_split(root, split):
    frames = []
    lineages = []
    for path in sorted((root/split).glob("*.lineage.json")):
        case = path.name.removesuffix(".lineage.json")
        lineage = json.loads(path.read_bytes())
        spec = lineage["spec"]
        task = f"{spec['suite']}/{spec['task_id']}"
        keys = dict(task=task, arm=spec["arm"], donor=spec["init_state_id"],
                    layout=lineage["layout"], original=lineage["original"], case=case)
        def load(suffix):
            return json.loads((path.parent/f"{case}.{suffix}.json").read_bytes())
        for x,y in zip(load("public"), load("truth"), strict=True):
            assert x["control"] == y["control"]
            frames.append(dict(**keys, **x, y=y["y"]))
        lineages.append(dict(**keys, controls=lineage["cpu_controls"], archive=lineage["source_controls"],
                             reward_equal=lineage["cpu_rewards"] == lineage["original_rewards"],
                             cpu_success=lineage["cpu_success"], source_success=lineage["original_success"],
                             censored=lineage["censored"], delta_tcp_mm=lineage["tcp_delta_mm"],
                             delta_jaw_mm=lineage["jaw_delta_mm"], seed=lineage["seed"],
                             input_path=lineage["input_path"],sha256=lineage["input_sha256"],
                             scene=lineage["reset"]["scene_fingerprint"],row=lineage["reset"]["released_row_fingerprint"]))
    expected = 9 if split=="source" else 44
    assert len(lineages) == expected and len({v["case"] for v in lineages}) == expected
    return pd.DataFrame(frames), lineages


def fit(df, cols):
    x = df[cols].to_numpy(dtype=float)
    y = df.y.to_numpy(dtype=float)
    w = 1 / df.groupby("case").y.transform("size").to_numpy()
    w *= len(w)/w.sum()
    mu = np.average(x, axis=0, weights=w)
    sd = np.sqrt(np.average((x-mu)**2, axis=0, weights=w))
    sd[sd < 1e-12] = 1.0
    prevalence = float(np.average(y,weights=w))
    params = dict(cols=cols,mean=mu.tolist(),scale=sd.tolist(),source_prevalence=prevalence,
                  source_windows=len(y),source_cases=sorted(set(df.case)))
    if len(np.unique(y))<2:
        return dict(**params,constant=prevalence)
    a = np.column_stack([np.ones(len(x)), (x-mu)/sd])
    def loss(theta):
        logit = a @ theta
        penalized = theta.copy()
        penalized[0] = 0
        value = w @ (np.logaddexp(0,logit) - y*logit) + .5*(penalized@penalized)
        return value, a.T @ (w*(expit(logit)-y)) + penalized
    result = minimize(loss,np.zeros(a.shape[1]), method="L-BFGS-B",jac=True,
                      options={"ftol":1e-13,"maxiter":2000,"gtol":1e-7})
    assert result.success, result.message
    return dict(**params,coef=result.x.tolist())


def predict(model,df):
    if "constant" in model:
        return np.full(len(df),model["constant"])
    x = (df[model["cols"]].to_numpy(dtype=float)-model["mean"])/model["scale"]
    # A BLAS matvec can change summation order with batch shape. Identical clock
    # inputs on the two physical branches must give exactly the same probability.
    logits = np.full(len(df),model["coef"][0],dtype=float)
    for j,coefficient in enumerate(model["coef"][1:]):
        logits += x[:,j] * coefficient
    return expit(logits)


def aggregate(df, variables, bout=False):
    # Equal weight from windows to bouts (sensitivity), tapes, arms within task, then tasks.
    keys = ["task","arm","donor"]
    if bout:
        df = df.groupby(keys+["bout"],as_index=False)[variables].mean()
    tapes = df.groupby(keys,as_index=False)[variables].mean()
    arms = tapes.groupby(["task","arm"],as_index=False)[variables].mean()
    return arms.groupby("task")[variables].mean(), arms


def summarize(df, variables, bout=False):
    if df.empty:
        return None
    tasks,arms = aggregate(df,variables,bout)
    x = tasks.to_numpy()
    rng = np.random.default_rng(1124)
    boot = x[rng.integers(len(x),size=(20000,len(x)))].mean(axis=1)
    return dict(n_task=len(tasks),by_task=tasks.to_dict(orient="index"),by_arm_task=arms.to_dict(orient="records"),
                mean=tasks.mean().to_dict(),cluster_95={v:list(np.quantile(boot[:,j],[.025,.975]))
                                                        for j,v in enumerate(variables)})


def paired_windows(scored):
    sides = [scored[scored.original==original][PAIRS+["y", "close_age"]+SCORES] for original in (True,False)]
    merged = sides[0].merge(sides[1],on=PAIRS,how="inner",suffixes=("_own","_new"),validate="one_to_one")
    assert (merged.close_age_own == merged.close_age_new).all()
    merged = merged[merged.y_own != merged.y_new].copy().sort_values(PAIRS)
    for m in SCORES:
        back = merged[m+"_own"] - merged[m+"_new"]
        sign = (2*merged.y_own-1)*back
        merged["rank_"+m] = (sign>0).astype(float)+.5*(sign==0)
        merged["loss_"+m] = .5*((merged[m+"_own"]-merged.y_own)**2+(merged[m+"_new"]-merged.y_new)**2)
    assert np.all(merged.rank_clock ==.5) and np.all(merged.clock_own==merged.clock_new)
    for m in SCORES[:-1]:
        merged["delta_"+m] = merged.loss_center-merged["loss_"+m]
    discontinuity = (merged.control.diff()!=5) | (merged.y_own.diff()!=0)
    for col in ("task","arm","donor"):
        discontinuity |= merged[col].ne(merged[col].shift())
    merged["bout"] = discontinuity.cumsum()
    return merged


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.item())+"\n")


def main(root):
    evidence = root/"analysis"
    evidence.mkdir(exist_ok=True)
    src, sources = load_split(root,"source")
    target, targets = load_split(root,"target")
    assert set(src.arm).isdisjoint(target.arm) and set(src.layout).isdisjoint(target.layout)
    assert set(x["scene"] for x in sources).isdisjoint(x["scene"] for x in targets)
    df = target[target.y.notna()].copy()
    models = {}
    for task in sorted(target.task.unique()):
        train = src[(src.task!=task) & src.y.notna()]
        assert len(train) and task not in set(train.task)
        models[task] = {}
        for method,cols in METHODS.items():
            model = fit(train,cols)
            models[task][method] = model
            mask = df.task==task
            df.loc[mask,method] = predict(model,df[mask])
    assert df[SCORES].notna().all().all()
    for method in SCORES:
        df["loss_"+method] = (df[method]-df.y)**2
    for method in SCORES[:-1]:
        df["delta_"+method] = df.loss_center - df["loss_"+method]
    pairs = paired_windows(df)
    loss = ["loss_"+m for m in SCORES]+["delta_"+m for m in SCORES[:-1]]
    pairvars = loss+["rank_"+m for m in SCORES]
    df.to_csv(evidence/"predictions.csv.gz",index=False)
    pairs.to_csv(evidence/"opposing.csv.gz",index=False)
    counts = pd.concat([src.assign(split="source"),target.assign(split="target")]).fillna({"y":-1})
    counts.groupby(["split","task","arm","case","y"]).size().rename("windows").reset_index().to_csv(evidence/"coverage.csv",index=False)
    result = dict(opposing_windows=len(pairs),opposing_tapes=len(pairs[PAIRS[:-1]].drop_duplicates()),
                  bouts=pairs.bout.nunique(),opposing=summarize(pairs,pairvars),by_bout=summarize(pairs,pairvars,True),
                  never_source_tasks=summarize(pairs[~pairs.task.isin(src.task.unique())],pairvars),
                  individual_arms={a:summarize(pairs[pairs.arm==a],pairvars) for a in sorted(target.arm.unique())},
                  original=summarize(df[df.original],loss),crossed=summarize(df[~df.original],loss))
    dump(evidence/"result.json",result)
    dump(evidence/"models.json",models)
    dump(evidence/"lineage.json",sources+targets)
    print("Complete",len(df),"eligible;",len(pairs),"opposite;",pairs.bout.nunique(),"bouts")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    main(parser.parse_args().root)
