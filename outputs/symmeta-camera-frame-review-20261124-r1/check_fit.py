"""Independently reproduce the real fit exposure IDs, source selection and validation errors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_bytes())


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def replay(seed, n_support, poses):
    draws = np.random.default_rng(seed*10007+31)
    cameras = np.random.default_rng(seed*4099+217)
    prob = np.full(len(poses), .8/(len(poses)-1))
    prob[0] = .2
    counts = np.zeros((len(poses), n_support), dtype=int)
    digest = hashlib.sha256()
    digests = {}
    for step in range(1,5001):
        idx = draws.choice(n_support,32,replace=True)
        draws.multinomial(32,[1.0])
        gain = np.exp(draws.uniform(np.log(.65), np.log(1.5), 32))
        yaw = draws.uniform(-180.,180.,32)
        sign = draws.choice([-1,1],32)
        view = cameras.choice(len(poses),32,p=prob)
        np.add.at(counts,(view,idx),1)
        for value in (idx,view,yaw.astype("f4"),gain.astype("f4"),sign.astype("i4")):
            digest.update(value.tobytes())
        if step in (2500,5000):
            digests[step]=digest.hexdigest()
    return dict(digest=digests[5000], first_cycle_digest=digests[2500],
                n_draws=int(counts.sum()), examples_seen_by_view=(counts>0).sum(axis=1).tolist(),
                view_counts=counts.sum(axis=1).tolist())


def check(fit, base, extra):
    seal = load(fit/"sealed.json")
    source = load(base/"source.json")
    source_sha = file_sha(base/"source.json")
    assert source_sha == seal["source"] == load(extra/"ready.json")["source"] == load(base/"ready.json")["source"]
    assert len(source["training"])==1466 and {x["arm"] for x in source["training"]}=={"Panda","Sawyer"}
    stored = load(fit/"source-curves.json")
    poses = seal["validation_poses"]
    source_tasks=np.array(source["val_task"])
    verified={}
    exposures={}
    for s in (41,42,43):
        exposures[s] = replay(s,1466,seal["source_poses"])
        assert exposures[s]["digest"] == seal["training_exposures"][str(s)]
        for arm in ("A","B"):
            for step in (1000,2000,2500,3500,4500,5000):
                row, = [r for r in stored if (r["seed"],r["arm"],r["steps"]) == (s,arm,step)]
                samples=np.load(fit/f"val-{arm}-{s}-{step}.npy",allow_pickle=False)
                assert samples.shape==(len(poses),len(source_tasks)) and np.isfinite(samples).all()
                assert (samples>=0).all() and (samples<=180).all()
                np.testing.assert_allclose(samples.mean(1),[p["mae"] for p in row["poses"]])
                np.testing.assert_allclose(samples.mean(),row["mae"])
                assert [p["pose"] for p in row["poses"]]==poses
                verified[arm,s,step]=samples.astype(float)
                if step in (2500,5000):
                    name=f"{arm}-seed{s}-step{step}.msgpack"
                    assert file_sha(fit/name)==seal["checkpoint_sha256"][name]
    by_budget={s:np.mean([verified["B",i,s].mean() for i in (41,42,43)]) for s in (2500,5000)}
    assert int(min(by_budget,key=by_budget.get))==seal["steps"]
    result={}
    for arm in ("A","B"):
        result[arm]={}
        for s in (1000,2000,2500,3500,4500,5000):
            mean=np.mean([verified[arm,k,s] for k in (41,42,43)],axis=0)
            result[arm][s]=dict(mae=float(mean.mean()), by_view=mean.mean(1).tolist(),
                                per_task={t:float(mean[:,source_tasks==t].mean()) for t in set(source_tasks)})
    return dict(seal_sha256=file_sha(fit/"sealed.json"),selection=int(seal["steps"]),B_budget_means=by_budget,
                poses=poses, source_ids=len(source_tasks), params=seal["params_per_arm"], readback=result,
                exposure=exposures, checkpoints_verified=len(seal["checkpoint_sha256"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for field in ("fit","base","extra"):
        parser.add_argument(field,type=Path)
    a=parser.parse_args()
    print(json.dumps(check(a.fit,a.base,a.extra),indent=2))
