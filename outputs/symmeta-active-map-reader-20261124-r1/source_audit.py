"""Descriptive physical source coverage and sampling variation in the shared law. No target data."""
import json
from pathlib import Path
import sys

import numpy as np

from observations import digest, save


def fit(x,y):
    a,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
    assert rank==6
    residual=np.square(y-x@a).sum()/np.square(x[:,:3]).sum()
    return a,residual


def main():
    bank,dest=map(Path,sys.argv[1:])
    commands,deltas,labels=[],[],[]
    source=["Panda","Sawyer"]
    for arm in source:
        with np.load(bank/f"law-{arm}.npz") as z:
            commands.extend(z["u"])
            deltas.extend(z["d"])
            labels.extend(z["task"])
    u,y=np.array(commands),np.array(deltas)
    x=np.concatenate((u,np.concatenate((np.zeros_like(u[:,:1]),u[:,:-1]),axis=1)),axis=-1)
    prior,noise=fit(x.reshape(-1,6),y.reshape(-1,3))
    rt=json.loads((bank/"runtime-Panda.json").read_text())
    np.testing.assert_allclose(prior,rt["prior"],atol=1e-14,rtol=0)
    np.testing.assert_allclose(noise,rt["residual"],atol=1e-16)
    groups=[np.where(np.asarray(labels)==t)[0] for t in sorted(set(labels))]
    rng=np.random.default_rng(24112488)
    coefs,scales=[],[]
    for ids in rng.integers(len(groups),size=(5000,len(groups))):
        i=np.concatenate([groups[t] for t in ids])
        a,k=fit(x[i].reshape(-1,6),y[i].reshape(-1,3))
        coefs.append(a[:,:2])
        scales.append(k)
    receipt=dict(bootstrap="5000 resamples of whole historical source task IDs; both bodies and all starts paired",
                 tasks=len(groups),nominal_supports=len(u),prior=prior.tolist(),residual=noise,
                 prior_xy_ci95=np.quantile(coefs,[.025,.975],axis=0).tolist(),
                 residual_ci95=np.quantile(scales,[.025,.975]).tolist())
    splits={}
    stats={}
    for arm in source:
        for folder in sorted(bank.glob(f"{arm}-libero*")):
            meta=json.loads((folder/"case.json").read_text())
            assert meta["complete"] and meta["arm"]==arm
            task=f'{meta["suite"]}/{meta["task"]}'
            split=meta["split"]
            splits.setdefault(split,set()).add(task)
            for name,desc in meta["maps"].items():
                a=np.array(desc["A"])
                for mode in ("passive","along+","along-","orth+","orth-"):
                    with np.load(folder/f"{name}-{mode}.npz") as z:
                        u=z["issued"][:,:2].astype(float)
                        d=np.diff(z["fk"][::5,:2],axis=0)
                    s=np.linalg.svd(u.reshape(3,5,2).mean(axis=1),compute_uv=False)
                    stats.setdefault((split,mode.rstrip("+-")),[]).append([s[1]/s[0] if s[0] else 0.,
                         np.square(u).sum(),(abs(u)==1).sum(),np.linalg.norm(d,axis=1).mean()*1000,
                         np.linalg.det(a),np.linalg.norm(a,np.inf)])
    receipt["splits"]={s:sorted(v) for s,v in splits.items()}
    receipt["coverage_columns"]=["weak_strong_ratio","XY_effort","issued_components_at_bound",
                                 "mean_chunk_motion_mm","determinant","max_row_sum"]
    receipt["coverage"]={"/".join(k):dict(attempts=len(v),mean=np.mean(v,axis=0).tolist(),
                 quantiles=np.quantile(v,[0,.25,.5,.75,1],axis=0).tolist()) for k,v in stats.items()}
    save(dest,receipt)


if __name__=="__main__":
    main()
