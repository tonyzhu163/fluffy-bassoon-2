"""Fit direct latent-attention readers on disjoint source tasks and seal before target evaluation."""
import itertools
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fluffy_bassoon.xemb.equiv_v2 import _apply_perceiver, _init_perceiver
from observations import (HERE, IDENTITY, SCALE, case_rows, digest, family_group, grouped, interval,
                          mean_report, metrics, ridge, save, train_views)

STEPS = 6000
LOOKS = (2000,4000,6000)
SEEDS = (41,42,43)
RIDGE = (.01,.1,1.,10.,100.)
GRID = list(itertools.product((64,96), (3e-4,1e-4), (False,True)))


def model(p,t):
    valid = jnp.ones(t.shape[:2],dtype=bool)
    return jnp.eye(2) + _apply_perceiver(p,t,valid,4).reshape(-1,2,2)


predict = jax.jit(model)


def restore(path):
    with path.open("rb") as f:
        return pickle.load(f)


def init(width,seed):
    p = _init_perceiver(jax.random.PRNGKey(seed), 100, 3, 8, width, 2, 4)
    p["head_w"] *= .02
    return p


def smooth_correct(q,a):
    # Differentiable damped inverse used only during training. Exact pseudoinverse at validation/test.
    trans = jnp.swapaxes(a,-1,-2)
    gram = trans@a + .05**2*jnp.eye(2)
    inv = jnp.linalg.solve(gram,trans)
    return jnp.clip(q @ jnp.swapaxes(inv,-1,-2), -1, 1)


def make_step(lr, with_command):
    sched = optax.warmup_cosine_decay_schedule(0.,lr,400,STEPS,end_value=.05*lr)
    opt = optax.chain(optax.clip_by_global_norm(1.),optax.adamw(sched,weight_decay=1e-4))
    @jax.jit
    def step(p,state,b,progress):
        def loss(params):
            est = model(params,b["tok"])
            err = jnp.mean(jnp.sum(jnp.square(est-b["a"]),axis=(-1,-2)))
            response = jnp.einsum("bij,bj->bi",b["x"],est.reshape(-1,4))-b["y"]
            objective = err + jnp.mean(jnp.square(response))
            if with_command:
                # Give the full map an informative initialization before inverse gradients matter.
                pred = smooth_correct(b["q"],est)
                goal = jnp.clip(b["q"] @ jnp.swapaxes(jnp.linalg.inv(b["a"]),-1,-2),-1,1)
                objective += progress*jnp.mean(jnp.sum((pred-goal)**2,axis=-1))
            return objective
        val,grad = jax.value_and_grad(loss)(p)
        updates,state = opt.update(grad,state,p)
        return optax.apply_updates(p,updates),state,val
    return step,opt


def bank(rows,prior,residual):
    # Each group first samples a base physical acquisition, then its original or a feasible view.
    rng = np.random.default_rng(24112431)
    arrays = {k:[] for k in ("tok","x","y","a","q")}
    offsets,counts,weights = [],[],[]
    for r in rows:
        offsets.append(len(arrays["tok"]))
        for tok,x,y,a in train_views(r,prior,residual,rng):
            for k,v in zip(arrays,(tok,x,y,a,r["q"][::5])):
                arrays[k].append(v)
        counts.append(len(arrays["tok"])-offsets[-1])
        assert counts[-1] >= 1
        # Four train tasks per suite and both source bodies have identical balanced designs.
        weights.append((1 if r["mode"]=="passive" else .5)/(3 if family_group(r["family"])=="general" else 1))
    return ({k:jax.device_put(np.asarray(v,np.float32)) for k,v in arrays.items()}, np.array(offsets),
            np.array(counts), np.array(weights)/sum(weights))


def sampled(data,rng):
    arrays,starts,counts,weights = data
    rows = rng.choice(len(starts),size=128,p=weights)
    aug = rng.integers(counts[rows])
    index = starts[rows] + np.where(rng.random(len(rows))<.5, 0, aug)
    return {k:v[index] for k,v in arrays.items()}


def observations(rows,assumed,method,seed,prior):
    result = []
    for r,a in zip(rows,assumed,strict=True):
        info = {k:r[k] for k in ("task","arm","family","mode","strategy","split","rank_ratio","effort","bound_hits","prefix_drift")}
        info.update(method=method,seed=seed,family_group=family_group(r["family"]),estimate=a.tolist(),
                    actual=r["a"].tolist(),held_chunks=len(r["hd"]), query_controls=len(r["q"]))
        result.append(dict(**info,**metrics(r,a,prior)))
    return result


def evaluate(p,rows,seed,prior):
    tokens = np.stack([r["tok"] for r in rows])
    estimated = np.asarray(predict(p,tokens))
    if not np.isfinite(estimated).all():
        raise ValueError("Nonfinite learned operators")
    return observations(rows,estimated,"direct",seed,prior)


def train_one(out,trainbank,val,prior,hp,seed,stop,select,tag):
    width,lr,with_command = hp
    p = init(width,seed)
    step,opt = make_step(lr,with_command)
    st = opt.init(p)
    rng = np.random.default_rng(seed*10007+31)
    best,records = None,[]
    for n in range(1,stop+1):
        p,st,loss = step(p,st,sampled(trainbank,rng),np.float32(np.clip((n-500)/500,0,1)))
        if n in LOOKS or n==stop:
            scored = evaluate(p,val,seed,prior)
            rec = dict(step=n,loss=float(loss),command=mean_report(scored,"command"),motion=mean_report(scored,"motion"),
                       operator=mean_report(scored,"map"), hp=list(hp),seed=seed)
            records.append(rec)
            print(tag,rec,flush=True)
            key = (rec["command"],rec["motion"],n)
            if (select and (best is None or key<best)) or (not select and n==stop):
                best = key
                with (out/f"{tag}.pkl").open("wb") as handle:
                    pickle.dump(jax.device_get(p),handle)
    save(out/f"{tag}-curve.json",records)
    return best


def baselines(rows,prior,residual,chosen):
    result = []
    for name,arr in (
        ("identity",[IDENTITY for r in rows]),("true_map",[r["a"] for r in rows]),
        ("goal_fit",[r["ideal"] for r in rows]),
        ("incumbent",[ridge(r["x"],r["y"],residual) for r in rows]),
        ("ridge",[ridge(r["x"],r["y"],residual,chosen) for r in rows])
    ):
        result.extend(observations(rows,arr,name,0,prior))
    return result


def other_law(folder,arm,train_tasks):
    with np.load(folder/f"law-{arm}.npz") as f:
        mask=np.isin(f["task"],list(train_tasks))
        u,d = f["u"][mask],f["d"][mask]
    lag=np.concatenate((np.zeros_like(u[:,:1]),u[:,:-1]),axis=1)
    x=np.concatenate((u,lag),axis=2).reshape(-1,6)
    y=d.reshape(-1,3)
    p,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
    assert rank==6
    return p,float(np.sum((y-x@p)**2)/np.sum(u**2))


def source_rows(folder,prior,residual,arms,include):
    rows,used=[],{}
    for arm in arms:
        for d in sorted(folder.glob(f"{arm}-libero*")):
            meta=json.loads((d/"case.json").read_text())
            assert meta["arm"] in ("Panda","Sawyer")
            if meta["split"] not in include:
                continue
            rows.extend(case_rows(d,prior,residual))
            for f in sorted(d.iterdir()):
                if f.is_file():
                    used[str(f)] = digest(f)
    return rows,used


def evidence(rows):
    # Argument contains source test only. Report cluster intervals for method levels and paired lateral comparisons.
    result={}
    for subset in ("all","general-lateral"):
        sub=[r for r in rows if subset=="all" or (r["family_group"]=="general" and r["strategy"]=="lateral")]
        for m in ("command","motion","map"):
            means={}
            for method in sorted({r["method"] for r in sub}):
                rs=[r for r in sub if r["method"]==method]
                tasks=sorted({r["task"] for r in rs})
                vals=np.array([mean_report([r for r in rs if r["task"]==t],m) for t in tasks])
                means[method]=vals
                result[f"{subset}/{m}/{method}"]=dict(mean=float(vals.mean()),ci95=interval(vals),
                                                       per_task=dict(zip(tasks,vals.tolist())))
            for baseline in ("identity","incumbent"):
                if baseline in means:
                    v=means["direct"]-means[baseline]
                    result[f"{subset}/{m}/direct-minus-{baseline}"]=dict(mean=float(v.mean()),ci95=interval(v))
    return result


def main():
    out,folder=map(Path,sys.argv[1:])
    assert jax.default_backend()=="gpu"
    arms=("Panda","Sawyer")
    rt=json.loads((folder/"runtime-Panda.json").read_text())
    prior,residual=np.array(rt["prior"]),rt["residual"]
    rows,used=source_rows(folder,prior,residual,arms,{"train","val"})
    train=[r for r in rows if r["split"]=="train"]
    val=[r for r in rows if r["split"]=="val"]
    assert len(train)==2*16*8*5 and len(val)==2*4*8*5
    data=bank(train,prior,residual)
    distribution=dict(physical=len(train),views=len(data[0]["a"]),minimum=int(data[2].min()),maximum=int(data[2].max()),
                      rank_quantiles=np.quantile([r["rank_ratio"] for r in train],[0,.25,.5,.75,1]).tolist())
    save(out/"source-bank.json",distribution)
    print(distribution,flush=True)
    candidates=[]
    for i,hp in enumerate(GRID):
        cmd,motion,n=train_one(out,data,val,prior,hp,41,STEPS,True,f"grid-{i}")
        candidates.append((cmd,motion,i,n))
    _,_,best,stop=min(candidates)
    hp=GRID[best]
    p41=out/f"grid-{best}.pkl"
    import shutil
    shutil.copyfile(p41,out/"seed-41.pkl")
    for seed in SEEDS[1:]:
        train_one(out,data,val,prior,hp,seed,stop,False,f"seed-{seed}")
    valcost=[]
    for mult in RIDGE:
        arr=[ridge(r["x"],r["y"],residual,mult) for r in val]
        scored=observations(val,arr,"ridge",0,prior)
        valcost.append((mean_report(scored,"command"),mean_report(scored,"motion"),mult))
    chosen=min(valcost)[2]
    selected=dict(hp=list(hp),steps=stop,grid=candidates,ridge_multiplier=chosen,ridge_validation=valcost,
                  prior=prior.tolist(),residual=residual,feature_width=100,seeds=SEEDS,
                  checkpoints={f"seed-{s}.pkl":digest(out/f"seed-{s}.pkl") for s in SEEDS})
    save(out/"selected.json",selected)
    print("source selection frozen",selected,flush=True)
    del data
    test,usedtest=source_rows(folder,prior,residual,arms,{"test"})
    assert len(test)==2*4*8*5 and not ({r["task"] for r in test}&{r["task"] for r in rows})
    scored=baselines(test,prior,residual,chosen)
    for seed in SEEDS:
        scored+=evaluate(restore(out/f"seed-{seed}.pkl"),test,seed,prior)
    save(out/"source-test-rows.json",scored)
    save(out/"source-test.json",evidence(scored))
    # Carry the selected recipe across source bodies. No checkpoint or law is trained on the diagnostic body.
    for arm,held in (arms,arms[::-1]):
        law,noise=other_law(folder,arm,{r["task"] for r in train})
        own,_=source_rows(folder,law,noise,[arm],{"train"})
        validation,_=source_rows(folder,law,noise,[arm],{"val"})
        other,_=source_rows(folder,law,noise,[held],{"test"})
        tag=f"from-{arm}"
        train_one(out,bank(own,law,noise),validation,law,hp,41,stop,False,tag)
        scores=baselines(other,law,noise,chosen)+evaluate(restore(out/f"{tag}.pkl"),other,41,law)
        save(out/f"{tag}-held-{held}.json",dict(law=law.tolist(),residual=noise,results=evidence(scores)))
    save(out/"source-files.json",dict(**used,**usedtest))
    save(out/"sealed.json",dict(selected=digest(out/"selected.json"),source=digest(out/"source-files.json"),
                               test=digest(out/"source-test.json"),bank=str(folder),checkpoints=selected["checkpoints"]))
    print("sealed",digest(out/"sealed.json"),flush=True)


if __name__=="__main__":
    main()
