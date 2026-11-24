"""Source-only fit, calibration, sensitivity, immutable source diagnostic and freeze."""
from collections import defaultdict
from pathlib import Path
import sys
import os

import numpy as np
from scipy.special import logsumexp, ndtr

from bank import BANK, HERE, LAW_HASH, balanced, cases, digest, group, key, save, source_law, template_bank
from mechanism import SHRINK, TEMPS, actions, qualify, rms

META = ("task","arm","family","mode","strategy","rank_ratio","receipt_fit_max_error")


def prediction(bank, r, logw, var, coverage=False):
    if not len(r["hy"]):
        return None, None
    d = r["hy"] - bank.flat @ r["hx"].T
    k = d.shape[1]
    ll = -.5 * (np.square(d).sum(axis=1)/var + k*np.log(2*np.pi*var))
    nll = float(-logsumexp(logw+ll)/k)
    if not coverage:
        return nll, None
    pit = np.exp(logw) @ ndtr(d / np.sqrt(var))
    return nll, pit.tolist()


def validation(bank, rows, basevar):
    valid = [r for r in rows if len(r["hy"])]
    weights = balanced(valid)
    scores = np.zeros(len(TEMPS))
    raw = []
    for r,w in zip(valid,weights,strict=True):
        ss = bank.evidence(r["x"],r["y"])
        values = []
        for T in TEMPS:
            var = basevar*T
            nll,_ = prediction(bank,r,bank.logweights(ss,var),var)
            values.append(nll)
        scores += w*np.array(values)
        raw.append(dict(**{k:r[k] for k in META}, weight=w, nll=values))
    return float(TEMPS[np.argmin(scores)]), dict(temps=TEMPS.tolist(), score=scores.tolist(),
            valid=len(valid), total=len(rows), coordinate_counts=[len(r["hy"]) for r in valid], rows=raw)


def shrink(bank, rows, var):
    subset = [r for r in rows if r["mode"]=="passive" and group(r["family"])=="general"]
    weights = balanced(subset)
    costs = np.zeros(len(SHRINK))
    raw = []
    for r, w in zip(subset,weights,strict=True):
        logw = bank.logweights(bank.evidence(r["x"],r["y"]),var)
        M,Q = bank.moments(np.exp(logw))
        values = [50*rms(actions(r["q"],M,Q,s,r["R"])["isotropic"] @ r["A"].T-r["q"]) for s in SHRINK]
        costs += w*np.array(values)
        raw.append(dict(**{k:r[k] for k in META},weight=w,setpoint_mm=values))
    return SHRINK[np.argmin(costs)],dict(multipliers=SHRINK,means=costs.tolist(),rows=raw)


def diagnostics(bank, rows, basevar, T):
    records=[]
    for r in rows:
        sse = bank.evidence(r["x"],r["y"])
        for name,factor in (("prior",1.),("half",.5),("selected",1.),("double",2.)):
            var = basevar * T * factor
            logw = bank.logprior if name=="prior" else bank.logweights(sse,var)
            nll, pit = prediction(bank,r,logw,var,True)
            coverage = None if pit is None else float(np.mean((np.array(pit)>=.05) & (np.array(pit)<=.95)))
            records.append(dict(**{k:r[k] for k in META}, variant=name, nll=nll, pit=pit,
                                coverage90=coverage, hand=int(np.sign(np.linalg.det(r["A"]))),
                                **bank.diagnose(np.exp(logw),r["A"])))
    summary={}
    for name in ("prior","half","selected","double"):
        for subset in ("all","general_passive","general_lateral"):
            sel=[r for r in records if r["variant"]==name and (subset=="all" or
                 (group(r["family"])=="general" and r["strategy"]==subset.split("_")[1]))]
            tab={}
            for metric in ("nll","coverage90","ess","top","true_hand","hand_brier","exact_mass",
                           "nearest_dist","nearest_mass","cov_trace","rank_ratio"):
                present=[r for r in sel if r[metric] is not None]
                if present:
                    w=balanced(present)
                    tab[metric]=float(w @ [r[metric] for r in present])
            tab.update(rows=len(sel), held_rows=sum(r["nll"] is not None for r in sel))
            summary[name+"/"+subset]=tab
    return records, summary


def main(out):
    assert os.sched_getaffinity(0)=={49} and os.environ["CUDA_VISIBLE_DEVICES"]==""
    save(out/"solver.json",qualify())
    manifest = {}
    law,residual = source_law(manifest)
    train = list(cases("train",law,residual,manifest))
    valid = list(cases("val",law,residual,manifest))
    groups = [set(r["task"] for r in rows) for rows in (train,valid)]
    assert len(train)==1280 and len(valid)==320 and len(groups[0])==16 and len(groups[1])==4 and not groups[0]&groups[1]
    bank,lineage = template_bank(train)
    basevar=float(balanced(train) @ [np.square(r["x"]@r["A"].ravel()-r["y"]).mean() for r in train])
    assert basevar>0 and np.isfinite(basevar)
    T, grid=validation(bank,valid,basevar)
    lam, scalar=shrink(bank,valid,basevar*T)
    save(out/"lineage.json",lineage)
    save(out/"validation-temperature.json",grid)
    save(out/"validation-shrinkage.json",scalar)
    np.savez_compressed(out/"templates.npz",maps=bank.maps,weights=bank.prior)
    # Test task IDs are opened only after final hyperparameters are persisted.
    choice=dict(T=T, basevar=basevar, isotropic_multiplier=lam, law=law.tolist(),
                residual_power=residual, law_hash=LAW_HASH, bank=str(BANK))
    save(out/"selection.json",choice)
    test=list(cases("test",law,residual,manifest))
    groups.append(set(r["task"] for r in test))
    assert len(test)==320 and len(groups[2])==4 and not (groups[0]|groups[1])&groups[2]
    raw, report=diagnostics(bank,test,basevar,T)
    save(out/"source-test-rows.json",raw)
    save(out/"source-test.json",report)
    save(out/"inputs.json",manifest)
    payloads=("solver.json","templates.npz","lineage.json","selection.json","validation-temperature.json",
              "validation-shrinkage.json","source-test-rows.json","source-test.json","inputs.json")
    save(out/"sealed.json",dict(files={f:digest(out/f) for f in payloads}, registration=digest(HERE/"PLAN.md"),
         pin=os.environ["UNCERTAIN_PIN"], signed_views=len(lineage), measured_parents=len(lineage)//8,
         templates=len(bank.maps), source_tasks=[sorted(g) for g in groups], **choice))
    print("SOURCE SEALED",digest(out/"sealed.json"),"templates",len(bank.maps),"T",T,"lambda",lam,flush=True)


if __name__=="__main__":
    main(Path(sys.argv[1]))
