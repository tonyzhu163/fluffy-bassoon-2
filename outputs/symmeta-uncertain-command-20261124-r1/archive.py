"""One frozen comparison on independently archived query tapes; projected setpoints only."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import os
import sys

import numpy as np

from bank import BANK, HERE, SIGNS, UP, cases, digest, group, key, read, save
from mechanism import Templates, actions, correct, outcomes, point, rms

METRICS=("setpoint","oracle_gap","issued_to_box","issued_to_clip","progress","desired","shortfall",
         "orthogonal","world_norm","effort","saturation","reverse_share")


def load_direct(split, receipts):
    up=UP/"gpu01"
    f=up/f"{split}-rows.json"
    if not f.is_file():
        return {}, f"Final {f} unavailable at the single comparison."
    seal=read(up/"sealed.json")
    assert digest(up/"source-test.json")==seal["test"] and digest(up/"selected.json")==seal["selected"]
    assert read(up/"source-test.json")["general-lateral/command/direct-minus-identity"]["mean"]<0
    for ck in ("seed-41.pkl","seed-42.pkl","seed-43.pkl"):
        assert digest(up/ck)==seal["checkpoints"][ck]
    for p in (f,up/"sealed.json",up/"source-test.json",up/"selected.json"):
        receipts[str(p)]=digest(p)
    result=defaultdict(dict)
    for r in read(f):
        if r["method"]=="direct":
            result[key(r)][r["seed"]]=np.array(r["estimate"],dtype=np.float32)
        if r["method"]=="incumbent":
            result[key(r)]["R"]=np.array(r["estimate"])
    assert len(result)==(200 if split=="fresh" else 160)
    assert all(set(v)=={"R",41,42,43} for v in result.values())
    return result,"Three final selected fits with frozen upstream estimators."


def summarize(records):
    # D's three separately scored seed rows first form one within-support mean.
    per={}
    for record in records:
        name=record["method"]
        pooled="D_"+name.split("_")[1] if name.startswith("D") else name
        target=per.setdefault((key(record),pooled),[])
        target.append(np.array([record[v] for v in METRICS]))
    per={k:np.mean(v,axis=0) for k,v in per.items()}
    methods=sorted({method for _,method in per})
    keys=sorted({k for k,_ in per})
    index=METRICS.index("setpoint")
    bootstrap=np.array(list(product(range(4),repeat=4)))
    result={}
    for fam in ["general","native","family","swap"]:
        for strategy,modes in (("passive",["passive"]),("along",["along+","along-"]),("lateral",["orth+","orth-"])):
            use=[k for k in keys if group(k[2])==fam and k[3] in modes]
            if not use:
                continue
            tab={}
            for method in methods:
                values=np.array([per[k,method] for k in use])
                risk=np.array([per[k,"risk"] for k in use])
                paired=values[:,index]-risk[:,index]
                def avg_part(indices):
                    return float(np.mean(paired[indices]))
                tasks=sorted({k[0] for k in use})
                assert len(tasks)==4
                diffs={t:avg_part([i for i,k in enumerate(use) if k[0]==t]) for t in tasks}
                x=np.array(list(diffs.values()))
                metrics=dict(zip(METRICS,values.mean(axis=0),strict=True))
                tab[method]=dict(metrics=metrics, comparator_minus_risk=float(x.mean()), tasks=diffs,
                    ci95=np.quantile(x[bootstrap].mean(axis=1),[.025,.975]).tolist(),
                    by_arm={a:avg_part([i for i,k in enumerate(use) if k[1]==a]) for a in sorted({k[1] for k in use})},
                    by_family={f:avg_part([i for i,k in enumerate(use) if k[2]==f]) for f in sorted({k[2] for k in use})},
                    by_pulse={m:avg_part([i for i,k in enumerate(use) if k[3]==m]) for m in modes})
            result[fam+"/"+strategy]=tab
    return result


def evaluate(out, split, bank, seal, direct, manifest):
    rows=[]
    inference=[]
    branch={}
    for r in cases(split,np.array(seal["law"]),seal["residual_power"],manifest):
        weights=np.exp(bank.logweights(bank.evidence(r["x"],r["y"]),seal["basevar"]*seal["T"]))
        M,Q=bank.moments(weights)
        q,A=r["q"],r["A"]
        D={}
        if direct:
            es=direct[key(r)]
            np.testing.assert_allclose(es["R"],r["R"],atol=1e-10,rtol=0)
            D={s:es[s] for s in (41,42,43)}
        choices=actions(q,M,Q,seal["isotropic_multiplier"],r["R"],D)
        oracle=point(A,q)
        clipped=correct(q,A)
        choices.update(oracle_box=oracle,oracle_clip=clipped)
        optimum=50*rms(oracle @ A.T-q)
        info={k:r[k] for k in ("task","arm","family","mode","strategy","rank_ratio","support_effort",
                                 "receipt_fit_max_error","max_receipt_error_m")}
        info.update(query_controls=len(q), hand=int(np.sign(np.linalg.det(A))))
        inference.append(dict(**info, **bank.diagnose(weights,A), R=r["R"].tolist(), second=Q.tolist(),
                              A=A.tolist(), D={s:mat.tolist() for s,mat in D.items()}))
        for method,u in choices.items():
            rows.append(dict(**info,method=method, **outcomes(u,q,A,oracle,clipped,optimum)))
        if split=="fresh" and r["mode"]=="passive" and r["family"] in SIGNS:
            branch["|".join(key(r))]={m:u[:5].tolist() for m,u in choices.items()}
    assert len(inference)==(200 if split=="fresh" else 160)
    result=summarize(rows)
    save(out/f"{split}-rows.json",rows)
    save(out/f"{split}-posterior.json",inference)
    save(out/f"{split}-report.json",result)
    if split=="fresh":
        save(out/"first-five-commands.json",branch)
    return result


def main(out, source, independent_hash):
    assert os.sched_getaffinity(0)=={49} and os.environ["CUDA_VISIBLE_DEVICES"]==""
    assert source.is_dir() and digest(source/"sealed.json")==independent_hash
    seal=read(source/"sealed.json")
    assert seal["registration"]==digest(HERE/"PLAN.md")
    for name,sha in seal["files"].items():
        assert digest(source/name)==sha
    with np.load(source/"templates.npz") as z:
        bank=Templates(z["maps"],z["weights"])
    manifest={}
    available={k:load_direct(k,manifest) for k in ("fresh","developmental")}
    notes={k:n for k,(_,n) in available.items()}
    for split in ("fresh","developmental"):
        result=evaluate(out,split,bank,seal,available[split][0],manifest)
        if split=="fresh":
            primary=result["general/passive"]
            condition=(primary["mean_box"]["comparator_minus_risk"]>0 and
                       primary["isotropic"]["comparator_minus_risk"]>0 and
                       primary["risk"]["metrics"]["progress"]>0)
    save(out/"inputs.json",manifest)
    save(out/"completion.json",dict(source_seal=independent_hash, D=notes,
         branch_condition=bool(condition),first_five_sha256=digest(out/"first-five-commands.json"),
         fresh=digest(out/"fresh-report.json"),developmental=digest(out/"developmental-report.json")))
    print("Completed single archived comparison.",notes, "physical branch eligible",condition,flush=True)


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
