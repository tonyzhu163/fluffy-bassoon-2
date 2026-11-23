"""Switch only the 15-control support to a fixed, identical active probe; keep passive queries."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

import experiment as ex
from motion import fit_source, fit_target, correction_matrix

PROBE = np.array([[.62,0,0,0,0,0,1],[0,.62,0,0,0,0,-1],[-.35,-.35,.45,0,0,0,1]], dtype=np.float32)
UR_PANEL_SHA = "51a4575ff3105b0e7005befb598f94d8bf69ed0c068324c8833a92b11cfd05d0"


def active(row):
    old=Path(row["support"]["path"])
    path=old.with_name(old.name.replace("passive","active"))
    with np.load(path,allow_pickle=False) as tape:
        np.testing.assert_allclose(tape["issued_mean"][tape["valid"]],PROBE,rtol=1e-6,atol=1e-7)
    with np.load(path,allow_pickle=False) as a, np.load(old,allow_pickle=False) as p:
        np.testing.assert_array_equal(a["frame0_img"],p["frame0_img"])
        np.testing.assert_array_equal(a["frame0_state"],p["frame0_state"])
    ident={k:v for k,v in row["support"].items() if k not in ("path","sha256")}
    return ex.get_trace(path,**ident)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--splits",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--linear-scores",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    splits=json.loads(args.splits.read_bytes())
    source=ex.source_traces(args.data_root,splits)
    source_active=[active(s) for s in source]
    costs={m:np.zeros(len(ex.STRENGTHS)) for m in ("jacobian","planar")}
    for arm in ("Panda","Sawyer"):
        src=[s for s in source if s["arm"]!=arm]
        pseudo=[s for s in source if s["arm"]==arm]
        xx,yy=ex.query_panel(pseudo)
        prior,_=fit_source(src)
        for support in (s for s in source_active if s["arm"]==arm):
            keep=np.array([s["task"]!=support["task"] for s in pseudo])
            for m,cost in costs.items():
                for j,lam in enumerate(ex.STRENGTHS):
                    coef=fit_target(m,prior,support,lam)
                    cost[j]+=ex.errors(coef,xx[keep],yy[keep]).mean()/len(source)
    strength={m:ex.STRENGTHS[np.argmin(cost)] for m,cost in costs.items()}
    ex.dump(args.out/"active-source.json",dict(strength=strength,cv={m:c.tolist() for m,c in costs.items()},
            grid=ex.STRENGTHS,tapes=[r["support"] for r in source_active]))
    targets=ex.target_traces(args.data_root,splits)
    panel=(args.data_root/"symmeta-fk-ur5e-20261123-r1/gpu01/panel.json").read_bytes()
    ex.check(sha256(panel).hexdigest()==UR_PANEL_SHA,"Wrong UR5e reference")
    refs=json.loads(panel)["rows"][::12]
    for row,ref in zip(targets["UR5e"],refs,strict=True):
        s=row["support"]
        ex.check(f'{ref["suite"]}/{ref["task_id"]}'==row["task"] and ref["embodiment"]=="fam_nom" and
                 (s["sha256"],s["physics"],s["seed"],s["path"])==
                 tuple(ref["support"][k] for k in ("sha256","physics","seed","path")),s["task"])
    active_targets={arm:[active(r) for r in rows] for arm,rows in targets.items()}
    prior,noise=fit_source(source)
    historical=json.loads(args.linear_scores.read_bytes())
    ex.check(len(historical)==168,"Expected 168 archived passive pairs")
    passive={(r["arm"],r["support"],r["query"]):r["errors_mm"]["jacobian"] for r in historical}
    results,matrices={},{}
    for arm,target in targets.items():
        inputs,response=ex.query_panel(target)
        grids={m:np.full((8,8),np.nan) for m in ("active_jacobian","active_planar","active_wrong_arm","passive_jacobian","source")}
        for i,row in enumerate(active_targets[arm]):
            index=f'{arm}/{row["task"]}'
            matrices[index]=dict(support=row["support"])
            for m,lam in strength.items():
                coef=fit_target(m,prior,row,lam)
                matrices[index]["matrix" if m=="jacobian" else "planar_matrix"]=correction_matrix(prior,coef,noise).tolist()
                grids["active_"+m][i]=ex.errors(coef,inputs,response)
            wrong=[ex.errors(fit_target("jacobian",prior,rows[i],strength["jacobian"]),inputs,response)
                   for a,rows in active_targets.items() if a!=arm]
            grids["active_wrong_arm"][i]=np.mean(wrong,axis=0)
            grids["source"][i]=ex.errors(prior,inputs,response)
            for j,q in enumerate(target):
                if i!=j:
                    grids["passive_jacobian"][i,j]=passive[(arm,row["task"],q["task"])]
            for g in grids.values():
                g[i,i]=np.nan
        results[arm]=dict(errors={m:ex.summary(g) for m,g in grids.items()},
            active_jacobian_minus={m:ex.summary(grids["active_jacobian"]-g) for m,g in grids.items() if m!="active_jacobian"})
        print(arm, {m:round(np.nanmean(g),3) for m,g in grids.items()},flush=True)
    ex.dump(args.out/"active-matrices.json",matrices)
    ex.dump(args.out/"active-summary.json",results)
    ex.dump(args.out/"readback.json",dict(ur5e_panel_sha256=UR_PANEL_SHA,ur5e_task_hashes_verified=len(refs),
        source_active_and_passive_reset_pairs=len(source_active),target_active_and_passive_reset_pairs=24,
        commands_identical_across_all_216_probes=True,probe=PROBE.tolist(),
        probe_stable_rank=float((PROBE[:,:3]**2).sum()/np.linalg.norm(PROBE[:,:3],ord=2)**2)))


if __name__=="__main__":
    main()
