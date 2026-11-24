"""Conditional paired five-control physical replay from official matched reset."""
from collections import defaultdict
from pathlib import Path
import hashlib
import os
import sys

import numpy as np

from bank import BANK, FRESH, SIGNS, SUITES, TARGET_ARMS, digest, read, save
from fixture import ArmEnv, Numeric, restore, snapshot
from score import sensor_audit


def main(out, archive, seal):
    assert os.sched_getaffinity(0)=={49} and os.environ["CUDA_VISIBLE_DEVICES"]==""
    assert digest(archive/"completion.json")==seal
    complete=read(archive/"completion.json")
    assert complete["branch_condition"]
    command_path=archive/"first-five-commands.json"
    assert digest(command_path)==complete["first_five_sha256"]
    commands=read(command_path)
    assert len(commands)==16
    raw=[]
    for arm in TARGET_ARMS:
        for suite, ti in zip(SUITES,FRESH,strict=True):
            folder=BANK/f"{arm}-{suite}-{ti}"
            meta=read(folder/"case.json")
            with np.load(folder/"archive-controls.npz") as z:
                q=z["query"][:5].copy()
            with ArmEnv(arm,suite,ti) as env:
                first=env.reset(meta["query_layout"],meta["query_seed"])
                cur=first["environment_reproducibility"]
                for k in ("initial_state_fingerprint","released_row_fingerprint","bddl_sha256","scene_fingerprint","perturbation_plan_fingerprint"):
                    assert cur[k]==meta["held_reset"][k]
                sim=Numeric(env,first["observation"])
                stem=f"{arm}-{suite}-{ti}"
                _,reference=sim.run(q,np.eye(2),out/f"{stem}-nominal.npz")
                desired=1000*(reference["fk"][-1,:2]-reference["fk"][0,:2])
                size=np.linalg.norm(desired)
                axis=desired/size if size else np.zeros(2)
                for family in SIGNS:
                    A=np.array(meta["maps"][family]["A"])
                    key="|".join((f"{suite}/{ti}",arm,family,"passive"))
                    for method,xy in commands[key].items():
                        u=q.copy()
                        u[:,:2]=xy
                        restore(sim.env,sim.state)
                        np.testing.assert_array_equal(snapshot(sim.env)["x"],sim.state["x"])
                        name=f"{stem}-{family}-{method}.npz"
                        _,actual=sim.run(u,A,out/name)
                        n=len(actual["issued"])
                        np.testing.assert_array_equal(actual["issued"],u[:n])
                        assert sensor_audit(actual,A)<1e-8
                        for k in ("q","fk","time"):
                            np.testing.assert_array_equal(actual[k][0],reference[k][0])
                        movement=1000*(actual["fk"][-1,:2]-actual["fk"][0,:2])
                        comparable=n==len(reference["issued"])==5
                        along=float(movement@axis)
                        raw.append(dict(arm=arm,task=f"{suite}/{ti}",family=family,method=method,
                            file=name,sha256=digest(out/name),controls=n,nominal_controls=len(reference["issued"]),
                            terminal=bool(actual["terminal"].any()),nominal_terminal=bool(reference["terminal"].any()),
                            matched_five=comparable,delta_xy_mm=movement.tolist(),nominal_xy_mm=desired.tolist(),
                            error_mm=float(np.linalg.norm(movement-desired)) if comparable else None,
                            projection_mm=along if comparable else None,
                            orthogonal_mm=float(np.linalg.norm(movement-along*axis)) if comparable else None,
                            state_hash=hashlib.sha256(sim.state["x"].tobytes()).hexdigest()))
                save(out/"physical-rows.json",raw)
                print(stem,"finished",flush=True)
    if not all(r["matched_five"] for r in raw):
        print("Early terminations recorded; incomplete five-control endpoints remain missing.")
    summaries={}
    for field in ("error_mm","projection_mm","orthogonal_mm"):
        cells=defaultdict(list)
        for r in raw:
            name=r["method"]
            pooled="D_"+name.split("_")[1] if name.startswith("D") else name
            if r[field] is not None:
                cells[(pooled,r["task"],r["arm"],r["family"])].append(r[field])
        result={}
        for name in sorted(set(k[0] for k in cells)):
            diffs=defaultdict(list)
            for k,x in cells.items():
                if k[0]==name and ("risk",*k[1:]) in cells:
                    diffs[k[1]].append(float(np.mean(x)-np.mean(cells[("risk",*k[1:])])))
            result[name]=dict(paired_cells=sum(len(x) for x in diffs.values()),
                              paired_tasks={k:float(np.mean(v)) for k,v in diffs.items()},
                              method_mean=float(np.mean([np.mean(v) for k,v in cells.items() if k[0]==name])))
        summaries[field]=result
    save(out/"physical-summary.json",summaries)


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
