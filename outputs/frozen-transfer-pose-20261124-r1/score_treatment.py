"""Verify all paired wrist interventions and summarize the native rewards and measured motion."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def magnitude(v):
    return np.linalg.norm(v,axis=-1)


def rotdeg(a,b):
    return np.rad2deg(Rotation.from_matrix(a @ b.transpose(0,2,1)).magnitude())


def stats(row,key,v):
    row.update({key+"_"+k:float(x) for k,x in (("mean",np.mean(v)),("median",np.median(v)),("max",np.max(v)))})


def main(root,out):
    rows=[]
    files=sorted(root.glob("*-adaptive.json"))
    assert len(files)==12
    for f in files:
        stem=f.name.removesuffix("-adaptive.json")
        raw=np.load(root/(stem+"-raw.npz"),allow_pickle=False)
        shared=min(len(np.load(root/(stem+"-"+m+".npz"))["cmd"]) for m in ("raw","adaptive","sham"))
        for method in ("raw","adaptive","sham"):
            p=root/(stem+"-"+method+".npz")
            d=np.load(p,allow_pickle=False)
            meta=json.loads(p.with_suffix(".json").read_bytes())
            assert hashlib.sha256(p.read_bytes()).hexdigest()==meta["data_sha256"]
            assert meta["native_return"]==sum(d["reward"])==float(meta["success"])
            assert np.array_equal(d["cmd"][:,[0,1,2,6]],d["applied"][:,[0,1,2,6]])
            assert np.array_equal(raw["cmd"][:shared],d["cmd"][:shared])
            idx=slice(0,shared)
            row=dict(tape=stem,method=method,controls=len(d["cmd"]),success=meta["success"],
                     native_return=meta["native_return"],shared_controls=shared,original_controls=meta["original_tape_length"],
                     wrist_changed_controls=int(np.sum(np.any(d["cmd"][:,3:6]!=d["applied"][:,3:6],axis=1))),
                     reset_hash=meta["reset_sha"])
            stats(row,"requested_wrist_change_deg",rotdeg(d["raw_R"],d["issued_R"]))
            stats(row,"same_branch_raw_goal_tcp_mm",1000*magnitude(d["raw_p"][idx]-d["end_p"][idx]))
            stats(row,"same_branch_raw_goal_pinch_mm",1000*magnitude(d["raw_pad"][idx]-d["end_pad"][idx]))
            stats(row,"same_branch_raw_goal_ang_deg",rotdeg(d["raw_R"][idx],d["end_R"][idx]))
            stats(row,"vs_raw_absolute_tcp_mm",1000*magnitude(raw["end_p"][idx]-d["end_p"][idx]))
            stats(row,"vs_raw_absolute_pinch_mm",1000*magnitude(raw["end_pad"][idx]-d["end_pad"][idx]))
            stats(row,"vs_raw_absolute_ang_deg",rotdeg(raw["end_R"][idx],d["end_R"][idx]))
            stats(row,"vs_raw_original_pinch_goal_mm",1000*magnitude(raw["raw_pad"][idx]-d["end_pad"][idx]))
            row.update(end_pinch_xyz_m=json.dumps(d["end_pad"][-1].tolist()),end_wrist_xyzw=json.dumps(
                Rotation.from_matrix(d["end_R"][-1]).as_quat().tolist()))
            for k in ("finger_drawer","finger_bowl","finger_plate"):
                hits=np.flatnonzero(d[k]) if k in d else []
                row["first_"+k]=int(hits[0]+1) if len(hits) else ""
                row["n_"+k]=len(hits) if k in d else ""
            row["max_drawer_travel_mm"]=np.max(np.abs(d["drawer_travel_m"]))*1000 if "drawer_travel_m" in d else ""
            row["max_bowl_lift_mm"]=max(d["bowl_lift_m"])*1000 if "bowl_lift_m" in d else ""
            rows.append(row)
        assert len({r["reset_hash"] for r in rows[-3:]})==1
    with (out/"treatment.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    revision=subprocess.check_output(["git","-C",Path(__file__).parent,"rev-parse","HEAD"],text=True).strip()
    (out/"score-receipt.json").write_text(json.dumps(dict(revision=revision,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),affinity=sorted(os.sched_getaffinity(0)),
        cuda_visible=os.environ["CUDA_VISIBLE_DEVICES"],num_triplets=len(files)),indent=2)+"\n")


if __name__=="__main__":
    assert not os.environ["CUDA_VISIBLE_DEVICES"] and set(os.sched_getaffinity(0))<={46,47}
    main(Path(sys.argv[1]),Path(sys.argv[2]) if len(sys.argv)>2 else Path(sys.argv[1]))
