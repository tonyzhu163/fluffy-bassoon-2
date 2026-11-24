"""Read released A/B predictions and matched native outcomes; leaves per-control checksums to the native worker."""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rows(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def summary(run, preds):
    contract=json.loads((run/"panel.json").read_bytes())
    assert not contract["cpu_fixture"]
    selected={(r["index"],r["view"],r["model"]):r for r in rows(preds)
              if r["seed"]==41 and r["calibration"]=="true" and r["model"] in ("A","B","shipped")}
    by_task=defaultdict(list)
    counts=defaultdict(int)
    delta=[]
    opened=first_issued=first_physical=0
    planned=contract["rows"]
    assert len(planned)==64 and len({r["index"] for r in planned})==64
    for row in planned:
        i=row["index"]
        sign="orbit+" if row["orbit_deg"]>0 else "orbit-"
        assert row["study"]=="physical" and abs(row["orbit_deg"])==30
        ca,cb=(row["corrections"][m] for m in ("A","B"))
        for m,c in (("A",ca),("B",cb)):
            assert selected[i,sign,m]["yaw_deg"]==c["yaw_deg"]
            assert (row["nominal"]["xy_scale"],row["nominal"]["gripper_sign"])==(c["xy_scale"],c["gripper_sign"])
        def err(c):
            return abs((c["yaw_deg"]-row["true_map"]["yaw_deg"]+180)%360-180)
        delta.append(err(ca)-err(cb))
        pairfile=run/"pairs"/f"{i:02d}.json.gz"
        if not pairfile.exists():
            continue
        pair=rows(pairfile)
        assert i==pair["index"]
        arms=pair["arms"]
        assert set(arms)==set(row["corrections"])
        a,b=arms["A"],arms["B"]
        assert a["chunks"][0]["raw"]==b["chunks"][0]["raw"]
        assert a["native"]["reset"]==b["native"]["reset"]
        opened+=1
        for m,trial in arms.items():
            rec=trial["native"]
            assert rec["inferred"]==row["corrections"][m]
            assert sum(rec["native_rewards"])==rec["native_return"]==int(rec["success"])
            counts[m]+=int(rec["success"])
        first_issued+=int(a["chunks"][0]["issued"]!=b["chunks"][0]["issued"])
        first_physical+=int([r["action"] for r in a["chunks"][0]["steps"]] != [r["action"] for r in b["chunks"][0]["steps"]])
        by_task[row["task_shard"]].append(int(a["native"]["success"])-int(b["native"]["success"]))
    result=dict(planned=len(planned),matched_export_mae_difference=float(np.mean(delta)),
                completed=opened,matching_openings=opened,different_issued_opening=first_issued,
                different_physical_opening=first_physical,success=dict(counts))
    if opened==len(planned):
        diff=np.asarray([np.mean(by_task[k]) for k in sorted(by_task)])*100
        assert len(diff)==8 and all(len(ds)==8 for ds in by_task.values())
        rng=np.random.default_rng(1124)
        boots=rng.choice(diff,(20000,8)).mean(1)
        # Eight observations per task: convolutions count every possible whole-task resample.
        task_totals=np.array([sum(by_task[k]) for k in sorted(by_task)],dtype=int)
        m=task_totals.min()
        hist=np.bincount(task_totals-m)
        weights=np.array([1],dtype=np.int64)
        for _ in by_task:
            weights=np.convolve(weights,hist)
        assert weights.sum()==8**8
        ranks=np.ceil(weights.sum()*np.array([.025,.975])).astype(int)
        totals=np.searchsorted(weights.cumsum(),ranks)+8*m
        result.update(exact_paired_95_interval=(100*totals/64).tolist(),
                      per_task_pp=diff.tolist(),A_minus_B_pp=float(np.mean(diff)),
                      paired_95_interval=np.quantile(boots,[.025,.975]).tolist(),
                      A_only=sum(int(x==1) for v in by_task.values() for x in v),
                      B_only=sum(int(x==-1) for v in by_task.values() for x in v))
    return result


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("run",type=Path)
    cli.add_argument("preds",type=Path)
    args=cli.parse_args()
    print(json.dumps(summary(args.run,args.preds),indent=2))
